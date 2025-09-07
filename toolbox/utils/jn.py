import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.backends import cudnn

class LegibilityClassifier34(nn.Module):
    def __init__(self, train=False,  finetune=False):
        super().__init__()
        self.model_ft = models.resnet34(pretrained=True)
        if finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)
        self.model_ft.fc.requires_grad = True
        self.model_ft.layer4.requires_grad = True

    def forward(self, x):
        x = self.model_ft(x)
        x = F.sigmoid(x)
        return x


class Legibility():
    def __init__(self, legibility_model_path, device):
        self.device = device
        cudnn.benchmark = True

        # Initialize transforms
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize model
        self.model = LegibilityClassifier34()
        state_dict = torch.load(legibility_model_path, map_location=device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def process(self, image_list, threshold=0.5):
        img_crops_PIL = image_list
        img_crops_PIL_transformed = [self.transforms(img_crop_PIL) for img_crop_PIL in img_crops_PIL]
        img_crops_PIL_transformed = torch.stack(img_crops_PIL_transformed).to(self.device)

        # Get legibility scores
        outputs = self.model(img_crops_PIL_transformed)
        if threshold > 0:
            outputs = (outputs>threshold).float()
        else:
            outputs = outputs.float()
        legibility_scores = outputs[:,0].float().cpu().detach().numpy()

        return list(legibility_scores)


class QWEN2_5VL_OCR_BATCH():
    def __init__(self, qwen_model_path, device):
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model_path = qwen_model_path
        self.save_jersey_number_full_detection = True
        self.use_legibility_filter = True

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.processor.tokenizer.padding_side = "left"
        self.device = device

        self.text_prompt = "Analyze this image and determine if the player is facing away from the camera. If the player is facing away, output the jersey number on their back. If the player is not facing away from the camera, output 'No'."

    def no_jersey_number(self):
        return None, 0

    def extract_numbers(self, text):
        if text.strip() == "?":
            return None
        number = ''
        for char in text:
            if char.isdigit():
                number += char
        return number if number != '' else None

    @torch.no_grad()
    def process(self, batch, threshold=0.5, batch_size=64):
        real_bs = len(batch['imgs'])
        jersey_number_detection = [None] * real_bs
        jersey_number_confidence = [0.0] * real_bs
        jersey_number_full_detection = [''] * real_bs

        # Create a list of valid indices based on legibility filter
        idxs = []
        if self.use_legibility_filter:
            for i, score in enumerate(batch['legibility_score']):
                if score >= threshold:
                    idxs.append(i)
        else:
            idxs = list(range(len(batch['imgs'])))

        if len(idxs) > 0:
            print(f"Processing {len(idxs)} images")
            imgs = [batch['imgs'][idx] for idx in idxs] 
            num_batches = (len(idxs) + batch_size - 1) // batch_size
            all_generated_text = []

            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, len(idxs))
                batch_imgs = imgs[batch_start:batch_end]

                messages = [
                        [
                            {
                                "role": "user", 
                                "content":[
                                    {
                                        "type": "image",
                                        "image": img
                                    }, 
                                    {"type": "text", "text": self.text_prompt}
                                ]
                            }
                        ] for img in batch_imgs
                    ]

                texts = [
                    self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    for msg in messages
                ]
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                all_generated_text.extend(output_texts)

            for idx, output_text in zip(idxs, all_generated_text):
                jersey_number = self.extract_numbers(output_text)
                jersey_number_detection[idx] = jersey_number
                jersey_number_confidence[idx] = 1.0 if jersey_number is not None else 0.0
                if self.save_jersey_number_full_detection:
                    jersey_number_full_detection[idx] = output_text

        batch['jersey_number_detection'] = jersey_number_detection
        batch['jersey_number_confidence'] = jersey_number_confidence
        if self.save_jersey_number_full_detection:
            batch['jersey_number_full_detection'] = jersey_number_full_detection

        return batch


class MajorityVoteTrackletFilter():
    def __init__(self):
        pass

    def select_highest_voted_att(self, values, confidences):
        value_counts = {}
        for v, c in zip(values, confidences):
            if v is not None:
                if v not in value_counts:
                    value_counts[v] = 0
                value_counts[v] += c

        if len(value_counts) == 0:
            return None

        max_value = max(value_counts, key=value_counts.get)
        return max_value

    @torch.no_grad()
    def process(self, tracklet):
        attribute_detection = tracklet['jersey_number_detection']
        attribute_confidence = tracklet['jersey_number_confidence']

        # Convert to list for easier manipulation
        detection_list = list(attribute_detection)
        confidence_list = list(attribute_confidence)

        # First pass: filter out values without 3 consecutive matches
        filtered_detection = detection_list.copy()
        for i in range(len(detection_list)):
            # Get window of 3 centered at current position
            start = max(0, i-1)
            end = min(len(detection_list), i+2)
            window = detection_list[start:end]

            # Check if current value matches all values in window
            current_val = detection_list[i]
            if current_val is None or not all(v == current_val for v in window):
                filtered_detection[i] = None

        # Create filtered lists removing None values
        final_detection = []
        final_confidence = []
        for d, c in zip(filtered_detection, confidence_list):
            if d is not None:
                final_detection.append(d)
                final_confidence.append(c)

        # Get majority vote from filtered values
        if final_detection:
            attribute_value = self.select_highest_voted_att(final_detection, final_confidence)
        else:
            attribute_value = None

        tracklet['jn_final'] = [attribute_value] * len(detection_list)

        return attribute_value, tracklet


def run(device, img_list, model_path, qwen_path, threshold=0.5):
    lc = Legibility(model_path, device)
    lc_score = lc.process(img_list, threshold)
    # print(lc_score)

    qwen = QWEN2_5VL_OCR_BATCH(qwen_model_path=qwen_path, device=device)
    results = qwen.process({"imgs": img_list, "legibility_score": lc_score}, threshold)
    # print(results)

    filter = MajorityVoteTrackletFilter()
    ans, results = filter.process(results)

    # print(ans)
    return ans, results
