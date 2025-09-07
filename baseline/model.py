import os
import torch
from transformers import AutoProcessor
import json
import csv
from tqdm import tqdm
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import openai
from openai import OpenAI
from utils import encode_image, encode_video, extract_option, count_csv_rows, sort_files_by_number_in_name, compress_video, compress_image, videolist2imglist


class BaselineModel:
    def __init__(self):
        pass

    def test_qa(self, json_file, csv_file):
        raise NotImplementedError

    def cal_acc(self, csv_file, json_file):
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            csv_data = list(reader)

        with open(json_file, mode='r', encoding='utf-8') as f:
            json_data = json.load(f)

        if len(csv_data) != len(json_data):
            raise ValueError(f"The number of rows in the output file ({len(csv_data)}) does not match the number of items in the gt file ({len(json_data)}).")

        correct_count = 0
        for i in range(len(csv_data)):
            csv_answer = csv_data[i]['answer']
            num = csv_data[i]['option_num']
            json_answer = json_data[i]['closeA']
            
            if csv_answer == json_answer:
                correct_count += 1
            elif csv_answer == None:
                correct_count += 1 / int(num)

        accuracy = correct_count / len(csv_data) * 100
        return accuracy


class QwenVL(BaselineModel):
    def __init__(self, model_path):
        self.instruction = f"""
        You are a football expert. You are provided with a question 'Q' and multiple options like: 'O1', 'O2', 'O3', 'O4'...
        Please answer the question with one option that best matches the question (replay with 'O1', 'O2', 'O3', 'O4'...). 
        Do not include any other text or explanations.
        """

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def chat_video(self, input_text, video_path, max_tokens=512):
        conversation = [
            {"role": "system", "content": self.instruction},
            {
                "role": "user",
                "content": []
            },
        ]
        for i in range(len(video_path)):
            conversation[1]["content"].append(
                {
                    "type": "video", 
                    "video": "file://" + video_path[i],
                    "max_pixels": 360 * 640,
                    "fps": 1.0,
                }
            )
        conversation[1]["content"].append(
            {"type": "text", "text": input_text}
        )
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text
    
    def chat_img(self, input_text, image_path, max_tokens=512):
        conversation = [
            {"role": "system", "content": self.instruction},
            {
                "role": "user",
                "content": []
            },
        ]
        for i in range(len(image_path)):
            conversation[1]["content"].append(
                {"type": "image", "image": "file://" + image_path[i] }
            )
        conversation[1]["content"].append(
            {"type": "text", "text": input_text}
        )
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text
    
    def test_qa(self, json_file, csv_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        num = count_csv_rows(csv_file)
        data = data[num:]
        if data == []:
            return
        for item in tqdm(data):
            question = item["Q"]
            options = []
            for i in range (1, 10):
                options.append(item.get(f"O{i}", None))
            option_num = 0
            prompt = f"Q: {question}\n"
            for i in range(1, 10):
                if options[i-1]:
                    prompt += f"O{i}: {options[i-1]}\n"
                    option_num += 1
            
            material_path = item["materials"]
            if material_path:
                if not os.path.isfile(material_path[0]):
                    material_path = sort_files_by_number_in_name(material_path[0])
                image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic'}
                video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg'}
                if os.path.splitext(material_path[0])[1].lower() in image_exts:
                    reply = self.chat_img(prompt, material_path)
                elif os.path.splitext(material_path[0])[1].lower() in video_exts:
                    reply = self.chat_video(prompt, material_path)
            else:
                reply = self.chat_img(prompt, [])
            answer = extract_option(reply)
            with open(csv_file, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([answer, option_num])
        return

    
class GPT4o(BaselineModel):
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.instruction = f"""
        You are a football expert. You are provided with a question 'Q' and four options 'O1', 'O2', 'O3', and 'O4'.
        Please answer the question with one option that best matches the question (replay with 'O1', 'O2', 'O3', or 'O4'). 
        Do not include any other text or explanations.
        """
        self.model = "gpt-4o"

    def chat_img(self, input_text, image_path, max_tokens=512):
        try:
            base64_images = []
            for image in image_path:
                base64_image = encode_image(image)
                base64_images.append(base64_image)
            
            messages = [
                {"role": "system", "content": self.instruction},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": input_text},
                    ]
                }
            ]
            for image in base64_images:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}"
                    }
                })
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except openai.APIStatusError as e:
            if "413" in str(e):
                print("Image too large, compressing and retrying...")
                compressed_images = []
                i = 0
                for image in image_path:
                    i += 1
                    compressed_image_path = compress_image(image, f"tmp_file/compressed_image{i}.jpg", quality=50)
                    compressed_images.append(compressed_image_path)
                return self.chat_img(input_text, compressed_images, max_tokens)
            else:
                raise e

    def chat_video(self, input_text, video_path, max_tokens=512):
        try:
            base64_videos = []
            base64_videos = videolist2imglist(video_path, 25)
            
            messages = [
                {"role": "system", "content": self.instruction},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": input_text},
                    ]
                }
            ]
            i = 0
            for video in base64_videos:
                intro = f"These images are uniformly captured from the {i}th video in chronological order. There are a total of {len(video)} pictures."
                messages[1]["content"].append({
                    "type": "text",
                    "text": intro
                })
                for image in video:
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}"
                        }
                    })
                i += 1
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except openai.APIStatusError as e:
            if "413" in str(e):
                print("video too large, compressing and retrying...")
                compressed_videos = []
                i = 0
                for video in video_path:
                    i += 1
                    compressed_video_path = compress_video(video, f"tmp_file/compressed_video{i}.mp4")
                    compressed_videos.append(compressed_video_path)
                return self.chat_video(input_text, compressed_videos, max_tokens)
            else:
                raise e

    def test_qa(self, json_file, csv_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        num = count_csv_rows(csv_file)
        data = data[num:]
        if data == []:
            return
        for item in tqdm(data):
            question = item["Q"]
            options = []
            for i in range (1, 10):
                options.append(item.get(f"O{i}", None))
            option_num = 0
            prompt = f"Q: {question}\n"
            for i in range(1, 10):
                if options[i-1]:
                    prompt += f"O{i}: {options[i-1]}\n"
                    option_num += 1
            
            material_path = item["materials"]
            if material_path:
                if not os.path.isfile(material_path[0]):
                    material_path = sort_files_by_number_in_name(material_path[0])
                image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic'}
                video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg'}
                if os.path.splitext(material_path[0])[1].lower() in image_exts:
                    reply = self.chat_img(prompt, material_path)
                elif os.path.splitext(material_path[0])[1].lower() in video_exts:
                    reply = self.chat_video(prompt, material_path)
            else:
                reply = self.chat_img(prompt, [])
            answer = extract_option(reply)
            with open(csv_file, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([answer, option_num])
        return
