from qwen_vl_utils import process_vision_info
import torch
from utils.vlm_distribution import vlm_model, vlm_processor
import torch
from PIL import Image
import cv2
import os

PROJECT_PATH = "YOUR_PROJECT_PATH"  # Replace with your actual project path

import subprocess
def compress_video(input_video_path, output_video_path, target_width=224, target_height=400, target_fps=1, codec="libx264"):
    """
    Compresses the video by reducing the resolution, frame rate and compressing the video with ffmpeg.
    """
    command = [
        "ffmpeg", "-i", input_video_path,  # input video path
        "-vf", f"scale={target_width}:{target_height}",  # scale to target resolution
        "-r", str(target_fps),  # set frame rate
        "-c:v", codec,  # use libx264 codec for compression
        "-crf", "23",  # constant rate factor for quality (lower means better quality)
        "-preset", "fast",  # preset for compression speed/quality tradeoff
        "-y",  # overwrite output file without asking
        output_video_path  # output video path
    ]
    
    subprocess.run(command, check=True)


def chat_video(input_text, Instruction, video_path, model=vlm_model, processor=vlm_processor, max_tokens=512):
    conversation = [
        {"role": "system", "content": Instruction},
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
                "max_pixels": 224 * 400,
                "fps": 1.0,
            }
        )
    conversation[1]["content"].append(
        {"type": "text", "text": input_text}
    )
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    try:
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text

    except torch.cuda.OutOfMemoryError:
        print("CUDA memory overflow detected. Compressing video and retrying...")
        compressed_video_paths = []
        for i, video in enumerate(video_path):
            compressed_video_path = f"HELPER_FILE/{i}.mp4" # Replace with your temporary path for compressed video
            os.makedirs(os.path.dirname(compressed_video_path), exist_ok=True)
            compress_video(video, compressed_video_path, target_width=168, target_height=300, target_fps=0.5)
            compressed_video_paths.append(compressed_video_path)
        
        conversation[1]["content"] = []
        for i in range(len(compressed_video_paths)):
            conversation[1]["content"].append(
                {
                    "type": "video",
                    "video": "file://" + compressed_video_paths[i],
                    "max_pixels": 224 * 400,
                    "fps": 1,
                }
            )
        conversation[1]["content"].append({"type": "text", "text": input_text})
        
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text


def REPLAY_GROUNDING(query=None, material=[]):
    instruction = "You are a football expert. You are given five video clips, with the first being a replay. Your task is to identify the clip being replayed from the next four."
    text = "There are five video clips in total. The first clip is a replay. Which of the other four clips is the one it is replaying? Reply with 'the 1st clip' or 'the 2nd clip' or 'the 3rd clip' or 'the 4th clip'."
    response = chat_video(text, instruction, material)
    return response

