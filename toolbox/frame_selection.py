
import os
import cv2
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel
import ffmpeg
import random

def select_rand_frame(video_path):
    output_dir = "YOUR_CACHE_FOLDER"
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"FRAME_SELECTION_{timestamp}.jpg")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"The video file cannot be opened: {video_path}")
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            raise ValueError("No available frame in the video")
        
        random_frame = random.randint(0, total_frames - 1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Random frames cannot be read")
        
        cv2.imwrite(output_path, frame)
        
        return output_path
        
    finally:
        cap.release()


def FRAME_SELECTION(query, material, output_dir="YOUR_FOLDER_PATH_TO_SOCCERAGENT_CODEBASE/log/"):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)
    
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    video_path = material[0]
    best_similarity = -np.inf
    best_frame = None
    
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    process = (
        ffmpeg.input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=1)  # 每秒1帧
        .run_async(pipe_stdout=True, quiet=True)
    )
    
    frame_count = 0
    while True:
        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            break
            
        frame = Image.frombytes('RGB', (width, height), in_bytes)
        frame_count += 1
        
        inputs = processor(text=[query], images=frame, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            similarity = model(**inputs).logits_per_image.item()
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_frame = frame.copy() 
    
    process.wait()
    
    if best_frame:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"FRAME_SELECTION_{timestamp}.jpg")
        best_frame.save(output_path, quality=95, subsampling=0)
        return f"The selected frame according to the prompt is save in {output_path}."
    try:
        output_path = select_rand_frame(material[0])
        return f"Cannot match the exact frame, so random selected a frame and saved in {output_path}."
    except:
        return "Failed in selecting frame!"