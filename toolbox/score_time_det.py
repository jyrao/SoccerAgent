import easyocr, sys
import re
import os
import time
from urllib.error import URLError
sys.path.append('YOUR_FOLDER_PATH_TO_SOCCERAGENT_CODEBASE/pipeline/toolbox')
from vlm import VLM

def extract_timestamp(image, max_retries=3):

    reader = None
    for attempt in range(max_retries):
        try:
            reader = easyocr.Reader(['en'], download_enabled=True, gpu=False)
            break
        except URLError as e:
            print(f"Download failed easyocr model, Attempt: {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                return "Cannot download easyocr model."
            time.sleep(5)  
    
    try:
        results = reader.readtext(image)
        
        time_pattern = re.compile(r'(\d{1,2})[:.-](\d{2})')
        
        for (bbox, text, confidence) in results:
            match = time_pattern.search(text)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                
                if 0 <= minutes <= 90 and 0 <= seconds < 60:
                    return f"The timestamp detected by easy easyocr is {minutes} minutes {seconds} seconds."
        
        return "Cannot find timestamp via easyocr."
    
    except Exception as e:
        return f"Failed in processing with pic: {e}"
    

import os
from datetime import datetime
import cv2

def SCORE_TIME_DETECTION(query, material):
    """
    Detect timestamp and scoreboard information in football broadcast footage
    
    Args:
        query: Query prompt
        material: List containing file paths (typically length 1)
    
    Returns:
        Text result from VLM model
    """
    if not material or len(material) == 0:
        return "Error: No material provided"
    
    file_path = material[0]
    
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = file_path
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return "Error: Failed to extract middle frame from video"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("CACHE_FOLDER_PATH", exist_ok=True)
        image_path = f"CACHE_FOLDER_PATH/SCORE_TIME_DETECTION_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
    else:
        return "Error: Unsupported file format"
    
    timestamp_info = extract_timestamp(image_path)
    
    vlm_prompt = f"We first used easyocr to analyze this football footage and obtained: {timestamp_info}. {query}"
    
    vlm_result = VLM(vlm_prompt, [image_path])
    
    return vlm_result
