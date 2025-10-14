
import base64
from PIL import Image
import json
import os
import csv
import subprocess
import re
from tqdm import tqdm
import cv2


def extract_option(reply):
    if reply:
        if reply[0] in 'ABCD':
            mapping = {'A': 'O1', 'B': 'O2', 'C': 'O3', 'D': 'O4'}
            reply = mapping[reply[0]] + reply[1:]

    options = re.findall(r'O[1-9]', reply)
    unique_options = list(set(options))
    
    if len(unique_options) == 1:
        return unique_options[0]
    else:
        return None


def count_csv_rows(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    return row_count - 1


def compress_video(input_path, output_path, crf=28):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    command = [
        "ffmpeg", "-i", input_path, "-vcodec", "libx264", "-crf", str(crf), output_path
    ]
    subprocess.run(command, check=True)
    return output_path


def compress_image(image_path, output_path, quality=50):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with Image.open(image_path) as img:
        img.save(output_path, format="JPEG", quality=quality)
    return output_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image =  base64.b64encode(image_file.read()).decode('utf-8')
    # print(f"Base64 image size: {sys.getsizeof(base64_image)} bytes")
    return base64_image
    

def encode_video(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def sort_files_by_number_in_name(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    stride = len(files) // 25
    files = files[::stride]
    return files


def videolist2imglist(video_path, num):
    base64_videos = []
    assign = []
    video_num = len(video_path)
    for i in range(video_num):
        assign.append(num / video_num)
    j = 0
    for video in video_path:
        video = cv2.VideoCapture(video)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        base64_images = []
        fps = assign[j]
        interval = int(frame_count / fps)
        for i in range(0, frame_count, interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                _, buffer = cv2.imencode(".jpg", frame)
                base64_images.append(base64.b64encode(buffer).decode("utf-8"))
        base64_videos.append(base64_images)
        j += 1
    return base64_videos