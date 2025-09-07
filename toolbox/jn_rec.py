import os
from .utils.jn import run
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import re
import csv

PROJECT_PATH = "YOUR_PROJECT_PATH"  # Replace with your actual project path
def JERSEY_NUMBER_RECOGNITION(query=None, material=[]):
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    cudnn.benchmark = True

    root_dir = material[0]
    material = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                material.append(os.path.join(root, file))
    material.sort(key=lambda x: int(re.search(r'(\d+)\.jpg', x).group(1)))
    image_list = []
    for path in material:
        img = Image.open(path)
        image_list.append(img)

    model_path = "YOUR_MODEL_PATH/legibility_resnet34_soccer_20240215.pth" # Replace with your actual model path
    qwen_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    ans, result = run(device, image_list, model_path, qwen_path, threshold=0.5)
    ans = -1 if ans == None else ans
    ans = f"The jersey number in the pictures is {ans}."
    return ans
