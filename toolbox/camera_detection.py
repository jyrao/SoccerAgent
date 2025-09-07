from openai import OpenAI
import base64
import re
import os
import cv2
from io import BytesIO
from PIL import Image
from collections import Counter

PROJECT_PATH = "PROJECT_PATH" # Replace with actual project path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def extract_camera_position(reply):
    options = [
        "Main camera center", "Close-up player or field referee", "Close-up side staff", 
        "Main camera left", "Main behind the goal", "Close-up behind the goal", 
        "Spider camera", "Main camera right", "Public", "Goal line technology camera", 
        "Close-up corner", "Inside the goal", "Other"
    ]
    
    pattern = "|".join([re.escape(option) for option in options])
    
    match = re.search(pattern, reply)
    
    if match:
        return match.group(0)
    else:
        return "None"


def send_request_with_background(prompt, img_64=None, background=[], api_key="YOUR_API_KEY"):
    client = OpenAI(
        base_url='YOUR_API_BASE_URL',  # Replace with your OpenAI API base URL
        api_key=api_key
    )
    messages = background.copy()
    messages.append({"role": "user", "content": []})
    messages[-1]["content"].append({"type": "text", "text": prompt})
    if img_64:
        base64_image = img_64
        messages[-1]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                }
            }
        )
    # print(f"request size in bytes: {sys.getsizeof(messages)}")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=512,
    )

    model_reply = response.choices[0].message.content
    return model_reply


def CAMERA_DETECTION(query=None, material=[]):
    example_path = f"{PROJECT_PATH}/pipeline/toolbox/utils/example_tiny" # Example images for learning camera positions
    example_img = [os.path.join(example_path, f) for f in os.listdir(example_path)]
    example_img = sorted(example_img)
    camera_position = ["Close-up behind the goal", "Close-up corner", "Close-up player or field referee", "Close-up side staff", "Goal line technology camera", "Inside the goal", "Main behind the goal", "Main camera center", "Main camera left", "Main camera right", "Other", "Public", "Spider camera"]

    learn_prompt = "I want you to help me identify the camera position of a football game photo. Now I will give you some example images, each of which corresponds to a specific camera position. Please learn the characteristics of these images for classification of new photos."

    history = [{"role": "system", "content": learn_prompt}]
    for i in range(len(example_img)):
        content = [{"type": "text", "text": f"The camera position corresponding to this photo is: {camera_position[i]}"}]
        base64_image = encode_image(example_img[i])
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                }
            }
        )
        history.append({"role": "user", "content": content})

    img_path = material[0]
    file_extension = os.path.splitext(img_path)[1].lower()
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']

    if file_extension in image_extensions:
        base64_image = encode_image(img_path)
        ask_prompt = "What is the camera position in this picture? The answer should be chosen from the following options: [Main camera center, Close-up player or field referee, Close-up side staff, Main camera left, Main behind the goal, Close-up behind the goal, Spider camera, Main camera right, Public, Goal line technology camera, Close-up corner, Inside the goal, Other]."
        reply = send_request_with_background(ask_prompt, base64_image, history)
        ans = extract_camera_position(reply)

        return f"The camera position in the photo is: {ans}."
    
    elif file_extension in video_extensions:
        video_path = img_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannnot open video: {video_path}")
        frames_base64 = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 10 != 0:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            frames_base64.append(img_base64)
        cap.release()
        ask_prompt = "What is the camera position in this picture? The answer should be chosen from the following options: [Main camera center, Close-up player or field referee, Close-up side staff, Main camera left, Main behind the goal, Close-up behind the goal, Spider camera, Main camera right, Public, Goal line technology camera, Close-up corner, Inside the goal, Other]."
        reply = []
        for frame in frames_base64:
            reply.append(send_request_with_background(ask_prompt, frame, history))
        ans = [extract_camera_position(r) for r in reply]
        count = Counter(ans)
        most_common_str, most_common_count = count.most_common(1)[0]
        return f"The camera position in the video is: {most_common_str}."
