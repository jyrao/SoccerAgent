import os

def is_image(file_path):
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg'}
    if os.path.splitext(file_path)[1].lower() in image_exts:
        return True
    return False
    
def is_video(file_path):
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg'}
    if os.path.splitext(file_path)[1].lower() in video_exts:
        return True
    return False