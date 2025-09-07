from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
# from moviepy.editor import VideoFileClip
from moviepy import *
import os

PROJECT_PATH = "PROJECT_PATH" # Replace with actual project path

def is_video_file(file_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions

def SHOT_CHANGE(query=None, material=[]):
    if len(material) == 0:
        return "No video material provided."
    if len(material) > 1:
        return "Only one video material is supported at a time. But you provided more than one."
    video_path = material[0]
    if not is_video_file(video_path):
        return "The provided file is not a valid video file."
    
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    
    scene_manager.add_detector(ContentDetector())
    
    video_manager.set_downscale_factor()
    video_manager.start()
    
    scene_manager.detect_scenes(video_manager)

    scene_list = scene_manager.get_scene_list()

    video_clip = VideoFileClip(video_path)
    if len(scene_list) == 0:
        return "No scene changes detected in the video."
    output_path = []
    change_time = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        # print(f"Scene {i+1}: Start = {start_time} sec, End = {end_time} sec")
        change_time.append(end_time)
        scene_clip = video_clip.subclipped(start_time, end_time)
        shot_path = f"{PROJECT_PATH}/YOUR_HELPER_FILES" # Replace with actual helper files path to save the temporary clips
        os.makedirs(shot_path, exist_ok=True)
        new_video_path = f"{shot_path}/scene_{i+1}.mp4"
        output_path.append(new_video_path)
        scene_clip.write_videofile(new_video_path, codec="libx264")

    video_clip.close()
    return f"Shot change detection completed. {len(scene_list)} scenes detected. The clips are saved in {output_path}. Change occurred at {change_time[:-1]} seconds."
