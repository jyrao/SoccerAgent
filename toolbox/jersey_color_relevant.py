from qwen_vl_utils import process_vision_info
import sys, torch
sys.path.append('YOUR_FOLDER_PATH_TO_SOCCERAGENT_CODEBASE/pipeline/toolbox')
from utils.vlm_distribution import vlm_model, vlm_processor


def JERSEY_COLOR_VLM(query, material, vlm_model=vlm_model, vlm_processor=vlm_processor):

    material_path = material[0] if material else None
    if not material_path:
        raise ValueError("Material list is empty")
    
    file_extension = material_path.split('.')[-1].lower()
    media_type = None
    media_key = None
    
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp']
    video_extensions = ['mp4', 'mkv', 'avi', 'mov', 'flv', 'wmv', 'webm']
    
    if file_extension in image_extensions:
        media_type = "image"
        media_key = "image"
    elif file_extension in video_extensions:
        media_type = "video"
        media_key = "video"
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": media_type,
                    media_key: material_path,
                },
                {
                    "type": "text", 
                    "text": f"Tell me about: the jersey color of two teams in this soccer video clip, then tell me about: {query}"
                },
            ],
        }
    ]
    
    with torch.no_grad():
        text = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(vlm_model.device)
        
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0] if output_text else ""