import torch
from einops import rearrange
import sys
sys.path.append('YOUR_FOLDER_PATH_TO_SOCCERAGENT_CODEBASE/pipeline/toolbox/unisoccer')
from model.MatchVision_classifier import MatchVision_Classifier
from dataset.video_utils_siglip import read_frames_decord, set_transform
sys.path.append('YOUR_FOLDER_PATH_TO_SOCCERAGENT_CODEBASE/pipeline/toolbox')
sys.path.append('YOUR_FOLDER_PATH_TO_SOCCERAGENT_CODEBASE')
from utils.all_devices import unisoccer_device

DEVICE = unisoccer_device
print("Unisoccer on:",DEVICE)

class VideoPreprocessor:
    def __init__(self, num_frames=30, sampling_method="middle", device=DEVICE):
        """
        Initialize video preprocessor with transformation pipeline.
        
        Args:
            num_frames (int): Number of frames to sample from video
            sampling_method (str): Frame sampling method ('middle', 'uniform', etc.)
        """
        self.num_frames = num_frames
        self.sampling_method = sampling_method
        self.transform_func = set_transform()
        self.device = device

    def __call__(self, video_path):  # ✅ 添加 __call__ 方法
        return self.preprocess(video_path)

    def preprocess(self, video_path):
        """
        Process video into input tensor for models.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            torch.Tensor: Processed video tensor of shape (1, C, T, H, W)
            tuple: Additional info (frame_indices, duration)
        """
        # Read and sample frames from video
        frames, frame_indices, duration = read_frames_decord(
            video_path, 
            self.num_frames, 
            self.sampling_method
        )
        
        # Apply transformations to each frame
        frames = torch.cat([
            self.transform_func(images=frame, return_tensors="pt")["pixel_values"] 
            for frame in frames
        ], dim=0)
        
        # Rearrange dimensions: (T, C, H, W) -> (C, T, H, W)
        frames = rearrange(frames, 't c h w -> c t h w')
        
        # Add batch dimension: (C, T, H, W) -> (1, C, T, H, W)
        return frames.unsqueeze(dim=0).to(self.device)

# ----------------------------
# Classification Module
# ----------------------------
class VideoClassifier:
    def __init__(self, checkpoint_path, device=DEVICE):
        """
        Initialize the classifier with pretrained weights.
        
        Args:
            checkpoint_path (str): Path to the pretrained model checkpoint
        """
        self.classifier = MatchVision_Classifier()
        self._load_checkpoint(checkpoint_path, device)
        
    def _load_checkpoint(self, checkpoint_path, device):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        # Remove 'module.' prefix if present (for DDP trained models)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.classifier.load_state_dict(new_state_dict)
        self.classifier.to(device)
        self.classifier.eval()  # Set to evaluation mode
    
    def classify(self, video_tensor):
        """
        Run classification on preprocessed video tensor.
        
        Args:
            video_tensor (torch.Tensor): Preprocessed video tensor from VideoPreprocessor
            
        Returns:
            torch.Tensor: Classification logits
        """
        with torch.no_grad():
            return self.classifier.get_logits(video_tensor).to("cpu")


CHECKPOINT_PATH_CLASSIFICATION = "PATH_of_UniSoccer_with_Pretrained_Classification_task" # Refer to https://huggingface.co/Homie0609/UniSoccer/blob/main/pretrained_classification.pth

preprocessor = VideoPreprocessor(num_frames=30, sampling_method="middle", device=DEVICE)
classifier = VideoClassifier(CHECKPOINT_PATH_CLASSIFICATION, DEVICE)


from model.matchvoice_model_all_blocks import matchvoice_model_all_blocks
CHECKPOINT_PATH_COMMENTARY = "PATH_of_UniSoccer_with_Finetuned_Commentary_Generation_task" # Refer to https://huggingface.co/Homie0609/UniSoccer/blob/main/downstream_commentary_all_open.pth
commentary_model = matchvoice_model_all_blocks(open_llm_decoder=True, num_features=768)
state_dict = torch.load(CHECKPOINT_PATH_COMMENTARY, map_location="cpu")['state_dict']
commentary_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

commentary_model.load_state_dict(commentary_state_dict)
commentary_model.to(DEVICE)
