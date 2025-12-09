import torch
import numpy as np
from skimage import transform
from PIL import Image
from types import SimpleNamespace
import os
import sys

# Import the new HRNet class
from src.pyceph.hrnet import get_hrnet_w32
from src.pyceph import utils

# Updated Config
DEFAULT_CONFIG = {
    'image_scale': [768, 768], 
    'use_gpu': 0,
    'landmarkNum': 19,
    'R2': 41, 
    'model_path': 'model/hrnet_model.pth' 
}

@torch.no_grad()
def load_model(model_path, device='cpu'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = get_hrnet_w32()
    print(f"Loading model from {model_path}...")
    
    # Load with security fix
    try:
        checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=torch.device(device))
    
    # Handle dictionary structure
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint 
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

def process_image(image_file, model, config_dict=DEFAULT_CONFIG):
    config = SimpleNamespace(**config_dict)
    
    original_pil = Image.open(image_file).convert('RGB')
    image = np.array(original_pil)
    
    # Resize
    new_h, new_w = config.image_scale
    processed_image = transform.resize(image, (new_h, new_w), mode='constant')
    
    # Tensor Prep
    torched_image = processed_image.transpose((2, 0, 1)) 
    torched_image = torch.from_numpy(torched_image).float()
    
    # --- CRITICAL FIX: Disable Gradients to save RAM ---
    with torch.no_grad():
        heatmaps = model(torched_image.unsqueeze(0))
        # Post-Processing
        raw_predicted_landmarks = utils.regression_voting([heatmaps], config.R2)
    
    if isinstance(raw_predicted_landmarks, torch.Tensor):
        raw_predicted_landmarks = raw_predicted_landmarks.cpu()

    landmarks = []
    for idx, [y, x] in enumerate(raw_predicted_landmarks[0]):
        y_ = int(y * new_h)
        x_ = int(x * new_w)
        landmarks.append((x_, y_))

    return processed_image, landmarks
