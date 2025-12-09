import torch
import gzip
import numpy as np
from skimage import transform
from PIL import Image
from types import SimpleNamespace
import sys
import os

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- IMPORTS ---
from src.pyceph import utils
import src.pyceph.models as models_module 

# Configuration matching your input.yml defaults
DEFAULT_CONFIG = {
    'image_scale': [800, 640],
    'use_gpu': 0, 
    'landmarkNum': 19,
    'R2': 41,
    'model_path': 'model/12-26-22.pkl.gz'
}

@torch.no_grad()
def load_model(model_path, device='cpu'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # --- CRITICAL FIX 1: Namespace Patch for Pickle ---
    sys.modules['models'] = models_module
    
    print(f"Loading model from: {model_path}")
    
    with gzip.open(model_path, 'rb') as zipped_model:
        # --- CRITICAL FIX 2: Security Flag for PyTorch 2.6+ ---
        try:
            model = torch.load(
                zipped_model, 
                map_location=torch.device(device), 
                weights_only=False 
            )
        except TypeError:
            # Fallback for older PyTorch versions
            model = torch.load(
                zipped_model, 
                map_location=torch.device(device)
            )
    
    model.eval()
    return model

def process_image(image_file, model, config_dict=DEFAULT_CONFIG):
    config = SimpleNamespace(**config_dict)
    
    original_pil = Image.open(image_file).convert('RGB')
    image = np.array(original_pil)
    
    new_h, new_w = config.image_scale
    processed_image = transform.resize(image, (new_h, new_w), mode='constant')
    
    torched_image = processed_image.transpose((2, 0, 1)) 
    torched_image = torch.from_numpy(torched_image).float()
    
    # --- CRITICAL FIX 3: Disable Gradients to prevent RAM Crash ---
    # This stops the "Uh oh" error on Streamlit Cloud
    with torch.no_grad():
        heatmaps = model(torched_image.unsqueeze(0))
        raw_predicted_landmarks = utils.regression_voting(heatmaps, config.R2)
    
    if isinstance(raw_predicted_landmarks, torch.Tensor):
        raw_predicted_landmarks = raw_predicted_landmarks.cpu()

    landmarks = []
    for idx, [y, x] in enumerate(raw_predicted_landmarks[0]):
        y_ = int(y * new_h)
        x_ = int(x * new_w)
        landmarks.append((x_, y_))

    return processed_image, landmarks
