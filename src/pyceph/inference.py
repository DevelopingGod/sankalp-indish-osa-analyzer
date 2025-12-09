import torch
import gzip
import numpy as np
from skimage import transform
from PIL import Image
from types import SimpleNamespace
import sys
import os

# --- PATH SETUP ---
# Ensure the system path can find our modules for the unpickler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- IMPORTS ---
from src.pyceph import utils
import src.pyceph.models as models_module  # We import this to patch it below

# Configuration matching your input.yml defaults
DEFAULT_CONFIG = {
    'image_scale': [800, 640],
    'use_gpu': 0,  # Set to 0 (CPU) or specific CUDA ID
    'landmarkNum': 19,
    'R2': 41,
    'model_path': 'model/12-26-22.pkl.gz'
}

@torch.no_grad()
def load_model(model_path, device='cpu'):
    """
    Loads the zipped pickle model.
    Includes patches for PyTorch 2.6+ security and directory structure changes.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # --- CRITICAL FIX 1: Namespace Patch ---
    # The pickled model expects a top-level module named 'models' containing 'fusionVGG19'.
    # Since we moved 'models.py' inside 'src/pyceph/', we map 'models' to 'src.pyceph.models'.
    sys.modules['models'] = models_module
    # ---------------------------------------

    print(f"Loading model from: {model_path}")
    
    with gzip.open(model_path, 'rb') as zipped_model:
        # --- CRITICAL FIX 2: Security Flag ---
        # PyTorch 2.6+ defaults weights_only=True, which breaks old pickles.
        # We set it to False because we trust this specific model file.
        model = torch.load(
            zipped_model, 
            map_location=torch.device(device), 
            weights_only=False 
        )
    
    model.eval()
    return model

def process_image(image_file, model, config_dict=DEFAULT_CONFIG):
    """
    Takes a file-like object (from Streamlit), preprocesses it,
    runs the model, and returns the image with landmarks and coordinates.
    """
    config = SimpleNamespace(**config_dict)
    
    # 1. Load and Resize Image
    # Streamlit passes a BytesIO object, open as PIL and convert to RGB
    original_pil = Image.open(image_file).convert('RGB')
    image = np.array(original_pil)
    
    new_h, new_w = config.image_scale
    # skimage.transform.resize returns floats in [0, 1]
    processed_image = transform.resize(image, (new_h, new_w), mode='constant')
    
    # 2. Prepare Tensor (HWC -> CHW)
    torched_image = processed_image.transpose((2, 0, 1)) 
    torched_image = torch.from_numpy(torched_image).float()
    
    # 3. Inference
    # Add batch dimension: (1, C, H, W)
    heatmaps = model(torched_image.unsqueeze(0))
    
    # 4. Post-processing (Regression Voting)
    # The model output is often a list [tensor], so we pass it directly to regression_voting
    # as per the original CephImage.process logic.
    raw_predicted_landmarks = utils.regression_voting(heatmaps, config.R2)
    
    # Ensure landmarks are on CPU for numpy conversion
    if isinstance(raw_predicted_landmarks, torch.Tensor):
        raw_predicted_landmarks = raw_predicted_landmarks.cpu()

    # 5. Map coordinates back to image scale
    landmarks = []
    # raw_predicted_landmarks is shape (1, 19, 2), so we iterate over the first batch
    for idx, [y, x] in enumerate(raw_predicted_landmarks[0]):
        y_ = int(y * new_h)
        x_ = int(x * new_w)
        landmarks.append((x_, y_))

    return processed_image, landmarks
