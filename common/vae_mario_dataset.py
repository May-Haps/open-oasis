from pathlib import Path
import os

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class MarioPixelDataset(Dataset):
    """
    Dataset for VAE training. 
    Focuses on individual frames from the 737k frame pool.
    """
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        
        # Pre-scan the directory for all PNGs
        # With 737k files, this might take 10-20 seconds on first run.
        print(f"Scanning {data_root} for frames...")
        self.frame_paths = sorted(list(self.data_root.glob("**/*.png")))
        print(f"Found {len(self.frame_paths)} frames.")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        # Use PIL to open and convert to RGB (standardizes everything to 3 channels)
        img = Image.open(self.frame_paths[idx]).convert("RGB")
        
        # Convert to numpy and transpose to (Channels, Height, Width)
        pixel_array = np.array(img).transpose(2, 0, 1) 
        
        # Normalize to [0, 1] - consistent with standard VAE training
        pixel_tensor = torch.from_numpy(pixel_array).float() / 255.0
        
        return {
            "pixels": pixel_tensor
        }