import os
import torch
import random
import numpy as np
from pathlib import Path
from pprint import pprint

from model_comps.vae import AutoencoderKL
from common.vae_training_manager import VAETrainingManager, VAETrainingConfig

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_mario_vae(ckpt_path: str | None, config: dict) -> AutoencoderKL:
    """Initializes and optionally loads VAE weights."""
    model = AutoencoderKL(**config)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading VAE weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Handle wrapped state dicts
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
    else:
        print("No VAE checkpoint found. Training from scratch.")
    
    return model

def main():
    TRAIN_DIR = "../data/vae_dataset/train" # Path to your PNGs
    VAL_DIR = "../data/vae_dataset/val"
    VAE_CKPT = None # Path to .pt if resuming
    SAVE_DIR = "runs/mario_vae_v1"

    # Runtime
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = 'cuda'
    SEED = 42

    # VAE Architecture
    MARIO_VAE_CONFIG = {
        'latent_dim': 4,
        'input_height': 240,
        'input_width': 256,
        'patch_size': 4,
        'enc_dim': 256,
        'enc_depth': 4,
        'enc_heads': 8,
        'dec_dim': 256,
        'dec_depth': 4,
        'dec_heads': 8,
        'use_variational': True
    }


    # Training Hyperparameters
    TRAIN_CONFIG: VAETrainingConfig = {
        'epochs': 5,
        'batch_size': 20,      # 256x240 is small, can likely go higher if VRAM allows
        'lr': 1e-4,
        'kl_weight': 1e-6,     # Beta: keep low to prevent blurry reconstructions
        'save_dir': SAVE_DIR
    }
    # ==============================================================

    set_seeds(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("--- VAE Training Start ---")
    pprint({
        "Config": MARIO_VAE_CONFIG,
        "Hyperparams": TRAIN_CONFIG,
        "Device": DEVICE
    })

    # 1. Initialize Model
    vae = load_mario_vae(VAE_CKPT, MARIO_VAE_CONFIG)

    # 2. Setup Manager
    manager = VAETrainingManager(vae, device=DEVICE)

    # 3. Start Training
    manager.train_vae(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        config=TRAIN_CONFIG
    )

if __name__ == "__main__":
    main()