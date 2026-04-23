from __future__ import annotations
import os
import torch
from pathlib import Path
from pprint import pprint
from safetensors.torch import load_file, load_model

from model_comps.mario_action_dit import MarioActionDiT
from model_comps.vae import AutoencoderKL, VAE_models
from common.dit_training_manager import TrainingManager, ModelTrainingConfig

MARIO_DIT_CONFIG = {
    'input_h': 60,
    'input_w': 64,
    'patch_size': 2,      
    'in_channels': 4,
    'hidden_size': 256,   
    'depth': 6,
    'num_heads': 8,
    'max_frames': 32,
    'action_dim': 8
}

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


def load_mario_dit(ckpt_path: str | None) -> MarioActionDiT:
    """Initializes and loads DiT weights using MARIO_DIT_CONFIG."""
    model = MarioActionDiT(**MARIO_DIT_CONFIG)

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("No DiT checkpoint found. Training from scratch.")
        return model

    print(f"Loading DiT weights from {ckpt_path}...")
    state_dict = (load_file(ckpt_path) if ckpt_path.endswith(".safetensors") 
                  else torch.load(ckpt_path, map_location="cpu", weights_only=True))
    
    # Clean state dict if wrapped in a "model" key
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    # Note: If loading from a generic DiT/Oasis checkpoint that lacks 
    # Mario action layers, you must change strict to False.
    model.load_state_dict(state_dict, strict=True)
    
    print("DiT Load complete.")
    return model

def load_mario_vae(ckpt_path: str | None) -> AutoencoderKL:
    """Initializes and loads VAE weights using MARIO_VAE_CONFIG."""
    model = AutoencoderKL(**MARIO_VAE_CONFIG)

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("No VAE checkpoint found. Initializing VAE from scratch.")
        return model

    print(f"Loading VAE weights from {ckpt_path}...")
    state_dict = (load_file(ckpt_path) if ckpt_path.endswith(".safetensors") 
                  else torch.load(ckpt_path, map_location="cpu", weights_only=True))
    
    # Clean state dict if wrapped in a "model" key
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=True)
    
    # VAE is typically frozen during DiT training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    print("VAE Load complete and parameters frozen.")
    return model

def main() -> None:
    # =========================== CONFIG ===========================
    TRAIN_DIR = "data/mario_latents_train"
    VAL_DIR = "data/mario_latents_val"
    DIT_CKPT = None             # Path to your Mario DiT checkpoint if resuming
    VAE_CKPT = "models/mario_vae.pt" 
    SAVE_DIR = "runs/mario_world_v1"

    DEVICE = 'cuda'

    CONFIG: ModelTrainingConfig = {
        "max_noise_level": 1000,
        "clip_len": 24,
        "clip_stride": 1,
        "epochs": 10,
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "warmup_steps": 1000,
        "grad_clip_max_norm": 1.0,
        "trainable_components": 'all', # NONE means train all components
        "save_dir": SAVE_DIR
    }
    # ==============================================================

    # 1. Load Models
    dit = load_mario_dit(DIT_CKPT)
    vae = load_mario_vae(VAE_CKPT)

    # 2. Total Parameter Check (Target: < 40M)
    total_params = sum(p.numel() for p in dit.parameters())
    print(f"Total DiT Parameters: {total_params / 1e6:.2f}M")

    # 3. Training
    manager = TrainingManager(dit=dit, vae=vae, device=DEVICE)
    results = manager.train_model(TRAIN_DIR, VAL_DIR, CONFIG)

    print(f"Training complete. Final Val Loss: {results['val_losses'][-1]:.5f}")

if __name__ == "__main__":
    main()