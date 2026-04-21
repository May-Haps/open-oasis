"""
Fine-tune Oasis DiT on processed MineRL latents.

Edit the CONFIG block at the top of main() and run:
    python run_training.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
from safetensors.torch import load_file, load_model

from dit.action_gated_dit import ActionGatedDiT
from dit.vae import AutoencoderKL, VAE_models
from common.training_manager import TrainingManager, ModelTrainingConfig

def set_seeds(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dit(ckpt_path: str | None) -> ActionGatedDiT:
    
    model = ActionGatedDiT(
        patch_size=2,
        hidden_size=1024,
        depth=16, 
        num_heads=16
    )

    if ckpt_path is None:
        print("No checkpoint provided, training from scratch.")
        return model

    print(f"Loading weights from {ckpt_path} into ActionGatedDiT...")
    
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Surgery complete. Missing keys (Action Gates): {len(missing)}")
    elif ckpt_path.endswith(".pt"):
        state = torch.load(ckpt_path, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    
    return model

def load_vae(ckpt_path: str) -> AutoencoderKL:
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    print(f"Loading VAE from {os.path.abspath(ckpt_path)}")
    if ckpt_path.endswith(".safetensors"):
        load_model(vae, ckpt_path)
    elif ckpt_path.endswith(".pt"):
        state = torch.load(ckpt_path, weights_only=True)
        vae.load_state_dict(state)
    else:
        raise ValueError(f"Unsupported VAE checkpoint format: {ckpt_path}")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae

def dump_run_metadata(save_dir: str | None, config: ModelTrainingConfig, extras: dict) -> None:
    if save_dir is None:
        return
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    meta_path = Path(save_dir) / "run_config.json"
    with meta_path.open("w") as f:
        json.dump({"training_config": dict(config), **extras}, f, indent=2)
    print(f"Wrote run config to {meta_path}")

def main() -> None:
    # =========================== CONFIG ===========================
    # Paths
    TRAIN_DIR = "data/processed_treechop_train"
    VAL_DIR = "data/processed_treechop_val"
    OASIS_CKPT = "models/oasis500m.safetensors"   # None to train from scratch
    VAE_CKPT = "models/vit-l-20.safetensors"
    SAVE_DIR = "runs/treechop_v1"          # None to disable checkpointing + rollouts

    # Runtime
    DEVICE = 'cuda:0'
    SEED = 0

    DEBUG = True

    # Training hyperparameters
    CONFIG: ModelTrainingConfig = {
        "max_noise_level": 1000,
        "clip_len": 12,
        "clip_stride": 8,
        "epochs": 3,
        "batch_size": 2,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "grad_clip_max_norm": 1.0,
        "trainable_components": ['action_routing', 'external_cond', 'adaLN_modulation'],
        "save_dir": SAVE_DIR
    }
    # ==============================================================

    assert torch.cuda.is_available(), "CUDA is required."

    print("Run configuration:")
    pprint({
        "train_dir": TRAIN_DIR,
        "val_dir": VAL_DIR,
        "oasis_ckpt": OASIS_CKPT,
        "vae_ckpt": VAE_CKPT,
        "save_dir": SAVE_DIR,
        "device": DEVICE,
        "seed": SEED,
        "training_config": dict(CONFIG),
        "debug": DEBUG
    })

    set_seeds(SEED)

    dit = load_dit(OASIS_CKPT)
    vae = load_vae(VAE_CKPT)

    dump_run_metadata(SAVE_DIR, CONFIG, extras={
        "train_dir": TRAIN_DIR,
        "val_dir": VAL_DIR,
        "oasis_ckpt": OASIS_CKPT,
        "vae_ckpt": VAE_CKPT,
        "device": DEVICE,
        "seed": SEED,
    })

    manager = TrainingManager(dit=dit, vae=vae, device=DEVICE, debug=DEBUG)
    results = manager.train_model(TRAIN_DIR, VAL_DIR, CONFIG)

    print("\n=== Training complete ===")
    print(f"Final train loss: {results['train_losses'][-1]:.5f}")
    print(f"Final val loss:   {results['val_losses'][-1]:.5f}")
    if SAVE_DIR is not None:
        print(f"Artifacts saved to: {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()