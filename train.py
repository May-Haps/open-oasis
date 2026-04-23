"""
Train Mario world model from scratch.

Edit the CONFIG block and run:
    python train.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from pprint import pprint

import torch

from model.dit import MarioWorldModel
from training.training_manager import TrainingManager, ModelTrainingConfig


def set_seeds(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    # =========================== CONFIG ===========================
    TRAIN_DIR = "data/processed/train"
    VAL_DIR   = "data/processed/val"
    CKPT      = None          # path to resume from, or None to train from scratch
    SAVE_DIR  = "runs/mario_v1"

    DEVICE = "cuda:0"
    SEED   = 0
    DEBUG  = False

    CONFIG: ModelTrainingConfig = {
        "max_noise_level":    1000,
        "clip_len":           32,
        "clip_stride":        8,
        "epochs":             30,
        "batch_size":         8,
        "lr":                 1e-4,
        "weight_decay":       0.01,
        "warmup_steps":       1000,
        "grad_clip_max_norm": 1.0,
        "trainable_components": [],   # empty = train all
        "save_dir":           SAVE_DIR,
    }
    ACTION_COND_DROPOUT = 0.1  # Set to 0.0 to disable action-conditioning dropout.
    # ==============================================================

    assert torch.cuda.is_available(), "CUDA is required."

    pprint({
        "train_dir": TRAIN_DIR,
        "val_dir":   VAL_DIR,
        "ckpt":      CKPT,
        "save_dir":  SAVE_DIR,
        "device":    DEVICE,
        "seed":      SEED,
        "config":    dict(CONFIG),
    })

    set_seeds(SEED)

    dit = MarioWorldModel(external_cond_dropout=ACTION_COND_DROPOUT)

    if CKPT is not None:
        state = torch.load(CKPT, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        dit.load_state_dict(state)
        print(f"Resumed from {CKPT}")

    if SAVE_DIR is not None:
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(SAVE_DIR, "run_config.json"), "w") as f:
            json.dump(
                {
                    "config": dict(CONFIG),
                    "action_cond_dropout": ACTION_COND_DROPOUT,
                    "ckpt": CKPT,
                    "device": DEVICE,
                    "seed": SEED,
                },
                f,
                indent=2,
            )

    manager = TrainingManager(dit=dit, device=DEVICE, seed=SEED, debug=DEBUG)
    results = manager.train_model(TRAIN_DIR, VAL_DIR, CONFIG)

    print("\n=== Training complete ===")
    print(f"Final train loss: {results['train_losses'][-1]:.5f}")
    print(f"Final val loss:   {results['val_losses'][-1]:.5f}")


if __name__ == "__main__":
    main()
