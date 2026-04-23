"""
Train CoinRun world model with WandB logging and checkpointing.

Setup:
    pip install wandb array-record grain-nightly

    # Preprocess (run once per split):
    python data/preprocess_coinrun.py --input-dir data/coinrun_raw/train --output-dir data/coinrun_processed/train
    python data/preprocess_coinrun.py --input-dir data/coinrun_raw/val   --output-dir data/coinrun_processed/val

Run:
    python train_coinrun.py

Resume from checkpoint:
    python train_coinrun.py --resume runs/coinrun_v1/ckpt_step_10000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from einops import rearrange
from torch import autocast
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data.dataset_coinrun import CoinRunDataset
from model.dit import CoinRunWorldModel
from model.utils import sigmoid_beta_schedule
from training.noise_scheduler import NoiseScheduler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    # Paths
    "train_dir":  "data/coinrun_processed/train",
    "val_dir":    "data/coinrun_processed/val",
    "save_dir":   "runs/coinrun_v1",

    # Model
    "max_noise_level": 1000,

    # Data
    "clip_len":   32,
    "clip_stride": 8,

    # Training
    "epochs":          10,       # stop early if time budget hit; checkpoints preserve progress
    "batch_size":      256,
    "lr":              1e-4,
    "weight_decay":    0.01,
    "warmup_steps":    2000,
    "grad_clip":       1.0,

    # Logging / saving
    "ckpt_every_steps":    2000,   # checkpoint every ~8-10 min on H100
    "rollout_every_steps": 5000,   # generate + log video every ~20-25 min
    "log_every_steps":     50,     # wandb loss every N steps
    "val_every_epoch":     True,

    # Rollout generation
    "ddim_steps":          10,
    "n_prompt_frames":     1,
    "rollout_frames":      16,     # frames to generate per sample
    "n_rollout_samples":   4,      # how many val clips to visualise

    # WandB
    "wandb_project": "coinrun-world-model",
    "wandb_entity":  None,         # set to your wandb username/org or leave None
}


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_rollout(
    model: nn.Module,
    prompt_frames: torch.Tensor,    # [B, n_prompt, 3, 64, 64] in [0,1]
    actions: torch.Tensor,          # [B, total_frames, 15]
    noise_scheduler_alphas: torch.Tensor,
    device: str,
    ddim_steps: int = 10,
    total_frames: int = 16,
    n_prompt: int = 1,
    noise_abs_max: float = 20.0,
    stabilization_level: int = 15,
) -> torch.Tensor:
    """Returns uint8 [B, total_frames, H, W, C]."""
    was_training = model.training
    model.eval()

    # scale to [-1, 1] for diffusion
    x = prompt_frames.to(device) * 2 - 1
    actions = actions.to(device)
    B = x.shape[0]

    alphas_cumprod = noise_scheduler_alphas
    noise_range = torch.linspace(-1, alphas_cumprod.shape[0] - 1, ddim_steps + 1)

    for i in range(n_prompt, total_frames):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_steps + 1)):
            t_ctx  = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=device)
            t      = torch.full((B, 1), noise_range[noise_idx],     dtype=torch.long, device=device)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            t      = torch.cat([t_ctx, t], dim=1)[:, start_frame:]
            t_next = torch.cat([t_ctx, t_next], dim=1)[:, start_frame:]

            x_curr = x[:, start_frame:]
            with autocast("cuda", dtype=torch.bfloat16):
                v = model(x_curr, t, actions[:, start_frame : i + 1])

            ac = alphas_cumprod[t]
            x_start = ac.sqrt() * x_curr - (1 - ac).sqrt() * v
            x_noise  = ((1 / ac).sqrt() * x_curr - x_start) / (1 / ac - 1).sqrt()

            an = alphas_cumprod[t_next]
            an[:, :-1] = 1.0
            if noise_idx == 1:
                an[:, -1:] = 1.0
            x[:, -1:] = an.sqrt() * x_start + x_noise * (1 - an).sqrt()

    out = (x.clamp(-1, 1) + 1) / 2                              # [B, T, 3, 64, 64] float
    out = rearrange(out, "b t c h w -> b t h w c")
    out = (out * 255).byte().cpu()

    if was_training:
        model.train()
    return out


def frames_to_grid(generated: torch.Tensor, ground_truth: torch.Tensor) -> np.ndarray:
    """
    Build a comparison grid: top row = ground truth, bottom row = generated.
    generated / ground_truth: uint8 [B, T, H, W, C]
    Returns numpy [H_grid, W_grid, 3] uint8.
    """
    B, T, H, W, C = generated.shape
    # pick first sample, all frames
    gen_row = generated[0]  # [T, H, W, C]
    gt_row  = ground_truth[0]

    def to_chw(frames):
        return [torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in frames.numpy()]

    gen_tensors = to_chw(gen_row)
    gt_tensors  = to_chw(gt_row)

    grid = make_grid(gt_tensors + gen_tensors, nrow=T, padding=2)
    return grid.permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    step: int,
    val_loss: float | None,
    config: dict,
) -> None:
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch":     epoch,
        "step":      step,
        "val_loss":  val_loss,
        "config":    config,
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, int, float | None]:
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["step"], ckpt.get("val_loss")


# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------
def train_step(
    model: nn.Module,
    batch: dict,
    noise_scheduler: NoiseScheduler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
    grad_clip: float,
) -> float:
    x0      = batch["frames"].to(device)
    actions = batch["actions"].to(device)
    B, T    = x0.shape[:2]

    optimizer.zero_grad()
    t     = torch.randint(0, noise_scheduler.timesteps, (B, T), device=device)
    noise = torch.randn_like(x0)
    xt, v_target = noise_scheduler.noised_sample_and_velocity_target(x0 * 2 - 1, t, noise)

    with autocast("cuda", dtype=torch.bfloat16):
        v_pred = model(xt, t, actions)
        loss = nn.functional.mse_loss(v_pred, v_target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    return loss.item()


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    noise_scheduler: NoiseScheduler,
    device: str,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x0      = batch["frames"].to(device)
        actions = batch["actions"].to(device)
        B, T    = x0.shape[:2]
        t       = torch.randint(0, noise_scheduler.timesteps, (B, T), device=device)
        noise   = torch.randn_like(x0)
        xt, v_target = noise_scheduler.noised_sample_and_velocity_target(x0 * 2 - 1, t, noise)
        with autocast("cuda", dtype=torch.bfloat16):
            v_pred = model(xt, t, actions)
        total += nn.functional.mse_loss(v_pred, v_target).item() * B
        n += B
    model.train()
    return total / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required"

    save_dir = Path(CONFIG["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- WandB ---
    run = wandb.init(
        project=CONFIG["wandb_project"],
        entity=CONFIG["wandb_entity"],
        config=CONFIG,
        resume="allow",
        id=wandb.util.generate_id() if not args.resume else None,
        dir=str(save_dir),
    )
    (save_dir / "wandb_run_id.txt").write_text(run.id)

    # --- Model ---
    model = CoinRunWorldModel().to(device)
    model = torch.compile(model)   # ~10-20% speedup on H100
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # --- Optimiser ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / CONFIG["warmup_steps"],
        total_iters=CONFIG["warmup_steps"],
    )
    noise_scheduler = NoiseScheduler(CONFIG["max_noise_level"], device)

    # --- Precompute alphas for DDIM rollout ---
    betas = sigmoid_beta_schedule(CONFIG["max_noise_level"]).float().to(device)
    alphas_cumprod = rearrange(torch.cumprod(1.0 - betas, dim=0), "T -> T 1 1 1")

    # --- Data ---
    train_ds = CoinRunDataset(CONFIG["train_dir"], CONFIG["clip_len"], CONFIG["clip_stride"])
    val_ds   = CoinRunDataset(CONFIG["val_dir"],   CONFIG["clip_len"], CONFIG["clip_stride"])

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"] // 4, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # Fixed val prompts for visual logging (same clips every time)
    val_samples = [val_ds[i * (len(val_ds) // CONFIG["n_rollout_samples"])]
                   for i in range(CONFIG["n_rollout_samples"])]
    prompt_frames = torch.stack([s["frames"][:CONFIG["n_prompt_frames"]] for s in val_samples])
    prompt_actions = torch.stack([s["actions"][:CONFIG["rollout_frames"]] for s in val_samples])
    gt_frames = torch.stack([s["frames"][:CONFIG["rollout_frames"]] for s in val_samples])
    gt_uint8 = (rearrange(gt_frames, "b t c h w -> b t h w c") * 255).byte()

    # --- Resume ---
    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume:
        start_epoch, global_step, best_val = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        best_val = best_val or float("inf")
        print(f"Resumed from {args.resume} (epoch={start_epoch}, step={global_step})")

    # Save config
    (save_dir / "config.json").write_text(json.dumps(CONFIG, indent=2))

    # --- Training loop ---
    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, CONFIG["epochs"]):
        epoch_loss, epoch_steps = 0.0, 0

        for batch in train_loader:
            loss = train_step(
                model, batch, noise_scheduler,
                optimizer, scheduler, device, CONFIG["grad_clip"],
            )
            global_step  += 1
            epoch_loss   += loss
            epoch_steps  += 1

            # --- Loss log ---
            if global_step % CONFIG["log_every_steps"] == 0:
                elapsed = (time.time() - t0) / 3600
                wandb.log({
                    "train/loss": loss,
                    "train/lr":   scheduler.get_last_lr()[0],
                    "perf/elapsed_hours": elapsed,
                    "perf/steps_per_sec": global_step / (time.time() - t0),
                }, step=global_step)

            # --- Checkpoint ---
            if global_step % CONFIG["ckpt_every_steps"] == 0:
                ckpt_path = str(save_dir / f"ckpt_step_{global_step:07d}.pt")
                save_checkpoint(ckpt_path, model, optimizer, scheduler,
                                epoch, global_step, None, CONFIG)
                wandb.save(ckpt_path, base_path=str(save_dir))
                print(f"[step {global_step}] checkpoint saved")

            # --- Rollout visual log ---
            if global_step % CONFIG["rollout_every_steps"] == 0:
                generated = generate_rollout(
                    model, prompt_frames, prompt_actions,
                    alphas_cumprod, device,
                    ddim_steps=CONFIG["ddim_steps"],
                    total_frames=CONFIG["rollout_frames"],
                    n_prompt=CONFIG["n_prompt_frames"],
                )
                # Log as video (wandb.Video expects [T, C, H, W] or [T, H, W, C])
                for i in range(CONFIG["n_rollout_samples"]):
                    wandb.log({
                        f"rollout/video_{i}": wandb.Video(
                            generated[i].numpy(), fps=15, format="mp4"
                        ),
                    }, step=global_step)

                # Log comparison grid (GT top, generated bottom)
                grid = frames_to_grid(generated, gt_uint8)
                wandb.log({"rollout/grid": wandb.Image(grid)}, step=global_step)

        # --- End of epoch ---
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        val_loss = val_epoch(model, val_loader, noise_scheduler, device)

        wandb.log({
            "epoch/train_loss": avg_train_loss,
            "epoch/val_loss":   val_loss,
            "epoch":            epoch + 1,
        }, step=global_step)

        print(f"Epoch {epoch+1} | train={avg_train_loss:.4f} val={val_loss:.4f} "
              f"| {(time.time()-t0)/3600:.2f}h elapsed")

        # Save epoch checkpoint
        ckpt_path = str(save_dir / f"ckpt_epoch_{epoch+1:03d}.pt")
        save_checkpoint(ckpt_path, model, optimizer, scheduler,
                        epoch + 1, global_step, val_loss, CONFIG)
        wandb.save(ckpt_path, base_path=str(save_dir))

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            best_path = str(save_dir / "ckpt_best.pt")
            save_checkpoint(best_path, model, optimizer, scheduler,
                            epoch + 1, global_step, val_loss, CONFIG)
            wandb.save(best_path, base_path=str(save_dir))
            print(f"  ↳ new best val loss: {best_val:.4f}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    main(parser.parse_args())
