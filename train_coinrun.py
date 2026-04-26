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
import torch.distributed as dist
import torch.nn as nn
import av
import wandb
from PIL import Image as PILImage, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data.dataset_coinrun_streaming import CoinRunStreamingDataset
from model.dit import CoinRunWorldModel5M, CoinRunWorldModel9M, CoinRunWorldModelSmall
from model.utils import sigmoid_beta_schedule
from training.noise_scheduler import NoiseScheduler

from eval import evaluate_noise_loss_and_recon, evaluate_rollouts


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
            x[:, -1:] = (an.sqrt() * x_start + x_noise * (1 - an).sqrt())[:, -1:]

    out = (x.clamp(-1, 1) + 1) / 2                              # [B, T, 3, 64, 64] float
    out = rearrange(out, "b t c h w -> b t h w c")
    out = (out * 255).byte().cpu()

    if was_training:
        model.train()
    return out


# ---------------------------------------------------------------------------
# Keyboard overlay helpers (shared with infer_coinrun.py)
# ---------------------------------------------------------------------------
_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
try:
    _FONT_SM  = ImageFont.truetype(_FONT_PATH, 14)
    _FONT_MD  = ImageFont.truetype(_FONT_PATH, 18)
    _FONT_XS  = ImageFont.truetype(_FONT_PATH, 8)   # compact overlay at native 64px
    _FONT_KEY = ImageFont.truetype(_FONT_PATH, 9)
except OSError:
    _FONT_SM = _FONT_MD = _FONT_XS = _FONT_KEY = ImageFont.load_default()

ACTION_NAMES = [
    "LEFT+DOWN", "LEFT", "LEFT+UP", "DOWN", "NOOP",
    "UP", "RIGHT+DOWN", "RIGHT", "RIGHT+UP",
    "D", "A", "W", "S", "Q", "E",
]
ACTION_KEYS = [
    (True,  False, False, True),  (True,  False, False, False),
    (True,  False, True,  False), (False, False, False, True),
    (False, False, False, False), (False, False, True,  False),
    (False, True,  False, True),  (False, True,  False, False),
    (False, True,  True,  False),
    *([(False, False, False, False)] * 6),
]
_BG      = (15, 15, 20)
_KEY_OFF = (55, 55, 65)
_KEY_ON  = (255, 210, 40)
_DIM     = (150, 150, 165)
_BRIGHT  = (255, 210, 40)


def _draw_keyboard(draw: ImageDraw.Draw, action: int, panel_w: int, y0: int) -> None:
    """D-pad sized to fit inside panel_w (designed for 64px columns)."""
    left, right, up, down = ACTION_KEYS[action]
    ksz, gap = 10, 2        # key size chosen so 3 keys fit in 64px (10+2+10+2+10=34)
    step = ksz + gap
    cx = panel_w // 2       # 32 for 64px column
    cy_top = y0 + 6
    cy_bot = cy_top + step
    for kx, ky, active, label in [
        (cx,        cy_top, up,    "^"),
        (cx - step, cy_bot, left,  "<"),
        (cx,        cy_bot, down,  "v"),
        (cx + step, cy_bot, right, ">"),
    ]:
        fill    = _KEY_ON if active else _KEY_OFF
        outline = (210, 170, 10) if active else (100, 100, 115)
        draw.rectangle([kx - ksz//2, ky - ksz//2, kx + ksz//2, ky + ksz//2],
                       fill=fill, outline=outline, width=1)
        draw.text((kx, ky), label, font=_FONT_KEY,
                  fill=(10, 10, 10) if active else _DIM, anchor="mm")
    # action name centred below d-pad
    name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
    draw.text((cx, cy_bot + ksz//2 + 5), name, font=_FONT_XS, fill=_BRIGHT, anchor="mt")


def _build_frame(gt_f, gen_f, action: int, scale: int, panel_h: int) -> np.ndarray:
    """gt_f / gen_f: uint8 [64, 64, 3]. Returns composite column array."""
    fw, fh = 64 * scale, 64 * scale
    canvas = PILImage.new("RGB", (fw, fh * 2 + panel_h), _BG)
    canvas.paste(PILImage.fromarray(gt_f).resize((fw, fh),  PILImage.NEAREST), (0, 0))
    canvas.paste(PILImage.fromarray(gen_f).resize((fw, fh), PILImage.NEAREST), (0, fh))
    draw = ImageDraw.Draw(canvas)
    draw.line([(0, fh),     (fw, fh)],     fill=(60, 60, 80), width=2)
    draw.line([(0, fh * 2), (fw, fh * 2)], fill=(40, 40, 55), width=1)
    _draw_keyboard(draw, action, fw, fh * 2 + 4)
    return np.array(canvas)


def save_rollout_mp4(
    frames: torch.Tensor,
    path: str,
    gt: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,  # int [B, T]
    fps: int = 15,
    panel_h: int = 80,
) -> None:
    """frames/gt: uint8 [B, T, H, W, C]. GT top, generated bottom, keyboard panel below."""
    B, T = frames.shape[:2]
    col_w = 64
    col_h = 64 * 2 + panel_h

    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = col_w * B, col_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}

    for t in range(T):
        columns = []
        for b in range(B):
            gt_f   = gt[b, t].numpy()     if gt      is not None else np.zeros((64, 64, 3), np.uint8)
            gen_f  = frames[b, t].numpy()
            action = int(actions[b, t])   if actions is not None else 4
            columns.append(_build_frame(gt_f, gen_f, action, scale=1, panel_h=panel_h))  # scale=1: no upscale
        row = np.concatenate(columns, axis=1)
        for pkt in stream.encode(av.VideoFrame.from_ndarray(row, format="rgb24")):
            container.mux(pkt)

    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


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
    accum_step: int,        # which micro-step within the accumulation window (0-indexed)
    grad_accum_steps: int,  # total micro-steps before an optimizer update
) -> tuple[float, bool]: 
    x0      = batch["frames"].to(device)
    actions = batch["actions"].to(device)
    B, T    = x0.shape[:2]

    
    t     = torch.randint(0, noise_scheduler.timesteps, (B, T), device=device)
    noise = torch.randn_like(x0)
    xt, v_target = noise_scheduler.noised_sample_and_velocity_target(x0 * 2 - 1, t, noise)

    with autocast("cuda", dtype=torch.bfloat16):
        v_pred = model(xt, t, actions)
        loss = nn.functional.mse_loss(v_pred, v_target) / grad_accum_steps

    loss.backward()
    stepped = False

    if (accum_step + 1) % grad_accum_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        stepped = True

    return loss.item() * grad_accum_steps, stepped


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    noise_scheduler: NoiseScheduler,
    device: str,
    alphas_cumprod: torch.Tensor | None = None,
    max_batches: int | None = None,
) -> tuple[float, float, float]:
    """Returns (val_loss, psnr, ssim). Pass max_batches for a fast subset check."""
    model.eval()
    loss_total, n = 0.0, 0
    gt_frames, pred_frames = [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x0      = batch["frames"].to(device)          # [B, T, 3, 64, 64] in [0, 1]
        actions = batch["actions"].to(device)
        B, T    = x0.shape[:2]
        t       = torch.randint(0, noise_scheduler.timesteps, (B, T), device=device)
        noise   = torch.randn_like(x0)
        x0_scaled = x0 * 2 - 1                        # [-1, 1]
        xt, v_target = noise_scheduler.noised_sample_and_velocity_target(x0_scaled, t, noise)
        with autocast("cuda", dtype=torch.bfloat16):
            v_pred = model(xt, t, actions)
        loss_total += nn.functional.mse_loss(v_pred, v_target).item() * B
        n += B

        # recover x0 prediction for pixel-space metrics
        if alphas_cumprod is not None:
            ac = alphas_cumprod[t]                     # [B, T, 1, 1, 1]
            x0_pred = ac.sqrt() * xt - (1 - ac).sqrt() * v_pred.float()
            # [B, T, 3, 64, 64] → [B*T, 64, 64, 3] uint8
            x0_pred = ((x0_pred.clamp(-1, 1) + 1) / 2 * 255).byte()
            x0_pred = x0_pred.permute(0, 1, 3, 4, 2).reshape(-1, 64, 64, 3).cpu().numpy()
            x0_gt   = (x0 * 255).byte().permute(0, 1, 3, 4, 2).reshape(-1, 64, 64, 3).cpu().numpy()
            gt_frames.append(x0_gt)
            pred_frames.append(x0_pred)

    model.train()
    val_loss = loss_total / n

    if alphas_cumprod is None or not gt_frames:
        return val_loss, float("nan"), float("nan")

    gt_all   = np.concatenate(gt_frames,   axis=0).astype(np.float32)
    pred_all = np.concatenate(pred_frames, axis=0).astype(np.float32)
    psnr = peak_signal_noise_ratio(gt_all, pred_all, data_range=255)
    ssim = float(np.mean([
        structural_similarity(gt_all[i], pred_all[i], data_range=255, channel_axis=-1)
        for i in range(len(gt_all))
    ]))
    return val_loss, psnr, ssim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    CONFIG = {
        "train_dir":  "data/coinrun_raw/train",
        "val_dir":    "data/coinrun_raw/val",
        "save_dir":   f"runs/coinrun_{args.model_size}_lin",
        "max_noise_level": 1000,
        "clip_len":   32,
        "clip_stride": 8,
        "epochs":     999,          # effectively unlimited — time stops the run
        "batch_size": 128,          # per-step batch (physical)
        "grad_accum_steps": 2,      # effective batch = 128 × 2 = 256
        "lr":         1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 2000,
        "grad_clip":  1.0,
        "max_hours":  args.max_hours,
        "ckpt_every_steps":    10000,
        "rollout_every_steps": 1000,
        "val_every_steps":     5000,
        "val_subset_batches":  50,
        "log_every_steps":     50,
        "ddim_steps":          10,
        "n_prompt_frames":     1,
        "rollout_frames":      16,
        "n_rollout_samples":   4,
        "wandb_project": "coinrun-scaling",
        "wandb_entity":  "spring26-gen-ai",
        "action_cond_mode": "linear",
        "eval_rollout_samples": 4,
    }
    # --- DDP setup ---
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        is_main = local_rank == 0
    else:
        local_rank = 0
        world_size = 1
        device = "cuda"
        is_main = True

    assert torch.cuda.is_available(), "CUDA required"

    model_config = dict(CONFIG)
    if args.resume:
        resume_ckpt = torch.load(args.resume, weights_only=True, map_location="cpu")
        resume_config = resume_ckpt.get("config", {})
        model_config["action_cond_mode"] = resume_config.get("action_cond_mode", "linear")

    save_dir = Path(CONFIG["save_dir"])
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)

    # --- WandB (rank 0 only) ---
    if is_main:
        wandb_id_file = save_dir / "wandb_run_id.txt"
        if args.resume and wandb_id_file.exists():
            wandb_id = wandb_id_file.read_text().strip()
        else:
            wandb_id = wandb.util.generate_id()

        run = wandb.init(
            project=CONFIG["wandb_project"],
            entity=CONFIG["wandb_entity"],
            config=model_config,
            resume="allow",
            id=wandb_id,
            dir=str(save_dir),
        )
        (save_dir / "wandb_run_id.txt").write_text(run.id)

    # --- Model ---
    model_map = {
        "5m":    CoinRunWorldModel5M,
        "9m":    CoinRunWorldModel9M,
        "small": CoinRunWorldModelSmall,
    }
    raw_model = model_map[args.model_size](
        external_cond_mode=model_config["action_cond_mode"],
    ).to(device)
    if is_main:
        print(f"Parameters: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M")
    model = DDP(raw_model, device_ids=[local_rank]) if is_ddp else raw_model
    model = torch.compile(model)

    # --- Optimiser ---
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
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
    train_ds = CoinRunStreamingDataset(CONFIG["train_dir"], CONFIG["clip_len"], CONFIG["clip_stride"],
                                       ddp_rank=local_rank, ddp_world_size=world_size)
    val_ds   = CoinRunStreamingDataset(CONFIG["val_dir"],   CONFIG["clip_len"], CONFIG["clip_stride"], seed=0,
                                       ddp_rank=local_rank, ddp_world_size=world_size)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              num_workers=16, pin_memory=True, multiprocessing_context="spawn")
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"] // 4,
                              num_workers=4, pin_memory=True, multiprocessing_context="spawn")

    # Pre-collect a pool of val clips for visual logging (rank 0 only)
    if is_main:
        val_pool = []
        for item in val_ds:
            val_pool.append(item)
            if len(val_pool) >= 200:
                break
        rng_rollout = torch.Generator()

    def sample_val_prompts(seed: int):
        """Randomly sample n_rollout_samples clips from val_pool."""
        rng_rollout.manual_seed(seed)
        idx = torch.randperm(len(val_pool), generator=rng_rollout)[:CONFIG["n_rollout_samples"]]
        chosen = [val_pool[i] for i in idx.tolist()]
        pf  = torch.stack([s["frames"][:CONFIG["n_prompt_frames"]] for s in chosen])
        pa  = torch.stack([s["actions"][:CONFIG["rollout_frames"]]  for s in chosen])
        gtf = torch.stack([s["frames"][:CONFIG["rollout_frames"]]   for s in chosen])
        gt  = (rearrange(gtf, "b t c h w -> b t h w c") * 255).byte()
        return pf, pa, gt

    # --- Resume ---
    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume:
        start_epoch, global_step, best_val = load_checkpoint(
            args.resume, raw_model, optimizer, scheduler
        )
        best_val = best_val or float("inf")
        if is_main:
            print(f"Resumed from {args.resume} (epoch={start_epoch}, step={global_step})")

    if is_main:
        (save_dir / "config.json").write_text(json.dumps(model_config, indent=2))

    if is_ddp:
        dist.barrier() 

    # --- Training loop ---
    optimizer.zero_grad()
    model.train()
    t0 = time.time()

    time_up = False

    for epoch in range(start_epoch, CONFIG["epochs"]):
        epoch_loss, epoch_steps = 0.0, 0
        optimizer.zero_grad()

        total_batches = len(train_ds) // (CONFIG["batch_size"] * world_size)  # len() returns full dataset size; dataset already sharded by rank
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", total=total_batches,
                    dynamic_ncols=True, disable=not is_main)
        for micro_step, batch in enumerate(pbar):
            loss, did_step = train_step(
                model, batch, noise_scheduler,
                optimizer, scheduler, device, CONFIG["grad_clip"],
                accum_step=micro_step,
                grad_accum_steps=CONFIG["grad_accum_steps"],
            )
            epoch_loss  += loss
            epoch_steps += 1

            if not did_step: # haven't completed an effective step yet
                    continue

            global_step  += 1
            elapsed = time.time() - t0
            remaining = max(0, CONFIG["max_hours"] * 3600 - elapsed)
            pbar.set_postfix(
                loss=f"{loss:.4f}",
                step=global_step,
                eta=f"{remaining/3600:.1f}h",
            )

            # ---- time-based stopping ----
            if (time.time() - t0) > CONFIG["max_hours"] * 3600:
                if is_main:
                    print(f"Reached {CONFIG['max_hours']}h limit at step {global_step}, stopping.")
                time_up = True
                break

            if is_main:
                # --- Loss log ---
                if global_step % CONFIG["log_every_steps"] == 0:
                    n_params = sum(p.numel() for p in raw_model.parameters())
                    flops_per_eff_step = 6 * n_params * CONFIG["batch_size"] * CONFIG["grad_accum_steps"] * CONFIG["clip_len"]
                    wandb.log({
                        "train/loss":             loss,
                        "train/lr":               scheduler.get_last_lr()[0],
                        "train/cumulative_flops": flops_per_eff_step * global_step,
                        "perf/elapsed_hours":     (time.time() - t0) / 3600,
                        "perf/steps_per_sec":     global_step / (time.time() - t0),
                    }, step=global_step)

                # --- Fast val loss (subset) ---
                if global_step % CONFIG["val_every_steps"] == 0:
                    fast_loss, _, _ = val_epoch(
                        model, val_loader, noise_scheduler, device,
                        alphas_cumprod=None,
                        max_batches=CONFIG["val_subset_batches"],
                    )
                    wandb.log({"val/loss": fast_loss}, step=global_step)

                # --- Checkpoint ---
                if global_step % CONFIG["ckpt_every_steps"] == 0:
                    ckpt_path = str(save_dir / f"ckpt_step_{global_step:07d}.pt")
                    save_checkpoint(ckpt_path, raw_model, optimizer, scheduler,
                                    epoch, global_step, None, model_config)
                    wandb.save(ckpt_path, base_path=str(save_dir))
                    print(f"[step {global_step}] checkpoint saved")

                # --- Rollout visual log ---
                if global_step % CONFIG["rollout_every_steps"] == 0:
                    prompt_frames, prompt_actions, gt_uint8 = sample_val_prompts(global_step)

                    # recover int action indices from one-hot
                    has_action    = prompt_actions.sum(dim=-1) > 0
                    action_idx    = prompt_actions.argmax(dim=-1)
                    action_idx[~has_action] = 4   # NOOP for zero-pad frame

                    generated = generate_rollout(
                        raw_model, prompt_frames, prompt_actions,
                        alphas_cumprod, device,
                        ddim_steps=CONFIG["ddim_steps"],
                        total_frames=CONFIG["rollout_frames"],
                        n_prompt=CONFIG["n_prompt_frames"],
                    )

                    rollout_dir = save_dir / "rollouts"
                    rollout_dir.mkdir(exist_ok=True)
                    local_path = str(rollout_dir / f"rollout_step_{global_step:07d}.mp4")
                    save_rollout_mp4(generated, local_path, gt=gt_uint8, actions=action_idx)
                    wandb.log({
                        "rollout/video": wandb.Video(local_path),
                        "rollout/grid":  wandb.Image(frames_to_grid(generated, gt_uint8)),
                    }, step=global_step)
                    print(f"[step {global_step}] rollout saved → {local_path}")

        if time_up:
            break 

        # --- End of epoch ---
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        val_loss, psnr, ssim = val_epoch(model, val_loader, noise_scheduler, device, alphas_cumprod)

        # aggregate metrics across ranks
        if is_ddp:
            metrics = torch.tensor([val_loss, psnr, ssim], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            val_loss, psnr, ssim = metrics.tolist()

        if is_main:
            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss":   val_loss,
                "epoch/psnr":       psnr,
                "epoch/ssim":       ssim,
                "epoch":            epoch + 1,
            }, step=global_step)
            print(f"Epoch {epoch+1} | train={avg_train_loss:.4f} val={val_loss:.4f} "
                  f"PSNR={psnr:.2f} SSIM={ssim:.4f} | {(time.time()-t0)/3600:.2f}h elapsed")

            ckpt_path = str(save_dir / f"ckpt_epoch_{epoch+1:03d}.pt")
            save_checkpoint(ckpt_path, raw_model, optimizer, scheduler,
                            epoch + 1, global_step, val_loss, model_config)
            wandb.save(ckpt_path, base_path=str(save_dir))

            if val_loss < best_val:
                best_val = val_loss
                best_path = str(save_dir / "ckpt_best.pt")
                save_checkpoint(best_path, raw_model, optimizer, scheduler,
                                epoch + 1, global_step, val_loss, model_config)
                wandb.save(best_path, base_path=str(save_dir))
                print(f"  ↳ new best val loss: {best_val:.4f}")

    if is_ddp:
        dist.barrier()

    if is_main:
        final_ckpt = str(save_dir / f"ckpt_step_{global_step:07d}_final.pt")
        save_checkpoint(final_ckpt, raw_model, optimizer, scheduler,
                        epoch, global_step, None, model_config)
        print(f"Final checkpoint saved → {final_ckpt}")

        final_eval_metrics = evaluate_noise_loss_and_recon(
            model=raw_model,
            loader=val_loader,
            noise_scheduler=noise_scheduler,
            device=device,
            max_noise_level=CONFIG["max_noise_level"],
            max_batches=None,
        )
        final_rollout_metrics = evaluate_rollouts(
            model=raw_model,
            dataset=val_ds,
            device=device,
            max_noise_level=CONFIG["max_noise_level"],
            ddim_steps=CONFIG["ddim_steps"],
            n_prompt_frames=CONFIG["n_prompt_frames"],
            rollout_frames=CONFIG["rollout_frames"],
            num_samples=CONFIG["eval_rollout_samples"],
            save_dir=save_dir / "final_eval_rollouts",
        )
        final_eval_metrics.update(final_rollout_metrics)
        wandb.log({
            "eval/noise_loss":           final_eval_metrics["noise_loss"],
            "eval/psnr":                 final_eval_metrics["psnr"],
            "eval/ssim":                 final_eval_metrics["ssim"],
            "eval/rollout_psnr":         final_eval_metrics["rollout_psnr"],
            "eval/rollout_ssim":         final_eval_metrics["rollout_ssim"],
            "eval/musiq":                final_eval_metrics["musiq"],
            "eval/laion_aes":            final_eval_metrics["laion_aes"],
            "eval/temporal_consistency": final_eval_metrics["temporal_consistency"],
        }, step=global_step)
        (save_dir / "final_eval_metrics.json").write_text(json.dumps(final_eval_metrics, indent=2))
        print(f"Final eval: {json.dumps(final_eval_metrics, indent=2)}")

        wandb.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["5m", "9m", "small"], default="small")
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    main(parser.parse_args())
