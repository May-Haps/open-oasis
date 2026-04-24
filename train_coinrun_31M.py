"""
Scaling ablation — ~31.4M parameter CoinRun world model (L variant).
1 epoch only; same hyperparameters as train_coinrun.py for fair comparison.

Model: DiT hidden=384, depth=6, heads=8  →  ~31.9M params
       (geometric series point between 17M and 57.8M)

FLOPs (6ND formula, S=2048 tokens, B=256 effective):
    Per sample : 6 × 31.9M × 2048 = 392 GFLOPs
    Per step   : 392 × 256         = 100 TFLOPs
    1 epoch    : 100 × 40,700      = 4.1 PFLOPs

Wall time on 2× H100 SXM:
    Speed   : ~5.5 it/s  (31.9/57.8 × 3.3 it/s baseline, with overhead)
    1 epoch : 40,700 / 5.5 ≈ 7,400 s ≈ 2.1 hours

Run:
    torchrun --nproc_per_node=2 train_coinrun_31M.py

Resume:
    torchrun --nproc_per_node=2 train_coinrun_31M.py --resume runs/coinrun_31M/ckpt_step_XXXXXXX.pt
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
from model.dit import DiT
from model.utils import sigmoid_beta_schedule
from training.noise_scheduler import NoiseScheduler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    # Paths
    "train_dir":  "data/coinrun/train",
    "val_dir":    "data/coinrun/val",
    "save_dir":   "runs/coinrun_31M",

    # Model
    "max_noise_level": 1000,

    # Data
    "clip_len":   32,
    "clip_stride": 8,

    # Training — 4 epochs for equal-FLOPs vs 57.8M baseline at 73K steps (13.3 ExaFLOPs)
    "epochs":          4,
    "batch_size":      128,
    "lr":              1e-4,
    "weight_decay":    0.01,
    "warmup_steps":    2000,
    "grad_clip":       1.0,

    # Logging / saving
    "ckpt_every_steps":    10000,
    "rollout_every_steps": 1000,
    "val_every_steps":     5000,
    "val_subset_batches":  50,
    "log_every_steps":     50,

    # Rollout generation
    "ddim_steps":          10,
    "n_prompt_frames":     1,
    "rollout_frames":      16,
    "n_rollout_samples":   4,

    # WandB
    "wandb_project": "coinrun-world-model",
    "wandb_entity":  "spring26-gen-ai",
    "wandb_name":    "ablation-31M",
}


def make_model():
    return DiT(
        input_h=64, input_w=64,
        patch_size=8,
        in_channels=3,
        hidden_size=384,
        depth=6,
        num_heads=8,
        external_cond_dim=15,
        max_frames=32,
    )


# ---------------------------------------------------------------------------
# Everything below is identical to train_coinrun.py
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_rollout(
    model, prompt_frames, actions, noise_scheduler_alphas, device,
    ddim_steps=10, total_frames=16, n_prompt=1,
    noise_abs_max=20.0, stabilization_level=15,
):
    was_training = model.training
    model.eval()
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

    out = (x.clamp(-1, 1) + 1) / 2
    out = rearrange(out, "b t c h w -> b t h w c")
    out = (out * 255).byte().cpu()
    if was_training:
        model.train()
    return out


_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
try:
    _FONT_SM  = ImageFont.truetype(_FONT_PATH, 14)
    _FONT_MD  = ImageFont.truetype(_FONT_PATH, 18)
    _FONT_XS  = ImageFont.truetype(_FONT_PATH, 8)
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
_BG = (15, 15, 20); _KEY_OFF = (55, 55, 65); _KEY_ON = (255, 210, 40)
_DIM = (150, 150, 165); _BRIGHT = (255, 210, 40)


def _draw_keyboard(draw, action, panel_w, y0):
    left, right, up, down = ACTION_KEYS[action]
    ksz, gap = 10, 2
    step = ksz + gap
    cx = panel_w // 2
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
    name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
    draw.text((cx, cy_bot + ksz//2 + 5), name, font=_FONT_XS, fill=_BRIGHT, anchor="mt")


def _build_frame(gt_f, gen_f, action, scale, panel_h):
    fw, fh = 64 * scale, 64 * scale
    canvas = PILImage.new("RGB", (fw, fh * 2 + panel_h), _BG)
    canvas.paste(PILImage.fromarray(gt_f).resize((fw, fh),  PILImage.NEAREST), (0, 0))
    canvas.paste(PILImage.fromarray(gen_f).resize((fw, fh), PILImage.NEAREST), (0, fh))
    draw = ImageDraw.Draw(canvas)
    draw.line([(0, fh),     (fw, fh)],     fill=(60, 60, 80), width=2)
    draw.line([(0, fh * 2), (fw, fh * 2)], fill=(40, 40, 55), width=1)
    _draw_keyboard(draw, action, fw, fh * 2 + 4)
    return np.array(canvas)


def save_rollout_mp4(frames, path, gt=None, actions=None, fps=15, panel_h=80):
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
            gt_f   = gt[b, t].numpy()   if gt      is not None else np.zeros((64, 64, 3), np.uint8)
            gen_f  = frames[b, t].numpy()
            action = int(actions[b, t]) if actions is not None else 4
            columns.append(_build_frame(gt_f, gen_f, action, scale=1, panel_h=panel_h))
        row = np.concatenate(columns, axis=1)
        for pkt in stream.encode(av.VideoFrame.from_ndarray(row, format="rgb24")):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def frames_to_grid(generated, ground_truth):
    B, T, H, W, C = generated.shape
    gen_row = generated[0]
    gt_row  = ground_truth[0]
    def to_chw(frames):
        return [torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in frames.numpy()]
    grid = make_grid(to_chw(gt_row) + to_chw(gen_row), nrow=T, padding=2)
    return grid.permute(1, 2, 0).numpy()


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, val_loss, config):
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": epoch,
        "step": step, "val_loss": val_loss, "config": config,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["step"], ckpt.get("val_loss")


def train_step(model, batch, noise_scheduler, optimizer, scheduler, device, grad_clip):
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
def val_epoch(model, loader, noise_scheduler, device, alphas_cumprod=None, max_batches=None):
    model.eval()
    loss_total, n = 0.0, 0
    gt_frames, pred_frames = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x0      = batch["frames"].to(device)
        actions = batch["actions"].to(device)
        B, T    = x0.shape[:2]
        t       = torch.randint(0, noise_scheduler.timesteps, (B, T), device=device)
        noise   = torch.randn_like(x0)
        x0_scaled = x0 * 2 - 1
        xt, v_target = noise_scheduler.noised_sample_and_velocity_target(x0_scaled, t, noise)
        with autocast("cuda", dtype=torch.bfloat16):
            v_pred = model(xt, t, actions)
        loss_total += nn.functional.mse_loss(v_pred, v_target).item() * B
        n += B
        if alphas_cumprod is not None:
            ac = alphas_cumprod[t]
            x0_pred = ac.sqrt() * xt - (1 - ac).sqrt() * v_pred.float()
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


def main(args):
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

    save_dir = Path(CONFIG["save_dir"])
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        run = wandb.init(
            project=CONFIG["wandb_project"],
            entity=CONFIG["wandb_entity"],
            name=CONFIG["wandb_name"],
            config=CONFIG,
            resume="allow",
            id=wandb.util.generate_id() if not args.resume else None,
            dir=str(save_dir),
        )
        (save_dir / "wandb_run_id.txt").write_text(run.id)

    raw_model = make_model().to(device)
    if is_main:
        print(f"Parameters: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M")
    model = DDP(raw_model, device_ids=[local_rank]) if is_ddp else raw_model
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / CONFIG["warmup_steps"],
        total_iters=CONFIG["warmup_steps"],
    )
    noise_scheduler = NoiseScheduler(CONFIG["max_noise_level"], device)

    betas = sigmoid_beta_schedule(CONFIG["max_noise_level"]).float().to(device)
    alphas_cumprod = rearrange(torch.cumprod(1.0 - betas, dim=0), "T -> T 1 1 1")

    train_ds = CoinRunStreamingDataset(CONFIG["train_dir"], CONFIG["clip_len"], CONFIG["clip_stride"],
                                       ddp_rank=local_rank, ddp_world_size=world_size)
    val_ds   = CoinRunStreamingDataset(CONFIG["val_dir"],   CONFIG["clip_len"], CONFIG["clip_stride"], seed=0,
                                       ddp_rank=local_rank, ddp_world_size=world_size)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              num_workers=16, pin_memory=True, multiprocessing_context="spawn")
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"] // 4,
                              num_workers=4, pin_memory=True, multiprocessing_context="spawn")

    if is_main:
        val_pool = []
        for item in val_ds:
            val_pool.append(item)
            if len(val_pool) >= 200:
                break
        rng_rollout = torch.Generator()

    def sample_val_prompts(seed):
        rng_rollout.manual_seed(seed)
        idx = torch.randperm(len(val_pool), generator=rng_rollout)[:CONFIG["n_rollout_samples"]]
        chosen = [val_pool[i] for i in idx.tolist()]
        pf  = torch.stack([s["frames"][:CONFIG["n_prompt_frames"]] for s in chosen])
        pa  = torch.stack([s["actions"][:CONFIG["rollout_frames"]]  for s in chosen])
        gtf = torch.stack([s["frames"][:CONFIG["rollout_frames"]]   for s in chosen])
        gt  = (rearrange(gtf, "b t c h w -> b t h w c") * 255).byte()
        return pf, pa, gt

    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume:
        start_epoch, global_step, best_val = load_checkpoint(
            args.resume, raw_model, optimizer, scheduler
        )
        best_val = best_val or float("inf")
        if is_main:
            print(f"Resumed from {args.resume} (epoch={start_epoch}, step={global_step})")

    if is_main:
        (save_dir / "config.json").write_text(json.dumps(CONFIG, indent=2))

    if is_ddp:
        dist.barrier()

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, CONFIG["epochs"]):
        epoch_loss, epoch_steps = 0.0, 0
        total_batches = len(train_ds) // (CONFIG["batch_size"] * world_size)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", total=total_batches,
                    dynamic_ncols=True, disable=not is_main)

        for batch in pbar:
            loss = train_step(model, batch, noise_scheduler,
                              optimizer, scheduler, device, CONFIG["grad_clip"])
            global_step  += 1
            epoch_loss   += loss
            epoch_steps  += 1
            pbar.set_postfix(loss=f"{loss:.4f}", step=global_step)

            if is_main:
                if global_step % CONFIG["log_every_steps"] == 0:
                    elapsed = (time.time() - t0) / 3600
                    wandb.log({
                        "train/loss": loss, "train/lr": scheduler.get_last_lr()[0],
                        "perf/elapsed_hours": elapsed,
                        "perf/steps_per_sec": global_step / (time.time() - t0),
                    }, step=global_step)

                if global_step % CONFIG["val_every_steps"] == 0:
                    fast_loss, _, _ = val_epoch(model, val_loader, noise_scheduler, device,
                                                alphas_cumprod=None,
                                                max_batches=CONFIG["val_subset_batches"])
                    wandb.log({"val/loss": fast_loss}, step=global_step)

                if global_step % CONFIG["ckpt_every_steps"] == 0:
                    ckpt_path = str(save_dir / f"ckpt_step_{global_step:07d}.pt")
                    save_checkpoint(ckpt_path, raw_model, optimizer, scheduler,
                                    epoch, global_step, None, CONFIG)
                    wandb.save(ckpt_path, base_path=str(save_dir))
                    print(f"[step {global_step}] checkpoint saved")

                if global_step % CONFIG["rollout_every_steps"] == 0:
                    prompt_frames, prompt_actions, gt_uint8 = sample_val_prompts(global_step)
                    has_action = prompt_actions.sum(dim=-1) > 0
                    action_idx = prompt_actions.argmax(dim=-1)
                    action_idx[~has_action] = 4
                    generated = generate_rollout(
                        raw_model, prompt_frames, prompt_actions, alphas_cumprod, device,
                        ddim_steps=CONFIG["ddim_steps"], total_frames=CONFIG["rollout_frames"],
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

        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        val_loss, psnr, ssim = val_epoch(model, val_loader, noise_scheduler, device, alphas_cumprod)

        if is_ddp:
            metrics = torch.tensor([val_loss, psnr, ssim], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            val_loss, psnr, ssim = metrics.tolist()

        if is_main:
            wandb.log({
                "epoch/train_loss": avg_train_loss, "epoch/val_loss": val_loss,
                "epoch/psnr": psnr, "epoch/ssim": ssim, "epoch": epoch + 1,
            }, step=global_step)
            print(f"Epoch {epoch+1} | train={avg_train_loss:.4f} val={val_loss:.4f} "
                  f"PSNR={psnr:.2f} SSIM={ssim:.4f} | {(time.time()-t0)/3600:.2f}h elapsed")

            ckpt_path = str(save_dir / f"ckpt_epoch_{epoch+1:03d}.pt")
            save_checkpoint(ckpt_path, raw_model, optimizer, scheduler,
                            epoch + 1, global_step, val_loss, CONFIG)
            wandb.save(ckpt_path, base_path=str(save_dir))

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(str(save_dir / "ckpt_best.pt"), raw_model, optimizer, scheduler,
                                epoch + 1, global_step, val_loss, CONFIG)
                print(f"  ↳ new best val loss: {best_val:.4f}")

    if is_main:
        wandb.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    main(parser.parse_args())
