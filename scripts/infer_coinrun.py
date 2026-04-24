#!/venv/open-oasis/bin/python3
"""
Quick CoinRun inference from a checkpoint.

Usage:
    python infer_coinrun.py --ckpt runs/coinrun_v1/ckpt_step_0004000.pt
    python infer_coinrun.py --ckpt runs/coinrun_v1/ckpt_step_0004000.pt \
        --frames 32 --n-samples 4 --output generated.mp4

Action selection:
    By default uses ground-truth actions from the val set (--action-source gt).
    Pass --action-source random to sample random actions instead.
"""

from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random

import av
import numpy as np
import torch
from einops import rearrange
from PIL import Image as PILImage, ImageDraw, ImageFont

from data.dataset_coinrun_streaming import CoinRunStreamingDataset
from model.dit import CoinRunWorldModelSmall
from model.utils import sigmoid_beta_schedule
from train_coinrun import generate_rollout

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/coinrun/val"

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
try:
    FONT_SM = ImageFont.truetype(FONT_PATH, 14)
    FONT_MD = ImageFont.truetype(FONT_PATH, 18)
except OSError:
    FONT_SM = FONT_MD = ImageFont.load_default()

# Procgen 15-action layout
ACTION_NAMES = [
    "LEFT+DOWN", "LEFT", "LEFT+UP", "DOWN", "NOOP",
    "UP", "RIGHT+DOWN", "RIGHT", "RIGHT+UP",
    "D", "A", "W", "S", "Q", "E",
]
# (left, right, up, down) active per action
ACTION_KEYS = [
    (True,  False, False, True),   # 0
    (True,  False, False, False),  # 1
    (True,  False, True,  False),  # 2
    (False, False, False, True),   # 3
    (False, False, False, False),  # 4 NOOP
    (False, False, True,  False),  # 5
    (False, True,  False, True),   # 6
    (False, True,  False, False),  # 7
    (False, True,  True,  False),  # 8
    (False, False, False, False),  # 9-14
    (False, False, False, False),
    (False, False, False, False),
    (False, False, False, False),
    (False, False, False, False),
    (False, False, False, False),
]

BG          = (15, 15, 20)
KEY_OFF     = (55, 55, 65)
KEY_ON      = (255, 210, 40)
KEY_OUTLINE = (130, 130, 150)
TEXT_DIM    = (150, 150, 165)
TEXT_ON     = (255, 210, 40)


def _draw_keyboard(draw: ImageDraw.Draw, action: int, panel_w: int, y0: int) -> None:
    left, right, up, down = ACTION_KEYS[action]
    ksz, gap = 28, 4
    step = ksz + gap

    cx = panel_w // 2
    cy_top = y0 + 16
    cy_bot = cy_top + step

    for kx, ky, active, label in [
        (cx,        cy_top, up,    "^"),
        (cx - step, cy_bot, left,  "<"),
        (cx,        cy_bot, down,  "v"),
        (cx + step, cy_bot, right, ">"),
    ]:
        fill    = KEY_ON  if active else KEY_OFF
        outline = (210, 170, 10) if active else KEY_OUTLINE
        draw.rectangle([kx - ksz//2, ky - ksz//2, kx + ksz//2, ky + ksz//2],
                       fill=fill, outline=outline, width=2)
        draw.text((kx, ky), label, font=FONT_MD,
                  fill=(10, 10, 10) if active else TEXT_DIM, anchor="mm")

    name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
    draw.text((cx + ksz * 2 + gap * 3, y0 + 14), f"#{action}", font=FONT_SM, fill=TEXT_DIM, anchor="lm")
    draw.text((cx + ksz * 2 + gap * 3, y0 + 34), name,         font=FONT_SM, fill=TEXT_ON,  anchor="lm")


def build_column(
    gt_frame: np.ndarray,    # uint8 [64, 64, 3]
    gen_frame: np.ndarray,   # uint8 [64, 64, 3]
    action: int,
    scale: int,
    panel_h: int,
) -> np.ndarray:
    """Return a single sample column: GT / divider / generated / keyboard panel."""
    fw = 64 * scale
    fh = 64 * scale

    canvas = PILImage.new("RGB", (fw, fh * 2 + panel_h), BG)

    gt_up  = PILImage.fromarray(gt_frame).resize((fw, fh),   PILImage.NEAREST)
    gen_up = PILImage.fromarray(gen_frame).resize((fw, fh),  PILImage.NEAREST)
    canvas.paste(gt_up,  (0, 0))
    canvas.paste(gen_up, (0, fh))

    draw = ImageDraw.Draw(canvas)

    # labels
    draw.text((4, 4),      "GT",        font=FONT_SM, fill=(180, 255, 180))
    draw.text((4, fh + 4), "generated", font=FONT_SM, fill=(180, 180, 255))

    # divider
    draw.line([(0, fh), (fw, fh)], fill=(60, 60, 80), width=2)
    draw.line([(0, fh * 2), (fw, fh * 2)], fill=(40, 40, 55), width=1)

    # keyboard
    _draw_keyboard(draw, action, fw, fh * 2 + 4)

    return np.array(canvas)


def save_mp4(
    gt: torch.Tensor,        # uint8 [B, T, H, W, C]
    gen: torch.Tensor,       # uint8 [B, T, H, W, C]
    actions: torch.Tensor,   # int   [B, T]
    path: str,
    fps: int = 15,
    scale: int = 5,
    panel_h: int = 80,
) -> None:
    B, T = gt.shape[:2]
    col_w = 64 * scale
    col_h = 64 * scale * 2 + panel_h

    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width   = col_w * B
    stream.height  = col_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}

    for t in range(T):
        columns = [
            build_column(
                gt[b, t].numpy(),
                gen[b, t].numpy(),
                int(actions[b, t]),
                scale,
                panel_h,
            )
            for b in range(B)
        ]
        frame_np = np.concatenate(columns, axis=1)   # [col_h, B*col_w, 3]
        av_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        for pkt in stream.encode(av_frame):
            container.mux(pkt)

    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def main(args: argparse.Namespace) -> None:
    print(f"Checkpoint    : {args.ckpt}")
    print(f"Action source : {args.action_source}")
    print(f"Device        : {DEVICE}")

    ckpt  = torch.load(args.ckpt, weights_only=True, map_location=DEVICE)
    ckpt_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    action_cond_mode = ckpt_config.get("action_cond_mode", "linear")

    model = CoinRunWorldModelSmall(
        external_cond_mode=action_cond_mode,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params        : {n_params:.1f}M  (step={ckpt.get('step')}, epoch={ckpt.get('epoch')})")
    print(f"Action cond   : mode={action_cond_mode}")

    betas          = sigmoid_beta_schedule(1000).float().to(DEVICE)
    alphas_cumprod = rearrange(torch.cumprod(1.0 - betas, dim=0), "T -> T 1 1 1")

    ds = CoinRunStreamingDataset(DATA_DIR, clip_len=args.frames, stride=8, seed=0)
    samples = []
    for item in ds:
        samples.append(item)
        if len(samples) >= args.n_samples:
            break

    prompt_frames  = torch.stack([s["frames"][:args.n_prompt]  for s in samples])  # [B,1,3,64,64]
    gt_actions_oh  = torch.stack([s["actions"][:args.frames]   for s in samples])  # [B,T,15] one-hot
    gt_frames      = torch.stack([s["frames"][:args.frames]    for s in samples])  # [B,T,3,64,64]

    if args.action_source == "random":
        # sample random action indices, convert to one-hot
        B = args.n_samples
        rand_ids = torch.randint(0, 15, (B, args.frames))
        import torch.nn.functional as F
        prompt_actions = F.one_hot(rand_ids, num_classes=15).float()
        action_indices = rand_ids
        print("Using random actions")
    else:
        prompt_actions = gt_actions_oh
        # recover int indices (frame 0 is zero-padded → NOOP=4)
        has_action = gt_actions_oh.sum(dim=-1) > 0   # [B, T]
        action_indices = gt_actions_oh.argmax(dim=-1)  # [B, T]
        action_indices[~has_action] = 4               # NOOP for the zero-pad frame
        print("Using ground-truth actions from val set")

    print(f"Generating {args.n_samples} samples × {args.frames} frames …")
    generated = generate_rollout(
        model, prompt_frames, prompt_actions,
        alphas_cumprod, DEVICE,
        ddim_steps=args.ddim_steps,
        total_frames=args.frames,
        n_prompt=args.n_prompt,
    )   # uint8 [B, T, H, W, C]

    gt_uint8 = (rearrange(gt_frames, "b t c h w -> b t h w c") * 255).byte()

    save_mp4(gt_uint8, generated, action_indices, args.output, fps=args.fps)
    print(f"Saved → {args.output}  (top=GT, bottom=generated, panel=action)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",          required=True)
    parser.add_argument("--frames",        type=int, default=24)
    parser.add_argument("--n-samples",     type=int, default=4)
    parser.add_argument("--n-prompt",      type=int, default=1)
    parser.add_argument("--ddim-steps",    type=int, default=10)
    parser.add_argument("--fps",           type=int, default=15)
    parser.add_argument("--output",        default="generated.mp4")
    parser.add_argument("--action-source", choices=["gt", "random"], default="gt",
                        help="gt = replay val actions, random = sample random actions")
    main(parser.parse_args())
