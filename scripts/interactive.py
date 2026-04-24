#!/venv/open-oasis/bin/python3
"""
Interactive CoinRun world model — browser UI via Gradio.

Usage:
    python interactive.py --ckpt runs/coinrun_v1/ckpt_step_0110000.pt

Then open the URL printed in the terminal.
Controls: arrow buttons to move/jump, Reset for a new prompt frame.
"""

from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from data.dataset_coinrun_streaming import CoinRunStreamingDataset
from model.dit import CoinRunWorldModelSmall
from model.utils import sigmoid_beta_schedule
from train_coinrun import generate_rollout

SCALE    = 6      # 64 → 384px display
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/coinrun/val"

# Procgen action indices
ACTIONS = {
    "⬅️  Left":       1,
    "➡️  Right":      7,
    "⬆️  Jump":       5,
    "↖️  Left+Jump":  2,
    "↗️  Right+Jump": 8,
    "⏹️  Wait":       4,
}

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------
_model          = None
_alphas_cumprod = None
_ds             = None


def load_model(ckpt_path: str):
    global _model, _alphas_cumprod, _ds
    ckpt   = torch.load(ckpt_path, weights_only=True, map_location=DEVICE)
    ckpt_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    action_cond_mode = ckpt_config.get("action_cond_mode", "linear")
    _model = CoinRunWorldModelSmall(
        external_cond_mode=action_cond_mode,
    ).to(DEVICE)
    _model.load_state_dict(ckpt["model"])
    _model.eval()
    print(
        f"Loaded {ckpt_path}  step={ckpt.get('step')}  "
        f"action_cond={action_cond_mode}"
    )

    betas           = sigmoid_beta_schedule(1000).float().to(DEVICE)
    _alphas_cumprod = rearrange(torch.cumprod(1.0 - betas, dim=0), "T -> T 1 1 1")

    _ds = CoinRunStreamingDataset(DATA_DIR, clip_len=32, stride=8, seed=0)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def _fresh_context(seed: int = 0) -> torch.Tensor:
    """Pull one prompt frame from val set → [1, 1, 3, 64, 64] on DEVICE."""
    for i, s in enumerate(_ds):
        if i >= seed:
            return s["frames"][:1].unsqueeze(0).to(DEVICE)
    return s["frames"][:1].unsqueeze(0).to(DEVICE)


def _frame_to_image(frame_chw: torch.Tensor) -> Image.Image:
    """[3,64,64] float [0,1] → upscaled PIL image."""
    np_hwc = (frame_chw.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_hwc).resize((64 * SCALE, 64 * SCALE), Image.NEAREST)


# ---------------------------------------------------------------------------
# Core step function
# ---------------------------------------------------------------------------
def step(context: list, action_name: str, recording: list) -> tuple:
    """Generate one frame, optionally append to recording."""
    action_idx = ACTIONS[action_name]

    frames = torch.from_numpy(np.stack(context)).float().unsqueeze(0).to(DEVICE)
    T = frames.shape[1]

    ctx_actions  = torch.zeros(1, T, 15, device=DEVICE)
    new_action   = F.one_hot(torch.tensor([[action_idx]], device=DEVICE), 15).float()
    full_actions = torch.cat([ctx_actions, new_action], dim=1)

    t0 = time.perf_counter()
    with torch.no_grad():
        rollout = generate_rollout(
            _model, frames, full_actions, _alphas_cumprod, DEVICE,
            ddim_steps=10, total_frames=T + 1, n_prompt=T,
        )
    ms = (time.perf_counter() - t0) * 1000

    new_frame_np  = rollout[0, -1].numpy()                         # [64,64,3] uint8
    new_frame_chw = torch.from_numpy(new_frame_np).permute(2,0,1).float() / 255.

    context = context + [new_frame_chw.numpy()]
    if len(context) > _model.max_frames:
        context = context[-_model.max_frames:]

    recording = recording + [(new_frame_np, action_idx)]            # (frame, action)

    img     = _frame_to_image(new_frame_chw)
    counter = f"🔴 {len(recording)} frames recorded"
    status  = f"{action_name}  |  {ms:.0f}ms  |  context={len(context)}"
    return img, context, recording, counter, status


_PANEL_H  = 120
_BG_COLOR = (15, 15, 20)
_KEY_OFF_C = (55, 55, 65)
_KEY_ON_C  = (255, 210, 40)
_DIM_C     = (150, 150, 165)
_BRIGHT_C  = (255, 210, 40)
try:
    from PIL import ImageFont as _IFont
    _F_SM  = _IFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    _F_MD  = _IFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except Exception:
    _F_SM = _F_MD = None

from train_coinrun import ACTION_NAMES, ACTION_KEYS


def _draw_episode_frame(frame_np: np.ndarray, action_idx: int) -> np.ndarray:
    """Upscale frame and draw full-size keyboard panel below. Returns [col_h, col_w, 3]."""
    from PIL import Image as PILImg, ImageDraw
    fw, fh = 64 * SCALE, 64 * SCALE
    canvas = PILImg.new("RGB", (fw, fh + _PANEL_H), _BG_COLOR)
    canvas.paste(PILImg.fromarray(frame_np).resize((fw, fh), PILImg.NEAREST), (0, 0))
    canvas.paste(PILImg.new("RGB", (fw, 1), (50, 50, 65)), (0, fh))

    draw = ImageDraw.Draw(canvas)
    left, right, up, down = ACTION_KEYS[action_idx]
    ksz, gap = 28, 6
    step = ksz + gap
    cx   = fw // 2
    cy_t = fh + 20
    cy_b = cy_t + step
    for kx, ky, active, label in [
        (cx,        cy_t, up,    "^"),
        (cx - step, cy_b, left,  "<"),
        (cx,        cy_b, down,  "v"),
        (cx + step, cy_b, right, ">"),
    ]:
        fill    = _KEY_ON_C  if active else _KEY_OFF_C
        outline = (210, 170, 10) if active else (110, 110, 130)
        draw.rectangle([kx-ksz//2, ky-ksz//2, kx+ksz//2, ky+ksz//2],
                       fill=fill, outline=outline, width=2)
        draw.text((kx, ky), label, font=_F_MD,
                  fill=(10,10,10) if active else _DIM_C, anchor="mm")
    name = ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else str(action_idx)
    draw.text((cx + ksz*2 + gap*2, cy_t + (step+ksz)//2 - 10), f"#{action_idx}", font=_F_SM, fill=_DIM_C,    anchor="lm")
    draw.text((cx + ksz*2 + gap*2, cy_t + (step+ksz)//2 + 12), name,            font=_F_SM, fill=_BRIGHT_C, anchor="lm")
    return np.array(canvas)


def save_episode(recording: list) -> tuple[str | None, str]:
    """Encode recorded frames with full-size keyboard overlay to mp4."""
    if not recording:
        return None, "Nothing recorded yet — step some frames first."
    import av
    from pathlib import Path
    ep_dir = Path("/workspace/open-oasis/episodes")
    ep_dir.mkdir(exist_ok=True)
    path = str(ep_dir / f"episode_{int(time.time())}.mp4")

    col_w, col_h = 64 * SCALE, 64 * SCALE + _PANEL_H

    container = av.open(path, mode="w")
    stream    = container.add_stream("libx264", rate=15)
    stream.width, stream.height = col_w, col_h
    stream.pix_fmt  = "yuv420p"
    stream.options  = {"crf": "18", "preset": "fast"}

    for frame_np, action_idx in recording:
        composite = _draw_episode_frame(frame_np, action_idx)
        for pkt in stream.encode(av.VideoFrame.from_ndarray(composite, format="rgb24")):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()

    return path, f"Saved {len(recording)} frames → {path}"


def clear_recording(_recording: list) -> tuple[list, str]:
    return [], "Recording cleared."


def reset(seed_val: int) -> tuple:
    seed_val   = int(seed_val)
    ctx_tensor = _fresh_context(seed=seed_val)
    frame_chw  = ctx_tensor[0, 0].cpu()
    context    = [frame_chw.numpy()]
    img        = _frame_to_image(frame_chw)
    return img, context, [], "🔴 0 frames recorded", f"Reset — val clip #{seed_val}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="CoinRun World Model") as demo:
        gr.Markdown("## CoinRun World Model — Interactive")

        with gr.Row():
            # --- Main display ---
            with gr.Column(scale=2):
                display = gr.Image(label="Generated frame", type="pil", height=64*SCALE)
                status  = gr.Textbox(label="", interactive=False, show_label=False)

            # --- Controls ---
            with gr.Column(scale=1):
                gr.Markdown("### Action")
                action = gr.Radio(list(ACTIONS.keys()), value="➡️  Right", label="")
                step_btn = gr.Button("▶ Step", variant="primary")

                gr.Markdown("### Recording")
                counter   = gr.Textbox(value="🔴 0 frames recorded", interactive=False, show_label=False)
                save_btn  = gr.Button("💾 Save episode", variant="secondary")
                clear_btn = gr.Button("🗑 Clear recording")
                video_out = gr.Video(label="Saved episode", interactive=False)

                gr.Markdown("### Reset")
                seed_sl   = gr.Slider(0, 199, value=0, step=1, label="Val clip #")
                reset_btn = gr.Button("↺ Reset")

        # hidden states
        ctx_state = gr.State([])
        rec_state = gr.State([])   # list of [64,64,3] uint8 numpy arrays

        step_btn.click(
            fn=step,
            inputs=[ctx_state, action, rec_state],
            outputs=[display, ctx_state, rec_state, counter, status],
        )
        save_btn.click(
            fn=save_episode,
            inputs=[rec_state],
            outputs=[video_out, status],
        )
        clear_btn.click(
            fn=clear_recording,
            inputs=[rec_state],
            outputs=[rec_state, status],
        ).then(fn=lambda: "🔴 0 frames recorded", outputs=[counter])
        reset_btn.click(
            fn=reset,
            inputs=[seed_sl],
            outputs=[display, ctx_state, rec_state, counter, status],
        )
        demo.load(
            fn=lambda: reset(0),
            outputs=[display, ctx_state, rec_state, counter, status],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--share",  action="store_true", help="create public gradio.live URL")
    args = parser.parse_args()

    load_model(args.ckpt)
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
              allowed_paths=["/workspace/open-oasis/episodes"])
