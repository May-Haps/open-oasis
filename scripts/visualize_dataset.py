#!/venv/open-oasis/bin/python3
"""
Visualize CoinRun dataset episodes with action + keyboard overlay.

Usage:
    python visualize_dataset.py
    python visualize_dataset.py --shard data/coinrun/train/data_0001.array_record
    python visualize_dataset.py --n-episodes 5 --fps 15 --output preview.mp4
"""

from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Procgen 15-action layout
# ---------------------------------------------------------------------------
ACTION_NAMES = [
    "LEFT+DOWN", "LEFT", "LEFT+UP", "DOWN", "NOOP",
    "UP", "RIGHT+DOWN", "RIGHT", "RIGHT+UP",
    "D", "A", "W", "S", "Q", "E",
]

# (left, right, up, down) keys active for each action
ACTION_KEYS = [
    (True,  False, False, True),   # 0  LEFT+DOWN
    (True,  False, False, False),  # 1  LEFT
    (True,  False, True,  False),  # 2  LEFT+UP
    (False, False, False, True),   # 3  DOWN
    (False, False, False, False),  # 4  NOOP
    (False, False, True,  False),  # 5  UP
    (False, True,  False, True),   # 6  RIGHT+DOWN
    (False, True,  False, False),  # 7  RIGHT
    (False, True,  True,  False),  # 8  RIGHT+UP
    (False, False, False, False),  # 9  D
    (False, False, False, False),  # 10 A
    (False, False, False, False),  # 11 W
    (False, False, False, False),  # 12 S
    (False, False, False, False),  # 13 Q
    (False, False, False, False),  # 14 E
]

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
SCALE    = 8
GAME_W   = 64 * SCALE   # 512
GAME_H   = 64 * SCALE   # 512
PANEL_H  = 130
TOTAL_W  = GAME_W
TOTAL_H  = GAME_H + PANEL_H

BG          = (15, 15, 20)
KEY_OFF     = (55, 55, 65)
KEY_ON      = (255, 210, 40)
KEY_OUTLINE = (140, 140, 160)
TEXT_DIM    = (160, 160, 175)
TEXT_BRIGHT = (240, 240, 255)
TEXT_ACTION = (255, 210, 40)

FONT_PATH   = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
try:
    FONT_SM = ImageFont.truetype(FONT_PATH, 18)
    FONT_MD = ImageFont.truetype(FONT_PATH, 22)
    FONT_LG = ImageFont.truetype(FONT_PATH, 28)
except OSError:
    FONT_SM = FONT_MD = FONT_LG = ImageFont.load_default()


def _key_rect(cx: int, cy: int, size: int) -> tuple[int, int, int, int]:
    h = size // 2
    return (cx - h, cy - h, cx + h, cy + h)


def draw_keyboard_panel(draw: ImageDraw.Draw, action: int, y0: int) -> None:
    left, right, up, down = ACTION_KEYS[action] if action < len(ACTION_KEYS) else (False,)*4

    ksz = 40   # key size (full width/height)
    gap = 6    # gap between keys
    step = ksz + gap

    # Centre the d-pad at x=160
    cx = 160
    cy_top = y0 + 28
    cy_bot = cy_top + step

    keys = [
        (cx,          cy_top, up,    "^"),
        (cx - step,   cy_bot, left,  "<"),
        (cx,          cy_bot, down,  "v"),
        (cx + step,   cy_bot, right, ">"),
    ]

    for kx, ky, active, label in keys:
        fill    = KEY_ON    if active else KEY_OFF
        outline = (220, 180, 20) if active else KEY_OUTLINE
        draw.rectangle(_key_rect(kx, ky, ksz), fill=fill, outline=outline, width=2)
        lc = (10, 10, 10) if active else TEXT_DIM
        draw.text((kx, ky), label, font=FONT_LG, fill=lc, anchor="mm")

    # Action info on the right side
    name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else f"action_{action}"
    draw.text((cx + 140, y0 + 22), f"action {action:2d}", font=FONT_SM, fill=TEXT_DIM,    anchor="lm")
    draw.text((cx + 140, y0 + 52), name,                  font=FONT_MD, fill=TEXT_ACTION, anchor="lm")


def render_frame(frame_np: np.ndarray, action: int, ep: int, frame_idx: int) -> np.ndarray:
    """frame_np: uint8 [64, 64, 3] → composite uint8 [TOTAL_H, TOTAL_W, 3]."""
    game_img = Image.fromarray(frame_np).resize((GAME_W, GAME_H), Image.NEAREST)

    canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), BG)
    canvas.paste(game_img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # Top-left HUD
    draw.text((10, 10), f"ep {ep:03d}  frame {frame_idx:04d}", font=FONT_SM, fill=TEXT_DIM)

    # Separator line
    draw.line([(0, GAME_H), (TOTAL_W, GAME_H)], fill=(50, 50, 60), width=2)

    # Keyboard panel
    draw_keyboard_panel(draw, action, GAME_H + 5)

    return np.array(canvas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    from array_record.python.array_record_module import ArrayRecordReader

    shard = Path(args.shard)
    out   = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Shard  : {shard}")
    print(f"Output : {out}")
    print(f"Episodes: {args.n_episodes}  |  FPS: {args.fps}")

    reader = ArrayRecordReader(str(shard))
    n_records = reader.num_records()
    n_ep = min(args.n_episodes, n_records)
    print(f"Records in shard: {n_records}  (rendering {n_ep})")

    container = av.open(str(out), mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width  = TOTAL_W
    stream.height = TOTAL_H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}

    total_frames = 0
    for ep_idx in range(n_ep):
        raw = reader.read([ep_idx])[0]
        rec = pickle.loads(raw)

        T         = int(rec["sequence_length"])
        frames_np = np.frombuffer(rec["raw_video"], dtype=np.uint8).reshape(T, 64, 64, 3)
        actions   = np.asarray(rec["actions"], dtype=np.int64)   # [T]

        print(f"  ep {ep_idx:03d}: {T} frames")
        for fi in range(T):
            action = int(actions[fi]) if fi < len(actions) else 4
            composite = render_frame(frames_np[fi], action, ep_idx, fi)
            av_frame = av.VideoFrame.from_ndarray(composite, format="rgb24")
            for pkt in stream.encode(av_frame):
                container.mux(pkt)
            total_frames += 1

    for pkt in stream.encode():
        container.mux(pkt)
    container.close()

    print(f"\nWrote {total_frames} frames → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard",       default="data/coinrun/train/data_0000.array_record")
    parser.add_argument("--n-episodes",  type=int, default=3)
    parser.add_argument("--fps",         type=int, default=15)
    parser.add_argument("--output",      default="dataset_preview.mp4")
    main(parser.parse_args())
