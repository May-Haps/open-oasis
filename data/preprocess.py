"""
Preprocess Mario NES dataset into per-episode tensors.

Expected input layout:
  <data_dir>/
    <user>_<sessid>_e<episode>_<world>-<level>_<outcome>/
      <user>_<sessid>_e<episode>_<world>-<level>_f<frame>_a<action>_<datetime>.<outcome>.png
      ...

Run:
  python data/preprocess.py --input-dir /path/to/raw --output-dir /path/to/processed
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize
from tqdm import tqdm

from model.utils import action_int_to_bits

TARGET_SIZE = (64, 64)  # (H, W)

FRAME_RE = re.compile(
    r"_f(?P<frame>\d+)_a(?P<action>\d+)_"
)


def load_episode_frames(episode_dir: Path) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Returns (frames [T, 3, 256, 256] float16, actions [T-1, 8] float32)
    or None if the episode has fewer than 2 frames.
    """
    png_files = sorted(episode_dir.glob("*.png"), key=_frame_number)
    if len(png_files) < 2:
        return None

    frames_list = []
    action_ints = []

    for png_path in png_files:
        m = FRAME_RE.search(png_path.name)
        if m is None:
            continue
        action_int = int(m.group("action"))
        action_ints.append(action_int)

        img = Image.open(png_path).convert("RGB")
        frame = to_tensor(img)                          # [3, H, W] float32 in [0,1]
        frame = resize(frame, list(TARGET_SIZE))        # [3, 256, 256]
        frames_list.append(frame)

    if len(frames_list) < 2:
        return None

    frames = torch.stack(frames_list).half()            # [T, 3, 256, 256] float16

    # actions[i] = button state that produced frames[i+1] from frames[i]
    action_tensor = torch.tensor(action_ints[:-1], dtype=torch.long)
    actions = action_int_to_bits(action_tensor)         # [T-1, 8] float32

    return frames, actions


def _frame_number(p: Path) -> int:
    m = FRAME_RE.search(p.name)
    return int(m.group("frame")) if m else 0


def preprocess_episode(episode_dir: Path, out_dir: Path) -> dict | None:
    result = load_episode_frames(episode_dir)
    if result is None:
        return None

    frames, actions = result
    episode_id = episode_dir.name
    ep_out = out_dir / episode_id
    ep_out.mkdir(parents=True, exist_ok=True)

    frames_path = ep_out / "frames.pt"
    actions_path = ep_out / "actions.pt"
    meta_path = ep_out / "meta.json"

    torch.save(frames, frames_path)
    torch.save(actions, actions_path)

    meta = {
        "episode_id": episode_id,
        "frames_path": str(frames_path),
        "actions_path": str(actions_path),
        "frame_count": int(frames.shape[0]),
        "action_count": int(actions.shape[0]),
        "frame_shape": list(frames.shape),
        "action_shape": list(actions.shape),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not episode_dirs:
        raise ValueError(f"No episode directories found under {input_dir}")

    manifest = []
    for ep_dir in tqdm(episode_dirs, desc="Episodes"):
        meta = preprocess_episode(ep_dir, output_dir)
        if meta is not None:
            manifest.append(meta)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    print(f"Processed {len(manifest)} episodes → {manifest_path}")


if __name__ == "__main__":
    main()
