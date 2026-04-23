from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from tqdm import tqdm

ACTION_KEYS = ["A", "up", "left", "B", "start", "right", "down", "select"]
ACTION_DIM = len(ACTION_KEYS)
IMAGE_EXTENSIONS = {".png"}

EPISODE_SUFFIX_RE = re.compile(r"_e(?P<episode>\d+)_(?P<world>\d+)-(?P<level>\d+)_(?P<outcome>win|fail)$")
FRAME_RE = re.compile(
    r"_e(?P<episode>\d+)_(?P<world>\d+)-(?P<level>\d+)_f(?P<frame>\d+)_a(?P<action>\d+)_"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.(?P<outcome>win|fail)\.png$"
)


def parse_episode_dir(dir_path: Path) -> dict:
    match = EPISODE_SUFFIX_RE.search(dir_path.name)
    if match is None:
        raise ValueError(f"Episode directory name does not match SMB format: {dir_path.name}")

    return {
        "episode_id": dir_path.name,
        "episode": int(match.group("episode")),
        "world": int(match.group("world")),
        "level": int(match.group("level")),
        "outcome": match.group("outcome"),
    }


def read_png_action(frame_path: Path) -> int | None:
    with Image.open(frame_path) as image:
        text_fields = {}
        text_fields.update(getattr(image, "text", {}))
        text_fields.update(image.info)

    for key in ("BP1", "tEXtBP1"):
        value = text_fields.get(key)
        if value is None:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        return int(value)

    return None


def parse_frame_path(frame_path: Path) -> dict:
    match = FRAME_RE.search(frame_path.name)
    if match is not None:
        action_int = int(match.group("action"))
        frame_index = int(match.group("frame"))
        return {
            "frame_index": frame_index,
            "action_int": action_int,
            "episode": int(match.group("episode")),
            "world": int(match.group("world")),
            "level": int(match.group("level")),
            "outcome": match.group("outcome"),
            "timestamp": match.group("timestamp"),
            "path": frame_path,
        }

    action_int = read_png_action(frame_path)
    if action_int is None:
        raise ValueError(f"Could not parse action from filename or PNG metadata: {frame_path}")

    frame_match = re.search(r"_f(?P<frame>\d+)_", frame_path.name)
    if frame_match is None:
        raise ValueError(f"Could not parse frame index from filename: {frame_path}")

    episode_meta = parse_episode_dir(frame_path.parent)
    return {
        "frame_index": int(frame_match.group("frame")),
        "action_int": action_int,
        "episode": episode_meta["episode"],
        "world": episode_meta["world"],
        "level": episode_meta["level"],
        "outcome": episode_meta["outcome"],
        "timestamp": None,
        "path": frame_path,
    }


def decode_action(action_int: int) -> torch.Tensor:
    if not 0 <= action_int <= 255:
        raise ValueError(f"Expected SMB action in [0, 255], got {action_int}")
    bits = [(action_int >> shift) & 1 for shift in range(7, -1, -1)]
    return torch.tensor(bits, dtype=torch.float32)


def load_frame_tensor(frame_path: Path) -> torch.Tensor:
    with Image.open(frame_path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
    return rearrange(torch.from_numpy(array), "h w c -> c h w")


def find_episodes(input_dir: Path) -> list[dict]:
    frames_by_episode: dict[Path, list[dict]] = defaultdict(list)

    for frame_path in sorted(input_dir.rglob("*")):
        if not frame_path.is_file() or frame_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        frames_by_episode[frame_path.parent].append(parse_frame_path(frame_path))

    episodes = []
    for episode_dir, frame_entries in sorted(frames_by_episode.items(), key=lambda item: item[0].name):
        episode_meta = parse_episode_dir(episode_dir)
        sorted_frames = sorted(frame_entries, key=lambda row: row["frame_index"])

        for expected, frame_entry in enumerate(sorted_frames, start=1):
            if frame_entry["frame_index"] != expected:
                raise ValueError(
                    f"Non-consecutive frame numbering in {episode_dir}: "
                    f"expected frame {expected}, found {frame_entry['frame_index']}"
                )

        first = sorted_frames[0]
        if first["world"] != episode_meta["world"] or first["level"] != episode_meta["level"]:
            raise ValueError(f"Frame metadata mismatch inside {episode_dir}")

        episodes.append(
            {
                **episode_meta,
                "source_dir": episode_dir,
                "frames": sorted_frames,
            }
        )

    return episodes


def preprocess_episode(episode: dict, out_dir: Path) -> dict:
    episode_id = episode["episode_id"]
    episode_out = out_dir / episode_id
    episode_out.mkdir(parents=True, exist_ok=True)

    frames = torch.stack([load_frame_tensor(row["path"]) for row in episode["frames"]], dim=0)
    frame_actions = torch.stack([decode_action(row["action_int"]) for row in episode["frames"]], dim=0)

    # Each stored action describes the transition from frame t to frame t + 1.
    actions = frame_actions[:-1]

    frames_rel_path = Path(episode_id) / "frames.pt"
    actions_rel_path = Path(episode_id) / "actions.pt"
    meta_rel_path = Path(episode_id) / "meta.json"

    frames_path = out_dir / frames_rel_path
    actions_path = out_dir / actions_rel_path
    meta_path = out_dir / meta_rel_path

    # Keep frames compact on disk; the dataset normalizes to [0, 1] on load.
    torch.save(frames.contiguous(), frames_path)
    torch.save(actions.contiguous(), actions_path)

    meta = {
        "episode_id": episode_id,
        "source_dir": str(episode["source_dir"]),
        "world": episode["world"],
        "level": episode["level"],
        "outcome": episode["outcome"],
        "frame_count": int(len(frames)),
        "action_count": int(len(actions)),
        "frame_shape": list(frames.shape),
        "action_shape": list(actions.shape),
        "frame_dtype": str(frames.dtype),
        "action_dtype": str(actions.dtype),
        "frame_size_hw": [int(frames.shape[-2]), int(frames.shape[-1])],
        "action_keys": ACTION_KEYS,
        "frames_path": str(frames_rel_path),
        "actions_path": str(actions_rel_path),
        "meta_path": str(meta_rel_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--use-vae",
        action="store_true",
        help="Reserved for future Mario latent preprocessing. The current SMB pipeline saves RGB frames directly.",
    )
    parser.add_argument(
        "--vae-ckpt",
        type=str,
        default=None,
        help="Reserved for future use when --use-vae is implemented.",
    )
    args = parser.parse_args()

    if args.use_vae:
        raise NotImplementedError(
            "use_vae=True is not implemented for the SMB pipeline yet. "
            "Run without --use-vae to preprocess RGB frame clips directly."
        )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = find_episodes(input_dir)
    if not episodes:
        raise ValueError(f"No SMB episode frames found under {input_dir}")

    manifest = []
    for episode in tqdm(episodes, desc="Episodes"):
        manifest.append(preprocess_episode(episode, output_dir))

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
