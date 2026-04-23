"""
Convert the CoinRun ArrayRecord dataset to per-episode numpy memmap files.

Install dependencies:
    pip install array-record grain-nightly

Download dataset:
    huggingface-cli download --repo-type dataset p-doom/coinrun-dataset \
        --local-dir data/coinrun_raw

Run:
    python data/preprocess_coinrun.py \
        --input-dir data/coinrun_raw/train \
        --output-dir data/coinrun_processed/train

    python data/preprocess_coinrun.py \
        --input-dir data/coinrun_raw/val \
        --output-dir data/coinrun_processed/val

    # To inspect the raw record format before converting:
    python data/preprocess_coinrun.py --input-dir data/coinrun_raw/train --inspect
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_records(input_dir: Path):
    """
    Yield deserialized records from all ArrayRecord shards in input_dir.
    Each record is expected to contain:
        obs:    uint8 [64, 64, 3]  — RGB frame
        action: int                — Procgen action index (0–14)
        done:   bool               — episode boundary flag
    """
    try:
        from array_record.python.array_record_module import ArrayRecordReader
    except ImportError:
        raise ImportError(
            "Install array-record: pip install array-record\n"
            "If unavailable, try: pip install grain-nightly"
        )

    shard_paths = sorted(input_dir.glob("*.array_record")) or \
                  sorted(input_dir.glob("*.arrayrecord")) or \
                  sorted(input_dir.glob("*.riegeli"))

    if not shard_paths:
        raise FileNotFoundError(
            f"No ArrayRecord shards found in {input_dir}.\n"
            "Expected files matching *.array_record or *.arrayrecord"
        )

    for shard_path in shard_paths:
        reader = ArrayRecordReader(str(shard_path))
        for raw_bytes in reader:
            yield _deserialize(raw_bytes)


def _deserialize(raw_bytes: bytes) -> dict:
    """
    Deserialize a record. Tries pickle then msgpack.
    If neither works, prints the raw bytes prefix and raises so you can adapt.
    """
    # Try pickle (common for numpy-based datasets)
    try:
        return pickle.loads(raw_bytes)
    except Exception:
        pass

    # Try msgpack
    try:
        import msgpack
        import msgpack_numpy
        msgpack_numpy.patch()
        return msgpack.unpackb(raw_bytes, raw=False)
    except Exception:
        pass

    raise ValueError(
        f"Could not deserialize record. Raw bytes prefix: {raw_bytes[:64]!r}\n"
        "Adapt _deserialize() in preprocess_coinrun.py to match the actual format.\n"
        "Run with --inspect to examine the first record."
    )


def inspect(input_dir: Path) -> None:
    """Print the first record to reveal the schema."""
    print(f"Inspecting first record from {input_dir} ...")
    for record in load_records(input_dir):
        print("Keys:", list(record.keys()) if isinstance(record, dict) else type(record))
        if isinstance(record, dict):
            for k, v in record.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: dtype={v.dtype} shape={v.shape}")
                else:
                    print(f"  {k}: {type(v).__name__} = {v!r}")
        break


def convert(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_frames: list[np.ndarray] = []
    episode_actions: list[int] = []
    episode_idx = 0
    manifest = []

    def flush_episode():
        nonlocal episode_idx
        if len(episode_frames) < 2:
            return

        T = len(episode_frames)
        ep_dir = output_dir / f"ep_{episode_idx:06d}"
        ep_dir.mkdir(exist_ok=True)

        # frames: uint8 [T, 64, 64, 3]
        frames_path = ep_dir / "frames.bin"
        frames_mm = np.memmap(frames_path, dtype=np.uint8, mode="w+", shape=(T, 64, 64, 3))
        frames_mm[:] = np.stack(episode_frames)
        del frames_mm  # flush

        # actions: uint8 [T-1] — transition action at each step
        actions_path = ep_dir / "actions.bin"
        actions_mm = np.memmap(actions_path, dtype=np.uint8, mode="w+", shape=(T - 1,))
        actions_mm[:] = np.array(episode_actions[:T - 1], dtype=np.uint8)
        del actions_mm

        meta = {
            "episode_id": f"ep_{episode_idx:06d}",
            "frames_path": str(frames_path),
            "actions_path": str(actions_path),
            "frame_count": T,
            "action_count": T - 1,
        }
        (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        manifest.append(meta)
        episode_idx += 1

    for record in tqdm(load_records(input_dir), desc="Converting"):
        # --- adapt these keys to match what --inspect shows ---
        obs    = record.get("obs") or record.get("observation") or record.get("frame")
        action = record.get("action")
        done   = record.get("done") or record.get("is_last") or False
        # ------------------------------------------------------

        if obs is None or action is None:
            raise KeyError(
                f"Expected keys 'obs' and 'action'. Got: {list(record.keys())}\n"
                "Run --inspect and update the key names above."
            )

        obs = np.asarray(obs, dtype=np.uint8)
        if obs.shape != (64, 64, 3):
            raise ValueError(f"Expected obs shape (64, 64, 3), got {obs.shape}")

        episode_frames.append(obs)
        episode_actions.append(int(action))

        if done:
            flush_episode()
            episode_frames = []
            episode_actions = []

    # flush any trailing partial episode
    flush_episode()

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    print(f"Converted {episode_idx} episodes → {manifest_path}")
    print(f"Total frames: {sum(m['frame_count'] for m in manifest):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  type=str, required=True)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--inspect",    action="store_true",
                        help="Print the first record and exit")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if args.inspect:
        inspect(input_dir)
        return

    if not args.output_dir:
        parser.error("--output-dir is required unless using --inspect")

    convert(input_dir, Path(args.output_dir))


if __name__ == "__main__":
    main()
