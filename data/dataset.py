from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from model.utils import ACTION_DIM


class MarioPixelDataset(Dataset):
    """
    Loads preprocessed Mario episodes produced by data/preprocess.py.
    Each episode folder contains:
      frames.pt   — float16 tensor [T, 3, 256, 256] in [0, 1]
      actions.pt  — float32 tensor [T-1, 8] binary NES button bits
      meta.json   — episode metadata
    """

    def __init__(
        self,
        processed_dir: str | Path,
        clip_len: int = 32,
        stride: int = 8,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.clip_len = clip_len
        self.stride = stride

        self.episodes = self._load_manifest()
        self.index = self._build_index()

        if not self.index:
            raise ValueError(f"No valid clips found in {self.processed_dir} for clip_len={clip_len}")

    def _load_manifest(self) -> list[dict]:
        manifest_path = self.processed_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.jsonl found in {self.processed_dir}. Run data/preprocess.py first.")
        with manifest_path.open() as f:
            return [json.loads(line) for line in f]

    def _build_index(self) -> list[tuple[int, int]]:
        index = []
        for ep_idx, ep in enumerate(self.episodes):
            frame_count = ep["frame_count"]
            if frame_count < self.clip_len:
                continue
            for start in range(0, frame_count - self.clip_len + 1, self.stride):
                index.append((ep_idx, start))
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.index[idx]
        ep = self.episodes[ep_idx]

        frames = torch.load(ep["frames_path"], weights_only=True).float()  # [T, 3, 256, 256]
        actions = torch.load(ep["actions_path"], weights_only=True)        # [T-1, 8]

        frames_clip = frames[start : start + self.clip_len]                # [clip_len, 3, 256, 256]
        transition_actions = actions[start : start + self.clip_len - 1]   # [clip_len-1, 8]

        # First frame is a prompt — prepend a zero action to align indices
        clip_actions = torch.zeros(self.clip_len, ACTION_DIM, dtype=torch.float32)
        clip_actions[1:] = transition_actions

        return {
            "frames": frames_clip,
            "actions": clip_actions,
            "episode_id": ep["episode_id"],
            "start": start,
        }
