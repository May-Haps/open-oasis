from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

NUM_ACTIONS = 15  # Procgen discrete action space


class CoinRunDataset(Dataset):
    """
    Loads preprocessed CoinRun episodes produced by data/preprocess_coinrun.py.

    Each episode folder contains:
        frames.bin   — uint8 [T, 64, 64, 3] raw binary (numpy memmap)
        actions.bin  — uint8 [T-1] raw Procgen action indices (0–14)
        meta.json    — episode metadata

    Returns:
        frames:     float32 [clip_len, 3, 64, 64] in [0, 1]
        actions:    float32 [clip_len, 15] one-hot encoded
        episode_id: str
        start:      int
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
            raise ValueError(
                f"No valid clips found in {self.processed_dir} for clip_len={clip_len}"
            )

    def _load_manifest(self) -> list[dict]:
        manifest_path = self.processed_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.jsonl in {self.processed_dir}. "
                "Run data/preprocess_coinrun.py first."
            )
        with manifest_path.open() as f:
            return [json.loads(line) for line in f]

    def _build_index(self) -> list[tuple[int, int]]:
        index = []
        for ep_idx, ep in enumerate(self.episodes):
            T = ep["frame_count"]
            if T < self.clip_len:
                continue
            for start in range(0, T - self.clip_len + 1, self.stride):
                index.append((ep_idx, start))
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.index[idx]
        ep = self.episodes[ep_idx]
        T = ep["frame_count"]

        # memmap — O(1) seek, no full-episode load
        frames_mm = np.memmap(ep["frames_path"], dtype=np.uint8, mode="r",
                               shape=(T, 64, 64, 3))
        actions_mm = np.memmap(ep["actions_path"], dtype=np.uint8, mode="r",
                                shape=(T - 1,))

        frames_np  = frames_mm[start : start + self.clip_len].copy()   # [clip_len, 64, 64, 3]
        action_ids = actions_mm[start : start + self.clip_len - 1].copy()  # [clip_len-1]

        del frames_mm, actions_mm

        # [clip_len, H, W, C] → [clip_len, C, H, W], float [0,1]
        frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0

        # one-hot actions; prepend zero vector for the prompt frame
        action_tensor = torch.from_numpy(action_ids.astype(np.int64))
        transition_actions = F.one_hot(action_tensor, num_classes=NUM_ACTIONS).float()
        clip_actions = torch.zeros(self.clip_len, NUM_ACTIONS)
        clip_actions[1:] = transition_actions

        return {
            "frames":     frames,
            "actions":    clip_actions,
            "episode_id": ep["episode_id"],
            "start":      start,
        }
