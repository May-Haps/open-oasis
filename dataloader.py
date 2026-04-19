from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

ACTION_DIM = 25


class MinecraftLatentDataset(Dataset):
    def __init__(
        self,
        processed_dir: str | Path,
        clip_len: int = 32,
        stride: int = 1,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.clip_len = clip_len
        self.stride = stride

        self.episodes = self._load_manifest(self.processed_dir)
        self.index = self._build_index()

        if not self.index:
            raise ValueError(f"No valid clips found in {self.processed_dir} for clip_len={clip_len}")

    def _load_manifest(self, processed_dir: Path) -> list[dict]:
        manifest_path = processed_dir / "manifest.jsonl"
        if manifest_path.exists():
            with manifest_path.open() as handle:
                return [json.loads(line) for line in handle]

        episodes = []
        for meta_path in sorted(processed_dir.rglob("meta.json")):
            episodes.append(json.loads(meta_path.read_text()))
        return episodes

    def _build_index(self) -> list[tuple[int, int]]:
        index = []
        for episode_idx, episode in enumerate(self.episodes):
            frame_count = episode["frame_count"]
            action_count = episode["action_count"]

            if action_count != frame_count - 1:
                continue
            if frame_count < self.clip_len:
                continue

            max_start = frame_count - self.clip_len
            for start in range(0, max_start + 1, self.stride):
                index.append((episode_idx, start))
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        episode_idx, start = self.index[idx]
        episode = self.episodes[episode_idx]

        latents = torch.load(episode["latents_path"], weights_only=True).float()
        actions = torch.load(episode["processed_actions_path"], weights_only=True).float()

        latents_clip = latents[start : start + self.clip_len]
        transition_actions = actions[start : start + self.clip_len - 1]

        # Match Oasis inference behavior: the first frame in the clip is a prompt frame.
        clip_actions = torch.zeros(self.clip_len, ACTION_DIM, dtype=actions.dtype)
        clip_actions[1:] = transition_actions

        return {
            "latents": latents_clip,
            "actions": clip_actions,
            "episode_id": episode["episode_id"],
            "start": start,
        }
