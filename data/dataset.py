from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

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

        zip_paths = sorted(processed_dir.glob("*.zip"))
        if zip_paths:
            return self._load_zipped_manifest(zip_paths)

        episodes = []
        for meta_path in sorted(processed_dir.rglob("meta.json")):
            episodes.append(json.loads(meta_path.read_text()))
        return episodes

    def _load_zipped_manifest(self, zip_paths: list[Path]) -> list[dict]:
        episodes_by_id: dict[str, dict] = {}

        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path) as archive:
                for member_name in archive.namelist():
                    if member_name.endswith("/meta.json"):
                        episode = json.loads(archive.read(member_name).decode())
                        episode_id = episode["episode_id"]
                        existing = episodes_by_id.get(episode_id, {})
                        episodes_by_id[episode_id] = {**existing, **episode}
                    elif member_name.endswith("/latents.pt"):
                        episode_id = Path(member_name).parent.name
                        episode = episodes_by_id.setdefault(episode_id, {"episode_id": episode_id})
                        episode["latents_zip_path"] = str(zip_path)
                        episode["latents_zip_member"] = member_name
                    elif member_name.endswith("/actions.one_hot.pt"):
                        episode_id = Path(member_name).parent.name
                        episode = episodes_by_id.setdefault(episode_id, {"episode_id": episode_id})
                        episode["actions_zip_path"] = str(zip_path)
                        episode["actions_zip_member"] = member_name

        return sorted(episodes_by_id.values(), key=lambda episode: episode["episode_id"])

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

    def _load_tensor(self, episode: dict, path_key: str, zip_path_key: str, zip_member_key: str) -> torch.Tensor:
        direct_path = episode.get(path_key)
        if direct_path and Path(direct_path).exists():
            return torch.load(direct_path, weights_only=True).float()

        zip_path = episode.get(zip_path_key)
        zip_member = episode.get(zip_member_key)
        if zip_path and zip_member:
            with zipfile.ZipFile(zip_path) as archive:
                with archive.open(zip_member) as handle:
                    buffer = io.BytesIO(handle.read())
            return torch.load(buffer, weights_only=True).float()

        raise FileNotFoundError(
            f"Could not find tensor for episode={episode['episode_id']} "
            f"using keys path={path_key}, zip_path={zip_path_key}, zip_member={zip_member_key}"
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        episode_idx, start = self.index[idx]
        episode = self.episodes[episode_idx]

        latents = self._load_tensor(episode, "latents_path", "latents_zip_path", "latents_zip_member")
        actions = self._load_tensor(episode, "processed_actions_path", "actions_zip_path", "actions_zip_member")

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
