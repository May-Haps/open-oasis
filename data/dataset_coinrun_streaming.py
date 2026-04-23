from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

NUM_ACTIONS = 15


class CoinRunStreamingDataset(IterableDataset):
    """
    Streams clips directly from ArrayRecord shards. No preprocessing required.

    Each record in the ArrayRecord is one full episode:
        raw_video:       bytes  — uint8 [T, 64, 64, 3] packed flat
        sequence_length: int    — T
        actions:         int32  [T] — action at each frame

    Each DataLoader worker owns a disjoint slice of shards.
    Shards are shuffled per-worker using seed + worker_id.

    Returns dicts with:
        frames:  float32 [clip_len, 3, 64, 64] in [0, 1]
        actions: float32 [clip_len, 15] one-hot (zero vector for first frame)
    """

    def __init__(
        self,
        shard_dir: str | Path,
        clip_len: int = 32,
        stride: int = 8,
        seed: int = 42,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.clip_len = clip_len
        self.stride = stride
        self.seed = seed

        self.shards = sorted(self.shard_dir.glob("*.array_record"))
        if not self.shards:
            raise FileNotFoundError(f"No .array_record files found in {self.shard_dir}")

        self._total_clips = self._load_or_compute_len()

    def __len__(self) -> int:
        return self._total_clips

    def _load_or_compute_len(self) -> int:
        cache_path = self.shard_dir / f".clip_count_cl{self.clip_len}_s{self.stride}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())["total_clips"]

        from array_record.python.array_record_module import ArrayRecordReader

        # Read sequence_length from one episode to check uniformity
        reader0 = ArrayRecordReader(str(self.shards[0]))
        rec0 = pickle.loads(reader0.read([0])[0])
        seq_len = int(rec0["sequence_length"])
        clips_per_episode = max(0, (seq_len - self.clip_len) // self.stride + 1)

        # Count total episodes across all shards (num_records reads only the index)
        total_episodes = sum(
            ArrayRecordReader(str(s)).num_records() for s in self.shards
        )
        total_clips = total_episodes * clips_per_episode

        cache_path.write_text(json.dumps({
            "total_clips": total_clips,
            "total_episodes": total_episodes,
            "seq_len": seq_len,
            "clip_len": self.clip_len,
            "stride": self.stride,
        }))
        return total_clips

    def __iter__(self) -> Iterator[dict]:
        try:
            from array_record.python.array_record_module import ArrayRecordReader
        except ImportError:
            raise ImportError("pip install array-record")

        import torch.distributed as dist

        worker = get_worker_info()
        shards = list(self.shards)

        # split shards across DDP ranks first, then across DataLoader workers
        if dist.is_available() and dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
            shards = shards[rank::world_size]

        if worker is not None:
            shards = shards[worker.id :: worker.num_workers]
            rng = random.Random(self.seed + worker.id)
        else:
            rng = random.Random(self.seed)

        rng.shuffle(shards)

        for shard_path in shards:
            yield from self._stream_shard(shard_path, ArrayRecordReader)

    def _stream_shard(self, shard_path: Path, ArrayRecordReader) -> Iterator[dict]:
        reader = ArrayRecordReader(str(shard_path))
        n = reader.num_records()

        for i in range(n):
            raw = reader.read([i])[0]
            rec = pickle.loads(raw)
            T          = int(rec["sequence_length"])
            frames_np  = np.frombuffer(rec["raw_video"], dtype=np.uint8).reshape(T, 64, 64, 3)
            actions_np = np.asarray(rec["actions"], dtype=np.int64)  # [T]

            yield from self._clips(frames_np, actions_np, T)

    def _clips(
        self,
        frames_np: np.ndarray,   # [T, 64, 64, 3]
        actions_np: np.ndarray,  # [T]
        T: int,
    ) -> Iterator[dict]:
        if T < self.clip_len:
            return

        for start in range(0, T - self.clip_len + 1, self.stride):
            f = (
                torch.from_numpy(frames_np[start : start + self.clip_len].copy())
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )  # [clip_len, 3, 64, 64]

            # action[t] transitions frame t → t+1; zero-pad for the first frame
            ids = torch.from_numpy(actions_np[start : start + self.clip_len - 1])
            a = torch.zeros(self.clip_len, NUM_ACTIONS)
            a[1:] = F.one_hot(ids, num_classes=NUM_ACTIONS).float()

            yield {"frames": f, "actions": a}
