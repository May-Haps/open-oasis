# Data Layout

Put processed Super Mario datasets under `data/` like this:

```text
data/
  processed_smb/
    manifest.jsonl
    episode_1/
      frames.pt
      actions.pt
      meta.json
    episode_2/
      ...
  processed_smb_holdout/
    manifest.jsonl
    ...
```

Each `processed_*` folder is one dataset root. `dataset.py` should point at one of those roots.

Example:

```python
from dataset import ProcessedGameDataset

smb = ProcessedGameDataset("data/processed_smb", clip_len=32, stride=8)
```

# Training On Multiple Processed Datasets

If you want to train on multiple processed datasets, combine them in training code:

```python
from torch.utils.data import ConcatDataset, DataLoader
from dataset import ProcessedGameDataset

train_a = ProcessedGameDataset("data/processed_smb", clip_len=32, stride=8)
train_b = ProcessedGameDataset("data/processed_smb_holdout", clip_len=32, stride=8)

train_ds = ConcatDataset([train_a, train_b])
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

batch = next(iter(train_loader))
print(batch["frames"].shape)   # [4, 32, 3, 240, 256]
print(batch["actions"].shape)  # [4, 32, 8]
```

In short:

- one task split = one `processed_*` folder
- one `processed_*` folder = one dataset root for `ProcessedGameDataset`
- combine multiple dataset roots in training with `ConcatDataset`
