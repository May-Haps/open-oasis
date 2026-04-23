# Data Pipeline Quick Guide

This repo now uses a Super Mario Bros. preprocessing pipeline for training directly on frame clips plus 8-button NES actions.

## Files

### `data_utils.py`

What it does:
- preprocesses raw SMB episode folders into frame-and-action tensors

Input:
- a raw SMB dataset folder
- one episode folder per run
- one PNG per frame, with the action integer in the filename or PNG metadata

Output:
- one processed folder per episode containing:
  - `frames.pt`
  - `actions.pt`
  - `meta.json`
- one dataset manifest:
  - `manifest.jsonl`

High-level behavior:
- reads PNG frames in episode order
- decodes the action integer into 8 buttons in this order:
  - `A, up, left, B, start, right, down, select`
- stores RGB frames on disk as compact `uint8` tensors
- stores transition actions as `[T - 1, 8]`

### `dataset.py`

What it does:
- defines the PyTorch `Dataset` used during training

Input:
- a processed dataset folder created by `data_utils.py`

Output:
- one training sample at a time:
  - `frames`: `[32, 3, 240, 256]`
  - `actions`: `[32, 8]`
  - `episode_id`
  - `start`

High-level behavior:
- loads processed frames and actions from disk
- normalizes frames to `[0, 1]` on load
- slices episodes into fixed-length clips
- prepends a zero action at the first timestep so the clip format stays aligned with prompt-frame conditioning

## How To Run Preprocessing

```bash
python data_utils.py \
  --input-dir /path/to/smb_raw \
  --output-dir /path/to/processed_smb
```

Future-facing flag:

```bash
python data_utils.py \
  --input-dir /path/to/smb_raw \
  --output-dir /path/to/processed_smb \
  --use-vae
```

Right now `--use-vae` intentionally fails fast because the Mario VAE path is not implemented yet.

## How To Use The Dataset

`dataset.py` is not a script you run over the whole dataset. You import it from training code.

Example:

```python
from torch.utils.data import DataLoader
from dataset import ProcessedGameDataset

ds = ProcessedGameDataset(
    "/path/to/processed_smb",
    clip_len=32,
    stride=8,
)

loader = DataLoader(ds, batch_size=4, shuffle=True)
batch = next(iter(loader))

print(batch["frames"].shape)   # [4, 32, 3, 240, 256]
print(batch["actions"].shape)  # [4, 32, 8]
```

## Important Shape Convention

Processed episode files on disk:
- `frames.pt`: `[T, 3, 240, 256]` stored as `uint8`
- `actions.pt`: `[T - 1, 8]`

Samples returned by the dataset:
- `frames`: `[32, 3, 240, 256]` normalized to `[0, 1]`
- `actions`: `[32, 8]`

Why the difference:
- the processed action file stores transition actions
- the dataset adds a zero action at the first timestep for the prompt frame

## In One Sentence

- `data_utils.py` builds the processed SMB dataset
- `dataset.py` loads that processed dataset for training
