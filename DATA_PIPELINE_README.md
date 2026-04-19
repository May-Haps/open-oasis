# Data Pipeline Quick Guide

This is the minimal guide for using the MineRL preprocessing pipeline in this repo.

## Files

### `data_utils.py`

What it does:
- preprocesses raw MineRL trajectories into Oasis-ready tensors

Input:
- a raw MineRL dataset folder
- each trajectory folder should contain:
  - `recording.mp4`
  - `rendered.npz`
  - `metadata.json`
- the Oasis VAE checkpoint: `vit-l-20.safetensors`

Output:
- one processed folder per trajectory containing:
  - `latents.pt`
  - `actions.one_hot.pt`
  - `meta.json`
- one dataset manifest:
  - `manifest.jsonl`

High-level behavior:
- reads MineRL video
- maps MineRL actions into the 25-dim Oasis/VPT action schema
- encodes video frames into VAE latents
- saves processed tensors to disk

### `dataset.py`

What it does:
- defines the PyTorch `Dataset` used during training

Input:
- a processed dataset folder created by `data_utils.py`

Output:
- one training sample at a time:
  - `latents`: `[32, 16, 18, 32]`
  - `actions`: `[32, 25]`
  - `episode_id`
  - `start`

High-level behavior:
- loads processed latents and actions from disk
- slices them into 32-frame clips
- prepends a zero action at the first timestep so the format matches Oasis generation-time conditioning

### `utils.py`

What it does:
- shared helpers for action formatting and prompt loading

Relevant to this pipeline:
- defines the 25-dim `ACTION_KEYS`
- provides shared action formatting logic
- provides video/image loading used elsewhere in the repo

## How To Run Preprocessing

```bash
python data_utils.py \
  --input-dir /path/to/MineRLTreechop-v0 \
  --output-dir /path/to/processed_treechop \
  --vae-ckpt /path/to/vit-l-20.safetensors \
  --device cpu
```

On GPU:

```bash
python data_utils.py \
  --input-dir /path/to/MineRLTreechop-v0 \
  --output-dir /path/to/processed_treechop \
  --vae-ckpt /path/to/vit-l-20.safetensors \
  --device cuda:0
```

## How To Use The Dataset

`dataset.py` is not a script you run over the whole dataset. You import it from training code.

Example:

```python
from torch.utils.data import DataLoader
from dataset import MinecraftLatentDataset

ds = MinecraftLatentDataset(
    "/path/to/processed_treechop",
    clip_len=32,
    stride=8,
)

loader = DataLoader(ds, batch_size=4, shuffle=True)
batch = next(iter(loader))

print(batch["latents"].shape)  # [4, 32, 16, 18, 32]
print(batch["actions"].shape)  # [4, 32, 25]
```

## Important Shape Convention

Processed episode files on disk:
- `latents.pt`: `[T, 16, 18, 32]`
- `actions.one_hot.pt`: `[T - 1, 25]`

Samples returned by the dataset:
- `latents`: `[32, 16, 18, 32]`
- `actions`: `[32, 25]`

Why the difference:
- the processed action file stores transition actions
- the dataset adds a zero action at the first timestep for the prompt frame

## In One Sentence

- `data_utils.py` builds the processed dataset
- `dataset.py` loads that processed dataset for training
