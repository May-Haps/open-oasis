# Data Layout

Put processed MineRL datasets under `data/` like this:

```text
data/
  processed_treechop/
    manifest.jsonl
    episode_1/
      latents.pt
      actions.one_hot.pt
      meta.json
    episode_2/
      ...
  processed_obtainironpickaxe/
    manifest.jsonl
    episode_1/
      latents.pt
      actions.one_hot.pt
      meta.json
    episode_2/
      ...
  processed_diamond/
    manifest.jsonl
    ...
```

Each `processed_*` folder is one dataset root. `dataset.py` should point at one of those roots.

Example:

```python
from dataset import MinecraftLatentDataset

treechop = MinecraftLatentDataset("data/processed_treechop", clip_len=32, stride=8)
```

# Training On Multiple MineRL Datasets

If you want to train on all processed MineRL datasets, combine them in training code:

```python
from torch.utils.data import ConcatDataset, DataLoader
from dataset import MinecraftLatentDataset

treechop = MinecraftLatentDataset("data/processed_treechop", clip_len=32, stride=8)
iron = MinecraftLatentDataset("data/processed_obtainironpickaxe", clip_len=32, stride=8)
diamond = MinecraftLatentDataset("data/processed_diamond", clip_len=32, stride=8)

train_ds = ConcatDataset([treechop, iron, diamond])
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

batch = next(iter(train_loader))
print(batch["latents"].shape)   # [4, 32, 16, 18, 32]
print(batch["actions"].shape)   # [4, 32, 25]
```

In short:

- one MineRL task = one `processed_*` folder
- one `processed_*` folder = one dataset root for `MinecraftLatentDataset`
- combine multiple dataset roots in training with `ConcatDataset`
