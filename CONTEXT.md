# open-oasis — Project Context

Drop this file into a new Claude conversation to resume with full context.

---

## What this project is

A video-generating world model trained on CoinRun (Procgen) gameplay.
Based on the original Oasis 500M (Minecraft), stripped of VAE and Minecraft-specific code,
retargeted to 64×64 pixel-space diffusion on CoinRun with WandB logging and H100 training.

---

## Architecture decisions

| Decision | Choice | Reason |
|---|---|---|
| VAE | No — pixel space | 64×64 is small enough for direct diffusion. VAE was only needed for 360×640. |
| Model | Spatiotemporal DiT | Alternating spatial + temporal axial attention, AdaLN-zero conditioning |
| Diffusion objective | v-parameterization | Standard for video diffusion |
| Noise schedule | Sigmoid beta | Smooth schedule, works well empirically |
| Action encoding | 15-dim one-hot | Procgen has 15 discrete actions |
| Positional encoding | Rotary (RoPE) | Spatial 2D axial + temporal 1D |
| Inference | DDIM (10 steps) | Fast, deterministic |

---

## Model size — CoinRunWorldModel (current)

```python
# model/dit.py — CoinRunWorldModel()
DiT(
    input_h=64, input_w=64,
    patch_size=8,
    in_channels=3,
    hidden_size=640,   # bumped from 512 → better H100 utilization
    depth=8,           # bumped from 6
    num_heads=8,
    external_cond_dim=15,
    max_frames=32,
)
# ~100M params, ~50% H100 SXM utilization, ~2 epochs in 8h on H100 SXM
```

Previous sizes considered:
- 170M (hidden=1024, depth=12) — original, too large for first run
- 58M (hidden=512, depth=6) — too small, only ~35% H100 util
- **100M (hidden=640, depth=8) — current, good balance**

---

## File structure

```
open-oasis/
  model/
    dit.py                  # DiT model + MarioWorldModel() + CoinRunWorldModel()
    attention.py            # SpatialAxialAttention, TemporalAxialAttention
    rotary_embedding.py     # RoPE (moved from rotary_embedding_torch.py)
    utils.py                # sigmoid_beta_schedule, action_int_to_bits (Mario 8-bit)
  data/
    preprocess_coinrun.py   # ArrayRecord → per-episode numpy memmap
    dataset_coinrun.py      # CoinRunDataset (memmap, O(1) clip access)
    preprocess.py           # Mario PNG → frames.pt/actions.pt
    dataset.py              # MarioPixelDataset
  training/
    noise_scheduler.py      # NoiseScheduler (v-param, sigmoid schedule)
    model_trainer.py        # Generic training step wrapper
    rollout_sampler.py      # DDIM rollout, no VAE
    training_manager.py     # Mario training manager (not used for CoinRun)
  train_coinrun.py          # Main CoinRun training script (use this)
  train.py                  # Mario training script
  generate.py               # Inference script
  setup.sh                  # Conda env + pip install script
```

---

## Training script — train_coinrun.py

Key CONFIG values at top of file:

```python
CONFIG = {
    "train_dir":  "data/coinrun_processed/train",
    "val_dir":    "data/coinrun_processed/val",
    "save_dir":   "runs/coinrun_v1",
    "max_noise_level": 1000,
    "clip_len":   32,
    "clip_stride": 8,
    "epochs":     10,
    "batch_size": 256,
    "lr":         1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 2000,
    "grad_clip":  1.0,
    "ckpt_every_steps":    2000,
    "rollout_every_steps": 5000,
    "log_every_steps":     50,
    "ddim_steps":          10,
    "n_prompt_frames":     1,
    "rollout_frames":      16,
    "n_rollout_samples":   4,
    "wandb_project": "coinrun-world-model",
}
```

WandB logs: `train/loss`, `train/lr`, `perf/elapsed_hours`, `epoch/val_loss`,
`rollout/video_0..3` (mp4), `rollout/grid` (GT top / generated bottom comparison).

Checkpoints saved at: every 2000 steps, every epoch, and `ckpt_best.pt` (best val loss).

---

## Data pipeline — CoinRun

**Dataset source:** `p-doom/coinrun-dataset` on HuggingFace (~50M frames, 64×64 uint8)

**Format:** Google ArrayRecord (per-step records: `obs [64,64,3]`, `action int`, `done bool`)

**Why we convert to memmap instead of reading ArrayRecord directly:**
ArrayRecord stores individual steps, not episodes. Clip sampling (32-frame windows with stride 8)
requires episode-grouped access. Preprocessing does this grouping once; training then uses
O(1) numpy memmap reads with standard PyTorch DataLoader.

**Preprocessing output:**
```
data/coinrun_processed/train|val/
  manifest.jsonl
  ep_000000/
    frames.bin    # uint8 [T, 64, 64, 3] memmap
    actions.bin   # uint8 [T-1] memmap
    meta.json
```

**Dataset class returns:**
- `frames`: float32 `[clip_len, 3, 64, 64]` in `[0, 1]`
- `actions`: float32 `[clip_len, 15]` one-hot (zero vector prepended for first frame)

---

## Setup

```bash
# Create env and install all deps
bash setup.sh              # CUDA 12.1 (default)
bash setup.sh --cuda 124   # CUDA 12.4 for newer drivers

conda activate open-oasis

# Download dataset (~hundreds of GB)
huggingface-cli download --repo-type dataset p-doom/coinrun-dataset \
    --local-dir data/coinrun_raw

# Inspect schema (run once — verify key names match obs/action/done)
python data/preprocess_coinrun.py --input-dir data/coinrun_raw/train --inspect

# Preprocess
python data/preprocess_coinrun.py \
    --input-dir data/coinrun_raw/train \
    --output-dir data/coinrun_processed/train
python data/preprocess_coinrun.py \
    --input-dir data/coinrun_raw/val \
    --output-dir data/coinrun_processed/val

# Train
python train_coinrun.py
python train_coinrun.py --resume runs/coinrun_v1/ckpt_step_0002000.pt
```

**If --inspect shows different key names** than `obs`/`action`/`done`, edit
`data/preprocess_coinrun.py` lines ~149-152 (marked `# --- adapt these keys ---`).

---

## GPU / cost guidance

| GPU | BF16 TFLOPS | ~Wall time (2 epochs) | ~Cost |
|---|---|---|---|
| 1× H100 SXM | 494 | 8 hrs | $36 |
| 2× H100 SXM | 494×2 | 4.5 hrs | $40 |
| 1× H100 PCIe | 204 | 19 hrs | $62 |
| 1× RTX 6000 Ada | 97 | 40 hrs | $36 |

**Recommendation:** 1× H100 SXM. Same cost as RTX 6000 Ada, 5× faster.
2× GPUs only worth it for wall time, not cost — and requires adding DDP to `train_coinrun.py`
(currently single-GPU only, ~15 lines to add).

Code uses `torch.compile(model)` for ~10-20% extra speedup on H100.

---

## Mario dataset (secondary, not currently training)

- 737k frames, 256×240 PNG, 8-bit bitmask actions
- Action encoding: 8-dim binary vector via `action_int_to_bits()` in `model/utils.py`
- Preprocessed to 64×64 float16 tensors by `data/preprocess.py`
- Model factory: `MarioWorldModel()` in `model/dit.py` (~15M params, `external_cond_dim=8`)
- Training script: `train.py`

---

## Known issues / watch-outs

- `preprocess_coinrun.py` deserialization tries pickle then msgpack — if neither works,
  the `--inspect` output will tell you the raw bytes prefix; adapt `_deserialize()`.
- `train_coinrun.py` `wandb_entity` is `None` — set to your WandB username before running.
- `torch.load(..., weights_only=True)` in `load_checkpoint` — requires checkpoint was saved
  with only serializable objects (it was — just model/optimizer state dicts).
- At batch_size=256 on RTX 6000 Ada (48GB), may need to drop to batch_size=128 if OOM.
