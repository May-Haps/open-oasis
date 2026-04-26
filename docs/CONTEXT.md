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

## Model in use — CoinRunWorldModelSmall (current)

```python
# model/dit.py — CoinRunWorldModelSmall()
# ~57.8M params, hidden_size=512, depth=6, num_heads=8, max_frames=32
# Runs at ~3.3 it/s on 2× H100 SXM with batch_size=128 per GPU (GPU-bound, 98-100% util)
```

`CoinRunWorldModel()` (~100M, hidden=640, depth=8) exists in the file but is not currently used.

**VRAM at batch_size=128:** ~55 GB. Max tested: 160 (68.5 GB). 192 OOMs.

---

## File structure

```
open-oasis/
  model/
    dit.py                        # DiT + CoinRunWorldModelSmall + CoinRunWorldModel + MarioWorldModel
    attention.py                  # SpatialAxialAttention, TemporalAxialAttention (uses F.scaled_dot_product_attention / flash attn)
    rotary_embedding.py           # RoPE — cached_freqs_seq_len is a plain int (NOT a buffer) to avoid torch.compile graph break
    utils.py                      # sigmoid_beta_schedule, action_int_to_bits
  data/
    dataset_coinrun_streaming.py  # CoinRunStreamingDataset (IterableDataset, reads ArrayRecord directly — NO preprocessing needed)
    dataset_coinrun.py            # CoinRunDataset (memmap, not currently used)
    preprocess_coinrun.py         # ArrayRecord → memmap (not needed for current pipeline)
    preprocess.py                 # Mario preprocessing
    dataset.py                    # MarioPixelDataset
  training/
    noise_scheduler.py            # NoiseScheduler (v-param, sigmoid schedule)
    model_trainer.py              # Generic training step wrapper
    rollout_sampler.py            # DDIM rollout (not used — generate_rollout is in train_coinrun.py)
    training_manager.py           # Mario training manager
  scripts/
    infer_coinrun.py              # CoinRun inference: load ckpt, run DDIM, save mp4 with keyboard overlay
    interactive.py                # Gradio web UI for interactive real-time inference
    visualize_dataset.py          # Render dataset episodes with action/keyboard overlay to mp4
    upload_checkpoints.py         # Upload .pt files to HuggingFace Hub via HTTP (no git needed)
    generate.py                   # Legacy Mario inference (not used for CoinRun)
  train_coinrun.py                # Main CoinRun training script — USE THIS
  train.py                        # Legacy Mario training script
  setup.sh                        # Creates /venv/open-oasis venv and installs all deps
  docs/CONTEXT.md
  docs/SCALING_ABLATION.md
```

**Python env:** `/venv/open-oasis/bin/python3` (NOT conda, NOT system python)

---

## Training script — train_coinrun.py

### Current CONFIG (as of last run)

```python
CONFIG = {
    "train_dir":  "data/coinrun/train",     # ArrayRecord shards, read directly — no preprocessing
    "val_dir":    "data/coinrun/val",
    "save_dir":   "runs/coinrun_v1",
    "max_noise_level": 1000,
    "clip_len":   32,
    "clip_stride": 8,
    "epochs":     10,
    "batch_size": 128,                      # per GPU; effective batch = 256 with 2× DDP
    "lr":         1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 2000,
    "grad_clip":  1.0,
    "ckpt_every_steps":    10000,
    "rollout_every_steps": 1000,
    "val_every_steps":     5000,            # fast val loss (50 batches subset)
    "val_subset_batches":  50,
    "log_every_steps":     50,
    "ddim_steps":          10,
    "n_prompt_frames":     1,
    "rollout_frames":      16,
    "n_rollout_samples":   4,
    "wandb_project": "coinrun-world-model",
    "wandb_entity":  "spring26-gen-ai",
}
```

### Logging cadence

| Event | Frequency | Time cost | WandB key |
|---|---|---|---|
| Train loss + LR | every 50 steps | instant | `train/loss`, `train/lr` |
| Fast val loss (50 batches) | every 5000 steps | ~9s | `val/loss` |
| Rollout video + grid | every 1000 steps | ~20s | `rollout/video`, `rollout/grid` |
| Checkpoint | every 10000 steps | instant | saved to `runs/coinrun_v1/` |
| Full val loss + PSNR + SSIM | every epoch | ~4.7 min | `epoch/val_loss`, `epoch/psnr`, `epoch/ssim` |

Rollout videos also saved locally to `runs/coinrun_v1/rollouts/rollout_step_XXXXXXX.mp4`.

### Training launch

```bash
# DDP on 2× H100 (required — use torchrun, not python)
torchrun --nproc_per_node=2 train_coinrun.py

# Resume from checkpoint
torchrun --nproc_per_node=2 train_coinrun.py --resume runs/coinrun_v1/ckpt_step_0110000.pt
```

**Do NOT use `python train_coinrun.py`** — that runs single GPU only, leaving GPU 1 idle.

### torch.compile

Applied after DDP wrap in `main()`. The rotary embedding had two compile bugs fixed:
- `copy_()` → `fill_()` (tensor vs scalar)
- `cached_freqs_seq_len.item()` → plain Python int attribute (eliminated graph break)

---

## Data pipeline — CoinRun

**Dataset source:** `p-doom/coinrun-dataset` on HuggingFace

**Actual size:** ~98M frames — 612,900 train episodes × 160 frames each (CONTEXT.md previously said ~50M, that was wrong)

**Format:** Google ArrayRecord — each shard has 100 episodes, each episode: `raw_video` (bytes, uint8 [T,64,64,3]), `sequence_length` (int), `actions` (int32 [T])

**No preprocessing required.** `CoinRunStreamingDataset` reads ArrayRecord directly with `pickle.loads()`.

**Clip stats:**
- clip_len=32, stride=8, seq_len=160 → 17 clips per episode
- Train: 612,900 × 17 = **10,419,300 clips** across 6,129 shards
- Val: 600 episodes → 10,200 clips across 6 shards
- Steps per epoch (batch=128, 2 GPUs): 10,419,300 / 256 ≈ **40,700 steps**

**DDP shard splitting:** Passed explicitly as `ddp_rank` / `ddp_world_size` constructor args to `CoinRunStreamingDataset`. Do NOT rely on `dist.is_initialized()` inside workers — spawned workers don't have the process group initialized.

**DataLoader:** `num_workers=16, pin_memory=True, multiprocessing_context="spawn"`

---

## Procgen action space (15 actions)

```
0: LEFT+DOWN   1: LEFT    2: LEFT+UP   3: DOWN    4: NOOP
5: UP          6: RIGHT+DOWN           7: RIGHT   8: RIGHT+UP
9-14: D, A, W, S, Q, E  (rarely used in CoinRun)
```

For CoinRun the meaningful actions are: 1 (LEFT), 7 (RIGHT), 5 (UP/jump), 2 (LEFT+UP), 8 (RIGHT+UP), 4 (NOOP).

---

## Inference

### Generate from checkpoint

```bash
/venv/open-oasis/bin/python3 scripts/infer_coinrun.py \
    --ckpt runs/coinrun_v1/ckpt_step_0110000.pt \
    --frames 24 --n-samples 4 --output generated.mp4

# Custom actions (gt = replay val actions, random = random actions)
--action-source gt | random
```

Output: mp4 with GT top / generated bottom / keyboard overlay, 4 samples side-by-side, scale=5.

### Interactive Gradio UI

```bash
/venv/open-oasis/bin/python3 scripts/interactive.py \
    --ckpt runs/coinrun_v1/ckpt_step_0110000.pt --share
```

Opens browser UI at a public `gradio.live` URL. Arrow buttons → generate next frame → record episode → save mp4 with keyboard overlay. Episodes saved to `episodes/`.

### Long rollouts with custom actions

```python
# run+jump pattern example (see inline script in conversation history)
pattern = [7, 7, 7, 8, 8]  # RIGHT×3, RIGHT+UP×2
action_ids = torch.tensor([pattern[i % len(pattern)] for i in range(FRAMES)])
actions_oh = F.one_hot(action_ids, 15).float()
```

Inference can go beyond `max_frames=32` — `generate_rollout` uses a sliding window automatically.

---

## Setup

```bash
# Install everything (creates /venv/open-oasis)
bash setup.sh              # CUDA 12.1 (default)
bash setup.sh --cuda 124   # CUDA 12.4

# Additional deps installed during development (already in venv):
# scikit-image, gradio, pygame, av, wandb, array-record, grain-nightly

# Dataset is already downloaded at data/coinrun/ — do not re-download
# No preprocessing needed — CoinRunStreamingDataset reads ArrayRecord directly
```

---

## GPU / performance

| Setup | it/s | Steps/epoch | Time/epoch |
|---|---|---|---|
| 1× H100 SXM, bs=128 | ~1.7 | ~81,400 | ~13h |
| 2× H100 SXM, bs=128 | ~3.3 | ~40,700 | ~3.4h |

GPU utilization: 98-100% on both GPUs — fully compute-bound, not data-bound.
Flash attention enabled via `F.scaled_dot_product_attention` (PyTorch built-in).

---

## Current training state (as of 2026-04-23)

- **Model:** CoinRunWorldModelSmall (57.8M params)
- **Latest checkpoint:** `runs/coinrun_v1/ckpt_step_0110000.pt` (epoch=1, step=110000)
- **WandB project:** `spring26-gen-ai / coinrun-world-model`
- **Checkpoints uploaded to HF:** `baseline/` folder in HF model repo (username/coinrunm)
- **Training command:** `torchrun --nproc_per_node=2 train_coinrun.py`
- Note: Before fixing DDP shard bug, epochs were 2× longer than expected (~81k steps instead of ~41k). Fixed by passing `ddp_rank`/`ddp_world_size` to dataset constructor.

---

## Known bugs fixed in this session

| Bug | Location | Fix |
|---|---|---|
| `copy_()` with scalar under torch.compile | `rotary_embedding.py:300` | Changed to `fill_()` |
| `cached_freqs_seq_len.item()` graph break | `rotary_embedding.py:134,292,300` | Replaced tensor buffer with plain Python int |
| Rollout shape error `x[:, -1:]` wrong size | `train_coinrun.py:139` | Added `[:, -1:]` to RHS |
| DDP shard splitting broken in spawned workers | `dataset_coinrun_streaming.py:95` | Pass `ddp_rank`/`ddp_world_size` explicitly to dataset |
| wandb.Video from numpy requires moviepy | `train_coinrun.py` rollout block | Pass local mp4 file path instead |

---

## Mario dataset (secondary, not currently training)

- 737k frames, 256×240 PNG, 8-bit bitmask actions
- Action encoding: 8-dim binary vector via `action_int_to_bits()` in `model/utils.py`
- Model factory: `MarioWorldModel()` in `model/dit.py` (~15M params, `external_cond_dim=8`)
- Training script: `train.py` (legacy, uses `TrainingManager`, no DDP)
- Inference: `scripts/generate.py` (requires PNG prompt + .pt action file)
