# Open-Oasis — CoinRun World Model

A video-generating world model trained on [CoinRun](https://github.com/openai/coinrun) (Procgen) gameplay.
Adapted from [Oasis](https://oasis-model.github.io/) (Minecraft), retargeted to 64×64 pixel-space diffusion with action conditioning.

Given a single prompt frame and a sequence of actions, the model autoregressively generates future frames using DDIM diffusion — functioning as a learned game simulator.

---

## How it works

The model is a **Spatiotemporal Diffusion Transformer (DiT)** with alternating spatial and temporal axial attention. At each generation step:

1. Context frames (already generated) are fixed at a low noise level
2. The next frame is initialised as pure noise
3. 10 DDIM denoising steps denoise the new frame, conditioned on context + action
4. The denoised frame is appended to the context (sliding window of 32 frames)

**Key choices:**
- No VAE — pixel-space diffusion directly on 64×64 RGB frames
- v-parameterization diffusion objective
- Sigmoid beta noise schedule
- Rotary positional encodings (RoPE) — 2D spatial + 1D temporal
- 15-dim one-hot action conditioning (Procgen action space)

---

## Model

| Variant | Params | Hidden size | Depth | Heads |
|---|---|---|---|---|
| `5m` | ~5M | 160 | 5 | 8 |
| `9m` | ~9M | 224 | 5 | 8 |
| `17m` | ~18M | 288 | 6 | 8 |
| `31m` | ~32M | 384 | 6 | 8 |
| `small` | 57.8M | 512 | 6 | 8 |
| `CoinRunWorldModel` | ~100M | 640 | 8 | 8 |

All CoinRun variants live in `model/dit.py` and are selected with `train_coinrun.py --model-size`.

---

## Baseline results

Trained on 2× H100 SXM, `batch_size=128` per GPU (effective 256), `lr=1e-4`.

| Checkpoint | Steps | Epoch | Notes |
|---|---|---|---|
| `baseline/ckpt_step_0110000.pt` | 110,000 | ~1.4 | Early training run |

Sample generations (run+jump action pattern from train set prompt frames):

| | | | |
|---|---|---|---|
| [train_clip_00](episodes/baseline/train_clip_00.mp4) | [train_clip_01](episodes/baseline/train_clip_01.mp4) | [train_clip_02](episodes/baseline/train_clip_02.mp4) | [train_clip_03](episodes/baseline/train_clip_03.mp4) |
| [train_clip_04](episodes/baseline/train_clip_04.mp4) | [train_clip_05](episodes/baseline/train_clip_05.mp4) | [train_clip_06](episodes/baseline/train_clip_06.mp4) | [train_clip_07](episodes/baseline/train_clip_07.mp4) |

Interactive episodes (Gradio UI, val set prompt frames):

- [episode_1](episodes/baseline/episode_1776969593.mp4)
- [episode_2](episodes/baseline/episode_1776969994.mp4)
- [episode_3](episodes/baseline/episode_1776970317.mp4)

Checkpoints available on HuggingFace: `username/coinrunm` under `baseline/`.

---

## Setup

### 1. Clone and install

```bash
git clone <this-repo>
cd open-oasis
bash setup.sh              # CUDA 12.1 (default)
bash setup.sh --cuda 124   # CUDA 12.4 for newer drivers
```

This creates a venv at `/venv/open-oasis` with all dependencies (PyTorch, av, wandb, gradio, scikit-image, array-record, etc.).

### 2. Download the CoinRun dataset

```bash
source /venv/open-oasis/bin/activate

hf download --repo-type dataset p-doom/coinrun-dataset \
    --local-dir data/coinrun
```

The dataset is ~60 GB (612,900 episodes × 160 frames, stored as Google ArrayRecord shards). No preprocessing is required — the dataloader reads ArrayRecord directly.

**Dataset stats:**
- Train: 6,129 shards, 612,900 episodes, ~98M frames, 10.4M clips (clip_len=32, stride=8)
- Val: 6 shards, 600 episodes, 10,200 clips
- Test: 6 shards, ~600 episodes

### 3. Verify data

```bash
/venv/open-oasis/bin/python3 scripts/visualize_dataset.py \
    --shard data/coinrun/train/data_0000.array_record \
    --n-episodes 3 --output preview.mp4
```

---

## Training

`train_coinrun.py` is the single training entry point for all scaling runs. It keeps the effective batch size at 256 by deriving gradient accumulation from the GPU count.

### Single GPU — 5M / 9M

```bash
/venv/open-oasis/bin/python3 train_coinrun.py --model-size 5m --max-hours 18
/venv/open-oasis/bin/python3 train_coinrun.py --model-size 9m --max-hours 15
```

### DDP — 17M / 31M / 57.8M

```bash
torchrun --nproc_per_node=2 train_coinrun.py --model-size 17m --max-hours 9
torchrun --nproc_per_node=2 train_coinrun.py --model-size 31m --max-hours 9
torchrun --nproc_per_node=2 train_coinrun.py --model-size small --max-hours 12

# Resume from checkpoint
torchrun --nproc_per_node=2 train_coinrun.py \
    --model-size small \
    --resume runs/coinrun_small_lin/ckpt_step_0110000.pt
```

At ~3.3 it/s on 2× H100 SXM, one epoch takes ~3.4 hours (~40,700 steps with bs=128×2).

### Key config (top of `train_coinrun.py`)

```python
CONFIG = {
    "batch_size":           128,     # physical batch per process
    "effective_batch_size": 256,
    "clip_len":             32,
    "lr":                   1e-4,
    "max_hours":            args.max_hours,
    "wandb_project":        "coinrun-scaling",
    "action_cond_mode":     "linear",
}
```

### WandB metrics

| Key | Description |
|---|---|
| `train/loss` | Training diffusion loss (every 50 steps) |
| `val/loss` | Val loss on 50-batch subset (every 5k steps) |
| `rollout/video` | Generated video with keyboard overlay (every 1k steps) |
| `rollout/grid` | GT vs generated comparison grid |
| `epoch/val_loss` | Full val loss over all 10,200 clips |
| `epoch/psnr` | PSNR from single-step x0 prediction on full val set |
| `epoch/ssim` | SSIM from single-step x0 prediction on full val set |

---

## Inference

### Generate from a checkpoint

```bash
/venv/open-oasis/bin/python3 scripts/infer_coinrun.py \
    --ckpt runs/coinrun_small_lin/ckpt_step_0110000.pt \
    --frames 32 --n-samples 4 --output generated.mp4

# Use random actions instead of ground-truth val actions
/venv/open-oasis/bin/python3 scripts/infer_coinrun.py \
    --ckpt runs/coinrun_small_lin/ckpt_step_0110000.pt \
    --action-source random --frames 64
```

Output: 4 samples side-by-side — GT frame on top, generated on bottom, keyboard overlay below.

### Interactive browser UI

```bash
/venv/open-oasis/bin/python3 scripts/interactive.py \
    --ckpt runs/coinrun_small_lin/ckpt_step_0110000.pt --share
```

Opens a Gradio web UI (public share URL valid 1 week). Choose actions with arrow buttons, step frame-by-frame, record and download episodes as mp4.

### Visualize dataset episodes

```bash
/venv/open-oasis/bin/python3 scripts/visualize_dataset.py \
    --shard data/coinrun/train/data_0000.array_record \
    --n-episodes 5 --fps 15 --output preview.mp4
```

---

## Procgen action space

| ID | Action | ID | Action |
|---|---|---|---|
| 0 | LEFT+DOWN | 5 | UP (jump) |
| 1 | LEFT | 6 | RIGHT+DOWN |
| 2 | LEFT+UP | 7 | RIGHT |
| 3 | DOWN | 8 | RIGHT+UP |
| 4 | NOOP | 9–14 | D, A, W, S, Q, E |

In CoinRun the meaningful actions are: `1` (left), `7` (right), `5` (jump), `2` (left+jump), `8` (right+jump), `4` (noop).

---

## Upload checkpoints to HuggingFace

```bash
/venv/open-oasis/bin/python3 scripts/upload_checkpoints.py \
    --token hf_xxx \
    --repo your-username/coinrunm \
    --folder baseline
```

---

## File structure

```
open-oasis/
  model/
    dit.py                        # DiT architecture + model factories
    attention.py                  # Spatial + temporal axial attention
    rotary_embedding.py           # RoPE positional encodings
    utils.py                      # Noise schedule, action utilities
  data/
    dataset_coinrun_streaming.py  # IterableDataset reading ArrayRecord directly
    dataset_coinrun.py            # Memmap-based dataset (alternative, not used)
    preprocess_coinrun.py         # ArrayRecord → memmap converter (not needed)
    preprocess.py / dataset.py    # Legacy Mario data pipeline
  training/
    noise_scheduler.py            # v-parameterization noise scheduler
    model_trainer.py / rollout_sampler.py / training_manager.py
  scripts/
    infer_coinrun.py              # Batch inference → mp4
    interactive.py                # Gradio interactive UI
    visualize_dataset.py          # Dataset visualization → mp4
    upload_checkpoints.py         # HuggingFace upload utility
    generate.py                   # Legacy Mario inference
  docs/
    CONTEXT.md                    # Development context snapshot
    SCALING_ABLATION.md           # Scaling-run notes
  train_coinrun.py                # Main training script
  train.py                        # Legacy Mario training
  episodes/                       # Generated episode videos
    baseline/                     # Videos from baseline checkpoint
  setup.sh                        # Environment setup
```

---

## Requirements

- Python 3.11
- CUDA 12.1+ (12.4 also tested)
- 2× H100 SXM recommended (80 GB VRAM each); single H100 also works
- ~60 GB disk for dataset, ~2 GB per checkpoint

---

## Credits

Based on [Oasis](https://oasis-model.github.io/) by Etched & decart.
Dataset: [p-doom/coinrun-dataset](https://huggingface.co/datasets/p-doom/coinrun-dataset) on HuggingFace.
CoinRun environment: [OpenAI Procgen](https://github.com/openai/procgen).
