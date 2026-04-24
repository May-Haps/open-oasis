# Scaling Ablation — CoinRun World Model

Drop this file into a new Claude conversation to resume with full context.

---

## Goal

Run a scaling law ablation across 5 model sizes (log-spaced from 5M to 57.8M params) on the CoinRun dataset. Compare val loss vs FLOPs to fit a power law: `L(N) = A · N^(−α)`.

---

## Baseline model

- **Script:** `train_coinrun.py`
- **Model:** `CoinRunWorldModelSmall` — hidden=512, depth=6, heads=8, **57.8M params**
- **Hardware:** 2× H100 SXM, batch=128/GPU (256 effective)
- **Speed:** 3.3 it/s, 1 epoch = 40,700 steps = 3.4 hours
- **Baseline run:** 73,000 steps = **13.3 ExaFLOPs**
- **Checkpoint:** `runs/coinrun_v1/ckpt_step_0073000.pt` (update path as needed)

---

## Five ablation model sizes

Log-spaced from 5M to 57.8M (ratio ≈ 1.84× each step).

| Label | Actual params | hidden | depth | heads | Script |
|---|---|---|---|---|---|
| XS | ~5.5M | 160 | 6 | 4 | `train_coinrun_5M.py` (TODO) |
| S | ~10.8M | 224 | 6 | 4 | `train_coinrun_9M.py` (TODO) |
| M | ~18.4M | 288 | 6 | 8 | `train_coinrun_17M.py` ✓ |
| L | ~31.9M | 384 | 6 | 8 | `train_coinrun_31M.py` ✓ |
| XL | 57.8M | 512 | 6 | 8 | `train_coinrun.py` ✓ |

Use the **actual** param count when plotting, not the label.

To instantiate any variant directly:
```python
from model.dit import DiT
model = DiT(
    input_h=64, input_w=64, patch_size=8, in_channels=3,
    hidden_size=<H>, depth=6, num_heads=<heads>,
    external_cond_dim=15, max_frames=32,
)
```

---

## FLOPs methodology

**Formula:** `C = 6 × N × B × T × S`

| Symbol | Value | Meaning |
|---|---|---|
| 6 | 6 | 2 (MAC) × 3 (fwd + 2× bwd) |
| N | params | model parameter count |
| B | 256 | effective batch size |
| T | 2048 | tokens/sample = 32 frames × 64 patches (patch_size=8 on 64×64) |
| S | steps | training steps |

**Unit:** results are in **ExaFLOPs** (10¹⁸ FLOPs). Earlier conversation mistakenly said "PFLOPs" — correct unit is ExaFLOPs.

**FLOPs per step for each model:**

| Model | Params | FLOPs/step |
|---|---|---|
| XS | 5.5M | 17 TFLOPs |
| S | 10.8M | 34 TFLOPs |
| M | 18.4M | 57 TFLOPs |
| L | 31.9M | 100 TFLOPs |
| XL | 57.8M | 182 TFLOPs |

**Worked example — baseline 57.8M at 73K steps:**
```
FLOPs/step = 6 × 57,800,000 × 2048 × 256 = 1.818 × 10¹⁴ = 181.8 TFLOPs
Total      = 1.818 × 10¹⁴ × 73,000       = 1.327 × 10¹⁹ = 13.3 ExaFLOPs
```

---

## Equal-FLOPs training plan

All runs target **13.3 ExaFLOPs** to match the baseline. Steps = `73,000 × (57.8M / N)`.

| Model | Steps | Epochs | Hardware | Batch | Est. wall time |
|---|---|---|---|---|---|
| XS ~5M | ~768,000 | ~19 | 1× H100 | 256 | ~18 hrs |
| S ~9M | ~391,000 | ~10 | 1× H100 | 256 | ~15 hrs |
| M ~18M | ~244,200 | 6 | 2× H100 | 128/GPU | ~8.5 hrs |
| L ~31M | ~162,800 | 4 | 2× H100 | 128/GPU | ~8.2 hrs |
| XL ~58M | 73,000 | 1.8 | 2× H100 | 128/GPU | ~6.1 hrs |

**Important:** XS and S use batch=256 on a single GPU to match effective batch of the 2× H100 runs. Do NOT use batch=128 for 1× H100 runs — it changes the gradient noise scale and confounds the comparison.

**Why 1× H100 runs don't OOM at batch=256:**
VRAM is dominated by activations ∝ batch × hidden_size. At hidden=160/224, batch=256 fits:
- XS (hidden=160): ~34 GB
- S (hidden=224): ~48 GB
Both well within 80 GB H100 limit.

---

## Existing training scripts

### `train_coinrun_17M.py`
- Model: hidden=288, depth=6, heads=8 → **18.4M params** (label says 17M)
- `epochs=6`, `save_dir=runs/coinrun_17M`, `wandb_name=ablation-17M`
- Run: `torchrun --nproc_per_node=2 train_coinrun_17M.py`

### `train_coinrun_31M.py`
- Model: hidden=384, depth=6, heads=8 → **31.9M params**
- `epochs=4`, `save_dir=runs/coinrun_31M`, `wandb_name=ablation-31M`
- Run: `torchrun --nproc_per_node=2 train_coinrun_31M.py`

### Run sequentially (only 2 GPUs available):
```bash
torchrun --nproc_per_node=2 train_coinrun_17M.py && \
torchrun --nproc_per_node=2 train_coinrun_31M.py
```

### TODO: create scripts for XS (5M) and S (9M)
- Use `python train_coinrun_XM.py` (single GPU, no torchrun)
- Set `batch_size=256` in CONFIG
- Epochs: ~19 for XS, ~10 for S

---

## Metric for scaling law comparison

**Primary:** `epoch/val_loss` — MSE on v-prediction target, already logged, directly comparable across all runs since same architecture family, same data, same noise schedule.

**Secondary:** `epoch/psnr`, `epoch/ssim` — already logged at epoch end.

**Plot:** log(val_loss) vs log(params) after equal-FLOPs training. Fit: `L(N) = A · N^(−α)`. If points fall on a line, α is your scaling exponent.

**Do NOT use FVD** — too expensive for a 5-run ablation.

---

## VRAM breakdown (why batch size dominates)

For 57.8M at batch=128 (~55 GB total):

| Component | Size |
|---|---|
| Parameters (bf16) | ~0.1 GB |
| Optimizer states (fp32 AdamW) | ~0.7 GB |
| Gradients (bf16) | ~0.1 GB |
| Activations | ~54 GB |

Activations ∝ `batch × hidden_size` — doubling batch ≈ doubles VRAM. Model size barely matters.

---

## Performance observations

- 57.8M on 2× H100: 3.3 it/s (compute-bound, 98-100% GPU util)
- 18.4M on 2× H100: ~4.3 it/s observed (lower than theoretical ~8 it/s — DataLoader becomes bottleneck for smaller/faster models)
- For smaller models, check GPU util with `watch -n 1 nvidia-smi`. If <80%, increase `num_workers` in DataLoader.
- **it/s does not affect FLOPs or scaling law validity** — only step count matters.

---

## Key facts to not get wrong

1. **FLOPs unit is ExaFLOPs** (10¹⁸), not PFLOPs. 13.3 ExaFLOPs = 13,300 PFLOPs.
2. **Effective batch must be 256 across all runs** — 2× H100 at 128/GPU = 256 effective; 1× H100 must use batch=256 directly.
3. **Steps per epoch = 40,700** (10,419,300 clips ÷ 256 batch). This is after the DDP shard bug fix — before the fix, epochs appeared 2× longer.
4. **Log-spaced targets vs actual params** — report actual params in results, not the round-number labels.
5. **Same LR (1e-4) across all runs** — do not tune LR per model size, it introduces a confound.
6. **torchrun for 2× H100, plain python for 1× H100** — `torchrun --nproc_per_node=2` for M and L; `python` for XS and S.

---

## WandB

- Project: `coinrun-world-model`
- Entity: `spring26-gen-ai`
- All 5 ablation runs log to same project for easy comparison
- Run names: `ablation-17M`, `ablation-31M` (set via `wandb_name` in CONFIG)
