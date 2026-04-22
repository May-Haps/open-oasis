"""
FLOPs estimates for two model configs using calflops:

  [A] Oasis DiT-S/2 (608M params) — MineRL-v0 Minecraft dataset
        x: (B=2, T=24, C=16, H=18, W=32),  actions: 25-dim

  [B] Small DiT (10M params, hidden=256 depth=4) — NES Mario dataset
        x: (B=2, T=24, C=4,  H=8,  W=8),   actions: 8-dim
        256×240 frames → VAE patch_size=16 → 16×15 latents → patch_size=2 → 8×7≈8×8
"""

import re
import torch
from calflops import calculate_flops
from dit import DiT_models, DiT

DEVICE = "cpu"
BATCH_SIZE = 2
CLIP_LEN   = 24

# ── GPU specs (dense bf16, no sparsity) ───────────────────────────────────────
# A100 SXM4 80GB : 312 TFLOPS  ~$2.50/hr RunPod community
# H100 SXM5 80GB : 989 TFLOPS  ~$4.00/hr RunPod community
GPUS = {
    "1× A100 SXM4": (312e12, 2.50, 1),
    "1× H100 SXM5": (989e12, 4.00, 1),
    "8× H100 SXM5": (989e12, 4.00, 8),
}
MFU = 0.35

def parse_flops(s):
    m = re.match(r"([\d.]+)\s*(K|M|G|T)?FLOPs?", s.strip(), re.IGNORECASE)
    if not m:
        return None
    val, suffix = float(m.group(1)), (m.group(2) or "").upper()
    return val * {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "": 1}[suffix]

def run_estimate(label, model, C, H, W, act_dim, dataset):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    kwargs = {
        "x":             torch.zeros(BATCH_SIZE, CLIP_LEN, C, H, W, device=DEVICE),
        "t":             torch.zeros(BATCH_SIZE, CLIP_LEN, dtype=torch.long, device=DEVICE),
        "external_cond": torch.zeros(BATCH_SIZE, CLIP_LEN, act_dim, device=DEVICE),
    }
    flops, macs, params = calculate_flops(model=model, kwargs=kwargs,
                                          print_results=True, print_detailed=False)

    print(f"\n  Params  : {params}")
    print(f"  MACs    : {macs}")
    print(f"  FLOPs   : {flops}")

    fwd_flops = parse_flops(flops)
    if not fwd_flops:
        return

    train_flops_per_step = fwd_flops * 3
    print(f"\n  Training FLOPs/step (fwd+bwd) : {train_flops_per_step/1e12:.4f} TFLOPs")

    # dataset stats
    total_frames = dataset["total_frames"]
    fps          = dataset["fps"]
    frames_ep    = dataset["frames_ep"]
    episodes     = total_frames // frames_ep
    clip_stride  = 16
    clips        = episodes * ((frames_ep - CLIP_LEN) // clip_stride + 1)
    steps_epoch  = clips // BATCH_SIZE

    print(f"\n  Dataset  : {dataset['name']}")
    print(f"  Frames   : {total_frames:,}  ({total_frames/fps/3600:.1f} hrs @ {fps} fps)")
    print(f"  Episodes : {episodes:,}  (~{frames_ep} frames each)")
    print(f"  Clips    : {clips:,}  (stride={clip_stride})")
    print(f"  Steps/ep : {steps_epoch:,}")

    epoch_options = dataset["epochs"]
    target_gpu    = dataset["target_gpu"]
    tflops, price_hr, n_gpus = GPUS[target_gpu]

    print(f"\n  Epoch sweep — {target_gpu} (${price_hr}/hr × {n_gpus}, MFU={MFU})")
    print(f"  {'Epochs':>7} {'Steps':>10} {'Time':>10} {'Cost':>8}")
    print(f"  {'-'*40}")
    for epochs in epoch_options:
        total_steps = steps_epoch * epochs
        wall_s      = (total_steps * train_flops_per_step) / (tflops * MFU * n_gpus)
        wall_h      = wall_s / 3600
        cost        = wall_h * price_hr * n_gpus
        time_str    = f"{wall_s/60:.1f} min" if wall_h < 1 else f"{wall_h:.1f} h"
        print(f"  {epochs:>7}  {total_steps:>10,}  {time_str:>10}  ${cost:>6.2f}")
    print(f"{'='*60}\n")


# ── [A] Oasis 608M — MineRL ───────────────────────────────────────────────────
model_a = DiT_models["DiT-S/2"]().to(DEVICE).eval()
run_estimate(
    label="[A] Oasis DiT-S/2  |  608M params  |  MineRL-v0",
    model=model_a,
    C=16, H=18, W=32, act_dim=25,
    dataset={
        "name":        "MineRL-v0 (Minecraft)",
        "total_frames": 60_000_000,
        "fps":          20,
        "frames_ep":    1_000,
        "epochs":       [3, 10, 30],
        "target_gpu":   "8× H100 SXM5",
    },
)

# ── [B] Small DiT 10M — NES Mario ────────────────────────────────────────────
# 256×240 → VAE patch_size=16 → 16×15 latents, C=4 → DiT patch_size=2 → 8×7
# Use 8×8 to keep grid even; act_dim=8 (NES 8-button binary)
model_b = DiT(
    input_h=8, input_w=8,
    patch_size=2,
    in_channels=4,
    hidden_size=256,
    depth=4,
    num_heads=4,
    external_cond_dim=8,
    max_frames=32,
).to(DEVICE).eval()
run_estimate(
    label="[B] Small DiT  |  ~10M params  |  NES Mario",
    model=model_b,
    C=4, H=8, W=8, act_dim=8,
    dataset={
        "name":         "NES Mario (737K frames, 280 episodes)",
        "total_frames":  737_134,
        "fps":           60,
        "frames_ep":     737_134 // 280,
        "epochs":        [10, 100, 300, 1000],
        "target_gpu":    "1× A100 SXM4",
    },
)
