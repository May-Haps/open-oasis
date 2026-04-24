"""
Compute training FLOPs for each CoinRun model variant.

Usage:
    /venv/open-oasis/bin/python3 scripts/count_flops.py

Reports:
    - MACs from fvcore (linear ops only — matmul, conv)
    - Attention quadratic MACs added manually (F.sdpa not counted by fvcore)
    - Training FLOPs per sample = total_MACs * 2 (MAC→FLOP) * 3 (fwd+2×bwd)
    - Training FLOPs per epoch
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fvcore.nn import FlopCountAnalysis

from model.dit import DiT

# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------
VARIANTS = [
    dict(name="XS  (~5M)",  hidden_size=160, depth=6, num_heads=4),
    dict(name="S  (~11M)",  hidden_size=224, depth=6, num_heads=4),
    dict(name="M  (~18M)",  hidden_size=288, depth=6, num_heads=8),
    dict(name="L  (~32M)",  hidden_size=384, depth=6, num_heads=8),
    dict(name="XL (~58M)",  hidden_size=512, depth=6, num_heads=8),
]

# ---------------------------------------------------------------------------
# Dataset / training constants (from train_coinrun.py CONFIG)
# ---------------------------------------------------------------------------
CLIP_LEN        = 32    # T — frames per clip
PATCH_SIZE      = 8
FRAME_H         = 64
FRAME_W         = 64
BATCH_SIZE      = 256   # effective (128 per GPU × 2)
STEPS_PER_EPOCH = 40_700

# Derived
S_SPATIAL = (FRAME_H // PATCH_SIZE) * (FRAME_W // PATCH_SIZE)  # 64 patches/frame
S_TOTAL   = CLIP_LEN * S_SPATIAL                                # 2048 tokens/sample


def attention_macs(hidden_size: int, depth: int) -> int:
    """
    Manual attention quadratic term — not counted by fvcore because
    F.scaled_dot_product_attention is a fused kernel.

    Spatial attention: (B*T) sequences of length s=64
        QK^T  : T * s * s * H  MACs  (per sample, T=32, s=64)
        A * V : T * s * s * H  MACs
        Total : 2 * T * s^2 * H

    Temporal attention: (B*s) sequences of length t=32
        QK^T  : s * t * t * H  MACs  (per sample)
        A * V : s * t * t * H  MACs
        Total : 2 * s * t^2 * H

    Per block = spatial + temporal, multiplied by depth.
    """
    H = hidden_size
    s = S_SPATIAL   # 64
    t = CLIP_LEN    # 32

    spatial_macs  = 2 * t * s * s * H     # 2 * 32 * 64 * 64 * H
    temporal_macs = 2 * s * t * t * H     # 2 * 64 * 32 * 32 * H
    return depth * (spatial_macs + temporal_macs)


def count(variant: dict, device: str = "cpu") -> dict:
    name        = variant["name"]
    hidden_size = variant["hidden_size"]
    depth       = variant["depth"]
    num_heads   = variant["num_heads"]

    model = DiT(
        input_h=FRAME_H, input_w=FRAME_W,
        patch_size=PATCH_SIZE,
        in_channels=3,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        external_cond_dim=15,
        max_frames=CLIP_LEN,
    ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    # fvcore counts MACs for linear/conv ops (batch=1)
    x       = torch.randn(1, CLIP_LEN, 3, FRAME_H, FRAME_W, device=device)
    t_in    = torch.randint(0, 1000, (1, CLIP_LEN), device=device)
    actions = torch.randn(1, CLIP_LEN, 15, device=device)

    fvcore_analysis = FlopCountAnalysis(model, (x, t_in, actions))
    fvcore_analysis.unsupported_ops_warnings(False)
    fvcore_analysis.uncalled_modules_warnings(False)
    linear_macs = fvcore_analysis.total()   # MACs from matmuls/convs

    attn_macs   = attention_macs(hidden_size, depth)
    total_macs  = linear_macs + attn_macs

    # 1 MAC = 2 FLOPs; training = 3× forward (fwd + 2× bwd)
    train_flops_per_sample = total_macs * 2 * 3
    train_flops_per_step   = train_flops_per_sample * BATCH_SIZE
    train_flops_per_epoch  = train_flops_per_step * STEPS_PER_EPOCH

    # Cross-check: 6ND formula
    formula_flops_per_sample = 6 * n_params * S_TOTAL

    return dict(
        name=name,
        n_params=n_params,
        linear_macs=linear_macs,
        attn_macs=attn_macs,
        total_macs=total_macs,
        train_flops_per_sample=train_flops_per_sample,
        train_flops_per_step=train_flops_per_step,
        train_flops_per_epoch=train_flops_per_epoch,
        formula_flops_per_sample=formula_flops_per_sample,
        attn_fraction=attn_macs / total_macs,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    header = f"{'Model':<12} {'Params':>8} {'LinMACs':>10} {'AttnMACs':>10} {'AttnFrac':>9} {'Train GFLOPs/sample':>20} {'6ND GFLOPs/sample':>18} {'Train PFLOPs/epoch':>19}"
    print(header)
    print("-" * len(header))

    results = []
    for v in VARIANTS:
        r = count(v, device=device)
        results.append(r)

        print(
            f"{r['name']:<12} "
            f"{r['n_params']/1e6:>7.1f}M "
            f"{r['linear_macs']/1e9:>9.1f}G "
            f"{r['attn_macs']/1e9:>9.1f}G "
            f"{r['attn_fraction']:>8.1%}  "
            f"{r['train_flops_per_sample']/1e9:>19.1f}G "
            f"{r['formula_flops_per_sample']/1e9:>17.1f}G "
            f"{r['train_flops_per_epoch']/1e15:>18.2f}P"
        )

    print()
    print("Notes:")
    print(f"  S_total (tokens/sample) = {CLIP_LEN} frames × {S_SPATIAL} patches = {S_TOTAL}")
    print(f"  Effective batch size    = {BATCH_SIZE} (128/GPU × 2 GPUs)")
    print(f"  Steps per epoch         = {STEPS_PER_EPOCH:,}")
    print("  Train FLOPs/sample      = total_MACs × 2 (MAC→FLOP) × 3 (fwd+2×bwd)")
    print("  6ND formula             = 6 × params × S_total  [cross-check]")
    print("  Attn fraction           = attention quadratic / total MACs")
    print()
    print("  Relative FLOPs (normalised to XS):")
    base = results[0]["train_flops_per_sample"]
    for r in results:
        print(f"    {r['name']:<12}  ×{r['train_flops_per_sample']/base:.1f}")


if __name__ == "__main__":
    main()
