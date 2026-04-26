#!/usr/bin/env bash
# setup.sh — create and configure the open-oasis conda environment
# Usage: bash setup.sh [--cuda 121|124]  (default: 121)
set -euo pipefail

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
CUDA_VERSION="121"
for arg in "$@"; do
    case $arg in
        --cuda) shift ;;
        121|124|118) CUDA_VERSION="$arg" ;;
        --cuda=*) CUDA_VERSION="${arg#*=}" ;;
    esac
done

TORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA_VERSION}"
ENV_NAME="open-oasis"

echo "=================================================="
echo "  open-oasis setup"
echo "  CUDA index : cu${CUDA_VERSION}  (${TORCH_INDEX})"
echo "  Conda env  : ${ENV_NAME}"
echo "=================================================="

# ---------------------------------------------------------------------------
# Conda env
# ---------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/5] Conda env '${ENV_NAME}' already exists — skipping creation"
else
    echo "[1/5] Creating conda env '${ENV_NAME}' (Python 3.11) ..."
    conda create -y -n "${ENV_NAME}" python=3.11
fi

# Activate the env for all subsequent installs
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
echo "[2/5] Installing PyTorch (cu${CUDA_VERSION}) ..."
pip install --quiet \
    torch torchvision \
    --index-url "${TORCH_INDEX}"

# ---------------------------------------------------------------------------
# Core dependencies
# ---------------------------------------------------------------------------
echo "[3/5] Installing core dependencies ..."
pip install --quiet \
    einops \
    diffusers \
    timm \
    av \
    tqdm \
    wandb

# ---------------------------------------------------------------------------
# CoinRun data preprocessing
# ---------------------------------------------------------------------------
echo "[4/5] Installing CoinRun preprocessing dependencies ..."
pip install --quiet \
    array-record \
    grain-nightly \
    huggingface_hub \
    msgpack \
    msgpack-numpy

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo "[5/5] Verifying installation ..."
python - <<'EOF'
import torch, einops, timm, wandb, tqdm, huggingface_hub
print(f"  torch      {torch.__version__}")
print(f"  CUDA avail {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU        {torch.cuda.get_device_name(0)}")
print(f"  einops     {einops.__version__}")
print(f"  timm       {timm.__version__}")
print(f"  wandb      {wandb.__version__}")
print("  All core imports OK")
EOF

echo ""
echo "=================================================="
echo "  Setup complete."
echo ""
echo "  Activate:  conda activate ${ENV_NAME}"
echo ""
echo "  Next steps:"
echo "    1. Download dataset:"
echo "       huggingface-cli download --repo-type dataset p-doom/coinrun-dataset \\"
echo "           --local-dir data/coinrun_raw"
echo ""
echo "    2. Inspect record schema (first time only):"
echo "       python data/preprocess_coinrun.py --input-dir data/coinrun_raw/train --inspect"
echo ""
echo "    3. Preprocess both splits:"
echo "       python data/preprocess_coinrun.py \\"
echo "           --input-dir data/coinrun_raw/train --output-dir data/coinrun_processed/train"
echo "       python data/preprocess_coinrun.py \\"
echo "           --input-dir data/coinrun_raw/val   --output-dir data/coinrun_processed/val"
echo ""
echo "    4. Train:"
echo "       python train_coinrun.py --model-size 5m"
echo "       torchrun --nproc_per_node=2 train_coinrun.py --model-size small"
echo "       python train_coinrun.py --model-size 5m --resume runs/coinrun_5m_lin/ckpt_step_0002000.pt"
echo "=================================================="
