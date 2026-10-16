#!/usr/bin/env bash
# One-off environment setup on a lab machine. Run from anywhere:
#
#   bash lab/setup.sh
#
# PyPI's Linux torch wheel bundles CUDA, so a plain install is GPU-enabled as long as
# the machine's NVIDIA driver is recent enough. If the check at the end reports no GPU
# on a machine that has one, the driver is older than the wheel's CUDA: reinstall torch
# from https://download.pytorch.org/whl/cu118 instead.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    # Login nodes on a cluster usually have no GPU; the training job itself will.
    print("No GPU visible from this machine. Fine on a cluster login node; not fine anywhere else.")
EOF
