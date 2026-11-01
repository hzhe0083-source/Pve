#!/usr/bin/env bash
# Python env for the WRWL experiments on 5090d (RTX 5090, sm_120): ManiSkill3 GPU sim + PyTorch CUDA 12.8.
# Reuses the CPython builds and the uv binary already under ~/a2r-train; the venv and uv cache live in ~/pve.
set -euo pipefail
R=${PVE_ROOT:-$HOME/pve}
UV=${UV:-$HOME/a2r-train/tools/bin/uv}
export UV_CACHE_DIR=$R/uv-cache
export UV_PYTHON_INSTALL_DIR=$HOME/a2r-train/python
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=8
PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/

trap 'echo SETUP_FAILED' ERR
"$UV" venv "$R/.venv" --python 3.11 --allow-existing
PY=$R/.venv/bin/python
# The PyPI build of torch 2.8.0 already targets CUDA 12.8 (sm_120), so the mirror alone is enough;
# download.pytorch.org kept timing out on small dependencies.
"$UV" pip install --python "$PY" --index-url "$PIP_MIRROR" "torch==2.8.0" mani_skill tyro tensorboard
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(),
      torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
assert torch.version.cuda.startswith("12.8") and torch.cuda.get_device_capability(0) == (12, 0)
x = torch.randn(1024, 1024, device="cuda")
print("matmul ok", torch.isfinite(x @ x).all().item())
import mani_skill, sapien
print("mani_skill", mani_skill.__version__, "sapien", sapien.__version__)
EOF
"$UV" pip freeze --python "$PY" > "$R/freeze.txt"
echo SETUP_DONE
