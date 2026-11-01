#!/usr/bin/env bash
# Python env for the WRWL / conflict-reuse experiments on a shared lab RTX 4090 (zhaobo account): everything lives in
# ~/zhr_pve, nothing outside it is touched. Same pins as the 5090 venv (torch 2.8.0 cu128, ManiSkill 3.0.1).
set -euo pipefail
cd "$HOME/zhr_pve"
MIRROR=https://mirrors.aliyun.com/pypi/simple/
if ! /usr/bin/python3 -m venv .venv; then
    # no python3-venv (ensurepip) on this host and no sudo: a venv without pip, then pip bootstrapped from its own wheel
    rm -rf .venv /tmp/zhr_pipwheel
    /usr/bin/python3 -m venv --without-pip .venv
    /usr/bin/python3 -m pip download -q -d /tmp/zhr_pipwheel -i $MIRROR pip
    W=$(ls /tmp/zhr_pipwheel/pip-*.whl)
    .venv/bin/python "$W/pip" install -q --no-index "$W"
fi
.venv/bin/pip install -q -i $MIRROR --upgrade pip
.venv/bin/pip install -q -i $MIRROR "torch==2.8.0" "mani_skill==3.0.1" tyro tensorboard
.venv/bin/python - <<"PY"
import torch, mani_skill, sapien
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda"); print("matmul ok", torch.isfinite(x @ x).all().item())
print("mani_skill", mani_skill.__version__, "sapien", sapien.__version__)
PY
.venv/bin/pip freeze > freeze.txt
echo SETUP_DONE
