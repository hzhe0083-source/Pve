#!/usr/bin/env bash
# Python env for the conflict-reuse experiments on 5090d (RTX 5090 D), independent of everything else on the host:
# everything lives in ~/zhr_pve, built from the system Python 3.10.12 with haha-c's exact package list (freeze.txt:
# torch 2.8.0 cu128, ManiSkill 3.0.1, SAPIEN 3.0.3), so the hosts differ only in the GPU. ~/pve and the a2r dirs are not
# used; pip's cache is ~/zhr_pve/pip-cache. Run once from ~/zhr_scratch/whye (the unpacked whye.tgz).
set -euo pipefail
W=$HOME/zhr_scratch/whye
R=$HOME/zhr_pve
[ -e "$R" ] && { echo "ABORT: $R exists"; exit 1; }
mkdir -p "$R/runs/experts_wc/runs" "$R/third_party"
cd "$R"
MIRROR=https://mirrors.ustc.edu.cn/pypi/simple/    # aliyun gave 1.3 MB/s on 5090d, USTC 4 MB/s
export PIP_CACHE_DIR=$R/pip-cache PIP_DISABLE_PIP_VERSION_CHECK=1
/usr/bin/python3 -m venv .venv
.venv/bin/pip install -q -i $MIRROR --upgrade pip
.venv/bin/pip install -q -i $MIRROR --retries 10 --timeout 120 -r "$W/freeze.txt"
cp -r "$W/ms_ppo" third_party/
cp -r "$W/PickCube_wc_c1" runs/experts_wc/runs/
cp "$W/freeze.txt" freeze_hahac.txt
.venv/bin/pip freeze > freeze.txt
if diff freeze_hahac.txt freeze.txt; then echo "FREEZE IDENTICAL to haha-c's"; else echo "FREEZE DIFFERS from haha-c's"; fi
.venv/bin/python - <<"PY"
import sys
import torch, mani_skill, sapien
print("python", sys.version.split()[0], "torch", torch.__version__, torch.version.cuda, torch.cuda.is_available(),
      torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda"); print("matmul ok", torch.isfinite(x @ x).all().item())
print("mani_skill", mani_skill.__version__, "sapien", sapien.__version__)
PY
md5sum runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt third_party/ms_ppo/ppo.py
echo SETUP_DONE
