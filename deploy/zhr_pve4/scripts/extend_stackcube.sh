#!/usr/bin/env bash
# StackCube's expert ended at 69% success after 25M steps, below G1's 80%. Continue it from its final checkpoint for
# 15M more steps and swap it in (old one kept as final_ckpt_25M.pt) if its last evals beat the original's.
# The latent pipeline waits for STACK_EXT_DONE, so the library and the labels use whichever expert is better.
set -u
cd "$HOME/pve/runs/experts" || exit 1
source "$HOME/pve/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890
echo "STACK_EXT_START $(date +%T)"
python "$HOME/pve/third_party/ms_ppo/ppo.py" --exp_name=StackCube_ext --env_id=StackCube-v1 --num_envs=1024 \
    --total_timesteps=15_000_000 --update_epochs=8 --num_minibatches=32 --num_eval_envs=16 --no-capture-video \
    --seed=2 --checkpoint=runs/StackCube/final_ckpt.pt > logs/StackCube_ext.txt 2>&1
rc=$?
python - <<'PY' && cp runs/StackCube/final_ckpt.pt runs/StackCube/final_ckpt_25M.pt && cp runs/StackCube_ext/final_ckpt.pt runs/StackCube/final_ckpt.pt && echo "SWAPPED"
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def last(d):
    acc = EventAccumulator(d, size_guidance={"scalars": 0})
    acc.Reload()
    v = [e.value for e in acc.Scalars("eval/success_once")][-3:]
    return sum(v) / len(v)
try:
    old, new = last("runs/StackCube"), last("runs/StackCube_ext")
except Exception as e:
    print("no eval to compare:", e)
    sys.exit(1)
print(f"last-3 eval success: original {old:.3f}, extended {new:.3f}")
sys.exit(0 if new > old else 1)
PY
echo "STACK_EXT_DONE rc=$rc $(date +%T)"
