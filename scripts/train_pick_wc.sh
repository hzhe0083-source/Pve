#!/usr/bin/env bash
# PickCube experts on panda_wristcam, the robot StackCube uses by default, so Stack<-Pick tests the grasp/release
# conflict rather than an embodiment change (code review, 2026-10-31). Two seeds: the copied source and a second
# seed for library disagreement. Dense reward (only competence matters here), early stop at 0.95.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
mkdir -p runs/experts_wc && cd runs/experts_wc || exit 1
echo "PICK_WC_START $(date +%T)"
for s in 1 2; do
    python -m wrwl.ppo --exp_name=PickCube_wc_s$s --env_id=PickCube-v1 --env_kwargs='{"robot_uids": "panda_wristcam"}' \
        --num_envs=1024 --num_steps=50 --total_timesteps=15_000_000 --eval_freq=10 --num_eval_envs=128 \
        --stop_success=0.95 --stop_patience=2 --seed=$s > PickCube_wc_s$s.log 2>&1 &
done
wait
for s in 1 2; do echo "PICK_WC_s$s $(tail -1 runs/PickCube_wc_s$s/eval.csv | cut -d, -f2,3) ckpt=$(ls runs/PickCube_wc_s$s/final_ckpt.pt 2>/dev/null)"; done
echo "PICK_WC_DONE $(date +%T)"
