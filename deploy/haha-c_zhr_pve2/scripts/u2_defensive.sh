#!/usr/bin/env bash
# Memo G0 item: does the defensive prior 1/2 p_L + 1/2 pi_base still pull? U2 again (theta 0, the source is the
# answer), now under sparse reward with the step-3 settings; plain reverse KL vs the defensive mixture. Compare T80
# with step 2's scratch at theta 0, gamma 0.95. The memo's bar: T80 <= 0.5 x scratch.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
mkdir -p runs/u2s && cd runs/u2s || exit 1
S0=$HOME/pve/runs/experts/runs/PushCube/final_ckpt.pt
C="--env_id=PushCubeDir-v1 --num_envs=2048 --num_steps=20 --total_timesteps=5_000_000 --eval_freq=2 --num_eval_envs=128 --stop_success=0.9 --stop_patience=2 --seed=1 --bootstrap_at_done=truncated --gamma=0.95 --prior_ckpt=$S0 --prior_coef=0.5 --gate=all --target_kl=1e9"
echo "U2S_START $(date +%T)"
python -m wrwl.ppo --exp_name=rkl $C --env_kwargs='{"theta_deg": 0, "reward_mode": "sparse"}' --copy_loss=rkl > rkl.log 2>&1 &
python -m wrwl.ppo --exp_name=rkl_def $C --env_kwargs='{"theta_deg": 0, "reward_mode": "sparse"}' --copy_loss=rkl_def > rkl_def.log 2>&1 &
wait
for n in rkl rkl_def; do
    t80=$(awk -F, 'NR>1 && $3>=0.8 {print $2/1e6; exit}' runs/$n/eval.csv)
    echo "U2S $n T80=${t80:-not_reached}M last=$(tail -1 runs/$n/eval.csv | cut -d, -f2,3)"
done
echo "U2S_END $(date +%T)"
