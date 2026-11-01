#!/usr/bin/env bash
# Follow-up to run_diag_stack.sh. Seed 1: the true-label oracle never succeeded, the label-shuffled control (same
# neighbourhoods, same copy fraction) went 0 -> 0.94 within 0.5M steps of its anchor ending, like kickstart.
#   shuffle s2, s3     is "the gold labels gate worse than random labels" reproducible? (the true-label oracle s2, s3 come
#                      from run_night1d.sh)
#   kickstart_ck s1    kickstart with checkpoints up to the end of its anchor: does it build up gripper exploration and
#                      sampled successes while anchored, as the shuffled control did (gripper std 0.75 -> 1.46, sampled
#                      success 22% at 12.5M), although it anchors every state?
# Starts after run_diag_stack.sh; run_diag_gamma.sh waits for this one.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
until grep -q DIAG_END logs/diag_stack.log 2>/dev/null; do sleep 60; done
echo "DIAG2_START $(date +%T)"
PICK=$HOME/pve/runs/experts_wc/runs/PickCube_wc_s1/final_ckpt.pt
ST="--env_id=StackCubeNarrow-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --obs_adapter=stack_to_pick --num_envs=1024 --num_steps=50 --eval_freq=10 --total_timesteps=25_000_000 --bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=2 --num_eval_envs=128 --target_kl=1e9 --prior_ckpt=$PICK --prior_coef=0.5 --copy_loss=rkl --save_every=2_500_000"
ORA="--gate=oracle --oracle_labels=$HOME/pve/data/copyloss_stack.pt --oracle_source=Pick --oracle_shuffle"
mkdir -p runs/diag && cd runs/diag || exit 1
: > jobs2.txt
job() { echo "python -m wrwl.ppo --exp_name=$1 $2 > logs_$1.txt 2>&1 && echo DONE $1" >> jobs2.txt; }
job stack_kickstart_ck_s1 "$ST --seed=1 --gate=all --halt_at=12_600_000"
job stack_shuffle_s2 "$ST --seed=2 $ORA --halt_at=17_600_000"
job stack_shuffle_s3 "$ST --seed=3 $ORA --halt_at=17_600_000"
xargs -P 2 -d '\n' -I CMD bash -c CMD < jobs2.txt
echo "DIAG2_END $(date +%T)"
