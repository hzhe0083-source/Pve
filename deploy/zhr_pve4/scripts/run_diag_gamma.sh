#!/usr/bin/env bash
# Step 2 picked gamma* = 0.8 after step 3 had started at 0.95: scratch learns T90 at 0.8 (AUC 0.84, T80 3.2M, 2 seeds)
# but not at 0.95 (0 in 25M). Does the step-3 ranking hold at 0.8? T45 scratch, and T90 finetune, kickstart, oracle,
# seed 1, same settings as run_night1d.sh except gamma. Starts after run_diag_stack2.sh.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
until grep -q DIAG2_END logs/diag_stack2.log 2>/dev/null; do sleep 60; done
echo "DIAG_GAMMA_START $(date +%T)"
S0=$HOME/pve/runs/experts/runs/PushCube/final_ckpt.pt
COMMON="--bootstrap_at_done=truncated --gamma=0.8 --stop_success=0.9 --stop_patience=2 --num_eval_envs=128 --target_kl=1e9 --num_envs=2048 --num_steps=20 --eval_freq=5 --total_timesteps=15_000_000 --seed=1"
P45="--env_id=PushCubeDir-v1 --env_kwargs='{\"theta_deg\": 45, \"reward_mode\": \"sparse\"}' $COMMON"
P90="--env_id=PushCubeDir-v1 --env_kwargs='{\"theta_deg\": 90, \"reward_mode\": \"sparse\"}' $COMMON"
KL="--prior_ckpt=$S0 --prior_coef=0.5 --copy_loss=rkl"
mkdir -p runs/diag && cd runs/diag || exit 1
: > jobs_gamma.txt
job() { echo "python -m wrwl.ppo --exp_name=$1 $2 > logs_$1.txt 2>&1 && echo DONE $1" >> jobs_gamma.txt; }
job t45_scratch_g08_s1 "$P45"
job t90_finetune_g08_s1 "$P90 --init_ckpt=$S0 --init_actor_only"
job t90_kickstart_g08_s1 "$P90 $KL --gate=all"
job t90_oracle_g08_s1 "$P90 $KL --gate=oracle --oracle_labels=$HOME/pve/data/copyloss_push90.pt --oracle_source=S0"
xargs -P 2 -d '\n' -I CMD bash -c CMD < jobs_gamma.txt
echo "DIAG_GAMMA_END $(date +%T)"
