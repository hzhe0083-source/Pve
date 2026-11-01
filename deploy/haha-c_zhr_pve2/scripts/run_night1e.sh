#!/usr/bin/env bash
# Step 3 add-on: the in-network form of WRWL. Start from the source's actor (like finetune) and anchor to it with the
# same gated KL (oracle) or everywhere (kickstart). Seed 1 night 1 showed finetune (warm start, no KL) far ahead of the
# cold-start KL arms on T45, so the fair test of gating is warm start + gated anchor vs warm start alone.
# Runs after step 2 (run_night1b.sh) frees its two lanes.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
until grep -q NIGHT1B_END logs/night1b.log 2>/dev/null; do sleep 60; done
echo "NIGHT1E_START $(date +%T)"
R=$HOME/pve/runs
S0=$R/experts/runs/PushCube/final_ckpt.pt
PICK=$R/experts_wc/runs/PickCube_wc_s1/final_ckpt.pt
COMMON="--bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=2 --num_eval_envs=128 --target_kl=1e9 --prior_coef=0.5 --copy_loss=rkl --init_actor_only"
cd runs/step3 || exit 1
: > jobs_e.txt
for seed in 1 2; do
    P45="--env_id=PushCubeDir-v1 --env_kwargs='{\"theta_deg\": 45, \"reward_mode\": \"sparse\"}' --num_envs=2048 --num_steps=20 --eval_freq=5 --total_timesteps=15_000_000 --seed=$seed $COMMON --init_ckpt=$S0 --prior_ckpt=$S0"
    P90="--env_id=PushCubeDir-v1 --env_kwargs='{\"theta_deg\": 90, \"reward_mode\": \"sparse\"}' --num_envs=2048 --num_steps=20 --eval_freq=5 --total_timesteps=15_000_000 --seed=$seed $COMMON --init_ckpt=$S0 --prior_ckpt=$S0"
    ST="--env_id=StackCubeNarrow-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --obs_adapter=stack_to_pick --num_envs=1024 --num_steps=50 --eval_freq=10 --total_timesteps=25_000_000 --seed=$seed $COMMON --init_ckpt=$PICK --prior_ckpt=$PICK"
    for spec in "t45|$P45|push45|S0" "stack|$ST|stack|Pick" "t90|$P90|push90|S0"; do
        IFS='|' read -r task args lab src <<< "$spec"
        echo "python -m wrwl.ppo --exp_name=${task}_warmoracle_s$seed $args --gate=oracle --oracle_labels=$HOME/pve/data/copyloss_$lab.pt --oracle_source=$src > logs_${task}_warmoracle_s$seed.txt 2>&1 && echo DONE ${task}_warmoracle_s$seed" >> jobs_e.txt
        echo "python -m wrwl.ppo --exp_name=${task}_warmkick_s$seed $args --gate=all > logs_${task}_warmkick_s$seed.txt 2>&1 && echo DONE ${task}_warmkick_s$seed" >> jobs_e.txt
    done
done
echo "NIGHT1E_JOBS $(wc -l < jobs_e.txt)"
xargs -P 2 -d '\n' -I CMD bash -c CMD < jobs_e.txt
echo "NIGHT1E_END $(date +%T)"
