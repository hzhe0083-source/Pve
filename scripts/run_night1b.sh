#!/usr/bin/env bash
# Rest of night 1 after the pivot memo (docs/pivot-memo-2026-10-31.md). E0, the src5 distillations and the dense-reward
# hand gate are cancelled. After M1 finishes:
#   1. restore fidelity under the experts                      G1 condition 2; the branch labels depend on it
#   2. KL scaling, PPO-clip vs covariant                         G1 condition 4; only the fallback paper needs it, cheap
#   3. memo step 2: can PPO learn from scratch under binary reward? PushCubeDir theta in {0, 90, 180} x gamma in
#      {0.8, 0.95}, PickCube and StackCube at gamma 0.95, 2 seeds each, seeds interleaved. reward_mode sparse (reward =
#      success, success ends the episode), value bootstrapped only at time-outs, budget 25M, stop at 0.9.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890
until grep -q M1_END logs/m1.log 2>/dev/null; do sleep 60; done
echo "NIGHT1B_START $(date +%T)"

python scripts/restore_check_expert.py > logs/restore_expert.log 2>&1
echo "RESTORE_DONE rc=$? $(date +%T) $(grep overall logs/restore_expert.log)"

# KL scaling: same physical task (2.5 s episodes, 1 s rollouts), decisions split m = 25 / 50 / 125 ways
mkdir -p runs/klscale && cd runs/klscale || exit 1
: > jobs.txt
for seed in 1 2; do
    for cf in 20 10 50; do
        H=$(( 5 * cf / 2 ))
        read -r G L <<< "$(python -c "print(0.8 ** (20 / $cf), 0.9 ** (20 / $cf))")"
        for loss in clip cov; do
            name=${loss}_cf${cf}_s$seed
            echo "python -m wrwl.ppo --exp_name=$name --env_id=PushCube-v1 --seed=$seed --loss=$loss --delta=0.5 \
--control_freq=$cf --max_episode_steps=$H --num_eval_steps=$H --num_steps=$cf --gamma=$G --gae_lambda=$L \
--num_envs=2048 --total_timesteps=$(( 150000 * cf )) --eval_freq=5 --num_eval_envs=128 \
> logs_$name.txt 2>&1 && echo DONE $name" >> jobs.txt
        done
    done
done
xargs -P 2 -d '\n' -I CMD bash -c CMD < jobs.txt
echo "KLSCALE_DONE $(date +%T)"
cd "$HOME/pve" || exit 1

mkdir -p runs/sparse && cd runs/sparse || exit 1
SP="--bootstrap_at_done=truncated --total_timesteps=25_000_000 --stop_success=0.9 --stop_patience=2 --num_eval_envs=128"
: > jobs.txt
for seed in 1 2; do
    for th in 0 90 180; do
        for g in 0.8 0.95; do
            name=push_t${th}_g${g}_s$seed
            echo "python -m wrwl.ppo --exp_name=$name --env_id=PushCubeDir-v1 \
--env_kwargs='{\"theta_deg\": $th, \"reward_mode\": \"sparse\"}' --num_envs=2048 --num_steps=20 --eval_freq=5 \
--gamma=$g --seed=$seed $SP > logs_$name.txt 2>&1 && echo DONE $name" >> jobs.txt
        done
    done
    for task in PickCube StackCube; do
        name=${task}_g0.95_s$seed
        echo "python -m wrwl.ppo --exp_name=$name --env_id=$task-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' \
--num_envs=1024 --num_steps=50 --eval_freq=10 --gamma=0.95 --seed=$seed $SP > logs_$name.txt 2>&1 && echo DONE $name" >> jobs.txt
    done
done
xargs -P 2 -d '\n' -I CMD bash -c CMD < jobs.txt
echo "SPARSE_DONE $(date +%T)"
echo "NIGHT1B_END $(date +%T)"
