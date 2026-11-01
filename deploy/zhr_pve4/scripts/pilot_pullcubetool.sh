#!/usr/bin/env bash
# PullCubeTool's expert (official ppo.py defaults, gamma 0.8, 25M steps) never succeeded. Its 100-step horizon is like
# PushT's, which the official baselines train with gamma 0.99. Two pilots, gamma 0.95 and 0.99, decide the E0 config.
set -u
cd "$HOME/pve/runs/experts" || exit 1
source "$HOME/pve/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890
PPO=$HOME/pve/third_party/ms_ppo/ppo.py
COMMON=(--env_id=PullCubeTool-v1 --num_envs=2048 --num-steps=50 --num_eval_steps=100 --total_timesteps=25_000_000
        --update_epochs=8 --num_minibatches=32 --num_eval_envs=64 --eval_freq=10 --gae_lambda=0.95 --no-capture-video --seed=1)
echo "PCT_PILOT_START $(date +%T)"
python "$PPO" --exp_name=PullCubeTool_g095 --gamma=0.95 "${COMMON[@]}" > logs/PullCubeTool_g095.txt 2>&1 &
python "$PPO" --exp_name=PullCubeTool_g099 --gamma=0.99 "${COMMON[@]}" > logs/PullCubeTool_g099.txt 2>&1 &
wait
echo "PCT_PILOT_DONE $(date +%T)"
