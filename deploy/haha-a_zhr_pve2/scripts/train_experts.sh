#!/usr/bin/env bash
# Train single-task state-PPO experts with the official ManiSkill ppo.py (v3.0.1), two lanes at a time on one GPU.
# Budgets follow examples/baselines/ppo/examples.sh; checkpoints land in ~/pve/runs/experts/runs/<name>/.
set -u
cd "$HOME/pve/runs/experts" || exit 1
source "$HOME/pve/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890
PPO=$HOME/pve/third_party/ms_ppo/ppo.py
COMMON=(--update_epochs=8 --num_minibatches=32 --num_eval_envs=16 --no-capture-video --seed=1)
mkdir -p logs

run() {
    local name=$1; shift
    echo "START $name $(date +%T)"
    python "$PPO" --exp_name="$name" "${COMMON[@]}" "$@" > "logs/$name.txt" 2>&1
    echo "END $name rc=$? $(date +%T)"
}

(
    run PushCube --env_id=PushCube-v1 --num_envs=2048 --total_timesteps=5_000_000 --eval_freq=10 --num-steps=20
    run PickCube --env_id=PickCube-v1 --num_envs=1024 --total_timesteps=10_000_000
    run PokeCube --env_id=PokeCube-v1 --num_envs=1024 --total_timesteps=10_000_000 --eval_freq=10 --num-steps=20
    run LiftPegUpright --env_id=LiftPegUpright-v1 --num_envs=2048 --total_timesteps=25_000_000
) &
(
    run PullCube --env_id=PullCube-v1 --num_envs=2048 --total_timesteps=5_000_000 --eval_freq=10 --num-steps=20
    run StackCube --env_id=StackCube-v1 --num_envs=1024 --total_timesteps=25_000_000
    run PlaceSphere --env_id=PlaceSphere-v1 --num_envs=1024 --total_timesteps=25_000_000
    run PullCubeTool --env_id=PullCubeTool-v1 --num_envs=1024 --total_timesteps=25_000_000 --num-steps=100 --num_eval_steps=100
) &
wait
echo EXPERTS_DONE
