#!/usr/bin/env bash
# Copy-term unit test U2: at theta=0 the source IS the target's solution, so KL-to-source from a fresh init (kickstart)
# must reach 80% success much faster than scratch (scratch T80 = 0.82M; the first kickstart, reverse KL with coef 0.5,
# took 1.97M). Three variants in parallel; the winner becomes the copy term for M1 and E0.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
mkdir -p runs/u2 && cd runs/u2 || exit 1
SRC=$HOME/pve/runs/m1/runs/source_t0/final_ckpt.pt
COMMON="--env_id=PushCubeDir-v1 --num_envs=2048 --num_steps=20 --total_timesteps=3_000_000 --eval_freq=2 --num_eval_envs=128 --stop_patience=2 --stop_success=0.9 --seed=1 --prior_ckpt=$SRC --gate=all"
echo "U2_START $(date +%T)"
# round 2: PPO's target-KL early stop (0.1 per decision) cuts the epoch short once the copy term moves the policy, so
# the copy term is tested with the early stop disabled, next to scratch with the same setting
python -m wrwl.ppo --exp_name=rkl_c05_notkl $COMMON --env_kwargs='{"theta_deg": 0}' --prior_coef=0.5 --copy_loss=rkl --target_kl=1e9 > rkl_c05_notkl.log 2>&1 &
python -m wrwl.ppo --exp_name=fkl_c05_notkl $COMMON --env_kwargs='{"theta_deg": 0}' --prior_coef=0.5 --copy_loss=fkl --target_kl=1e9 > fkl_c05_notkl.log 2>&1 &
python -m wrwl.ppo --exp_name=scratch_notkl --env_id=PushCubeDir-v1 --num_envs=2048 --num_steps=20 --total_timesteps=3_000_000 --eval_freq=2 --num_eval_envs=128 --stop_patience=2 --stop_success=0.9 --seed=1 --env_kwargs='{"theta_deg": 0}' --target_kl=1e9 > scratch_notkl.log 2>&1 &
wait
for d in runs/*/; do
    n=$(basename "$d")
    t80=$(awk -F, 'NR>1 && $3>=0.8 {print $2/1e6; exit}' "$d/eval.csv")
    echo "U2 $n T80=${t80:-not_reached}M last=$(tail -1 "$d/eval.csv" | cut -d, -f2,3)"
done
echo "U2_END $(date +%T)"
