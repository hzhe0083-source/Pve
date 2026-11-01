#!/usr/bin/env bash
# Pilot for "reuse the source's features, resolve the conflict locally" on Stack (sparse reward, PickCube expert as
# the source). scripts/conflict_probe.py found that a new linear readout on the frozen PickCube hidden layer predicts the
# StackCube expert's actions (held-out R^2 0.94 arm / 0.92 gripper; random features 0.84 / 0.80), and that the gripper
# conflict sits where the cube is held at the goal. No arm here except kick_boost has a copy term.
#   frozen_boost   PickCube actor as init, hidden layers frozen, gripper log-std +1.2 while the cube is held within
#                  2.5 cm of the goal (the conflict region, taken from the probe: an oracle for this pilot)
#   ft_boost       the same without freezing: does freezing matter?
#   frozen         frozen hidden layers, no boost: does the boost matter?
#   kick_boost     kickstarting as in step 3 (copy everywhere, coef 0.5 annealed over 12.5M) plus the boost: does the
#                  boost remove the coin flip of the anchored arms?
#   scratch_boost  random init plus the boost: does the boost need reuse to bring the policy to the conflict region?
# Step-3 baselines with the same settings: finetune 0.002 / 0, kickstart 0.47 / 0, scratch 0 / 0 (AUC, seeds 1 / 2).
# Every run halts at 15M steps (the schedules still span 25M); success = eval >= 0.8 by then.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
echo "PILOT_START $(date +%T)"
PICK=$HOME/pve/runs/experts_wc/runs/PickCube_wc_s1/final_ckpt.pt
ST="--env_id=StackCubeNarrow-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --obs_adapter=stack_to_pick --num_envs=1024 --num_steps=50 --eval_freq=10 --total_timesteps=25_000_000 --bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=2 --num_eval_envs=128 --target_kl=1e9 --halt_at=15_000_000"
INIT="--init_ckpt=$PICK --init_actor_only"
KICK="--prior_ckpt=$PICK --prior_coef=0.5 --copy_loss=rkl --gate=all"
BOOST="--explore_boost=1.2"
mkdir -p runs/pilot && cd runs/pilot || exit 1
: > jobs.txt
job() { echo "bash $HOME/pve/scripts/with_gpu_mem.sh 6000 python -m wrwl.ppo --exp_name=$1 $2 > logs_$1.txt 2>&1 && echo DONE $1" >> jobs.txt; }
for seed in 1 2 3; do
    job stack_frozen_boost_s$seed "$ST --seed=$seed $INIT --freeze_trunk $BOOST"
    job stack_ft_boost_s$seed "$ST --seed=$seed $INIT $BOOST"
    job stack_frozen_s$seed "$ST --seed=$seed $INIT --freeze_trunk"
done
for seed in 1 2; do job stack_kick_boost_s$seed "$ST --seed=$seed $KICK $BOOST"; done
job stack_scratch_boost_s1 "$ST --seed=1 $BOOST"
xargs -P 3 -d '\n' -I CMD bash -c CMD < jobs.txt
echo "PILOT_END $(date +%T)"
