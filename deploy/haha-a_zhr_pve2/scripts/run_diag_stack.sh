#!/usr/bin/env bash
# Why does the oracle gate fail on Stack (seed 1: oracle 0.00, kickstart 0.47 and 0.99 final) when it un-anchors only
# ~9% of visited states, mostly the early approach (stack labels: 41% unsafe at t < 10, 9-12% near the goal)?
#   oracle_ck   the oracle run again with checkpoints every 2.5M, halted when the anchor ends (12.5M): is the chain
#               already broken while anchored, or does it break after the anchor goes?
#   shuffle     the oracle with its labels permuted over the labelled states (same neighbourhoods, same copy fraction):
#               does it matter which states are un-anchored?
#   kickshort   kickstart with the anchor gone by 2.5M instead of 12.5M: kickstart's success came ~1M steps after its
#               anchor ended, so is the anchor's value only to hold competence while the critic warms up?
# Runs in runs/diag so analyze_sparse.py does not pick these up.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
echo "DIAG_START $(date +%T)"
PICK=$HOME/pve/runs/experts_wc/runs/PickCube_wc_s1/final_ckpt.pt
ST="--env_id=StackCubeNarrow-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --obs_adapter=stack_to_pick --num_envs=1024 --num_steps=50 --eval_freq=10 --total_timesteps=25_000_000 --seed=1 --bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=2 --num_eval_envs=128 --target_kl=1e9 --prior_ckpt=$PICK --prior_coef=0.5 --copy_loss=rkl --save_every=2_500_000"
ORA="--gate=oracle --oracle_labels=$HOME/pve/data/copyloss_stack.pt --oracle_source=Pick"
mkdir -p runs/diag && cd runs/diag || exit 1
: > jobs.txt
job() { echo "python -m wrwl.ppo --exp_name=$1 $2 > logs_$1.txt 2>&1 && echo DONE $1" >> jobs.txt; }
job stack_oracle_ck_s1 "$ST $ORA --halt_at=12_600_000"
job stack_shuffle_s1 "$ST $ORA --oracle_shuffle --halt_at=17_600_000"
job stack_kickshort_s1 "$ST --gate=all --prior_anneal_frac=0.1 --halt_at=10_100_000"
xargs -P 2 -d '\n' -I CMD bash -c CMD < jobs.txt
echo "DIAG_END $(date +%T)"
