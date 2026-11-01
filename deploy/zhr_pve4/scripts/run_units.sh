#!/usr/bin/env bash
# Key parameters + incremental training (the user's hypothesis): from the frozen source, train the output layer plus a
# small set of hidden units chosen by their policy-gradient importance on the new task (--train_units, picked after
# --select_iters head-only iterations), against the same number of random units, the output layer alone, and full
# fine-tuning. Two pairs:
#   stack  PickCube -> StackCube (sparse, gripper floor 1.0 everywhere, the best placement so far); the output layer
#          alone already solves it, so this pair can only show speed
#   pull   PushCube -> PullCube (dense reward, no floor): the motion is reversed, so the frozen features may not be
#          enough; check first that the output layer alone falls short, otherwise this pair cannot test the claim
# Arms: head, key5 / rand5 (5% of each hidden layer's units), key20 / rand20, full; pull adds scratch (no reuse).
#   UNITS_TASKS="stack pull" UNITS_SEEDS="1 2 3 4" UNITS_P=2 bash scripts/run_units.sh
set -u
R=${UNITS_ROOT:-$HOME/zhr_pve}
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
G="bash $R/scripts/with_gpu_mem.sh 6000"
PICK=$R/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
PUSH=$R/runs/experts/runs/PushCube_c1/final_ckpt.pt
D=${UNITS_DIR:-runs/units}
mkdir -p "$D" logs
J=$D/jobs_$(hostname).txt
: > "$J"
for task in ${UNITS_TASKS:-stack pull}; do
    if [ "$task" = stack ]; then
        COMMON="--env_id=StackCubeNarrow-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --obs_adapter=stack_to_pick \
--num_envs=1024 --num_steps=50 --eval_freq=2 --num_eval_envs=256 --total_timesteps=25_000_000 --halt_at=8_000_000 \
--bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=3 --target_kl=1e9 --min_grip_std=1.0"
        SRC=$PICK
    else
        COMMON="--env_id=PullCube-v1 --env_kwargs='{\"reward_mode\": \"normalized_dense\"}' --num_envs=2048 --num_steps=20 \
--eval_freq=5 --num_eval_envs=256 --total_timesteps=15_000_000 --halt_at=10_000_000 --stop_success=0.9 --stop_patience=3"
        SRC=$PUSH
    fi
    [ -f "$SRC" ] || { echo "MISSING source checkpoint $SRC on $(hostname)"; exit 1; }   # (21:40: a missing Push expert on haha-d crashed every pull job)
    INIT="--init_ckpt=$SRC --init_actor_only"
    for seed in ${UNITS_SEEDS:-1 2 3 4}; do
        n() { echo "$G python -m wrwl.ppo --exp_name=${task}_$1_s$seed $2 > logs_${task}_$1_s$seed.txt 2>&1 && echo DONE ${task}_$1_s$seed" >> "$J"; }
        n head "$COMMON $INIT --seed=$seed --freeze_trunk"
        n key5 "$COMMON $INIT --seed=$seed --freeze_trunk --train_units=0.05 --train_units_by=grad --select_iters=5"
        n rand5 "$COMMON $INIT --seed=$seed --freeze_trunk --train_units=0.05 --train_units_by=random --select_iters=5"
        n key20 "$COMMON $INIT --seed=$seed --freeze_trunk --train_units=0.2 --train_units_by=grad --select_iters=5"
        n rand20 "$COMMON $INIT --seed=$seed --freeze_trunk --train_units=0.2 --train_units_by=random --select_iters=5"
        n full "$COMMON $INIT --seed=$seed"
        [ "$task" = pull ] && n scratch "$COMMON --seed=$seed"
    done
done
echo "UNITS_START $(date +%T) $(hostname) jobs=$(wc -l < "$J")"
(cd "$D" && xargs -P "${UNITS_P:-2}" -d '\n' -I CMD bash -c CMD < "$(basename "$J")")
echo "UNITS_END $(date +%T)"
