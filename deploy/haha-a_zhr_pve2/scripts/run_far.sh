#!/usr/bin/env bash
# PushRetreat with a long retreat (env kwarg retreat_dist, e.g. 0.10): the PushCube source almost never backs away far
# enough, so the arm must explore after the push, while arm noise during the push spoils it. This is the case where
# seam-only exploration should beat global exploration, if locality matters at all (on Stack it did not: exploration
# was never the bottleneck there). Every floor arm (global, hand-written seam, done-probe seam) runs at each dose in
# FAR_DOSES, so each can be reported at its own best dose (tuned vs tuned); frozen and fine-tune without exploration
# are the no-exploration references.
#   FAR_DIST=0.10 FAR_DOSES="0.6 1.0" FAR_SEEDS="1 2 3" FAR_P=2 bash scripts/run_far.sh
set -u
R=${FAR_ROOT:-$HOME/zhr_pve}
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
G="bash $R/scripts/with_gpu_mem.sh 6000"
PUSH=$R/runs/experts/runs/PushCube_c1/final_ckpt.pt
DIST=${FAR_DIST:-0.10}
TAG=far$(echo "$DIST" | sed 's/^0\.//')
D=runs/$TAG
mkdir -p "$D" logs
ENV="--env_id=PushRetreat-v1 --env_kwargs='{\"reward_mode\": \"sparse\", \"retreat_dist\": $DIST}' --num_envs=2048 --num_steps=20 --eval_freq=5 --total_timesteps=15_000_000 --halt_at=10_000_000"
COMMON="$ENV --bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=2 --num_eval_envs=128 --target_kl=1e9 --init_ckpt=$PUSH --init_actor_only"
SEAM="--seam_rule=push --explore_radius=0.1 --seam_dims=arm"
J=$D/jobs.txt
: > "$J"
for seed in ${FAR_SEEDS:-1 2 3}; do
    n() { echo "$G python -m wrwl.ppo --exp_name=${TAG}_$1_s$seed $2 > logs_${TAG}_$1_s$seed.txt 2>&1 && echo DONE ${TAG}_$1_s$seed" >> "$J"; }
    n frozen "$COMMON --seed=$seed --freeze_trunk"
    n ft "$COMMON --seed=$seed"
    for dose in ${FAR_DOSES:-0.6 1.0}; do
        f=$(echo "$dose" | tr -d .)
        n global$f "$COMMON --seed=$seed --freeze_trunk $SEAM --min_grip_std=$dose"
        n seam$f "$COMMON --seed=$seed --freeze_trunk $SEAM --min_grip_std=$dose --grip_floor_seam"
        n probe$f "$COMMON --seed=$seed --freeze_trunk $SEAM --min_grip_std=$dose --grip_floor_seam --seam_probe=$R/data/done_probe_push_c1.pt"
    done
done
echo "FAR_START $(date +%T) dist=$DIST doses=${FAR_DOSES:-0.6 1.0} jobs=$(wc -l < "$J")"
(cd "$D" && xargs -P "${FAR_P:-2}" -d '\n' -I CMD bash -c CMD < jobs.txt)
echo "FAR_END $(date +%T)"
