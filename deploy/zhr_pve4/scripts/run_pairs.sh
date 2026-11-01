#!/usr/bin/env bash
# More extension pairs for the reuse-and-resolve recipe (the target is the source task plus one final step):
#   place   PickCube -> PlaceCube-v1    release the cube on the table goal (gripper; like Stack without cube B)
#   retreat PushCube -> PushRetreat-v1  back the arm away once the cube is in the goal (arm dims, not the gripper)
# Prep (each step waits for GPU memory): the PushCube source expert, its done-probe, the done-probe checks, and the
# exploration-yield checks that pick the floor value (the hand-seam layout with the best first-success yield; the
# global and probe arms get the same value). Then the RL arms, seeds 1-4, sparse reward, frozen hidden layers unless
# noted: frozen, ft, floor_global, floor_seam, floor_probe, kick, scratch_floor.
#   PAIRS_ROOT=$HOME/zhr_pve PAIRS_P=3 bash scripts/run_pairs.sh place retreat
set -u
R=${PAIRS_ROOT:-$HOME/pve}
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
G="bash $R/scripts/with_gpu_mem.sh 6000"
echo "PAIRS_START $(date +%T) on $(hostname): $*"
PICK=$R/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
PUSH=$R/runs/experts/runs/PushCube_c1/final_ckpt.pt
mkdir -p logs data runs/experts runs/pairs

pick_floor() {     # the hand-seam layout with the highest first-success yield in a yield-check JSON
    python - "$1" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
best = max((k for k in r if k.startswith("seam:")), key=lambda k: r[k]["success"])
print(best.split(":")[1])
PY
}

for pair in "$@"; do
    if [ "$pair" = place ]; then
        $G python scripts/explore_yield.py --pick=$PICK --env=PlaceCube-v1 --env_kwargs='{"robot_uids": "panda_wristcam"}' \
            --adapter=none --obs_dim=42 --seam_rule=pick --dims=gripper --probe=data/done_probe_pick_c1.pt \
            --layouts=source,global:0.6,global:1.0,seam:0.6,seam:1.0,pseam:1.0 --out=logs/yield_place.json \
            > logs/yield_place.txt 2>&1
        $G python scripts/train_done_probe.py --src=$PICK --out=data/done_probe_pick_c1.pt --check --tgt_env=PlaceCube-v1 \
            --tgt_kwargs='{"robot_uids": "panda_wristcam"}' --adapter=none > logs/probe_check_place.txt 2>&1
        F=$(pick_floor logs/yield_place.json)
        SRC=$PICK; ENV="--env_id=PlaceCube-v1 --env_kwargs='{\"robot_uids\": \"panda_wristcam\", \"reward_mode\": \"sparse\"}' --num_envs=1024 --num_steps=50 --eval_freq=10 --total_timesteps=25_000_000 --halt_at=15_000_000"
        SEAM="--seam_rule=pick --seam_dims=gripper"; PROBE=data/done_probe_pick_c1.pt
    elif [ "$pair" = retreat ]; then
        [ -f "$PUSH" ] || (cd runs/experts && $G python $R/third_party/ms_ppo/ppo.py --exp_name=PushCube_c1 \
            --env_id=PushCube-v1 --num_envs=2048 --total_timesteps=5_000_000 --eval_freq=10 --num-steps=20 \
            --update_epochs=8 --num_minibatches=32 --num_eval_envs=16 --no-capture-video --seed=1 > PushCube_c1.log 2>&1)
        [ -f data/done_probe_push_c1.pt ] || $G python scripts/train_done_probe.py --src=$PUSH --src_env=PushCube-v1 \
            --src_kwargs='{}' --obs_dim=35 --out=data/done_probe_push_c1.pt > logs/probe_push.txt 2>&1
        $G python scripts/train_done_probe.py --src=$PUSH --obs_dim=35 --out=data/done_probe_push_c1.pt --check \
            --tgt_env=PushRetreat-v1 --adapter=none --seam_rule=push --radius=0.1 > logs/probe_check_retreat.txt 2>&1
        $G python scripts/explore_yield.py --pick=$PUSH --env=PushRetreat-v1 --adapter=none --obs_dim=35 --seam_rule=push \
            --radius=0.1 --dims=arm --probe=data/done_probe_push_c1.pt \
            --layouts=source,global:0.3,global:0.6,global:1.0,seam:0.3,seam:0.6,seam:1.0,pseam:0.6 \
            --out=logs/yield_retreat.json > logs/yield_retreat.txt 2>&1
        F=$(pick_floor logs/yield_retreat.json)
        SRC=$PUSH; ENV="--env_id=PushRetreat-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --num_envs=2048 --num_steps=20 --eval_freq=5 --total_timesteps=15_000_000 --halt_at=10_000_000"
        SEAM="--seam_rule=push --explore_radius=0.1 --seam_dims=arm"; PROBE=data/done_probe_push_c1.pt
    else
        echo "unknown pair $pair"; continue
    fi
    echo "PAIR $pair floor=$F $(date +%T)"
    COMMON="$ENV --bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=2 --num_eval_envs=128 --target_kl=1e9"
    INIT="--init_ckpt=$SRC --init_actor_only"
    D=runs/pairs/$pair
    mkdir -p "$D"
    J=$D/jobs.txt
    : > "$J"
    for seed in ${PAIRS_SEEDS:-1 2 3 4}; do
        n() { echo "$G python -m wrwl.ppo --exp_name=${pair}_$1_s$seed $2 > logs_${pair}_$1_s$seed.txt 2>&1 && echo DONE ${pair}_$1_s$seed" >> "$J"; }
        n frozen "$COMMON $INIT --seed=$seed --freeze_trunk"
        n ft "$COMMON $INIT --seed=$seed"
        n floor_global "$COMMON $INIT --seed=$seed --freeze_trunk $SEAM --min_grip_std=$F"
        n floor_seam "$COMMON $INIT --seed=$seed --freeze_trunk $SEAM --min_grip_std=$F --grip_floor_seam"
        n floor_probe "$COMMON $INIT --seed=$seed --freeze_trunk $SEAM --min_grip_std=$F --grip_floor_seam --seam_probe=$R/$PROBE"
        n kick "$COMMON --seed=$seed --prior_ckpt=$SRC --prior_coef=0.5 --copy_loss=rkl --gate=all"
        n scratch_floor "$COMMON --seed=$seed $SEAM --min_grip_std=$F"
    done
    (cd "$D" && xargs -P "${PAIRS_P:-3}" -d '\n' -I CMD bash -c CMD < jobs.txt)
    echo "PAIR_END $pair $(date +%T)"
done
echo "PAIRS_END $(date +%T)"
