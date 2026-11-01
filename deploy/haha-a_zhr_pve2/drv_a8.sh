#!/usr/bin/env bash
# One-off on haha-a (01 Nov, ~16:10): after drv_a7.sh (chain + nsa_eval) ends, learn PlaceCube (seeds 1-4) and
# StackCubeNarrow (seeds 1-3) from scratch with scripts/run_scratch.sh: the reference for what adding a skill to an
# existing network saves (rules S1/S2 registered in docs/conflict-reuse-2026-10-31.md, 16:05). A smoke run of the
# scratch path first. Every job yields the card to other users (gpu_polite.sh); up to 4 passes per task.
R=$HOME/zhr_pve2
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
SRC=$R/runs/src_c
NPS=$R/runs/protect_np/runs
PICK=$R/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
P="bash $R/scripts/gpu_polite.sh"
polite() {      # gpu_polite.sh exits 75 when it yields the card to another user: wait, then run the command again
    local rc
    until $P "$@"; rc=$?; [ $rc != 75 ]; do echo "yielded $(date +%T), retry in 300 s"; sleep 300; done
    return $rc
}
until [ -f logs/drv_a6.done ]; do sleep 60; done
rm -f logs/drv_a8.done
echo "start $(date +%T)"
mkdir -p smoke
r_s=$(cd smoke && polite python -m wrwl.ppo --exp_name=smoke_scratch --env_id=PlaceCube-v1 \
    --env_kwargs={\"robot_uids\":\"panda_wristcam\",\"reward_mode\":\"sparse\"} --num_envs=256 --num_steps=50 \
    --eval_freq=1000 --num_eval_envs=16 --total_timesteps=38400 --bootstrap_at_done=truncated --gamma=0.95 \
    --target_kl=1e9 --min_grip_std=1.0 > smoke_scratch.txt 2>&1; echo $?)
echo "smoke scratch $r_s $(date +%T)"
passes() {      # $1 = label, $2 = done-check command, rest = the queue command; up to 4 passes
    local label=$1 check=$2; shift 2
    for pass in 1 2 3 4; do
        "$@"
        left=$(eval "$check")
        echo "$label pass $pass $(date +%T) unfinished $left"
        [ "$left" = 0 ] && return
        sleep 300
    done
}
left_s() { for s in $SEEDS_S; do [ -f runs/scratch/runs/${TASK_S}_scratch_s$s/final_ckpt.pt ] || echo x; done | wc -l; }
run_s() { PROT_ROOT=$R SCR_TASK=$TASK_S SCR_SEEDS="$SEEDS_S" SCR_P=3 bash scripts/run_scratch.sh >> logs/run_scratch_a8.log 2>&1; }
if [ "$r_s" = 0 ]; then
    TASK_S=place SEEDS_S="1 2 3 4"
    passes place left_s run_s
    TASK_S=stack SEEDS_S="1 2 3"
    passes stack left_s run_s
fi
echo "PROT_END $(date +%T)" > logs/drv_a8.done
