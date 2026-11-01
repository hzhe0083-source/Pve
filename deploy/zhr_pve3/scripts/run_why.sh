#!/usr/bin/env bash
# Why is the projected adapter faster than the prefix sum on PlaceCube? Mechanism queue for scripts/why_test.py (rules
# registered there before any of these runs). Every arm is run_protect.sh's PlaceCube nsa64 command (same COMMON, stop,
# halt, old-skill rollouts) plus its own flags; runs/why/runs/place_<arm>_s<seed>, one code version for all.
#   nsa64   the projected adapter, rerun with the stage logging     npr   the prefix sum from nsa64's initial network
#   rp      same-rank random projection (--nsa_proj random)
#   rf01 rf1 rf10      npr + --nsa_restore 0.1 / 1 / 10 (restoring force on W_old g, on-policy states)
#   nsa64d nprd        nsa64 / npr with the stage-1 cost 0.05 (env kwarg stage1_cost)
#   nsa64lo nsa64hi nprlo nprhi   --adapter_lr 1e-4 / 1e-3 (head, log-std and critic stay at 3e-4)
#   nsa64vlo nsa64vhi nprvlo nprvhi   3e-5 / 3e-3: only when why_test.py asks for them (best rate at an edge)
#   vis none nsa64z    rule F: write only what the old head reads (--nsa_mask visible), no write (--nsa_mask zero),
#                      and nsa64 with a zero last adapter layer (--nsa_init zero)
# WHY_SMOKE=1: every listed arm on seed 1 for two iterations (exp names smoke_<arm>, not read by why_test.py).
#   WHY_ARMS="nsa64 npr rp" WHY_SEEDS="1 2 3 4 5 6 7 8" WHY_P=3 bash scripts/run_why.sh
# Each job line ends with DONE <run> or EXIT <code> <run> (75: yielded to another user; 137: killed); runs that already
# have final_ckpt.pt are skipped, so rerunning the script resumes the queue.
set -u
R=${PROT_ROOT:-$HOME/zhr_pve}
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
G="bash $R/scripts/with_gpu_mem.sh 6000 bash $R/scripts/gpu_polite.sh"
PICK=$R/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
[ -f "$PICK" ] || { echo "MISSING $PICK on $(hostname)"; exit 1; }
D=${WHY_DIR:-runs/why}
mkdir -p "$D"
J=$D/jobs_$(hostname)_$$.txt
: > "$J"
STOP="--halt_at=8_000_000 --stop_success=0.9 --stop_patience=3 --save_every=500000"
[ "${WHY_SMOKE:-0}" = 1 ] && STOP="--halt_at=102400"
COMMON="--env_id=PlaceCube-v1 --num_envs=1024 --num_steps=50 --eval_freq=2 --num_eval_envs=256 \
--total_timesteps=25_000_000 $STOP --bootstrap_at_done=truncated --gamma=0.95 --target_kl=1e9 --min_grip_std=1.0 \
--init_ckpt=$PICK --init_actor_only --nullspace_adapter=64 --protect_env=PickCube-v1 --protect_dagger=10"
envk() {    # the env kwargs, with the stage-1 cost for the dense arms
    local c=""
    case $1 in nsa64d|nprd) c=', "stage1_cost": 0.05' ;; esac
    echo "--env_kwargs='{\"robot_uids\": \"panda_wristcam\", \"reward_mode\": \"sparse\"$c}'"
}
flags() {
    local npr="--nsa_project=False --nsa_init=rand"
    case $1 in
        nsa64|nsa64d) echo "" ;;
        npr|nprd) echo "$npr" ;;
        rp) echo "--nsa_proj=random" ;;
        vis) echo "--nsa_mask=visible" ;;
        none) echo "--nsa_mask=zero" ;;
        nsa64z) echo "--nsa_init=zero" ;;
        rf01) echo "$npr --nsa_restore=0.1" ;;
        rf1) echo "$npr --nsa_restore=1" ;;
        rf10) echo "$npr --nsa_restore=10" ;;
        nsa64lo) echo "--adapter_lr=1e-4" ;;
        nsa64hi) echo "--adapter_lr=1e-3" ;;
        nsa64vlo) echo "--adapter_lr=3e-5" ;;
        nsa64vhi) echo "--adapter_lr=3e-3" ;;
        nprlo) echo "$npr --adapter_lr=1e-4" ;;
        nprhi) echo "$npr --adapter_lr=1e-3" ;;
        nprvlo) echo "$npr --adapter_lr=3e-5" ;;
        nprvhi) echo "$npr --adapter_lr=3e-3" ;;
        *) echo "UNKNOWN" ;;
    esac
}
ARMS=${WHY_ARMS:-"nsa64 npr rp"}
for a in $ARMS; do [ "$(flags $a)" = UNKNOWN ] && { echo "UNKNOWN ARM $a"; exit 1; }; done
SEEDS=${WHY_SEEDS:-"1 2 3 4 5 6 7 8"}
[ "${WHY_SMOKE:-0}" = 1 ] && SEEDS=1
for seed in $SEEDS; do
    for a in $ARMS; do
        e=place_${a}_s$seed
        [ "${WHY_SMOKE:-0}" = 1 ] && e=smoke_$a
        [ -f "$D/runs/$e/final_ckpt.pt" ] && continue
        echo "$G python -m wrwl.ppo --exp_name=$e $(envk $a) $COMMON --seed=$seed $(flags $a) > logs_$e.txt 2>&1 && echo DONE $e || echo EXIT \$? $e" >> "$J"
    done
done
echo "WHY_START $(date +%T) $(hostname) jobs=$(wc -l < "$J")"
(cd "$D" && xargs -P "${WHY_P:-3}" -d '\n' -I CMD bash -c CMD < "$(basename "$J")")
echo "WHY_END $(date +%T)"
