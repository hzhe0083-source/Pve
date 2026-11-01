#!/usr/bin/env bash
# One-off on haha-c (01 Nov, ~16:10): after drv_c8.sh ends, (1) the mechanism queue's learning-rate block E
# (nsa64lo nsa64hi nprlo nprhi, seeds 1-8; rules in scripts/why_test.py) from ~/zhr_pve3, the same files as haha-d's,
# with a smoke run of its arms first; (2) FN-PPO's PlaceCube replication (rule registered 15:10 for haha-d; moved here
# because haha-d lacks data/bc_pick_c1.pt) from ~/zhr_pve2. Every job yields the card to other users; up to 4 passes.
until [ -f "$HOME/zhr_pve2/logs/drv_c8.done" ]; do sleep 60; done
echo "start $(date +%T)"
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
cd "$HOME/zhr_pve3" || exit 1
mkdir -p logs
SMOKE="nsa64lo nsa64hi nprlo nprhi"
left_smoke() { for a in $SMOKE; do [ -f runs/why/runs/smoke_$a/final_ckpt.pt ] || echo x; done | wc -l; }
run_smoke() { PROT_ROOT=$HOME/zhr_pve3 WHY_SMOKE=1 WHY_ARMS="$SMOKE" WHY_P=3 bash scripts/run_why.sh >> logs/why_smoke_c.log 2>&1; }
passes smoke left_smoke run_smoke
if [ "$(left_smoke)" = 0 ]; then
    left_e() { for a in $SMOKE; do for s in 1 2 3 4 5 6 7 8; do [ -f runs/why/runs/place_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
    run_e() { PROT_ROOT=$HOME/zhr_pve3 WHY_ARMS="$SMOKE" WHY_SEEDS="1 2 3 4 5 6 7 8" WHY_P=3 bash scripts/run_why.sh >> logs/why_E.log 2>&1; }
    passes E left_e run_e
else
    echo "E skipped: smoke failed $(date +%T)"
fi
R=$HOME/zhr_pve2
cd "$R" || exit 1
rm -f logs/drv_c9.done      # a stale "smoke failed" from an earlier start would stop ret_watch.sh at once
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
echo "FN place start $(date +%T)"
mkdir -p smoke
SM="--num_envs=256 --num_steps=50 --eval_freq=1000 --num_eval_envs=16 --total_timesteps=38400 --bootstrap_at_done=truncated \
--gamma=0.95 --target_kl=1e9 --min_grip_std=1.0 --init_ckpt=$PICK --init_actor_only --protect_states=$R/data/bc_pick_c1.pt"
STK="--env_id=StackCubeNarrow-v1 --env_kwargs={\"reward_mode\":\"sparse\"} --obs_adapter=stack_to_pick"
PLC="--env_id=PlaceCube-v1 --env_kwargs={\"robot_uids\":\"panda_wristcam\",\"reward_mode\":\"sparse\"}"
smoke() {       # $1 = name, rest = trainer flags; prints the exit code
    (cd smoke && polite python -m wrwl.ppo --exp_name=smoke_$1 "${@:2}" > smoke_$1.txt 2>&1; echo $?)
}
r_fn=$(smoke fn_c9 $PLC $SM --protect_old=fn --protect_env=PickCube-v1 --protect_dagger=1)
echo "smoke fn $r_fn $(date +%T)"
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
FA="monitor fncl fnbank klclp1"
left_fn() { for a in $FA; do for s in 1 2 3; do [ -f runs/protect_3m_place/runs/place_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
run_fn() {
    PROT_ROOT=$R PROT_TASK=place PROT_DIR=runs/protect_3m_place PROT_STOP="--halt_at=3_000_000 --save_every=102400" \
        PROT_SEEDS="1 2 3" PROT_P=3 PROT_ARMS="$FA" bash scripts/run_protect.sh >> logs/run_fn_c9.log 2>&1
}
[ "$r_fn" = 0 ] && passes fn left_fn run_fn
echo "PROT_END $(date +%T)" > logs/drv_c9.done
