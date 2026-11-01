#!/usr/bin/env bash
# One-off on haha-c (01 Nov, ~13:15): the FN-PPO block of haha-a's drv_a6.sh, moved here because a lab job has held
# haha-a's card since 12:32 while this card is idle. Same code as haha-a (md5-checked). On haha-a drv_a7.sh replaces
# drv_a6.sh and only finishes the chain and its nsa_eval, so nothing runs twice. Every job yields the card to other
# users (gpu_polite.sh); a yielded smoke test is run again, and the queue gets up to 4 passes.
R=$HOME/zhr_pve2
cd "$R" || exit 1
rm -f logs/drv_c7.done      # a stale "smoke failed" from an earlier start would stop ret_watch.sh at once
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
echo "start $(date +%T)"
mkdir -p smoke
SM="--num_envs=256 --num_steps=50 --eval_freq=1000 --num_eval_envs=16 --total_timesteps=38400 --bootstrap_at_done=truncated \
--gamma=0.95 --target_kl=1e9 --min_grip_std=1.0 --init_ckpt=$PICK --init_actor_only --protect_states=$R/data/bc_pick_c1.pt"
STK="--env_id=StackCubeNarrow-v1 --env_kwargs={\"reward_mode\":\"sparse\"} --obs_adapter=stack_to_pick"
PLC="--env_id=PlaceCube-v1 --env_kwargs={\"robot_uids\":\"panda_wristcam\",\"reward_mode\":\"sparse\"}"
smoke() {       # $1 = name, rest = trainer flags; prints the exit code
    (cd smoke && polite python -m wrwl.ppo --exp_name=smoke_$1 "${@:2}" > smoke_$1.txt 2>&1; echo $?)
}
r_fn=$(smoke fn_c7 $STK $SM --protect_old=fn --protect_env=PickCube-v1 --protect_dagger=1)
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
FA="monitor fncl fnbank fncl_w5 klcl1 klclp3 klclp1"
left_fn() { for a in $FA; do for s in 1 2 3; do [ -f runs/protect_3m/runs/stack_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
run_fn() {
    PROT_ROOT=$R PROT_DIR=runs/protect_3m PROT_STOP="--halt_at=3_000_000 --save_every=102400" PROT_SEEDS="1 2 3" \
        PROT_P=3 PROT_ARMS="$FA" bash scripts/run_protect.sh >> logs/run_fn_c7.log 2>&1
}
[ "$r_fn" = 0 ] && passes fn left_fn run_fn
echo "PROT_END $(date +%T)" > logs/drv_c7.done
