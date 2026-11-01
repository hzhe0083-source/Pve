#!/usr/bin/env bash
# One-off on haha-a (01 Nov, ~10:45): the queue of haha-c's drv_c6b.sh, moved here because haha-c has been held by the
# lab's a2r jobs since 03:45 (that host only finishes its Pick->PlaceCube queue). Code in ~/zhr_pve2 (the same files as
# haha-c's; same Pick expert and BC data on both hosts, checked by md5). In order:
#   GPU smoke tests of every new code path; each stage runs only if its own smoke test passed;
#   the projection test, all arms under this one code version and paired by seed: nsa64 (projected), nsa64np (prefix
#     sum, zero-init adapter), nsa64npr (prefix sum with nsa64's own random adapter, so it differs from nsa64 only in
#     the projection), on PlaceCube seeds 1-8 (decides; rule in docs/conflict-reuse-2026-10-31.md) and Stack seeds 1-3;
#   a frozen-adapter control on PlaceCube (nsa64fz);
#   the skill chain Pick -> Stack -> PlaceCube (with its prefix-sum version) and scripts/nsa_eval.py on every chain
#     run; its Stack bases are haha-c's runs/protect_c Stack runs, copied to runs/src_c (OK file after the md5 check);
#   FN-PPO and its comparisons on Stack to 3M.
# Every job yields the card to other users (gpu_polite.sh); a yielded smoke test or nsa_eval is run again, and each
# training queue gets up to 4 passes.
R=$HOME/zhr_pve2
cd "$R" || exit 1
rm -f logs/drv_a6.done      # a stale "smoke failed" from an earlier start would stop ret_watch.sh at once
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
r_fn=$(smoke fn $STK $SM --protect_old=fn --protect_env=PickCube-v1 --protect_dagger=1)
r_nsa=$(smoke nsa $PLC $SM --nullspace_adapter=64 --protect_env=PickCube-v1 --protect_dagger=1)
r_np=$(smoke np $STK $SM --nullspace_adapter=64 --nsa_project=False --protect_env=PickCube-v1 --protect_dagger=1)
r_npr=$(smoke npr $STK $SM --nullspace_adapter=64 --nsa_project=False --nsa_init=rand --protect_env=PickCube-v1 \
    --protect_dagger=1)
r_fz=$(smoke fz $PLC $SM --nullspace_adapter=64 --nsa_freeze_adapter --protect_env=PickCube-v1 --protect_dagger=1)
echo "smoke fn $r_fn nsa $r_nsa np $r_np npr $r_npr fz $r_fz $(date +%T)"
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
NPA=""
[ "$r_nsa" = 0 ] && NPA="nsa64"
[ "$r_np" = 0 ] && NPA="$NPA nsa64np"
[ "$r_npr" = 0 ] && NPA="$NPA nsa64npr"
left_np() {
    for a in $NPA; do
        for s in 1 2 3 4 5 6 7 8; do [ -f $NPS/place_${a}_s$s/final_ckpt.pt ] || echo x; done
        for s in 1 2 3; do [ -f $NPS/stack_${a}_s$s/final_ckpt.pt ] || echo x; done
    done | wc -l
}
run_np() {      # PlaceCube first: it decides
    PROT_ROOT=$R PROT_TASK=place PROT_DIR=runs/protect_np PROT_ARMS="$NPA" PROT_SEEDS="1 2 3 4 5 6 7 8" PROT_P=3 \
        bash scripts/run_protect.sh >> logs/run_np_a6.log 2>&1
    PROT_ROOT=$R PROT_TASK=stack PROT_DIR=runs/protect_np PROT_ARMS="$NPA" PROT_SEEDS="1 2 3" PROT_P=3 \
        bash scripts/run_protect.sh >> logs/run_np_a6.log 2>&1
}
case "$NPA" in
    nsa64\ *) passes np left_np run_np ;;
    *) echo "projection test skipped: arms that passed smoke '$NPA'" ;;
esac
left_fz() { for s in 1 2 3; do [ -f runs/protect_fz/runs/place_nsa64fz_s$s/final_ckpt.pt ] || echo x; done | wc -l; }
run_fz() {      # the adapter frozen at its random start: only the head, log-std and critic learn
    PROT_ROOT=$R PROT_TASK=place PROT_DIR=runs/protect_fz PROT_ARMS="nsa64fz" PROT_SEEDS="1 2 3" PROT_P=3 \
        PROT_STOP="--halt_at=3_000_000 --stop_success=0.9 --stop_patience=3 --save_every=500000" \
        bash scripts/run_protect.sh >> logs/run_fz_a6.log 2>&1
}
[ "$r_fz" = 0 ] && passes fz left_fz run_fz
for i in $(seq 1 120); do [ -f "$SRC/OK" ] && break; sleep 60; done
if [ -f "$SRC/OK" ]; then
    r_chain=$(smoke chain $PLC $SM --nsa_base=$SRC/stack_nsa64_s1/final_ckpt.pt --nullspace_adapter=64)
    echo "smoke chain $r_chain $(date +%T)"
else
    r_chain=nobase
    echo "chain skipped: no $SRC/OK after 2 h $(date +%T)"
fi
CA="chain64 chain64p ftchain chain64np"
left_chain() { for a in $CA; do for s in 1 2 3; do [ -f runs/chain/runs/place_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
run_chain() {
    CHAIN_ROOT=$R CHAIN_SRC=$SRC CHAIN_SRC_NP=$NPS CHAIN_DIR=runs/chain CHAIN_ARMS="$CA" CHAIN_SEEDS="1 2 3" CHAIN_P=3 \
        bash scripts/run_chain.sh >> logs/run_chain_a6.log 2>&1
}
if [ "$r_chain" = 0 ]; then
    passes chain left_chain run_chain
    for s in 1 2 3; do
        for a in $CA; do
            c=runs/chain/runs/place_${a}_s$s/final_ckpt.pt
            [ -f "$c" ] || continue
            rd=last
            case $a in
                ftchain) st=$SRC/stack_monitor_s$s/final_ckpt.pt; ls=0; lp=0 ;;
                chain64np) st=$NPS/stack_nsa64np_s$s/final_ckpt.pt; ls=1; lp=2; rd=own ;;
                *) st=$SRC/stack_nsa64_s$s/final_ckpt.pt; ls=1; lp=2 ;;
            esac
            polite python scripts/nsa_eval.py --ckpt=$c --read=$rd \
                --skills=pick:PickCube-v1:$PICK:0,stack:StackCubeNarrow-v1:$st:$ls,place:PlaceCube-v1:$c:$lp \
                --out=runs/chain/eval_place_${a}_s$s.json > runs/chain/eval_place_${a}_s$s.log 2>&1
            echo "nsa_eval place_${a}_s$s rc=$? $(date +%T)"
        done
    done
fi
FA="monitor fncl fnbank fncl_w5 klcl1 klclp3 klclp1"
left_fn() { for a in $FA; do for s in 1 2 3; do [ -f runs/protect_3m/runs/stack_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
run_fn() {
    PROT_ROOT=$R PROT_DIR=runs/protect_3m PROT_STOP="--halt_at=3_000_000 --save_every=102400" PROT_SEEDS="1 2 3" \
        PROT_P=3 PROT_ARMS="$FA" bash scripts/run_protect.sh >> logs/run_fn_a6.log 2>&1
}
[ "$r_fn" = 0 ] && passes fn left_fn run_fn
echo "PROT_END $(date +%T)" > logs/drv_a6.done
