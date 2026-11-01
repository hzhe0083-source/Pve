#!/usr/bin/env bash
# One-off on haha-c (01 Nov, ~10:40; replaces drv_c6.sh while it was still waiting, before it started anything), code
# in ~/zhr_pve2. After the Pick->PlaceCube queue from ~/zhr_pve ends: GPU smoke tests; the projection test, all arms
# under this one code version and paired by seed: nsa64 (projected), nsa64np (prefix sum, zero-init adapter) and
# nsa64npr (prefix sum with nsa64's own random adapter, so it differs from nsa64 only in the projection), on PlaceCube
# seeds 1-8 (decides, rule in docs/conflict-reuse-2026-10-31.md) and Stack seeds 1-3; a frozen-adapter control on
# PlaceCube; the skill chain Pick -> Stack -> PlaceCube (with its prefix-sum version) and scripts/nsa_eval.py on every
# chain run; then FN-PPO and its comparisons on Stack to 3M. Every job yields the card to other users (gpu_polite.sh);
# a yielded smoke test or nsa_eval is run again, and each training queue gets up to 4 passes.
R=$HOME/zhr_pve2
cd "$R" || exit 1
rm -f logs/drv_c6.done      # a stale "smoke failed" from an earlier start would stop ret_watch.sh at once
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
SRC=$HOME/zhr_pve/runs/protect_c/runs
NPS=$R/runs/protect_np/runs
PICK=$R/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
P="bash $R/scripts/gpu_polite.sh"
polite() {      # gpu_polite.sh exits 75 when it yields the card to another user: wait, then run the command again
    local rc
    until $P "$@"; rc=$?; [ $rc != 75 ]; do echo "yielded $(date +%T), retry in 300 s"; sleep 300; done
    return $rc
}
until [ -f "$HOME/zhr_pve/logs/drv_c4.done" ]; do sleep 60; done
echo "start $(date +%T)"
mkdir -p smoke && cd smoke
SM="--num_envs=256 --num_steps=50 --eval_freq=1000 --num_eval_envs=16 --total_timesteps=38400 --bootstrap_at_done=truncated \
--gamma=0.95 --target_kl=1e9 --min_grip_std=1.0 --init_ckpt=$PICK --init_actor_only --protect_states=$R/data/bc_pick_c1.pt"
STK="--env_id=StackCubeNarrow-v1 --env_kwargs={\"reward_mode\":\"sparse\"} --obs_adapter=stack_to_pick"
PLC="--env_id=PlaceCube-v1 --env_kwargs={\"robot_uids\":\"panda_wristcam\",\"reward_mode\":\"sparse\"}"
polite python -m wrwl.ppo --exp_name=smoke_fn $STK $SM --protect_old=fn --protect_env=PickCube-v1 --protect_dagger=1 \
    > smoke_fn.txt 2>&1; r1=$?
polite python -m wrwl.ppo --exp_name=smoke_chain $PLC $SM --nsa_base=$SRC/stack_nsa64_s1/final_ckpt.pt \
    --nullspace_adapter=64 > smoke_chain.txt 2>&1; r2=$?
polite python -m wrwl.ppo --exp_name=smoke_np $STK $SM --nullspace_adapter=64 --nsa_project=False \
    --protect_env=PickCube-v1 --protect_dagger=1 > smoke_np.txt 2>&1; r3=$?
# the two new arms do not gate the queue: an arm whose smoke test fails is left out
polite python -m wrwl.ppo --exp_name=smoke_npr $STK $SM --nullspace_adapter=64 --nsa_project=False --nsa_init=rand \
    --protect_env=PickCube-v1 --protect_dagger=1 > smoke_npr.txt 2>&1; r4=$?
polite python -m wrwl.ppo --exp_name=smoke_fz $PLC $SM --nullspace_adapter=64 --nsa_freeze_adapter \
    --protect_env=PickCube-v1 --protect_dagger=1 > smoke_fz.txt 2>&1; r5=$?
cd "$R"
echo "smoke fn $r1 chain $r2 np $r3 npr $r4 fz $r5 $(date +%T)"
if [ $r1 != 0 ] || [ $r2 != 0 ] || [ $r3 != 0 ]; then echo "PROT_END smoke failed $(date +%T)" > logs/drv_c6.done; exit 1; fi
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
NPA="nsa64 nsa64np"
[ $r4 = 0 ] && NPA="$NPA nsa64npr"
left_np() {
    for a in $NPA; do
        for s in 1 2 3 4 5 6 7 8; do [ -f $NPS/place_${a}_s$s/final_ckpt.pt ] || echo x; done
        for s in 1 2 3; do [ -f $NPS/stack_${a}_s$s/final_ckpt.pt ] || echo x; done
    done | wc -l
}
run_np() {      # PlaceCube first: it decides
    PROT_ROOT=$R PROT_TASK=place PROT_DIR=runs/protect_np PROT_ARMS="$NPA" PROT_SEEDS="1 2 3 4 5 6 7 8" PROT_P=3 \
        bash scripts/run_protect.sh >> logs/run_np_c6.log 2>&1
    PROT_ROOT=$R PROT_TASK=stack PROT_DIR=runs/protect_np PROT_ARMS="$NPA" PROT_SEEDS="1 2 3" PROT_P=3 \
        bash scripts/run_protect.sh >> logs/run_np_c6.log 2>&1
}
passes np left_np run_np
left_fz() { for s in 1 2 3; do [ -f runs/protect_fz/runs/place_nsa64fz_s$s/final_ckpt.pt ] || echo x; done | wc -l; }
run_fz() {      # the adapter frozen at its random start: only the head, log-std and critic learn
    PROT_ROOT=$R PROT_TASK=place PROT_DIR=runs/protect_fz PROT_ARMS="nsa64fz" PROT_SEEDS="1 2 3" PROT_P=3 \
        PROT_STOP="--halt_at=3_000_000 --stop_success=0.9 --stop_patience=3 --save_every=500000" \
        bash scripts/run_protect.sh >> logs/run_fz_c6.log 2>&1
}
[ $r5 = 0 ] && passes fz left_fz run_fz
CA="chain64 chain64p ftchain chain64np"
left_chain() { for a in $CA; do for s in 1 2 3; do [ -f runs/chain/runs/place_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
run_chain() {
    CHAIN_ROOT=$R CHAIN_SRC=$SRC CHAIN_SRC_NP=$NPS CHAIN_DIR=runs/chain CHAIN_ARMS="$CA" CHAIN_SEEDS="1 2 3" CHAIN_P=3 \
        bash scripts/run_chain.sh >> logs/run_chain_c6.log 2>&1
}
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
FA="monitor fncl fnbank fncl_w5 klcl1 klclp3 klclp1"
left_fn() { for a in $FA; do for s in 1 2 3; do [ -f runs/protect_3m/runs/stack_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
run_fn() {
    PROT_ROOT=$R PROT_DIR=runs/protect_3m PROT_STOP="--halt_at=3_000_000 --save_every=102400" PROT_SEEDS="1 2 3" \
        PROT_P=3 PROT_ARMS="$FA" bash scripts/run_protect.sh >> logs/run_fn_c6.log 2>&1
}
passes fn left_fn run_fn
echo "PROT_END $(date +%T)" > logs/drv_c6.done
