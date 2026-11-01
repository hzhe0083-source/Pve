#!/usr/bin/env bash
# One-off on haha-c (01 Nov, ~03:40; replaces drv_c5.sh before it started anything), code in ~/zhr_pve2. After the
# Pick->PlaceCube queue from ~/zhr_pve ends: GPU smoke tests; the prefix-sum ablation (nsa64np: no projection) on Stack
# and PlaceCube, which the novelty check asked for before anything else; the skill chain Pick -> Stack -> PlaceCube
# (with its prefix-sum version) and scripts/nsa_eval.py on every chain run; then FN-PPO and its comparisons on Stack
# to 3M. Every job yields the card to other users (gpu_polite.sh); up to 4 passes per queue.
R=$HOME/zhr_pve2
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
SRC=$HOME/zhr_pve/runs/protect_c/runs
NPS=$R/runs/protect_np/runs
PICK=$R/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
P="bash $R/scripts/gpu_polite.sh"
until [ -f "$HOME/zhr_pve/logs/drv_c4.done" ]; do sleep 60; done
echo "start $(date +%T)"
mkdir -p smoke && cd smoke
SM="--num_envs=256 --num_steps=50 --eval_freq=1000 --num_eval_envs=16 --total_timesteps=38400 --bootstrap_at_done=truncated \
--gamma=0.95 --target_kl=1e9 --min_grip_std=1.0 --init_ckpt=$PICK --init_actor_only --protect_states=$R/data/bc_pick_c1.pt"
STK="--env_id=StackCubeNarrow-v1 --env_kwargs={\"reward_mode\":\"sparse\"} --obs_adapter=stack_to_pick"
PLC="--env_id=PlaceCube-v1 --env_kwargs={\"robot_uids\":\"panda_wristcam\",\"reward_mode\":\"sparse\"}"
$P python -m wrwl.ppo --exp_name=smoke_fn $STK $SM --protect_old=fn --protect_env=PickCube-v1 --protect_dagger=1 \
    > smoke_fn.txt 2>&1; r1=$?
$P python -m wrwl.ppo --exp_name=smoke_chain $PLC $SM --nsa_base=$SRC/stack_nsa64_s1/final_ckpt.pt \
    --nullspace_adapter=64 > smoke_chain.txt 2>&1; r2=$?
$P python -m wrwl.ppo --exp_name=smoke_np $STK $SM --nullspace_adapter=64 --nsa_project=False \
    --protect_env=PickCube-v1 --protect_dagger=1 > smoke_np.txt 2>&1; r3=$?
cd "$R"
echo "smoke fn $r1 chain $r2 np $r3 $(date +%T)"
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
left_np() { for t in stack place; do for s in 1 2 3; do [ -f $NPS/${t}_nsa64np_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
run_np() {
    for t in stack place; do
        PROT_ROOT=$R PROT_TASK=$t PROT_DIR=runs/protect_np PROT_ARMS="nsa64np" PROT_SEEDS="1 2 3" PROT_P=3 \
            bash scripts/run_protect.sh >> logs/run_np_c6.log 2>&1
    done
}
passes np left_np run_np
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
        $P python scripts/nsa_eval.py --ckpt=$c --read=$rd \
            --skills=pick:PickCube-v1:$PICK:0,stack:StackCubeNarrow-v1:$st:$ls,place:PlaceCube-v1:$c:$lp \
            --out=runs/chain/eval_place_${a}_s$s.json > runs/chain/eval_place_${a}_s$s.log 2>&1
        echo "nsa_eval place_${a}_s$s rc=$? $(date +%T)"
    done
done
FA="monitor fncl fnbank fncl_w5 klcl1 klcl3"
left_fn() { for a in $FA; do for s in 1 2 3; do [ -f runs/protect_3m/runs/stack_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
run_fn() {
    PROT_ROOT=$R PROT_DIR=runs/protect_3m PROT_STOP="--halt_at=3_000_000 --save_every=102400" PROT_SEEDS="1 2 3" \
        PROT_P=3 PROT_ARMS="$FA" bash scripts/run_protect.sh >> logs/run_fn_c6.log 2>&1
}
passes fn left_fn run_fn
echo "PROT_END $(date +%T)" > logs/drv_c6.done
