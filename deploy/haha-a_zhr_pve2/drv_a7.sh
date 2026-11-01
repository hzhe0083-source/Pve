#!/usr/bin/env bash
# One-off on haha-a (01 Nov, ~13:15): replaces drv_a6.sh during its chain stage. A lab job has held this card since
# 12:32, so the FN-PPO block moved to haha-c (drv_c7.sh). This finishes what drv_a6.sh had left here: wait for the
# chain queue it started at 12:00:58 (scripts/run_chain.sh, still running), give the chain up to 4 passes, score every
# chain run with scripts/nsa_eval.py, then write logs/drv_a6.done so the two retention watchers end.
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
echo "start $(date +%T)"
while pgrep -f "bash scripts/run_chain.sh" > /dev/null; do sleep 60; done
echo "chain queue from drv_a6.sh ended $(date +%T)"
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
CA="chain64 chain64p ftchain chain64np"
left_chain() { for a in $CA; do for s in 1 2 3; do [ -f runs/chain/runs/place_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
run_chain() {
    CHAIN_ROOT=$R CHAIN_SRC=$SRC CHAIN_SRC_NP=$NPS CHAIN_DIR=runs/chain CHAIN_ARMS="$CA" CHAIN_SEEDS="1 2 3" CHAIN_P=3 \
        bash scripts/run_chain.sh >> logs/run_chain_a6.log 2>&1
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
echo "PROT_END $(date +%T)" > logs/drv_a6.done
