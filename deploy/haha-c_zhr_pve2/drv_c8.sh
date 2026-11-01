#!/usr/bin/env bash
# One-off on haha-c (01 Nov, ~14:10): after drv_c7.sh (FN-PPO block) ends, full fine-tuning (monitor) and head-only on
# PlaceCube, seeds 1-8, with exactly the settings of haha-a's runs/protect_np (same code, md5-checked), so the four arms
# (ours, prefix-sum, fine-tuning, head-only) share one code version and 8 seeds. Rules registered in
# docs/conflict-reuse-2026-10-31.md (14:10). Every job yields the card to other users; up to 4 passes.
R=$HOME/zhr_pve2
cd "$R" || exit 1
rm -f logs/drv_c8.done      # a stale "smoke failed" from an earlier start would stop ret_watch.sh at once
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
until [ -f logs/drv_c7.done ]; do sleep 60; done
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
FH="monitor head"
left_fh() { for a in $FH; do for s in 1 2 3 4 5 6 7 8; do [ -f runs/protect_np/runs/place_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
run_fh() {
    PROT_ROOT=$R PROT_TASK=place PROT_DIR=runs/protect_np PROT_ARMS="$FH" PROT_SEEDS="1 2 3 4 5 6 7 8" PROT_P=3 \
        bash scripts/run_protect.sh >> logs/run_fh_c8.log 2>&1
}
passes fh left_fh run_fh
echo "PROT_END $(date +%T)" > logs/drv_c8.done
