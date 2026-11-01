#!/usr/bin/env bash
# One-off on haha-d (01 Nov, ~16:10): the mechanism queue (scripts/run_why.sh; rules in scripts/why_test.py and
# docs/conflict-reuse-2026-10-31.md, 15:25) from ~/zhr_pve3: the 15:25 patch plus the rule-F masks, CPU-tested. A smoke
# run of every new code path first; then, seeds 1-8 each: A+B (nsa64 npr rp), F (vis none nsa64z), C (rf01 rf1 rf10),
# D (nsa64d nprd). The learning-rate block E runs on haha-c (drv_c9.sh) from identical files. Every job yields the card
# to other users (gpu_polite.sh); smoke and every block get up to 4 passes.
R=$HOME/zhr_pve3
cd "$R" || exit 1
mkdir -p logs
rm -f logs/drv_why_d.done
export PROT_ROOT=$R
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
SMOKE="nsa64 npr rp vis none nsa64z rf1 nsa64d nsa64lo nprhi"
left_smoke() { for a in $SMOKE; do [ -f runs/why/runs/smoke_$a/final_ckpt.pt ] || echo x; done | wc -l; }
run_smoke() { WHY_SMOKE=1 WHY_ARMS="$SMOKE" WHY_P=3 bash scripts/run_why.sh >> logs/why_smoke_d.log 2>&1; }
passes smoke left_smoke run_smoke
if [ "$(left_smoke)" != 0 ]; then echo "PROT_END smoke failed $(date +%T)" > logs/drv_why_d.done; exit 1; fi
left_block() { for a in $ARMS_B; do for s in 1 2 3 4 5 6 7 8; do [ -f runs/why/runs/place_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
run_block() { WHY_ARMS="$ARMS_B" WHY_SEEDS="1 2 3 4 5 6 7 8" WHY_P=3 bash scripts/run_why.sh >> logs/why_$LABEL_B.log 2>&1; }
for spec in "AB:nsa64 npr rp" "F:vis none nsa64z" "C:rf01 rf1 rf10" "D:nsa64d nprd"; do
    LABEL_B=${spec%%:*} ARMS_B=${spec#*:}
    passes "$LABEL_B" left_block run_block
done
echo "PROT_END $(date +%T)" > logs/drv_why_d.done
