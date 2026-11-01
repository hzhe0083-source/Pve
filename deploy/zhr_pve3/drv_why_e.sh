#!/usr/bin/env bash
# One-off on 5090d (01 Nov, evening): the whole mechanism queue (scripts/run_why.sh; rules in scripts/why_test.py) on one
# card, from ~/zhr_pve3 (code md5 681f2d3a, as haha-c/d's), seeds 1-8: A+E (nsa64 npr and the adapter learning rates
# nsa64lo nsa64hi nprlo nprhi), E's registered extension when why_test.py asks for it, then F (vis none nsa64z), B (rp),
# C (rf01 rf1 rf10), D (nsa64d nprd). E came off haha-c's queue (drv_c10.sh) and F, C, D off haha-d's (drv_why_d2.sh);
# amendment in docs/conflict-reuse-2026-10-31.md, written before any of these runs. A smoke run of every code path first
# (the head-projected arms' old head must stay within 1e-4); every job yields the card to other users (gpu_polite.sh);
# smoke and every block get up to 4 passes. Ends with logs/drv_why_e.done.
R=${WHY_ROOT:-$HOME/zhr_pve3}
cd "$R" || exit 1
mkdir -p logs
rm -f logs/drv_why_e.done
export PROT_ROOT=$R
echo "start $(date +%T)"
fail() { echo "PROT_END failed: $1 $(date +%T)" | tee logs/drv_why_e.done; exit 1; }
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
SMOKE="nsa64 npr nsa64lo vis none nsa64z rf1 nsa64d rp"
left_smoke() { for a in $SMOKE; do [ -f runs/why/runs/smoke_$a/final_ckpt.pt ] || echo x; done | wc -l; }
run_smoke() { WHY_SMOKE=1 WHY_ARMS="$SMOKE" WHY_P=3 bash scripts/run_why.sh >> logs/why_smoke_e.log 2>&1; }
passes smoke left_smoke run_smoke
[ "$(left_smoke)" = 0 ] || fail "smoke runs unfinished (logs/why_smoke_e.log)"
for a in nsa64 nsa64lo nsa64z nsa64d; do        # projected onto the heads' null space: the old head reads nothing new
    g=$(grep "nullspace adapter: max" runs/why/logs_smoke_$a.txt | grep -oE "[0-9.]+e[-+][0-9]+$" | sort -g | tail -n 1)
    [ -n "$g" ] || fail "no 'nullspace adapter: max' line in runs/why/logs_smoke_$a.txt"
    .venv/bin/python -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 1e-4 else 1)" "$g" \
        || fail "smoke_$a: old head moved by $g (> 1e-4)"
    echo "smoke_$a old head max change $g"
done
echo "smoke passed $(date +%T)"
left_block() { for a in $ARMS_B; do for s in 1 2 3 4 5 6 7 8; do [ -f runs/why/runs/place_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
run_block() { WHY_ARMS="$ARMS_B" WHY_SEEDS="1 2 3 4 5 6 7 8" WHY_P=3 bash scripts/run_why.sh >> logs/why_$LABEL_B.log 2>&1; }
block() { LABEL_B=$1; ARMS_B=$2; passes "$1" left_block run_block; }
block AE "nsa64 npr nsa64lo nsa64hi nprlo nprhi"
for round in 1 2; do    # E's registered extension: the next rate out when an arm's best rate is at an edge
    .venv/bin/python scripts/why_test.py --runs=runs/why/runs > logs/why_test_E$round.txt 2>&1
    ext=$(sed -n 's/^E1: open, the best rate is at an edge: run \(.*\) on seeds 1-8.*/\1/p' logs/why_test_E$round.txt)
    echo "E extension round $round: ${ext:-none} $(date +%T)"
    [ -n "$ext" ] || break
    block Eext$round "$ext"
done
block F "vis none nsa64z"
block B "rp"
block C "rf01 rf1 rf10"
block D "nsa64d nprd"
.venv/bin/python scripts/why_test.py --runs=runs/why/runs > logs/why_test_all.txt 2>&1
echo "PROT_END $(date +%T)" | tee logs/drv_why_e.done
