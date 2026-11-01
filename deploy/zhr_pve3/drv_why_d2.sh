#!/usr/bin/env bash
# One-off on haha-d (01 Nov, evening), replacing drv_why_d.sh: the mechanism queue's later blocks (F, C, D) moved to
# 5090d with the rest of that queue (drv_why_e.sh there, every arm on one card), so this host only finishes block AB
# (nsa64 npr rp, seeds 1-8; kept as the 4090 replication of A and B) and then writes logs/drv_why_d.done, which
# drv_suite_d.sh waits for. $1 = the PID of the AB pass drv_why_d.sh had started (left running when it was stopped).
R=$HOME/zhr_pve3
cd "$R" || exit 1
export PROT_ROOT=$R
echo "start $(date +%T); waiting for the running AB pass, PID $1"
while grep -q run_why "/proc/$1/cmdline" 2>/dev/null; do sleep 60; done
left_ab() { for a in nsa64 npr rp; do for s in 1 2 3 4 5 6 7 8; do [ -f runs/why/runs/place_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l; }
for pass in 2 3 4; do       # pass 1 is the one drv_why_d.sh started; gpu_polite.sh makes every job wait for the card
    left=$(left_ab)
    echo "AB before pass $pass $(date +%T): unfinished $left"
    [ "$left" = 0 ] && break
    WHY_ARMS="nsa64 npr rp" WHY_SEEDS="1 2 3 4 5 6 7 8" WHY_P=3 bash scripts/run_why.sh >> logs/why_AB.log 2>&1
done
echo "PROT_END $(date +%T) AB unfinished $(left_ab)" > logs/drv_why_d.done
