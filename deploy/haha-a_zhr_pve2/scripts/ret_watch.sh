#!/usr/bin/env bash
# Evaluate PickCube retention (scripts/retention_eval.py: old head on the new trunk, and the new head) for every run under
# DIR/runs as soon as it finishes, on its 0.5M checkpoints and final checkpoint; stop once END_LOG says PROT_END and all
# finished runs are done. Each eval yields the card to other users (gpu_polite.sh) and is run again after a yield; one
# attempt otherwise (a failed run keeps its .log and no .json; delete the .log to retry). RET_SEL=t90: only the
# checkpoint at t90 (first of 3 evals >= 0.9; for runs saving a checkpoint at every eval) and the final one. END_LOG must
# get PROT_END only once the whole queue is over (a driver's done file, not a run_protect log, which gets it every pass).
#   PROT_ROOT=$HOME/zhr_pve2 nohup bash scripts/ret_watch.sh runs/protect_np logs/drv_c6.done > logs/ret_watch_np.log 2>&1 &
set -u
R=${PROT_ROOT:-$HOME/zhr_pve}
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
D=$1
END=$2
OLD=runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
mkdir -p "$D/ret"
while :; do
    ended=0 retry=0
    grep -q PROT_END "$END" 2>/dev/null && ended=1
    for r in "$D"/runs/*/; do
        n=$(basename "$r")
        [ -f "$r/final_ckpt.pt" ] && grep -q "^done $n" "$D/logs_$n.txt" 2>/dev/null || continue
        [ -f "$D/ret/$n.log" ] && continue
        if [ "${RET_SEL:-all}" = t90 ]; then
            t=$(python -c "import csv,sys;e=list(csv.DictReader(open(sys.argv[1])));s=[float(r['success_once']) for r in e];print(next((e[i]['global_step'] for i in range(len(s)-2) if min(s[i:i+3])>=0.9),''))" "$r/eval.csv")
            ck=$([ -n "$t" ] && [ -f "${r}ckpt_$t.pt" ] && echo "${r}ckpt_$t.pt,")${r}final_ckpt.pt
        else
            ck=$(ls "$r"ckpt_*.pt 2>/dev/null | sort -V | tr '\n' ,)${r}final_ckpt.pt
        fi
        bash scripts/with_gpu_mem.sh 6000 bash scripts/gpu_polite.sh \
            python scripts/retention_eval.py --old=$OLD --ckpts="$ck" --out="$D/ret/$n.json" > "$D/ret/$n.log" 2>&1
        rc=$?
        if [ $rc = 75 ]; then rm -f "$D/ret/$n.log" "$D/ret/$n.json"; retry=1; echo "$(date +%T) $n yielded, retry"; continue; fi
        echo "$(date +%T) $n $([ -f "$D/ret/$n.json" ] && echo ok || echo "FAILED rc=$rc")"
    done
    [ $ended = 1 ] && [ $retry = 0 ] && break
    sleep 60
done
echo "RET_END $(date +%T)"
