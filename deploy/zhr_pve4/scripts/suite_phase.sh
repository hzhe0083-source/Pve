#!/usr/bin/env bash
# One phase of the task suite on this host, with the registered settings (scripts/suite_gate.py, scripts/suite_test.py
# hold the rules). Run from the suite's checkout. r0, confirm and chain need SEEDS, this host's half of seeds 1-8
# ("1 3 5 7" on haha-c, "2 4 6 8" on haha-d: every arm of a seed on one host, so the pairs share a host).
#   zeroshot  gate Z: the Pick expert on every candidate, runs/suite_zs/<task>.json (scripts/nsa_eval.py, 256 envs)
#   pilot     gate L: seed 0, nsa64 and monitor, every candidate that passed Z, runs/suite_pilot (risky ones first)
#   gate      the gates' verdicts; with WRITE=1 freezes runs/suite/suite.json (UNAVAILABLE=task:reason,... drops)
#   r0        place nsa64 for SEEDS in runs/suite (rule R0, checked before anything else is read)
#   confirm   every task of suite.json (counted, near, dev) x nsa64 nsa64npr monitor head x SEEDS, runs/suite
#             (ARMS="nsa64 nsa64npr monitor head fncl" adds the optional FN-PPO arm, reported with suite_test.py --fncl)
#   chain     suite.json's chain, runs/suite_chain: chain64 chain64p for SEEDS, chain64npr ftchain for SEEDS <= 4
#   ret       retention of every finished runs/suite run (scripts/ret_watch.sh, RET_SEL=t90), until logs/suite.done
#             says PROT_END (the confirm phase writes it when its queue is over)
#   bash scripts/suite_phase.sh zeroshot
#   SEEDS="1 3 5 7" bash scripts/suite_phase.sh confirm
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
cd "$R" || exit 1
export PROT_ROOT=$R CHAIN_ROOT=$R
PICK=$R/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
need_seeds() { [ -n "${SEEDS:-}" ] || { echo "set SEEDS (\"1 3 5 7\" on haha-c, \"2 4 6 8\" on haha-d)"; exit 1; }; }
cands() { python -c "import sys; sys.path.insert(0, 'scripts'); import suite_gate as g; print(' '.join(g.COUNTED + g.NEAR))"; }
suite() { python -c "import json; s = json.load(open('runs/suite/suite.json')); print(' '.join($1))"; }
source "$R/scripts/suite_tasks.sh"
case ${1:-} in
    zeroshot)
        source .venv/bin/activate
        export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
        mkdir -p runs/suite_zs
        for t in $(cands); do
            [ -f "runs/suite_zs/$t.json" ] && continue
            until bash scripts/with_gpu_mem.sh 6000 bash scripts/gpu_polite.sh python scripts/nsa_eval.py --ckpt="$PICK" \
                    --skills="$t:$(task_env_id "$t"):$PICK:0" --out="runs/suite_zs/$t.json" > "runs/suite_zs/$t.log" 2>&1; do
                rc=$?; [ $rc = 75 ] || { echo "zero-shot $t failed (exit $rc): runs/suite_zs/$t.log"; break; }
                sleep 300      # yielded to another user: try again
            done
            [ -f "runs/suite_zs/$t.json" ] && echo "zero-shot $t done"
        done ;;
    pilot)
        source .venv/bin/activate
        pass=$(python - "$(cands)" <<'EOF'
import json, sys
from pathlib import Path
out = []
for t in sys.argv[1].split():
    f = Path("runs/suite_zs") / f"{t}.json"
    if f.exists() and json.load(open(f))["skills"][t]["success"] <= 0.30:
        out.append(t)
print(" ".join(out))
EOF
)
        [ -n "$pass" ] || { echo "no candidate has passed gate Z yet (run the zeroshot phase)"; exit 1; }
        echo "pilots for: $pass"
        PROT_TASKS="$pass" PROT_SEEDS=0 PROT_ARMS="nsa64 monitor" PROT_DIR=runs/suite_pilot bash scripts/run_protect.sh ;;
    gate)
        source .venv/bin/activate
        python scripts/suite_gate.py --zeroshot=runs/suite_zs --pilot=runs/suite_pilot/runs \
            ${UNAVAILABLE:+--unavailable=$UNAVAILABLE} $([ "${WRITE:-0}" = 1 ] && echo --write=runs/suite/suite.json) ;;
    r0)
        need_seeds
        PROT_TASKS=place PROT_ARMS=nsa64 PROT_SEEDS="$SEEDS" PROT_DIR=runs/suite bash scripts/run_protect.sh ;;
    confirm)
        need_seeds
        [ -f runs/suite/suite.json ] || { echo "runs/suite/suite.json missing: copy it from the host that ran the gate"; exit 1; }
        source .venv/bin/activate
        mkdir -p logs
        PROT_TASKS="$(suite 's["counted"] + s["near"] + s["dev"]')" PROT_ARMS="${ARMS:-nsa64 nsa64npr monitor head}" \
            PROT_SEEDS="$SEEDS" PROT_DIR=runs/suite bash scripts/run_protect.sh 2>&1 | tee -a logs/suite.done ;;
    chain)
        need_seeds
        [ -f runs/suite/suite.json ] || { echo "runs/suite/suite.json missing"; exit 1; }
        source .venv/bin/activate
        tasks=$(suite 's["chain"]')
        CHAIN_TASKS="$tasks" CHAIN_SRC=$R/runs/suite/runs CHAIN_DIR=runs/suite_chain CHAIN_ARMS="chain64 chain64p" \
            CHAIN_SEEDS="$SEEDS" bash scripts/run_chain.sh
        low=$(for s in $SEEDS; do [ "$s" -le 4 ] && echo -n "$s "; done)
        [ -n "$low" ] && CHAIN_TASKS="$tasks" CHAIN_SRC=$R/runs/suite/runs CHAIN_DIR=runs/suite_chain \
            CHAIN_ARMS="chain64npr ftchain" CHAIN_SEEDS="$low" bash scripts/run_chain.sh ;;
    ret)
        RET_SEL=t90 bash scripts/ret_watch.sh runs/suite logs/suite.done ;;
    *)
        echo "usage: bash scripts/suite_phase.sh zeroshot|pilot|gate|r0|confirm|chain|ret"; exit 1 ;;
esac
