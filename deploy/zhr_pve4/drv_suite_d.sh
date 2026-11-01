#!/usr/bin/env bash
# One-off on haha-d (01 Nov, ~17:50): the task suite's GPU smoke test, the protocol freeze and gates Z and L, from
# ~/zhr_pve4 (rules registered in docs/conflict-reuse-2026-10-31.md, 17:41), once the mechanism queue (drv_why_d.sh)
# has ended. It stops before the gate is written: suite.json, R0 and the confirmatory runs start by hand after a check.
# The protocol is frozen only if every smoke check passes; otherwise logs/suite_d.done says what failed.
R=$HOME/zhr_pve4
cd "$R" || exit 1
mkdir -p runs/suite_smoke logs
rm -f logs/suite_d.done
until [ -f "$HOME/zhr_pve3/logs/drv_why_d.done" ]; do sleep 60; done
echo "start $(date +%T)"
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
G="bash scripts/with_gpu_mem.sh 6000 bash scripts/gpu_polite.sh"
PICK=runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
retry() {       # run a command again while gpu_polite.sh yields the card (exit 75)
    local rc
    until "$@"; rc=$?; [ $rc != 75 ]; do echo "yielded $(date +%T), retry in 300 s"; sleep 300; done
    return $rc
}
fail() { echo "SUITE_SMOKE_FAILED $1 $(date +%T)" | tee logs/suite_d.done; exit 1; }
retry $G python scripts/smoke_envs.py --pick=$PICK --time_eval --out=runs/suite_smoke/smoke4.json \
    > logs/suite_smoke4.log 2>&1 || fail "smoke_envs --time_eval (logs/suite_smoke4.log)"
retry $G python scripts/smoke_envs.py --pick=$PICK --envs=256 --out=runs/suite_smoke/smoke256.json \
    > logs/suite_smoke256.log 2>&1 || fail "smoke_envs --envs=256 (logs/suite_smoke256.log)"
echo "smoke_envs done $(date +%T)"
TASKS="ycb pushng pullng liftpeg sphere beside onpair" ARMS="nsa64 nsa64npr monitor head"
want=$(( $(echo $TASKS | wc -w) * $(echo $ARMS | wc -w) ))
for pass in 1 2 3 4; do
    PROT_TASKS="$TASKS" PROT_ARMS="$ARMS" PROT_SEEDS=0 PROT_DIR=runs/suite_smoke PROT_STOP="--halt_at=102400" \
        bash scripts/run_protect.sh >> logs/suite_smoke_runs.log 2>&1
    got=$(ls runs/suite_smoke/runs/*/final_ckpt.pt 2>/dev/null | wc -l)
    echo "smoke runs pass $pass $(date +%T): $got of $want finished"
    [ "$got" -ge "$want" ] && break
    grep -E "^EXIT [0-9]+ " logs/suite_smoke_runs.log | grep -qv "^EXIT 75 " && fail "a smoke run crashed (logs/suite_smoke_runs.log)"
    sleep 300
done
[ "$got" -ge "$want" ] || fail "smoke runs unfinished after 4 passes"
[ -z "$(grep -L '^done ' runs/suite_smoke/logs_*.txt)" ] || fail "a smoke log has no done line"
gaps=$(grep -h "nullspace adapter: max" runs/suite_smoke/logs_*nsa64_s0.txt | grep -oE "[0-9.]+e[-+][0-9]+|[0-9]+\.[0-9]+" )
[ -n "$gaps" ] || fail "no 'nullspace adapter: max' line in the nsa64 smoke logs"
python -c "import sys; v = [float(x) for x in sys.argv[1:]]; sys.exit(0 if max(v) <= 1e-4 else 1)" $gaps \
    || fail "nsa64 old-head drift above 1e-4: $gaps"
echo "smoke checks passed $(date +%T): nsa64 drift $(echo $gaps | tr '\n' ' ')"
python scripts/suite_gate.py --freeze=runs/suite/protocol.json > logs/suite_freeze.log 2>&1 \
    || fail "freeze (logs/suite_freeze.log)"
echo "protocol frozen $(date +%T)"
bash scripts/suite_phase.sh zeroshot >> logs/suite_zeroshot.log 2>&1
for pass in 1 2 3 4; do       # pilots skip finished runs, so a pass after a yield reruns only the unfinished ones
    bash scripts/suite_phase.sh pilot >> logs/suite_pilot.log 2>&1
    bash scripts/suite_phase.sh gate > logs/suite_gate.log 2>&1
    echo "pilot pass $pass $(date +%T)"
    grep -q PENDING logs/suite_gate.log || break
    sleep 300
done
echo "PROT_END gates run $(date +%T); verdicts in logs/suite_gate.log, nothing written" | tee logs/suite_d.done
