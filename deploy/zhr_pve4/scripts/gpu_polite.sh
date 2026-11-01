#!/usr/bin/env bash
# Run a command on a shared GPU only while no one else uses it: wait until every compute process on the card is one of
# ours (our trainer and eval scripts), start the command, and stop it (its own PID only) as soon as another user's
# compute process appears; exit 75 then, so the run is left unfinished and a later pass of the queue picks it up.
# Innermost wrapper, so the PID it watches is the command itself:
#   bash scripts/with_gpu_mem.sh 6000 bash scripts/gpu_polite.sh python -m wrwl.ppo ...
OURS='wrwl\.ppo|retention_eval\.py|nsa_eval\.py|forgetting_closedloop\.py|posthoc_projection_eval\.py|smoke_envs\.py'
foreign() {
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        c=$(tr '\0' ' ' 2>/dev/null < "/proc/$p/cmdline")
        [ -n "$c" ] || continue     # no cmdline: the process has exited or is exiting (one of ours tearing down)
        grep -Eq "$OURS" <<< "$c" || return 0
    done
    return 1
}
while foreign; do
    sleep 60
done
"$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
    if foreign; then
        echo "gpu_polite: another user's GPU process appeared at $(date +%T), stopping $pid"
        kill "$pid"
        wait "$pid"
        exit 75
    fi
    sleep 20
done
wait "$pid"
