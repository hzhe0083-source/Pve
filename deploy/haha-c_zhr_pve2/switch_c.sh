#!/usr/bin/env bash
# Run once on haha-c: the projection test, chain and FN queue moved to haha-a (drv_a6.sh), so stop this host's copy of
# it (still waiting for the PlaceCube queue, nothing started) and its two idle watchers. Our processes only, each
# checked by command line. haha-c keeps its PlaceCube queue (drv_c4.sh) and the protect_c retention watcher.
set -u
grep -q "^start" "$HOME/zhr_pve2/logs/drv_c6b.log" 2>/dev/null && { echo "ABORT: drv_c6b.sh has started"; exit 1; }
stop() {    # $1 = PID, $2 = pattern its command line must contain
    local c
    c=$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null)
    case "$c" in
        *"$2"*) kill "$1" && echo "stopped $1: ${c:0:100}" ;;
        *) echo "SKIP $1 (not ours or gone): ${c:0:100}" ;;
    esac
}
stop 1473792 "bash drv_c6b.sh"
stop 1473795 "scripts/ret_watch.sh runs/protect_3m"
stop 1473796 "scripts/ret_watch.sh runs/protect_np"
echo "moved to haha-a (drv_a6.sh) $(date +%T)" >> "$HOME/zhr_pve2/logs/drv_c6b.log"
