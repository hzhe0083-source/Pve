#!/usr/bin/env bash
# Run once on haha-a: start drv_a6.sh and the retention watchers for its two training directories (the chain's
# Stack bases arrive in runs/src_c later; the driver waits for runs/src_c/OK).
set -u
cd "$HOME/zhr_pve2" || exit 1
pgrep -f "bash drv_a6.sh" > /dev/null && { echo "ABORT: drv_a6.sh already running"; exit 1; }
setsid nohup bash drv_a6.sh > logs/drv_a6.log 2>&1 < /dev/null &
PROT_ROOT=$HOME/zhr_pve2 setsid nohup bash scripts/ret_watch.sh runs/protect_np logs/drv_a6.done \
    > logs/ret_watch_np.log 2>&1 < /dev/null &
PROT_ROOT=$HOME/zhr_pve2 RET_SEL=t90 setsid nohup bash scripts/ret_watch.sh runs/protect_3m logs/drv_a6.done \
    > logs/ret_watch_3m.log 2>&1 < /dev/null &
sleep 2
ps -o pid,etime,args -u "$(id -u)" | grep -E "drv_a6|ret_watch" | grep -v grep | cut -c1-140
