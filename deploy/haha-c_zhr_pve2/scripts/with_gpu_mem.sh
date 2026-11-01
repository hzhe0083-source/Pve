#!/usr/bin/env bash
# Run a command once the GPU has more than $1 MB free and the host has more than $ZHR_RAM_GB (default 9) GB of
# available RAM (a PlaceCube job holds about 7.7 GB; on 2026-10-31 haha-d ran out of RAM and jobs were killed). The
# check and the command's first 90 s happen under a lock, so jobs released together do not all see the same free memory.
#   bash scripts/with_gpu_mem.sh 6000 python -m wrwl.ppo ...
need=$1
shift
ram_kb=$(( ${ZHR_RAM_GB:-9} * 1024 * 1024 ))
exec 9>/tmp/zhr_gpu_mem.lock
flock 9
until [ "$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}')" -gt "$need" ] &&
      [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -gt "$ram_kb" ]; do
    sleep 60
done
"$@" 9>&- &
pid=$!
sleep 90
flock -u 9
wait $pid
