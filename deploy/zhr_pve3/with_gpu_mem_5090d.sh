#!/usr/bin/env bash
# Run a command once the GPU has more than $1 MB free and the host has more than $ZHR_RAM_GB (default 9) GB of
# available RAM (a PlaceCube job holds about 7.7 GB; on 2026-10-31 haha-d ran out of RAM and jobs were killed). The
# check and the command's first $ZHR_GPU_HOLD s happen under a lock, so jobs released together do not all see the same
# free memory. The hold was 90 s; on 5090d (01 Nov, 18:57) a PlaceCube job reaches its full 5.8 GB within 28 s and
# finishes in about 60 s, so 90 s ran the jobs one at a time with the card idle in between: 30 s now.
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
sleep "${ZHR_GPU_HOLD:-30}"
flock -u 9
wait $pid
