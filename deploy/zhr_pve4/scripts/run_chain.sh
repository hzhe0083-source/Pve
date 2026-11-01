#!/usr/bin/env bash
# A chain of skills in one network: Pick (expert) -> CHAIN_TASKS[0] -> CHAIN_TASKS[1] -> ... (task keys of
# scripts/suite_tasks.sh, at least two). Step 1 is the single-step run of the same seed in CHAIN_SRC
# (scripts/run_protect.sh output with the same code version, default runs/suite/runs); each step k = 2..K continues the
# step before. One job per seed and arm runs its steps in order: a step with final_ckpt.pt is skipped, a step that exits
# non-zero (75: yielded to another user, 137: killed) ends the job and reruns from scratch on the next pass, and a step
# (step 1 included) that finishes without 3 consecutive evals >= 0.9 (it hit the 8M halt) stops that chain for good:
# $D/stopped/<arm>_s<seed>.txt names it and no later step runs. Then, unless CHAIN_EVAL=0, scripts/nsa_eval.py scores
# every skill so far through the deepest trained level: $D/eval/<arm>_s<seed>_L<levels>.json (chain64npr also
# ..._own.json, each skill through its own level). The levels are in the name, so a longer chain gets a new eval.
# Run names: <arm>-<k>-<task>_s<seed>. The first call writes CHAIN_TASKS to $D/chain_tasks.txt; a call with another
# list exits (start a new CHAIN_DIR). One queue per CHAIN_DIR at a time ($D/.queue.lock, as scripts/run_protect.sh).
# Arms, all with the gripper floor (step 1 in CHAIN_SRC in brackets):
#   chain64    [<task>_nsa64]     one more null-space level (width 64) per step; the new head and log-std start as the
#              previous skill's
#   chain64p   [<task>_nsa64]     the same, but every new head and log-std start as Pick's (--nsa_head_from init):
#              step k against the single-step nsa64 run of the same task and seed isolates what reading the earlier
#              adapters' features is worth; the arm whose t90 is read against the free width 256 - 8k
#   chain64npr [<task>_nsa64npr]  the prefix-sum chain: no projection at any level, each new adapter nsa64npr's random
#              draw written once into the null space and trained unprojected; a skill is exact only through its own
#              level (nsa_eval.py --read own); the last level is reported too
#   ftchain    [<task>_monitor]   plain full fine-tuning from the step before: does forgetting pile up?
#   chain64np  [<task>_nsa64np in CHAIN_SRC_NP]  legacy: the prefix sum with a zero-init adapter (init and projection
#              confounded); for the old runs only
# CHAIN_ARMS picks the arms (default chain64 chain64p); CHAIN_ROOT the checkout (default: the one holding this script).
#   CHAIN_TASKS="place stack beside onpair sphere ycb" CHAIN_DIR=runs/suite_chain CHAIN_ARMS="chain64 chain64p" \
#     CHAIN_SEEDS="1 3 5 7" bash scripts/run_chain.sh
set -u
R=${CHAIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
G="bash $R/scripts/with_gpu_mem.sh 6000 bash $R/scripts/gpu_polite.sh"   # yields to other users' GPU jobs
PICK=$R/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
[ -f "$PICK" ] || { echo "MISSING $PICK on $(hostname)"; exit 1; }
SRC=${CHAIN_SRC:-$R/runs/suite/runs}
D=${CHAIN_DIR:-runs/chain}
source "$R/scripts/suite_tasks.sh"      # task_env, task_env_id: each task key's env flags
read -ra TASKS <<< "${CHAIN_TASKS:-stack place}"
K=${#TASKS[@]}
[ "$K" -ge 2 ] || { echo "CHAIN_TASKS needs at least two tasks"; exit 1; }
for t in "${TASKS[@]}"; do [ "$(task_env "$t")" = UNKNOWN ] && { echo "UNKNOWN TASK $t"; exit 1; }; done
mkdir -p "$D/jobs" "$D/stopped" "$D/eval" logs
DA=$(cd "$D" && pwd)
command -v flock > /dev/null || { echo "flock not found"; exit 1; }
exec 7> "$D/.queue.lock"
flock -n 7 || { echo "BUSY: another queue (or a job it started) holds $D/.queue.lock on $(hostname); not started"; exit 1; }
if [ -f "$D/chain_tasks.txt" ]; then
    [ "$(cat "$D/chain_tasks.txt")" = "${TASKS[*]}" ] || {
        echo "$D was started with the chain '$(cat "$D/chain_tasks.txt")', not '${TASKS[*]}': use a new CHAIN_DIR"; exit 1; }
else
    echo "${TASKS[*]}" > "$D/chain_tasks.txt"
fi
J=$D/jobs_$(hostname).txt
: > "$J"
COMMON="--num_envs=1024 --num_steps=50 --eval_freq=2 --num_eval_envs=256 --total_timesteps=25_000_000 \
--halt_at=8_000_000 --bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=3 --target_kl=1e9 \
--min_grip_std=1.0 --init_ckpt=$PICK --init_actor_only --save_every=500000"
# exit 0 iff the eval.csv given has 3 consecutive evals with success_once >= 0.9 (the run learned its skill)
LEARNED="python -c \"import csv,sys;s=[float(r['success_once']) for r in csv.DictReader(open(sys.argv[1]))];\
sys.exit(0 if any(min(s[i:i + 3]) >= 0.9 for i in range(len(s) - 2)) else 1)\""
for seed in ${CHAIN_SEEDS:-1 2 3}; do
    for a in ${CHAIN_ARMS:-chain64 chain64p}; do
        case $a in
            chain64|chain64p) first=$SRC/${TASKS[0]}_nsa64_s$seed ;;
            chain64npr) first=$SRC/${TASKS[0]}_nsa64npr_s$seed ;;
            ftchain) first=$SRC/${TASKS[0]}_monitor_s$seed ;;
            chain64np) first=${CHAIN_SRC_NP:-$SRC}/${TASKS[0]}_nsa64np_s$seed ;;
            *) echo "UNKNOWN ARM $a"; exit 1 ;;
        esac
        [ -f "$first/final_ckpt.pt" ] || { echo "MISSING $first/final_ckpt.pt on $(hostname): ${a}_s$seed skipped"; continue; }
        lv() { if [ "$a" = ftchain ]; then echo 0; else echo "$1"; fi; }    # nsa_eval.py levels (full FT has none)
        js=$DA/jobs/${a}_s$seed.sh
        {
            echo "set -u"
            echo "cd '$DA'"
            echo "stop() { echo \"\$1\" > stopped/${a}_s$seed.txt; echo \"STOPPED ${a}_s$seed: \$1\"; }"
            echo "prev=$first/final_ckpt.pt"
            echo "skills=pick:PickCube-v1:$PICK:0,${TASKS[0]}:$(task_env_id "${TASKS[0]}"):$first/final_ckpt.pt:$(lv 1)"
            echo "while :; do"
            echo "  $LEARNED $first/eval.csv || { stop 'step 1 ${TASKS[0]} ($first) has no 3 evals >= 0.9'; exit 0; }"
            for ((k = 2; k <= K; k++)); do
                t=${TASKS[k-1]}
                e=$a-$k-${t}_s$seed
                case $a in
                    chain64) f="--nsa_base=\$prev --nullspace_adapter=64" ;;
                    chain64p) f="--nsa_base=\$prev --nullspace_adapter=64 --nsa_head_from=init" ;;
                    chain64npr) f="--nsa_base=\$prev --nullspace_adapter=64 --nsa_project=False --nsa_init=rand" ;;
                    ftchain) f="--init_ckpt=\$prev" ;;      # the last --init_ckpt wins
                    chain64np) f="--nsa_base=\$prev --nullspace_adapter=64 --nsa_project=False" ;;
                esac
                echo "  if [ ! -f runs/$e/final_ckpt.pt ]; then"
                echo "    $G python -m wrwl.ppo --exp_name=$e $(task_env "$t") $COMMON --seed=$seed $f > logs_$e.txt 2>&1 || { echo EXIT \$? $e; exit 1; }"
                echo "    echo DONE $e"
                echo "  fi"
                echo "  prev=\$PWD/runs/$e/final_ckpt.pt"
                echo "  skills=\$skills,$t:$(task_env_id "$t"):\$PWD/runs/$e/final_ckpt.pt:$(lv "$k")"
                echo "  $LEARNED runs/$e/eval.csv || { stop 'step $k $t ($e) has no 3 evals >= 0.9'; break; }"
            done
            echo "  break"
            echo "done"
            if [ "${CHAIN_EVAL:-1}" = 1 ]; then
                # skills' levels are their chain positions; the deepest trained level is the chain's length so far
                echo "n=\$(echo \"\$skills\" | tr ',' '\n' | wc -l); L=\$((n - 1))"
                for rd in last $([ "$a" = chain64npr ] && echo own); do
                    sfx=$([ "$rd" = own ] && echo _own)
                    echo "ev=eval/${a}_s${seed}_L\${L}$sfx"
                    echo "if [ ! -f \$ev.json ]; then"
                    echo "  $G python $R/scripts/nsa_eval.py --ckpt=\$prev --skills=\$skills --read=$rd --out=\$ev.json > \$ev.log 2>&1 || { echo EXIT \$? \$ev; exit 1; }"
                    echo "  echo DONE \$ev"
                    echo "fi"
                done
            fi
            echo "echo DONE ${a}_s$seed"
        } > "$js"
        echo "bash $js" >> "$J"
    done
done
echo "CHAIN_START $(date +%T) $(hostname) jobs=$(wc -l < "$J") chain=${TASKS[*]}"
(cd "$D" && xargs -P "${CHAIN_P:-3}" -d '\n' -I CMD bash -c CMD < "$(basename "$J")")
echo "CHAIN_END $(date +%T)"
