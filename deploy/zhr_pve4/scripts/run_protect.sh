#!/usr/bin/env bash
# Add a skill to the same network without breaking the old one: Stack learned from the PickCube expert with the
# shared hidden layers trained under --protect_old (each skill keeps its own output layer; the old one stays frozen).
# Arms, all full fine-tuning with the gripper floor 1.0 (the fastest unprotected recipe, which forgets 14-23 points):
#   monitor        no projection; logs the old skill's drift and the share a block projection would remove
#   block09/095/099  remove P_out dW P_in from every step at subspace energy 0.9 / 0.95 / 0.99
#   block095r      as block095, re-estimating the subspaces every 10 iterations
#   gpm095/099     remove every input direction the old skill occupies (GPM-style)
#   *g             (block09g, block095g, block099g, gpm095g) also project the gradient before Adam (--protect_grad)
# Baselines on the same old-skill states, nothing projected, a penalty added to the loss (--protect_coef swept):
#   distill001/01/1  the old skill's mean-action drift (LwF-style)   ewc10/100/1000  diagonal Fisher   l2_1/10/100  L2-SP
#   distill1d      distill 1 with half of each batch from states the old skill reaches now (rolled out every 5 iterations)
#   klcl1/klcl3    distill at coef 1 / 3 on FN's anchors (--protect_strat): every penalty on a fresh draw made as fncl
#                  draws its anchors (the same pool, with its rollout before training, the same strata and size);
#                  distill1d is the same penalty on 4096 uniform states, so distill1d vs klcl1 isolates the sampling
# FN-PPO (--protect_old fn): every gradient and every Adam step of the hidden layers loses its exact first-order effect
#   on the old skill's actions at 512 anchor states, redrawn and refactored every iteration
#   fncl           anchors half from the old skill's rollouts through the current hidden layers (every 5 iterations and
#                  once before training), half from the stored states
#   fnbank         anchors from the stored states only (--fn_pool_frac 0; the rollouts still run for old_eval.csv and
#                  fn.csv's drift_pool): is closed-loop anchoring needed?
#   fncl_w5        fncl after 5 iterations with the hidden layers fixed, head and log-std at LR 1e-3 (the speed rescue)
# nsa<H> (nsa16, nsa64, nsa256): the old hidden layers frozen, a width-H adapter writes only where the old head cannot
#   read (kept exactly, wrwl/nsa.py)
# nsa<H>np: the prefix-sum ablation of nsa<H> (no projection; the old skill reads its own level; zero-init adapter).
# nsa<H>npr: the prefix sum started from nsa<H>'s own random adapter (same initial network and random stream), so it
#   differs from nsa<H> only in the projection.   nsa<H>z: nsa<H> with a zero-init adapter.
# nsa<H>fz: nsa<H> with the adapter frozen at its random start (only the head, log-std and critic learn).
# klclp1/klclp3: closed-loop KL (klcl) at coefficient 0.1 / 0.3.   klcl1h: klcl1 with fncl_w5's head LR 1e-3.
# head: only the output layer trains (--freeze_trunk).
# PROT_TASKS (or PROT_TASK, the same) picks the new skills, task keys of scripts/suite_tasks.sh (default stack): place
#   (PlaceCube-v1, Pick's own layout, release on the table), stack, or a task of the suite: sphere, ycb, pushng,
#   pullng, beside, onpair, liftpeg. Runs are named <task>_<arm>_s<seed>, queued seed-major (every task and arm of a
#   seed before the next seed), so an interrupted queue leaves complete seeds. Every job goes through
#   scripts/gpu_polite.sh, which yields the card to other users. One queue per PROT_DIR at a time: a second call while
#   $PROT_DIR/.queue.lock is held (by the queue or any job it started) exits without touching the job file.
#   PROT_TASKS="ycb pushng sphere" PROT_DIR=runs/suite PROT_ARMS="nsa64 nsa64npr monitor head" PROT_SEEDS="1 3 5 7" \
#     bash scripts/run_protect.sh
# PROT_ROOT is the checkout the jobs run in (default: the one holding this script).
# PROT_ARMS picks the arms (default: the projection arms); runs that already have final_ckpt.pt are skipped.
# PROT_STOP replaces the stop / halt / checkpoint flags (unset or empty: stop after 3 evals >= 0.9, halt at 8M, and
# checkpoints every 0.5M steps, which give retention curves with scripts/retention_eval.py and the old head).
#   PROT_SEEDS="1 2 3" PROT_P=3 bash scripts/run_protect.sh
#   PROT_ARMS="block09g distill01" PROT_SEEDS="1 2 3" bash scripts/run_protect.sh
# The FN experiment runs every arm to 3M without early stop, with a checkpoint at every eval (102,400 steps = eval_freq
# x batch: ckpt_<t90>.pt holds the weights eval.csv scored at t90, ckpt_<t90 + 204800>.pt those an early-stopped run
# ends with, final_ckpt.pt those at 3,020,800 steps), in its own directory (runs/protect holds early-stopped runs that
# already have final_ckpt.pt, and its job list belongs to the other queue):
#   PROT_DIR=runs/protect_3m PROT_STOP="--halt_at=3_000_000 --save_every=102400" PROT_SEEDS="1 2 3" PROT_P=3 \
#     PROT_ARMS="monitor fncl fnbank fncl_w5 klcl1 klclp3 klclp1" bash scripts/run_protect.sh
# Each job line ends with DONE <run> or EXIT <code> <run> (75: yielded to another user; 137: killed).
set -u
R=${PROT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
G="bash $R/scripts/with_gpu_mem.sh 6000 bash $R/scripts/gpu_polite.sh"   # yields to other users' GPU jobs
PICK=$R/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
OLD=$R/data/bc_pick_c1.pt
for f in "$PICK" "$OLD"; do [ -f "$f" ] || { echo "MISSING $f on $(hostname)"; exit 1; }; done
D=${PROT_DIR:-runs/protect}
mkdir -p "$D" logs
command -v flock > /dev/null || { echo "flock not found"; exit 1; }
exec 7> "$D/.queue.lock"
flock -n 7 || { echo "BUSY: another queue (or a job it started) holds $D/.queue.lock on $(hostname); not started"; exit 1; }
J=$D/jobs_$(hostname).txt
: > "$J"
STOP=${PROT_STOP:-"--halt_at=8_000_000 --stop_success=0.9 --stop_patience=3 --save_every=500000"}
TASKS=${PROT_TASKS:-${PROT_TASK:-stack}}
source "$R/scripts/suite_tasks.sh"      # task_env: each task key's env flags
for T in $TASKS; do [ "$(task_env "$T")" = UNKNOWN ] && { echo "UNKNOWN TASK $T"; exit 1; }; done
COMMON="--num_envs=1024 --num_steps=50 --eval_freq=2 --num_eval_envs=256 --total_timesteps=25_000_000 $STOP \
--bootstrap_at_done=truncated --gamma=0.95 --target_kl=1e9 --min_grip_std=1.0 \
--init_ckpt=$PICK --init_actor_only --protect_states=$OLD"
flags() {
    case $1 in
        monitor) echo "--protect_old=monitor --protect_energy=0.95" ;;
        head) echo "--freeze_trunk --protect_env=PickCube-v1 --protect_dagger=10" ;;
        block09) echo "--protect_old=block --protect_energy=0.9" ;;
        block095) echo "--protect_old=block --protect_energy=0.95" ;;
        block099) echo "--protect_old=block --protect_energy=0.99" ;;
        block095r) echo "--protect_old=block --protect_energy=0.95 --protect_every=10" ;;
        gpm095) echo "--protect_old=gpm --protect_energy=0.95" ;;
        gpm099) echo "--protect_old=gpm --protect_energy=0.99" ;;
        block09g) echo "--protect_old=block --protect_energy=0.9 --protect_grad" ;;
        block095g) echo "--protect_old=block --protect_energy=0.95 --protect_grad" ;;
        block099g) echo "--protect_old=block --protect_energy=0.99 --protect_grad" ;;
        gpm095g) echo "--protect_old=gpm --protect_energy=0.95 --protect_grad" ;;
        distill001) echo "--protect_old=distill --protect_coef=0.01" ;;
        distill01) echo "--protect_old=distill --protect_coef=0.1" ;;
        distill1) echo "--protect_old=distill --protect_coef=1" ;;
        ewc10|ewc100|ewc1000) echo "--protect_old=ewc --protect_coef=${1#ewc}" ;;
        l2_1|l2_10|l2_100) echo "--protect_old=l2 --protect_coef=${1#l2_}" ;;
        distill1d) echo "--protect_old=distill --protect_coef=1 --protect_env=PickCube-v1 --protect_dagger=5" ;;
        klcl1|klcl3) echo "--protect_old=distill --protect_coef=${1#klcl} --protect_env=PickCube-v1 \
--protect_dagger=5 --protect_strat" ;;
        klclp1|klclp3) echo "--protect_old=distill --protect_coef=0.${1#klclp} --protect_env=PickCube-v1 \
--protect_dagger=5 --protect_strat" ;;
        klcl1h) echo "--protect_old=distill --protect_coef=1 --protect_env=PickCube-v1 --protect_dagger=5 --protect_strat --head_lr=1e-3" ;;
        fncl) echo "--protect_old=fn --protect_env=PickCube-v1 --protect_dagger=5" ;;
        fnbank) echo "--protect_old=fn --protect_env=PickCube-v1 --protect_dagger=5 --fn_pool_frac=0" ;;
        fncl_w5) echo "--protect_old=fn --protect_env=PickCube-v1 --protect_dagger=5 --fn_warmup=5 --head_lr=1e-3" ;;
        nsa[0-9]*npr) w=${1#nsa}; echo "--nullspace_adapter=${w%npr} --nsa_project=False --nsa_init=rand --protect_env=PickCube-v1 --protect_dagger=10" ;;
        nsa[0-9]*fz) w=${1#nsa}; echo "--nullspace_adapter=${w%fz} --nsa_freeze_adapter --protect_env=PickCube-v1 --protect_dagger=10" ;;
        nsa[0-9]*z) w=${1#nsa}; echo "--nullspace_adapter=${w%z} --nsa_init=zero --protect_env=PickCube-v1 --protect_dagger=10" ;;
        nsa[0-9]*np) w=${1#nsa}; echo "--nullspace_adapter=${w%np} --nsa_project=False --protect_env=PickCube-v1 --protect_dagger=10" ;;
        nsa[0-9]*) echo "--nullspace_adapter=${1#nsa} --protect_env=PickCube-v1 --protect_dagger=10" ;;
        *) echo "UNKNOWN" ;;
    esac
}
ARMS=${PROT_ARMS:-monitor block09 block095 block099 block095r gpm095 gpm099}
for a in $ARMS; do [ "$(flags $a)" = UNKNOWN ] && { echo "UNKNOWN ARM $a"; exit 1; }; done
for seed in ${PROT_SEEDS:-1 2 3}; do
    for T in $TASKS; do
        for a in $ARMS; do
            e=${T}_${a}_s$seed
            [ -f "$D/runs/$e/final_ckpt.pt" ] && continue
            echo "$G python -m wrwl.ppo --exp_name=$e $(task_env "$T") $COMMON --seed=$seed $(flags $a) > logs_$e.txt 2>&1 && echo DONE $e || echo EXIT \$? $e" >> "$J"
        done
    done
done
echo "PROT_START $(date +%T) $(hostname) jobs=$(wc -l < "$J")"
(cd "$D" && xargs -P "${PROT_P:-3}" -d '\n' -I CMD bash -c CMD < "$(basename "$J")")
echo "PROT_END $(date +%T)"
