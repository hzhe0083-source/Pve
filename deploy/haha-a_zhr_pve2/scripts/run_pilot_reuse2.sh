#!/usr/bin/env bash
# Round 2 of the reuse pilot on Stack (sparse reward, PickCube_wc_s1 as the source, hidden layers frozen unless noted).
# Round 1 (run_pilot_reuse.sh) put extra gripper noise in a hand-written seam. The review asked for two things:
#   a placement test   the lab's recipe (a2r: gripper std floor 0.6 everywhere) vs the same floor only at the seam;
#   a locator          the seam found automatically: where the source's own done-probe fires on its frozen hidden layer
#                      (scripts/train_done_probe.py, trained on PickCube rollouts only, no Stack labels)
# Arms (seeds 1-4 now, split across hosts with PILOT_SEEDS; 4 more only if undecided; floors are 1.0 except the
# lab recipe, see the yield check below):
#   frozen, frozen_boost   round 1's baseline and hand-seam boost (PILOT_BASELINES=1, for a different source expert)
#   floor_global06   --min_grip_std=0.6                                 (the lab's exact recipe)
#   floor_global10   --min_grip_std=1.0                                 (the same floor strength as the seam arms)
#   floor_seam10     --min_grip_std=1.0 --grip_floor_seam               (hand-written seam only: the placement test)
#   floor_probe10    --min_grip_std=1.0 --grip_floor_seam --seam_probe  (automatic seam: the full recipe)
#   boost_probe      --explore_boost=1.2 --seam_probe                    (round 1's boost with the automatic seam)
#   ft_floor_probe10 floor_probe10 without --freeze_trunk               (does freezing the impressions matter?)
# Primary outcome: ignition time (training success >= 2% for 3 consecutive iterations, train.csv), compared with
# Kaplan-Meier and a log-rank test; secondary: eval AUC. Every run halts at 15M steps (schedules span 25M).
#   PILOT_ROOT=$HOME/zhr_pve PILOT_P=4 bash scripts/run_pilot_reuse2.sh      (a lab 4090)
set -u
R=${PILOT_ROOT:-$HOME/pve}
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
echo "PILOT2_START $(date +%T) on $(hostname)"
PICK=${PILOT_PICK:-$R/runs/experts_wc/runs/PickCube_wc_s1/final_ckpt.pt}
PROBE=${PILOT_PROBE:-$R/data/done_probe_pick.pt}
RANDPROBE=${PILOT_RANDPROBE:-$R/data/random_probe.pt}
ST0="--env_id=StackCubeNarrow-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --obs_adapter=stack_to_pick --num_envs=1024 --num_steps=50 --eval_freq=10 --total_timesteps=25_000_000 --bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=2 --num_eval_envs=128 --target_kl=1e9 --halt_at=15_000_000"
ST="$ST0 --init_ckpt=$PICK --init_actor_only"
D=${PILOT_DIR:-runs/pilot2}     # a second queue on the same host needs its own directory: jobs.txt is rewritten here
mkdir -p "$D" && cd "$D" || exit 1
: > jobs.txt
T=${PILOT_TAG:+_$PILOT_TAG}     # e.g. c1 when the source expert was retrained on another machine
job() {     # PILOT_ARMS (space-separated arm names, e.g. "stack_floor_seam10 stack_floor_global10") keeps only those arms
    if [ -n "${PILOT_ARMS:-}" ] && [[ " $PILOT_ARMS " != *" ${1%_s*} "* ]]; then return; fi
    echo "bash $R/scripts/with_gpu_mem.sh 6000 python -m wrwl.ppo --exp_name=$1$T $2 > logs_$1$T.txt 2>&1 && echo DONE $1$T" >> jobs.txt
}
for seed in ${PILOT_EXTRA_SEEDS-}; do    # baselines on the same source expert: plain fine-tuning, step 3's kickstart
                                          # (copy everywhere, coef 0.5 annealed), and the noise without reuse
    job stack_ft_s$seed "$ST --seed=$seed"
    job stack_kick_s$seed "$ST0 --seed=$seed --prior_ckpt=$PICK --prior_coef=0.5 --copy_loss=rkl --gate=all"
    job stack_scratch_floor10_s$seed "$ST0 --seed=$seed --min_grip_std=1.0"
done
for seed in ${PILOT_SEEDS-1 2 3 4}; do
    if [ "${PILOT_BASELINES:-0}" = 1 ]; then     # round 1's arms, needed again when the source expert differs
        job stack_frozen_s$seed "$ST --seed=$seed --freeze_trunk"
        job stack_frozen_boost_s$seed "$ST --seed=$seed --freeze_trunk --explore_boost=1.2"
    fi
    # floors at 1.0: the exploration-yield check (scripts/explore_yield.py, 4096 episodes per layout) gave 0.98% first
    # successes for the lab's 0.6 everywhere, 10.9% for 1.0 everywhere, 14.7% for 1.0 at the hand-written seam and
    # 13.3% for 1.0 where the done-probe fires; 0.6 stays in as the lab's exact recipe
    job stack_floor_global06_s$seed "$ST --seed=$seed --freeze_trunk --min_grip_std=0.6"
    job stack_floor_global10_s$seed "$ST --seed=$seed --freeze_trunk --min_grip_std=1.0"
    job stack_floor_seam10_s$seed "$ST --seed=$seed --freeze_trunk --min_grip_std=1.0 --grip_floor_seam"
    job stack_floor_probe10_s$seed "$ST --seed=$seed --freeze_trunk --min_grip_std=1.0 --grip_floor_seam --seam_probe=$PROBE"
    job stack_boost_probe_s$seed "$ST --seed=$seed --freeze_trunk --explore_boost=1.2 --seam_probe=$PROBE"
    job stack_ft_floor_probe10_s$seed "$ST --seed=$seed --min_grip_std=1.0 --grip_floor_seam --seam_probe=$PROBE"
    # locality controls: the same floor everywhere except the seam, and in a random region of the probe's size
    # (scripts/make_random_probe.py)
    job stack_floor_anti10_s$seed "$ST --seed=$seed --freeze_trunk --min_grip_std=1.0 --grip_floor_seam --grip_floor_anti"
    job stack_floor_rand10_s$seed "$ST --seed=$seed --freeze_trunk --min_grip_std=1.0 --grip_floor_seam --seam_probe=$RANDPROBE"
done
xargs -P "${PILOT_P:-3}" -d '\n' -I CMD bash -c CMD < jobs.txt
echo "PILOT2_END $(date +%T)"
