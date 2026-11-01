#!/usr/bin/env bash
# Confirmatory batch for the core mechanism (pre-registered in docs/prereg-2026-11-01.md; seeds 1001+ are reserved
# for it, every earlier run is development data). One target task per call; arms, experts and seeds as below.
#   CONF_ROOT=$HOME/zhr_pve CONF_P=4 bash scripts/run_confirm.sh stack "c1 c2 c3" "1001 1002 1003 1004 1005" [ARMS]
# Measurement: eval every 2 iterations on 256 deterministic episodes; early stop (0.9 twice) only after H steps;
# halt at 15M (Stack, Place) or 10M (Retreat). Floors act on the gripper (Stack, Place) or the arm joints (Retreat).
# Arms (frozen hidden layers unless noted):
#   M    floor 1.0 where the expert's own done-probe fires (the method)
#   P    floor 1.0 in the hand-written seam (the source task's own success region): locator control
#   G6   global floor 0.6 (the lab's recipe)          G10  global floor 1.0
#   T    floor 1.0 from step 16 of the episode on (time-gated)
#   H    floor 1.0 whenever the object is held (grasp-gated; pick-layout tasks only)
#   R    floor 1.0 in a random half-space of the expert's hidden layer, matched to the probe's coverage
#   F    fine-tuning (not frozen) + global floor 0.6 + 5 critic-only iterations + target_kl 0.05 + lr 1e-4
#   K    kickstart (random init, copy term annealed over 2.5M) + global floor 0.6
set -u
R=${CONF_ROOT:-$HOME/pve}
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
task=$1; experts=$2; seeds=$3; arms=${4:-}
echo "CONFIRM_START $(date +%T) on $(hostname): $task experts=[$experts] seeds=[$seeds] arms=[${arms:-default}]"
case $task in
    stack)
        ENV="--env_id=StackCubeNarrow-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --obs_adapter=stack_to_pick --num_envs=1024 --num_steps=50 --total_timesteps=25_000_000 --halt_at=15_000_000 --stop_after=6_000_000"
        SEAM="--seam_rule=pick --seam_dims=gripper"; SRC_DIR=runs/experts_wc/runs/PickCube_wc_; PROBE=data/done_probe_pick_
        arms=${arms:-M P G6 G10 T H R F K}; LO=0.6; HI=1.0 ;;
    place)
        ENV="--env_id=PlaceCube-v1 --env_kwargs='{\"robot_uids\": \"panda_wristcam\", \"reward_mode\": \"sparse\"}' --num_envs=1024 --num_steps=50 --total_timesteps=25_000_000 --halt_at=15_000_000 --stop_after=6_000_000"
        SEAM="--seam_rule=pick --seam_dims=gripper"; SRC_DIR=runs/experts_wc/runs/PickCube_wc_; PROBE=data/done_probe_pick_
        arms=${arms:-M P G6 T F K}; LO=0.6; HI=1.0 ;;
    retreat)
        ENV="--env_id=PushRetreat-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --num_envs=2048 --num_steps=20 --total_timesteps=15_000_000 --halt_at=10_000_000 --stop_after=4_000_000"
        SEAM="--seam_rule=push --explore_radius=0.1 --seam_dims=arm"; SRC_DIR=runs/experts/runs/PushCube_; PROBE=data/done_probe_push_
        arms=${arms:-M P G6 T F K}; LO=${CONF_LO:-0.6}; HI=${CONF_HI:-1.0} ;;
    *) echo "unknown task $task"; exit 1 ;;
esac
COMMON="$ENV --eval_freq=2 --num_eval_envs=256 --bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=2"
D=runs/confirm/$task
mkdir -p "$D"
J=$D/jobs_$(date +%H%M%S).txt
: > "$J"
for seed in $seeds; do
    for ex in $experts; do
        SRC=$R/${SRC_DIR}$ex/final_ckpt.pt
        PR=$R/${PROBE}$ex.pt
        RP=$R/data/random_probe_${task}_$ex.pt
        FROZEN="$COMMON --init_ckpt=$SRC --init_actor_only --freeze_trunk --target_kl=1e9 $SEAM"
        for arm in $arms; do
            name=${task}_${arm}_${ex}_s$seed
            [ -f "$D/runs/$name/final_ckpt.pt" ] && continue         # skip what is already done
            case $arm in
                M)   a="$FROZEN --min_grip_std=$HI --grip_floor_seam --seam_probe=$PR" ;;
                P)   a="$FROZEN --min_grip_std=$HI --grip_floor_seam" ;;
                G6)  a="$FROZEN --min_grip_std=$LO" ;;
                G10) a="$FROZEN --min_grip_std=$HI" ;;
                T)   a="$FROZEN --min_grip_std=$HI --grip_floor_seam --seam_gate=time --seam_t0=16" ;;
                H)   a="$FROZEN --min_grip_std=$HI --grip_floor_seam --seam_gate=grasp" ;;
                R)   a="$FROZEN --min_grip_std=$HI --grip_floor_seam --seam_probe=$RP" ;;
                F)   a="$COMMON --init_ckpt=$SRC --init_actor_only $SEAM --min_grip_std=$LO --critic_warmup=5 --target_kl=0.05 --learning_rate=1e-4" ;;
                K)   a="$COMMON --target_kl=1e9 $SEAM --min_grip_std=$LO --prior_ckpt=$SRC --prior_coef=0.5 --copy_loss=rkl --gate=all --prior_anneal_frac=0.1" ;;
                *)   echo "unknown arm $arm"; continue ;;
            esac
            echo "bash $R/scripts/with_gpu_mem.sh 6000 python -m wrwl.ppo --exp_name=$name --seed=$seed $a > logs_$name.txt 2>&1 && echo DONE $name" >> "$J"
        done
    done
done
echo "$(wc -l < "$J") jobs in $J"
(cd "$D" && xargs -P "${CONF_P:-4}" -d '\n' -I CMD bash -c CMD < "$(basename "$J")")
echo "CONFIRM_END $task $(date +%T)"
