#!/usr/bin/env bash
# Is the seam floor local-but-incomplete? The hand-written seam and the done-probe both need the cube held, so the
# floor switches off as soon as the gripper opens and the hand closes again; the biggest gripper conflict ("released
# near the goal") is outside the seam and only a global floor covers it. Latched (--seam_gate latch) the floor stays on
# from the first seam step to the episode's end. Arms, all frozen trunk + gripper floor at the same dose: global,
# probe seam (the method so far), latched probe seam, latched hand-written seam. Evals every 2 iterations (0.1M steps)
# on 256 episodes so differences under 0.5M show; compare eval success at fixed steps and steps to 0.9.
#   LATCH_TASKS="stack place" LATCH_SEEDS="1 2 3 4" LATCH_FLOOR=1.0 LATCH_P=2 bash scripts/run_latch.sh
# Other doses go to their own directory (LATCH_DIR, e.g. runs/latch06), so run names stay comparable within one dose.
set -u
R=${LATCH_ROOT:-$HOME/zhr_pve}
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
G="bash $R/scripts/with_gpu_mem.sh 6000"
PICK=$R/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
PROBE=$R/data/done_probe_pick_c1.pt
F=${LATCH_FLOOR:-1.0}
D=${LATCH_DIR:-runs/latch}
mkdir -p "$D" logs
J=$D/jobs.txt
: > "$J"
for task in ${LATCH_TASKS:-stack place}; do
    if [ "$task" = stack ]; then
        ENV="--env_id=StackCubeNarrow-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --obs_adapter=stack_to_pick"
    else
        ENV="--env_id=PlaceCube-v1 --env_kwargs='{\"robot_uids\": \"panda_wristcam\", \"reward_mode\": \"sparse\"}'"
    fi
    COMMON="$ENV --num_envs=1024 --num_steps=50 --eval_freq=2 --num_eval_envs=256 --total_timesteps=25_000_000 \
--halt_at=8_000_000 --bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=3 --target_kl=1e9 \
--init_ckpt=$PICK --init_actor_only --freeze_trunk --seam_rule=pick --min_grip_std=$F"
    for seed in ${LATCH_SEEDS:-1 2 3 4}; do
        n() { echo "$G python -m wrwl.ppo --exp_name=${task}_$1_s$seed $2 > logs_${task}_$1_s$seed.txt 2>&1 && echo DONE ${task}_$1_s$seed" >> "$J"; }
        n global "$COMMON --seed=$seed"
        n probe "$COMMON --seed=$seed --grip_floor_seam --seam_probe=$PROBE"
        n latch_probe "$COMMON --seed=$seed --grip_floor_seam --seam_probe=$PROBE --seam_gate=latch"
        n latch_rule "$COMMON --seed=$seed --grip_floor_seam --seam_gate=latch"
    done
done
echo "LATCH_START $(date +%T) floor=$F jobs=$(wc -l < "$J")"
(cd "$D" && xargs -P "${LATCH_P:-2}" -d '\n' -I CMD bash -c CMD < jobs.txt)
echo "LATCH_END $(date +%T)"
