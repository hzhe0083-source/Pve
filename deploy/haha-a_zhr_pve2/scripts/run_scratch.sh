#!/usr/bin/env bash
# Learn PlaceCube or StackCubeNarrow from scratch: the reference for "what adding a skill to an existing network saves"
# (rule registered in docs/conflict-reuse-2026-10-31.md, 16:05). Exactly run_protect.sh's protocol (sparse reward,
# 1024 envs, gamma 0.95, gripper std floor 1.0, an eval every 102,400 steps, stop after 3 evals >= 0.9) except that no
# expert is loaded (a random network) and the halt is 20M; runs/scratch/runs/<task>_scratch_s<seed>.
#   SCR_TASK=place SCR_SEEDS="1 2 3 4" SCR_P=3 bash scripts/run_scratch.sh
# Each job line ends with DONE <run> or EXIT <code> <run> (75: yielded to another user); runs that already have
# final_ckpt.pt are skipped, so rerunning the script resumes the queue.
set -u
R=${PROT_ROOT:-$HOME/zhr_pve2}
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
G="bash $R/scripts/with_gpu_mem.sh 6000 bash $R/scripts/gpu_polite.sh"   # yields to other users' GPU jobs
D=${SCR_DIR:-runs/scratch}
mkdir -p "$D" logs
J=$D/jobs_$(hostname)_$$.txt
: > "$J"
T=${SCR_TASK:-place}
if [ "$T" = place ]; then
    ENV="--env_id=PlaceCube-v1 --env_kwargs='{\"robot_uids\": \"panda_wristcam\", \"reward_mode\": \"sparse\"}'"
else
    ENV="--env_id=StackCubeNarrow-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --obs_adapter=stack_to_pick"
fi
COMMON="$ENV --num_envs=1024 --num_steps=50 --eval_freq=2 --num_eval_envs=256 --total_timesteps=25_000_000 \
--halt_at=20_000_000 --stop_success=0.9 --stop_patience=3 --save_every=2_000_000 --bootstrap_at_done=truncated \
--gamma=0.95 --target_kl=1e9 --min_grip_std=1.0"
for seed in ${SCR_SEEDS:-1 2 3}; do
    e=${T}_scratch_s$seed
    [ -f "$D/runs/$e/final_ckpt.pt" ] && continue
    echo "$G python -m wrwl.ppo --exp_name=$e $COMMON --seed=$seed > logs_$e.txt 2>&1 && echo DONE $e || echo EXIT \$? $e" >> "$J"
done
echo "SCR_START $(date +%T) $(hostname) jobs=$(wc -l < "$J")"
(cd "$D" && xargs -P "${SCR_P:-3}" -d '\n' -I CMD bash -c CMD < "$(basename "$J")")
echo "SCR_END $(date +%T)"
