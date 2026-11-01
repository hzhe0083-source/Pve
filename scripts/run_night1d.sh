#!/usr/bin/env bash
# Pivot memo step 3: does gating pay with perfect knowledge? Sparse reward, value bootstrapped only at time-outs,
# gamma 0.95 (the memo's expected gamma*; step 2 runs in parallel and may revise it), stop at 0.9.
#   T45 and T90 (PushCubeDir, source S0 = PushCube expert), Stack (StackCube seen through PickCube's layout, source
#   Pick = PickCube expert); arms:
#     scratch     T45 and Stack here (T90 scratch comes from step 2)
#     finetune    the source's actor as init (critic not loaded)
#     kickstart   reverse KL to the source everywhere, coef 0.5 annealed over the first half, target-KL stop off
#     oracle      the same KL only where the gold copy loss of the 5 nearest labelled states is < 0.1
#   budgets: push 15M (the memo's clip(2.5 x scratch T80, 10M, 25M) before step 2 is in), Stack 25M.
# Seeds outer, so seed 1 of every arm finishes first. Jobs wait for the labels and checkpoints they need.
# After the code review (03:40): every arm runs with the target-KL stop off (finetune was throttled at 0.1); the copy
# term moves only the mean (wrwl/ppo.py --copy_std off); the oracle uses the 3-chunk loss only where pi* keeps 15 steps
# and drops states where the expert itself fails; Stack is StackCubeNarrow-v1 with a PickCube expert trained on the
# same robot (panda_wristcam). Starts once the panda_wristcam PickCube runs are done, to keep GPU memory in bounds.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890
until [ -f runs/experts_wc/runs/PickCube_wc_s1/final_ckpt.pt ] && [ -f runs/experts_wc/runs/PickCube_wc_s2/final_ckpt.pt ]; do sleep 60; done
echo "NIGHT1D_START $(date +%T)"
R=$HOME/pve/runs
S0=$R/experts/runs/PushCube/final_ckpt.pt
PICK=$R/experts_wc/runs/PickCube_wc_s1/final_ckpt.pt
COMMON="--bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=2 --num_eval_envs=128 --target_kl=1e9"
KL="--prior_coef=0.5 --copy_loss=rkl"
mkdir -p runs/step3 && cd runs/step3 || exit 1
: > jobs.txt
job() {   # name, wait-for-file ("" for none), args
    local pre=""
    [ -n "$2" ] && pre="until [ -f $HOME/pve/$2 ]; do sleep 60; done; "
    echo "${pre}python -m wrwl.ppo --exp_name=$1 $3 > logs_$1.txt 2>&1 && echo DONE $1" >> jobs.txt
}
for seed in 1 2 3; do
    P45="--env_id=PushCubeDir-v1 --env_kwargs='{\"theta_deg\": 45, \"reward_mode\": \"sparse\"}' --num_envs=2048 --num_steps=20 --eval_freq=5 --total_timesteps=15_000_000 --seed=$seed $COMMON"
    P90="--env_id=PushCubeDir-v1 --env_kwargs='{\"theta_deg\": 90, \"reward_mode\": \"sparse\"}' --num_envs=2048 --num_steps=20 --eval_freq=5 --total_timesteps=15_000_000 --seed=$seed $COMMON"
    ST="--env_id=StackCubeNarrow-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --obs_adapter=stack_to_pick --num_envs=1024 --num_steps=50 --eval_freq=10 --total_timesteps=25_000_000 --seed=$seed $COMMON"
    job t45_scratch_s$seed "" "$P45"
    job t45_finetune_s$seed "" "$P45 --init_ckpt=$S0 --init_actor_only"
    job t45_kickstart_s$seed "" "$P45 --prior_ckpt=$S0 $KL --gate=all"
    job t45_oracle_s$seed "" "$P45 --prior_ckpt=$S0 $KL --gate=oracle --oracle_labels=$HOME/pve/data/copyloss_push45.pt --oracle_source=S0"
    job stack_scratch_s$seed "" "$ST"
    job stack_finetune_s$seed "" "$ST --init_ckpt=$PICK --init_actor_only"
    job stack_kickstart_s$seed "" "$ST --prior_ckpt=$PICK $KL --gate=all"
    job stack_oracle_s$seed data/copyloss_stack.pt "$ST --prior_ckpt=$PICK $KL --gate=oracle --oracle_labels=$HOME/pve/data/copyloss_stack.pt --oracle_source=Pick"
    job t90_finetune_s$seed "" "$P90 --init_ckpt=$S0 --init_actor_only"
    job t90_kickstart_s$seed "" "$P90 --prior_ckpt=$S0 $KL --gate=all"
    job t90_oracle_s$seed data/copyloss_push90.pt "$P90 --prior_ckpt=$S0 $KL --gate=oracle --oracle_labels=$HOME/pve/data/copyloss_push90.pt --oracle_source=S0"
done
echo "STEP3_JOBS $(wc -l < jobs.txt)"
xargs -P 2 -d '\n' -I CMD bash -c CMD < jobs.txt
echo "STEP3_DONE $(date +%T)"
