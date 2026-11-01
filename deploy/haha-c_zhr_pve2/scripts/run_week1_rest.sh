#!/usr/bin/env bash
# The rest of week 1, queued behind M1 and the criticality labels. Most informative first, seeds interleaved, so a
# partial run still answers every question with fewer seeds.
#   1. restore fidelity under the experts                            G1 condition 2
#   2. the source library: src5 shared networks, 3 seeds             (targets excluded)
#   3. KL scaling: PushCube at 10/20/50 Hz, PPO-clip vs covariant     G1 condition 4
#   4. E0 + hand gate on the new tasks, unified observation           G1 condition 3 and the hand-gate evidence
#        scratch      no reuse
#        finetune     start from the library, the new task's one-hot copied from its designated source
#        kickstart    finetune + KL to the frozen initial network everywhere
#        handgate     finetune + KL only before anything is grasped (copy the approach, learn the contact phase)
#      PullCubeTool is out: its expert never succeeded (gamma 0.8, 0.95 and 0.99, 25M steps). The hand gate runs on
#      LiftPegUpright (copy PickCube's grasp, learn the upright rotation) and PlaceSphere (consistent with PickCube).
#      KL arms use the copy loss that passed the U2 unit test (COPY) with PPO's target-KL early stop off.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890
COPY=${COPY:-rkl}
until grep -q M1_END logs/m1.log 2>/dev/null && grep -q LABELS_END logs/labels.log 2>/dev/null; do sleep 60; done
echo "WEEK1_REST_START $(date +%T)"

python scripts/restore_check_expert.py > logs/restore_expert.log 2>&1
echo "RESTORE_DONE rc=$? $(date +%T) $(grep overall logs/restore_expert.log)"

SRC5=PushCube-v1,PullCube-v1,PickCube-v1,StackCube-v1,PokeCube-v1
for seed in 1 2 3; do
    python scripts/distill.py --tasks=$SRC5 --seed=$seed --out=runs/distill/src5_s$seed.pt > logs/distill_src5_s$seed.log 2>&1
    echo "SRC5_DONE seed=$seed rc=$? $(date +%T)"
done

# KL scaling: same physical task (2.5 s episodes, 1 s rollouts), decisions split m = 25 / 50 / 125 ways.
# gamma and lambda are physical time constants (0.8 and 0.9 per 50 ms), the step budget scales with the frequency.
mkdir -p runs/klscale && cd runs/klscale || exit 1
: > jobs.txt
for seed in 1 2; do
    for cf in 20 10 50; do
        H=$(( 5 * cf / 2 ))
        read -r G L <<< "$(python -c "print(0.8 ** (20 / $cf), 0.9 ** (20 / $cf))")"
        for loss in clip cov; do
            name=${loss}_cf${cf}_s$seed
            echo "python -m wrwl.ppo --exp_name=$name --env_id=PushCube-v1 --seed=$seed --loss=$loss --delta=0.5 \
--control_freq=$cf --max_episode_steps=$H --num_eval_steps=$H --num_steps=$cf --gamma=$G --gae_lambda=$L \
--num_envs=2048 --total_timesteps=$(( 150000 * cf )) --eval_freq=5 --num_eval_envs=128 \
> logs_$name.txt 2>&1; echo DONE $name" >> jobs.txt
        done
    done
done
xargs -P 2 -d '\n' -I CMD bash -c CMD < jobs.txt
echo "KLSCALE_DONE $(date +%T)"
cd "$HOME/pve" || exit 1

mkdir -p runs/e0 && cd runs/e0 || exit 1
declare -A LIKE=([PlaceSphere-v1]=PickCube-v1 [LiftPegUpright-v1]=PickCube-v1)
declare -A TASKARGS=(
    [PlaceSphere-v1]="--num_envs=1024 --num_steps=50 --num_eval_steps=50"
    [LiftPegUpright-v1]="--num_envs=2048 --num_steps=50 --num_eval_steps=50"
)
: > jobs.txt
for seed in 1 2 3; do
    LIB=$HOME/pve/runs/distill/src5_s$seed.pt
    for task in LiftPegUpright-v1 PlaceSphere-v1; do
        FT="--init_from_shared --init_task_like=${LIKE[$task]}"
        declare -A ARM=(
            [scratch]=""
            [finetune]="$FT"
            [kickstart]="$FT --prior_self --prior_coef=0.5 --gate=all --copy_loss=$COPY --target_kl=1e9"
            [handgate]="$FT --prior_self --prior_coef=0.5 --gate=pregrasp --copy_loss=$COPY --target_kl=1e9"
        )
        arms="scratch finetune kickstart handgate"
        for arm in $arms; do
            name=${task%-v1}_${arm}_s$seed
            echo "python -m wrwl.ppo --exp_name=$name --env_id=$task --seed=$seed --unified --shared_ckpt=$LIB \
${TASKARGS[$task]} ${ARM[$arm]} --total_timesteps=25_000_000 --eval_freq=10 --num_eval_envs=128 \
--stop_success=0.9 --stop_patience=2 > logs_$name.txt 2>&1; echo DONE $name" >> jobs.txt
        done
    done
done
xargs -P 2 -d '\n' -I CMD bash -c CMD < jobs.txt
echo "E0_DONE $(date +%T)"
cd "$HOME/pve" || exit 1
python scripts/analyze_g1.py > logs/g1.txt 2>&1
echo "WEEK1_REST_END $(date +%T)"
