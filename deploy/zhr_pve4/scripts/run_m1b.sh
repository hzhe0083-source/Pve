#!/usr/bin/env bash
# M1 restarted after the copy-term unit test (U2): the original design minus the runs already finished, with the KL
# arms using the copy loss that passed U2 (COPY, default rkl) with PPO's target-KL early stop turned off: the stop
# (0.1 nats per decision) cut every epoch short once the copy term moved the policy, so copying never happened.
# U2 at theta=0: T80 0.08M with the stop off vs 1.97M with it on; scratch is unchanged either way (0.82M). Finished runs are those with an eval.csv and a DONE line;
# the KL runs made with the failed copy term were moved to runs/m1/aborted.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890
COPY=${COPY:-rkl}
COEF=${COEF:-0.5}
cd runs/m1 || exit 1
echo "M1B_START $(date +%T) copy_loss=$COPY prior_coef=$COEF"
COMMON="--env_id=PushCubeDir-v1 --num_envs=2048 --num_steps=20 --total_timesteps=5_000_000 --eval_freq=2 --num_eval_envs=128 --stop_patience=2"
SRC=$HOME/pve/runs/m1/runs/source_t0/final_ckpt.pt
KL="--prior_ckpt=$SRC --prior_coef=$COEF --copy_loss=$COPY --target_kl=1e9"
declare -A ARM=(
    [scratch]=""
    [finetune]="--init_ckpt=$SRC"
    [kickstart]="$KL --gate=all"
    [gate_reach]="$KL --gate=reach"
    [gate_contact]="$KL --gate=contact"
)
: > jobs_b.txt
for seed in 1 2 3; do
    for th in 0 90 180 45 135; do
        for arm in scratch finetune kickstart gate_reach gate_contact; do
            name=t${th}_${arm}_s${seed}
            [ -f "runs/$name/eval.csv" ] && grep -q "^DONE $name\$" "$HOME/pve/logs/m1.log" && continue
            echo "python -m wrwl.ppo --exp_name=$name $COMMON --seed=$seed --env_kwargs='{\"theta_deg\": $th}' --stop_success=0.9 ${ARM[$arm]} > logs_$name.txt 2>&1 && echo DONE $name" >> jobs_b.txt
        done
    done
done
echo "M1B_JOBS $(wc -l < jobs_b.txt)"
xargs -P 2 -d '\n' -I CMD bash -c CMD < jobs_b.txt
echo "M1_END $(date +%T)"
