#!/usr/bin/env bash
# M1 conflict dose-response on PushCubeDir(theta): a source policy trained at theta = 0, then five arms per target theta.
#   scratch       no reuse
#   finetune      start from the source weights
#   kickstart     KL to the source everywhere (decays over the first half of training)
#   gate_reach    KL to the source only while reaching the cube (hand gate: copy the shared part, learn the contact)
#   gate_contact  KL only near contact (inverse gate, knock-out control)
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890
until grep -q PIPELINE_END logs/latent_pipeline.log 2>/dev/null; do sleep 30; done
mkdir -p runs/m1 && cd runs/m1 || exit 1
echo "M1_START $(date +%T)"

COMMON="--env_id=PushCubeDir-v1 --num_envs=2048 --num_steps=20 --total_timesteps=5_000_000 --eval_freq=2 --num_eval_envs=128 --stop_patience=2"
python -m wrwl.ppo --exp_name=source_t0 $COMMON --env_kwargs='{"theta_deg": 0}' --stop_success=0.95 > source_t0.log 2>&1
SRC=$HOME/pve/runs/m1/runs/source_t0/final_ckpt.pt
[ -f "$SRC" ] || { echo "M1_FAILED no source"; exit 1; }
echo "SOURCE_DONE $(date +%T) $(tail -1 runs/source_t0/eval.csv)"

declare -A ARM=(
    [scratch]=""
    [finetune]="--init_ckpt=$SRC"
    [kickstart]="--prior_ckpt=$SRC --prior_coef=0.5 --gate=all"
    [gate_reach]="--prior_ckpt=$SRC --prior_coef=0.5 --gate=reach"
    [gate_contact]="--prior_ckpt=$SRC --prior_coef=0.5 --gate=contact"
)
jobs_file=jobs.txt
: > $jobs_file
for seed in 1 2 3; do
    for th in 0 90 180 45 135; do
        for arm in scratch finetune kickstart gate_reach gate_contact; do
            name=t${th}_${arm}_s${seed}
            echo "python -m wrwl.ppo --exp_name=$name $COMMON --seed=$seed --env_kwargs='{\"theta_deg\": $th}' --stop_success=0.9 ${ARM[$arm]} > logs_$name.txt 2>&1; echo DONE $name" >> $jobs_file
        done
    done
done
xargs -P 2 -d '\n' -I CMD bash -c CMD < $jobs_file
echo "M1_END $(date +%T)"
