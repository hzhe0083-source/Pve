#!/usr/bin/env bash
# Pivot memo steps 0 and 1 (docs/pivot-memo-2026-10-31.md), one lane next to run_night1b.sh:
#   step 0   sources on their own task and on the targets under sparse reward; Pick on Stack with and without release
#   step 1   D1' copy-loss labels: T45 and Stack first (2048 states), then T90, T180, T0 (1024 states);
#            sources S0 = PushCube expert, S180 = PullCube expert, Pick = PickCube expert;
#            target experts: M1 scratch seed 1 at the target theta (source_t0 for theta 0), the StackCube expert
#   then     a second PickCube seed (library disagreement for Stack), and the D1' evaluation
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890
R=$HOME/pve/runs
PUSH_SRC="S0=$R/experts/runs/PushCube/final_ckpt.pt,S180=$R/experts/runs/PullCube/final_ckpt.pt"
echo "NIGHT1C_START $(date +%T)"
python scripts/step0_checks.py > logs/step0.txt 2>&1
echo "STEP0_DONE rc=$? $(date +%T)"

label() {   # target expert sources states
    python scripts/label_copyloss.py --target="$1" --expert="$2" --sources="$3" --states="$4" \
        --out="data/copyloss_$1.pt" > "logs/copyloss_$1.log" 2>&1
    echo "LABELS_$1 rc=$? $(date +%T)"
}
label push45 "$R/m1/runs/t45_scratch_s1/final_ckpt.pt" "$PUSH_SRC" 2048
label stack "$R/experts/runs/StackCube/final_ckpt.pt" "Pick=$R/experts/runs/PickCube/final_ckpt.pt" 2048
label push90 "$R/m1/runs/t90_scratch_s1/final_ckpt.pt" "$PUSH_SRC" 1024
label push180 "$R/m1/runs/t180_scratch_s1/final_ckpt.pt" "$PUSH_SRC" 1024
label push0 "$R/m1/runs/source_t0/final_ckpt.pt" "$PUSH_SRC" 1024

cd "$R/experts" || exit 1
python "$HOME/pve/third_party/ms_ppo/ppo.py" --exp_name=PickCube_s2 --env_id=PickCube-v1 --num_envs=1024 \
    --total_timesteps=10_000_000 --update_epochs=8 --num_minibatches=32 --num_eval_envs=16 --no-capture-video \
    --seed=2 > logs/PickCube_s2.txt 2>&1
echo "PICK_S2_DONE rc=$? $(date +%T)"
cd "$HOME/pve" || exit 1
python scripts/d1p_eval.py --labels data/copyloss_push45.pt data/copyloss_push90.pt data/copyloss_push180.pt \
    data/copyloss_push0.pt data/copyloss_stack.pt --out=logs/d1p_eval.json > logs/d1p_eval.txt 2>&1
echo "D1P_EVAL_DONE rc=$? $(date +%T)"
echo "NIGHT1C_END $(date +%T)"
