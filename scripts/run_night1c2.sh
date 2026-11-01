#!/usr/bin/env bash
# Steps 0-1 after the code review (replaces run_night1c.sh from the Stack labels on). Push labels first while the
# panda_wristcam PickCube experts train; then the Stack check and labels on StackCubeNarrow-v1 (both cubes inside
# PickCube's support, the StackCube robot), with PickCube_wc_s1 as the source; then the D1' evaluation.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
R=$HOME/pve/runs
PUSH_SRC="S0=$R/experts/runs/PushCube/final_ckpt.pt,S180=$R/experts/runs/PullCube/final_ckpt.pt"
PICK1=$R/experts_wc/runs/PickCube_wc_s1/final_ckpt.pt
echo "NIGHT1C2_START $(date +%T)"
label() {   # target expert sources states
    python scripts/label_copyloss.py --target="$1" --expert="$2" --sources="$3" --states="$4" \
        --out="data/copyloss_$1.pt" > "logs/copyloss_$1.log" 2>&1
    echo "LABELS_$1 rc=$? $(date +%T)"
}
label push90 "$R/m1/runs/t90_scratch_s1/final_ckpt.pt" "$PUSH_SRC" 1024
label push180 "$R/m1/runs/t180_scratch_s1/final_ckpt.pt" "$PUSH_SRC" 1024
label push0 "$R/m1/runs/source_t0/final_ckpt.pt" "$PUSH_SRC" 1024
until [ -f "$PICK1" ]; do sleep 60; done
python scripts/step0_checks.py --stack-only --pick="$PICK1" --out=logs/step0_stack.json > logs/step0_stack.txt 2>&1
echo "STEP0_STACK_DONE rc=$? $(date +%T)"
label stack "$R/experts/runs/StackCube/final_ckpt.pt" "Pick=$PICK1" 2048
until [ -f "$R/experts_wc/runs/PickCube_wc_s2/final_ckpt.pt" ]; do sleep 60; done
python scripts/d1p_eval.py --labels data/copyloss_push45.pt data/copyloss_push90.pt data/copyloss_push180.pt \
    data/copyloss_push0.pt data/copyloss_stack.pt --out=logs/d1p_eval.json > logs/d1p_eval.txt 2>&1
echo "D1P_EVAL_DONE rc=$? $(date +%T)"
echo "NIGHT1C2_END $(date +%T)"
