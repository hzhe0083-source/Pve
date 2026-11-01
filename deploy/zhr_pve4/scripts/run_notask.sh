#!/usr/bin/env bash
# After the labels: distil the shared network again without the task one-hot, and redraw both latents with
# PullCubeTool left out (its expert never succeeded). Shows whether the per-task islands come from the one-hot.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
until grep -q LABELS_END logs/labels.log 2>/dev/null; do sleep 20; done
[ -f staging/plot_latent.py ] && cp staging/plot_latent.py scripts/plot_latent.py   # swapped in only after the labels' own plot
echo "NOTASK_START $(date +%T)"
python scripts/distill.py --drop-task-id --out=runs/distill/shared_notask.pt && echo "DISTILL_NOTASK_DONE $(date +%T)" &&
python scripts/plot_latent.py --tag=_onehot --exclude=PullCubeTool-v1 && echo "PLOT_ONEHOT_DONE $(date +%T)" &&
python scripts/plot_latent.py --model=runs/distill/shared_notask.pt --drop-task-id --tag=_notask \
    --exclude=PullCubeTool-v1 && echo "PLOT_NOTASK_DONE $(date +%T)"
echo "NOTASK_END rc=$? $(date +%T)"
