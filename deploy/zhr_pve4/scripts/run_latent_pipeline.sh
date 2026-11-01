#!/usr/bin/env bash
# After the experts finish: collect expert rollouts, distil them into the shared network, plot the latent space.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890
until grep -q EXPERTS_DONE runs/experts/train_experts.log 2>/dev/null && grep -q STACK_EXT_DONE logs/stackcube_ext.log 2>/dev/null; do sleep 30; done
echo "PIPELINE_START $(date +%T)"
python scripts/collect_rollouts.py --num-envs 512 && echo "COLLECT_DONE $(date +%T)" &&
python scripts/distill.py && echo "DISTILL_DONE $(date +%T)" &&
python scripts/plot_latent.py && echo "PLOT_DONE $(date +%T)"
echo "PIPELINE_END rc=$? $(date +%T)"
