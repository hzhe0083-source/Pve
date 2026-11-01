#!/usr/bin/env bash
# After the latent pipeline: gold criticality labels for every expert task, then redraw the latent plots with them.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890
until grep -q PIPELINE_END logs/latent_pipeline.log 2>/dev/null; do sleep 30; done
echo "LABELS_START $(date +%T)"
python scripts/label_criticality.py && echo "LABELS_DONE $(date +%T)" &&
python scripts/plot_latent.py && echo "PLOT2_DONE $(date +%T)"
echo "LABELS_END rc=$? $(date +%T)"
