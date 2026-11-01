#!/usr/bin/env bash
# The last four step-2 jobs (seed 2 of T180 at both gammas, PickCube and StackCube at 0.95), moved to the back of the
# queue at 07:15 so the warm-start arms (run_night1e.sh) start ~2 h sooner. Seed 1 of all four reached <= 1.4% AUC.
set -u
cd "$HOME/pve" || exit 1
source .venv/bin/activate
export PYTHONPATH=$HOME/pve PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
until grep -q STEP3_DONE logs/night1d.log 2>/dev/null && grep -q NIGHT1E_END logs/night1e.log 2>/dev/null; do sleep 60; done
echo "STEP2_REST_START $(date +%T)"
cd runs/sparse || exit 1
sed -n 13,16p jobs.txt > jobs_rest.txt
xargs -P 2 -d '\n' -I CMD bash -c CMD < jobs_rest.txt
echo "STEP2_REST_END $(date +%T)"
