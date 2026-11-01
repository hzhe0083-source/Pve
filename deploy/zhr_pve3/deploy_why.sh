#!/usr/bin/env bash
# Run once per host from ~/zhr_scratch/whyc (the unpacked why_code.tgz): build ~/zhr_pve3 for the mechanism queue (our
# own dir; the venv, expert and data are links into our ~/zhr_pve), print one md5 over the code so hosts can be compared,
# and parse every queue arm's flags with the patched trainer on the CPU.
set -eu
W=$HOME/zhr_scratch/whyc
R=$HOME/zhr_pve3
[ -e "$R" ] && { echo "ABORT: $R exists"; exit 1; }
mkdir -p "$R/runs" "$R/logs" "$R/scripts"
cd "$R"
ln -s ~/zhr_pve/.venv .venv
ln -s ~/zhr_pve/third_party third_party
ln -s ~/zhr_pve/data data
ln -s ~/zhr_pve/runs/experts_wc runs/experts_wc
cp -r "$W/wrwl" .
cp "$W/scripts/run_why.sh" "$W/scripts/why_test.py" scripts/
cp ~/zhr_pve2/scripts/gpu_polite.sh ~/zhr_pve2/scripts/with_gpu_mem.sh scripts/
cp "$W/drv_why_d.sh" "$W/drv_c9.sh" .
find wrwl scripts -name "*.py" -o -name "*.sh" | sort | xargs md5sum | md5sum | sed 's/ .*/  code md5 over wrwl + scripts/'
ls -la runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
eval "$(sed -n '/^flags() {/,/^}/p' scripts/run_why.sh)"
for a in nsa64 npr rp vis none nsa64z rf01 rf1 rf10 nsa64d nprd nsa64lo nsa64hi nprlo nprhi; do
    echo "$a	$(flags $a) --nullspace_adapter=64 --init_ckpt=x --init_actor_only --protect_env=PickCube-v1"
done > logs/why_argv.txt
CUDA_VISIBLE_DEVICES= PYTHONPATH=$R .venv/bin/python - <<'EOF'
import shlex
import tyro
from wrwl.ppo import Args
bad = 0
for line in open("logs/why_argv.txt"):
    arm, argv = line.rstrip("\n").split("\t")
    try:
        a = tyro.cli(Args, args=shlex.split(argv))
        print(f"parse {arm:8s} project={a.nsa_project} init={a.nsa_init} proj={a.nsa_proj} mask={a.nsa_mask} "
              f"restore={a.nsa_restore} adapter_lr={a.adapter_lr}")
    except SystemExit as e:
        bad += 1
        print(f"PARSE FAILED {arm}: exit {e.code}")
print("ALL ARMS PARSE" if not bad else f"{bad} ARMS FAILED")
EOF
