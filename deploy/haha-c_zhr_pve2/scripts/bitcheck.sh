#!/usr/bin/env bash
# Bit-identity and smoke test of a new wrwl/ppo.py before it is deployed under running queues.
# OLD = the deployed checkout, NEW = a staging checkout. A short Stack run with default-path flags must give the same
# train.csv (on the columns both versions write) and the same final weights; the new options must run.
#   OLD=$HOME/zhr_pve NEW=$HOME/zhr_pve_test bash scripts/bitcheck.sh > bitcheck.log 2>&1
set -u
OLD=${OLD:-$HOME/zhr_pve}; NEW=${NEW:-$HOME/zhr_pve_test}
PY=$OLD/.venv/bin/python
export PVE_ROOT=$OLD CUDA_VISIBLE_DEVICES=0
P=$OLD/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
C="--env_id=StackCubeNarrow-v1 --env_kwargs={\"reward_mode\":\"sparse\"} --obs_adapter=stack_to_pick --num_envs=256 --num_steps=50 --eval_freq=2 --total_timesteps=25_000_000 --halt_at=40000 --seed=7 --bootstrap_at_done=truncated --gamma=0.95 --target_kl=1e9 --init_ckpt=$P --init_actor_only"
W=$HOME/zhr_bit
rm -rf "$W" && mkdir -p "$W" && cd "$W" || exit 1
run() { local name=$1 root=$2; shift 2; PYTHONPATH=$root bash $OLD/scripts/with_gpu_mem.sh 2500 $PY -m wrwl.ppo --exp_name=$name $C "$@" > $name.log 2>&1; echo "$name: $(tail -n 1 $name.log)"; }
run bit_old "$OLD" --freeze_trunk --min_grip_std=1.0 --grip_floor_seam
run bit_old2 "$OLD" --freeze_trunk --min_grip_std=1.0 --grip_floor_seam    # GPU determinism of the old code itself
run bit_new "$NEW" --freeze_trunk --min_grip_std=1.0 --grip_floor_seam
run gate_time "$NEW" --freeze_trunk --min_grip_std=1.0 --grip_floor_seam --seam_gate=time
run gate_grasp "$NEW" --freeze_trunk --min_grip_std=1.0 --grip_floor_seam --seam_gate=grasp
run warm "$NEW" --min_grip_std=0.6 --critic_warmup=3 --save_every=12800
$PY - <<'EOF'
import csv, torch
def rows(n): return list(csv.DictReader(open(f"runs/{n}/train.csv")))
a, a2, b = rows("bit_old"), rows("bit_old2"), rows("bit_new")
print("old code run twice, train.csv identical:", a == a2)
same = len(a) == len(b) and all(r1[k] == r2[k] for r1, r2 in zip(a, b) for k in r1)
wa, wb = torch.load("runs/bit_old/final_ckpt.pt"), torch.load("runs/bit_new/final_ckpt.pt")
diff = max((wa[k].float() - wb[k].float()).abs().max().item() for k in wa)
ea, eb = [r[:6] for r in csv.reader(open("runs/bit_old/eval.csv"))], [r[:6] for r in csv.reader(open("runs/bit_new/eval.csv"))]
print("DEFAULT PATH: train.csv common columns identical:", same, "| weights max diff:", diff, "| eval identical:", ea == eb)
for n in ("gate_time", "gate_grasp", "warm"):
    r = rows(n)
    print(n, "iterations", len(r), "| seam_frac", [round(float(x["seam_frac"]), 3) if x["seam_frac"] != "nan" else "nan" for x in r],
          "| applied gripper std", [round(float(x["applied_std_gripper"]), 3) for x in r])
init = torch.load("/home/zhaobo/zhr_pve/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt")
w = torch.load("runs/warm/ckpt_38400.pt")       # saved at the start of iteration 4, after the 3 warm-up iterations
print("critic warm-up: actor max change after the 3 value-only iterations (must be 0):",
      max((w[k] - init[k]).abs().max().item() for k in init if k.startswith("actor")),
      "| critic changed:", any((w[k] != init[k]).any().item() for k in init if k.startswith("critic") and k in w))
EOF
echo BITCHECK_END
