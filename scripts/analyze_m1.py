"""M1 analysis: steps to reach 80% success (T80) vs conflict angle theta, per arm, plus learning curves."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--runs", default=str(Path.home() / "pve/runs/m1/runs"))
p.add_argument("--out", default=str(Path.home() / "pve/figures/m1_dose_response.png"))
p.add_argument("--threshold", type=float, default=0.8)
a = p.parse_args()

ARMS = ["scratch", "finetune", "kickstart", "gate_reach", "gate_contact"]
LABEL = {"scratch": "scratch", "finetune": "fine-tune from source", "kickstart": "kickstart (KL everywhere)",
         "gate_reach": "gate: copy reach, learn contact", "gate_contact": "inverse gate (knock-out)"}
curves = defaultdict(list)
t80 = defaultdict(list)
for d in sorted(Path(a.runs).glob("t*_s*")):
    m = re.match(r"t(\d+)_(.+)_s(\d+)", d.name)
    if not m or not (d / "eval.csv").exists():
        continue
    th, arm = int(m.group(1)), m.group(2)
    rows = list(csv.DictReader(open(d / "eval.csv")))
    steps = np.array([float(r["global_step"]) for r in rows])
    succ = np.array([float(r["success_once"]) for r in rows])
    budget = json.load(open(d / "args.json"))["total_timesteps"]
    hit = np.nonzero(succ >= a.threshold)[0]
    t80[(th, arm)].append(steps[hit[0]] if len(hit) else np.inf)
    curves[(th, arm)].append((steps, succ, budget))

thetas = sorted({k[0] for k in t80})
print(f"T80 in millions of env steps (inf = not reached), threshold {a.threshold}")
print("theta  " + "  ".join(f"{arm:>13s}" for arm in ARMS))
for th in thetas:
    cells = []
    for arm in ARMS:
        v = np.array(t80.get((th, arm), []))
        cells.append("-" if len(v) == 0 else f"{np.median(v) / 1e6:5.2f} (n={len(v)})")
    print(f"{th:5d}  " + "  ".join(f"{c:>13s}" for c in cells))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, arm in enumerate(ARMS):
    xs, med, lo, hi = [], [], [], []
    for th in thetas:
        v = np.array(t80.get((th, arm), []), dtype=float)
        if len(v) == 0:
            continue
        cap = max(c[2] for c in curves[(th, arm)])
        v = np.where(np.isinf(v), cap, v) / 1e6
        xs.append(th)
        med.append(np.median(v))
        lo.append(np.percentile(v, 25))
        hi.append(np.percentile(v, 75))
    axes[0].errorbar(np.array(xs) + (i - 2) * 1.5, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)],
                     marker="o", capsize=3, label=LABEL[arm])
axes[0].set_xlabel("conflict angle theta between source and target push direction (deg)")
axes[0].set_ylabel("env steps to 80% success (M), median and IQR")
axes[0].set_title("M1: conflict dose-response")
axes[0].legend(fontsize=8)
th_show = max(thetas) if thetas else 0
for i, arm in enumerate(ARMS):
    for j, (s, y, _) in enumerate(curves.get((th_show, arm), [])):
        axes[1].plot(s / 1e6, y, color=f"C{i}", alpha=0.7, label=LABEL[arm] if j == 0 else None)
axes[1].set_xlabel("env steps (M)")
axes[1].set_ylabel("eval success_once")
axes[1].set_title(f"learning curves at theta = {th_show} deg")
axes[1].legend(fontsize=8)
fig.tight_layout()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=160)
print("saved", a.out)
