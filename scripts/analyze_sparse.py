"""Steps 2 and 3 of the pivot memo under sparse reward: learning speed per arm, and gate G3' (does the oracle pay?).

Per run (eval.csv): normalised success AUC over [0, B] (the last eval value carried to B after an early stop),
T1%, T10% and T80 (first eval at or above the level; censored at B), final success. Per (task, arm): medians and
the per-seed values. Step 3 also reports the oracle's copy and fallback fractions (train.csv, while the KL is on).
G3' (memo): the oracle's AUC >= 1.25 x the best baseline's in T45 or Stack, and >= 0.9 x scratch in T90; fallback
on <= 20% of states.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.home() / "pve"
p = argparse.ArgumentParser()
p.add_argument("--step2", default=str(ROOT / "runs/sparse/runs"))
p.add_argument("--step3", default=str(ROOT / "runs/step3/runs"))
p.add_argument("--out", default=str(ROOT / "logs/sparse_summary.json"))
a = p.parse_args()


def read(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def run_stats(d):
    rows = read(d / "eval.csv")
    if not rows:
        return None
    budget = json.load(open(d / "args.json"))["total_timesteps"]
    x = np.array([float(r["global_step"]) for r in rows] + [budget])
    y = np.array([float(r["success_once"]) for r in rows])
    y = np.append(y, y[-1])
    auc = float(np.sum(np.diff(x) * y[:-1]) / budget)          # step function, last value carried to the budget
    done = (d / "final_ckpt.pt").exists()

    def first(level):
        hit = np.nonzero(y[:-1] >= level)[0]
        return float(x[hit[0]]) if len(hit) else float("inf")
    res = dict(auc=auc, T1=first(0.01), T10=first(0.10), T80=first(0.80), final=float(y[-2]), budget=budget,
               finished=done, steps=float(x[-2]))
    tr = read(d / "train.csv")
    if tr and "copy_fraction" in tr[0]:
        cf = [float(r["copy_fraction"]) for r in tr if r["copy_fraction"] not in ("nan", "")]
        fb = [float(r["fallback_fraction"]) for r in tr if r["fallback_fraction"] not in ("nan", "")]
        res["copy_fraction"] = float(np.mean(cf)) if cf else float("nan")
        res["fallback_fraction"] = float(np.mean(fb)) if fb else float("nan")
    return res


def fmt_t(v):
    return "cens" if v == float("inf") else f"{v / 1e6:.1f}"


summary = {}
for step, root in (("step2", Path(a.step2)), ("step3", Path(a.step3))):
    groups = defaultdict(dict)
    for d in sorted(root.glob("*_s*")):
        m = re.match(r"(.+)_s(\d+)$", d.name)
        st = run_stats(d) if m else None
        if st:
            groups[m.group(1)][int(m.group(2))] = st
    if not groups:
        continue
    print(f"\n=== {step}: median over seeds (per-seed AUC) ===")
    print(f"{'group':30s} {'n':>2s} {'AUC':>6s} {'T1%':>6s} {'T10%':>6s} {'T80':>6s} {'final':>6s}  per-seed AUC / copy / fallback")
    for g, seeds in sorted(groups.items()):
        vals = list(seeds.values())
        med = {k: float(np.median([v[k] for v in vals])) for k in ("auc", "T1", "T10", "T80", "final")}
        extra = ""
        if "copy_fraction" in vals[0]:
            extra = (f" | copy {np.nanmean([v.get('copy_fraction', np.nan) for v in vals]):.2f}"
                     f" fallback {np.nanmean([v.get('fallback_fraction', np.nan) for v in vals]):.2f}")
        unfinished = sum(not v["finished"] for v in vals)
        print(f"{g:30s} {len(vals):2d} {med['auc']:6.3f} {fmt_t(med['T1']):>6s} {fmt_t(med['T10']):>6s} "
              f"{fmt_t(med['T80']):>6s} {med['final']:6.2f}  " + " ".join(f"{v['auc']:.3f}" for v in vals)
              + extra + (f"  ({unfinished} still running)" if unfinished else ""))
        summary.setdefault(step, {})[g] = dict(median=med, seeds=seeds)

s3 = summary.get("step3", {})
if s3:
    def auc(g):
        return s3[g]["median"]["auc"] if g in s3 else float("nan")
    print("\n=== G3' (oracle gate) ===")
    for task, base_arms in (("t45", ["scratch", "finetune", "kickstart"]), ("stack", ["scratch", "finetune", "kickstart"])):
        best = max((auc(f"{task}_{b}") for b in base_arms if f"{task}_{b}" in s3), default=float("nan"))
        o = auc(f"{task}_oracle")
        print(f"  {task}: oracle AUC {o:.3f} vs best baseline {best:.3f} -> ratio {o / best if best > 0 else float('nan'):.2f} (needs >= 1.25)")
    t90s = summary.get("step2", {}).get("push_t90_g0.95", {}).get("median", {}).get("auc", float("nan"))
    o90 = auc("t90_oracle")
    print(f"  t90: oracle AUC {o90:.3f} vs scratch (step 2, gamma 0.95) {t90s:.3f} -> ratio "
          f"{o90 / t90s if t90s > 0 else float('nan'):.2f} (needs >= 0.9)")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(a.out, "w"), indent=1, default=float)
print("saved", a.out)
