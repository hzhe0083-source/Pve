"""Ignition times and AUC for the reuse pilots (one run directory per run, e.g. runs/pilot2/runs).

Ignition is the first iteration at which training-rollout success (train.csv: train_success) stays at or above
--thresh for --k consecutive iterations; a run that never ignites is censored at its last step. Stack outcomes are
bimodal across seeds, so the primary comparison is on ignition time: per arm the number ignited, the Kaplan-Meier
median, and a log-rank test against --ref. Secondary: eval success AUC over [0, budget], last value carried forward.
Arms are run names without the _s<seed> part (and any _<tag> after it).
  python scripts/analyze_pilot.py --runs runs/pilot2/runs --ref stack_floor_global
"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", required=True)
p.add_argument("--thresh", type=float, default=0.02)
p.add_argument("--k", type=int, default=3)
p.add_argument("--level", type=float, default=0.9, help="report the first step with eval success >= this")
p.add_argument("--consecutive", type=int, default=1, help="... held for this many evals in a row (2 for the "
                                                          "confirmatory runs)")
p.add_argument("--ref", default="")
p.add_argument("--out", default="")
a = p.parse_args()


def read(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def run_stats(d):
    tr, ev = read(d / "train.csv"), read(d / "eval.csv")
    if not tr or "train_success" not in tr[0]:
        return None
    steps = np.array([float(r["global_step"]) for r in tr])
    succ = np.array([float(r["train_success"]) for r in tr])
    ok = succ >= a.thresh
    ign = None
    for i in range(len(ok) - a.k + 1):
        if ok[i:i + a.k].all():
            ign = steps[i]
            break
    budget = json.load(open(d / "args.json"))["total_timesteps"]
    auc = float("nan")
    if ev:
        x = np.array([float(r["global_step"]) for r in ev] + [budget])
        y = np.array([float(r["success_once"]) for r in ev])
        y = np.append(y, y[-1])
        auc = float(np.sum(np.diff(x) * y[:-1]) / budget)
    s = [float(r["success_once"]) for r in ev]       # first of --consecutive evals in a row at or above --level
    t_level = next((float(ev[i]["global_step"]) for i in range(len(s) - a.consecutive + 1)
                    if all(v >= a.level for v in s[i:i + a.consecutive])), float("inf"))
    return {"ignited": ign is not None, "time": ign if ign is not None else steps[-1], "auc": auc, "t_level": t_level,
            "max_eval": max((float(r["success_once"]) for r in ev), default=float("nan")), "last_step": steps[-1],
            "done": (d / "final_ckpt.pt").exists()}


def km_median(times, events):
    order = np.argsort(times)
    t, e = np.asarray(times)[order], np.asarray(events)[order]
    s, n = 1.0, len(t)
    for i in range(len(t)):
        if e[i]:
            s *= 1 - 1 / (n - i)
            if s <= 0.5:
                return t[i]
    return float("inf")


def logrank(t1, e1, t2, e2):
    t1, e1, t2, e2 = map(np.asarray, (t1, e1, t2, e2))
    o_minus_e, var = 0.0, 0.0
    for t in np.unique(np.concatenate([t1[e1.astype(bool)], t2[e2.astype(bool)]])):
        n1, n2 = (t1 >= t).sum(), (t2 >= t).sum()
        d1, d2 = ((t1 == t) & e1.astype(bool)).sum(), ((t2 == t) & e2.astype(bool)).sum()
        n, d = n1 + n2, d1 + d2
        if n < 2:
            continue
        o_minus_e += d1 - d * n1 / n
        var += d * (n1 / n) * (n2 / n) * (n - d) / (n - 1)
    if var == 0:
        return float("nan")
    return math.erfc(math.sqrt(o_minus_e ** 2 / var / 2))


arms = defaultdict(dict)
for root in a.runs:
    for d in sorted(Path(root).iterdir()):
        m = re.match(r"^(.*)_s(\d+)(_\w+)?$", d.name)
        st = run_stats(d) if m and d.is_dir() else None
        if st:
            arms[m.group(1)][int(m.group(2))] = st
res = {}
print(f"ignition = training success >= {a.thresh} for {a.k} consecutive iterations; times in M env steps")
print(f"  {'arm':28s} {'n':>2s} {'ignited':>8s} {'KM median':>9s} {'per seed (x = censored)':40s} {'AUC median':>10s}"
      f" {'log-rank p vs ref':>17s}  eval>={a.level} at (M steps; * = still running)")
ref = arms.get(a.ref)
for arm in sorted(arms):
    runs = arms[arm]
    t = [r["time"] / 1e6 for r in runs.values()]
    e = [r["ignited"] for r in runs.values()]
    med = km_median(t, e)
    per = " ".join(f"{r['time'] / 1e6:.1f}{'' if r['ignited'] else 'x'}" for _, r in sorted(runs.items()))
    pv = float("nan")
    if ref and arm != a.ref:
        pv = logrank(t, e, [r["time"] / 1e6 for r in ref.values()], [r["ignited"] for r in ref.values()])
    auc = float(np.median([r["auc"] for r in runs.values()]))
    res[arm] = {"seeds": runs, "km_median": med, "auc_median": auc, "logrank_p_vs_ref": pv}
    tl = " ".join(f"{r['t_level'] / 1e6:.1f}{'' if r['done'] else '*'}" for _, r in sorted(runs.items()))
    print(f"  {arm:28s} {len(runs):2d} {sum(e):4d}/{len(e):<3d} {med:9.2f} {per:40s} {auc:10.3f} {pv:17.3f}  {tl}")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1, default=float)
