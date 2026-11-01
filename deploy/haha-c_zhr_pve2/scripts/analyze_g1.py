"""Gate G1 of week 1: the four preconditions, plus the hand-gate evidence, as one table and two figures.

    1  every source expert reaches >= 80% success                      (collect_rollouts log, src5 distill json)
    2  restore-and-replay under the expert reproduces >= 95% of outcomes (restore_expert.json)
    3  transfer has room both ways: fine-tuning from the library is >= 1.3x faster than scratch on one new task and
       no better on another                                            (runs/e0, steps to 80% success)
    4  F8: covariant per-episode KL ~ Delta and flat in m (within +-20%); PPO-clip per-episode KL grows >= 1.8x per
       doubling of m; covariant final success within 0.10 of PPO-clip at 20 Hz   (runs/klscale)
"""
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

ROOT = Path.home() / "pve"
p = argparse.ArgumentParser()
p.add_argument("--threshold", type=float, default=0.8)
p.add_argument("--delta", type=float, default=0.5)
p.add_argument("--fig-dir", default=str(ROOT / "figures"))
a = p.parse_args()
fig_dir = Path(a.fig_dir)
fig_dir.mkdir(parents=True, exist_ok=True)
verdict = {}


def read_csv(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def steps_to(rows, thr):
    for r in rows:
        if float(r["success_once"]) >= thr:
            return float(r["global_step"])
    return np.inf


print("== 1. source experts and the src5 library ==")
log = ROOT / "logs/latent_pipeline.log"
expert = {}
if log.exists():
    for line in open(log):
        m = re.match(r"(\S+-v1)\s+horizon=\d+ expert success_once=([\d.]+)", line)
        if m:
            expert[m.group(1)] = float(m.group(2))
for t, v in expert.items():
    print(f"  expert {t:20s} {v:.3f}")
src5 = [json.load(open(f)) for f in sorted((ROOT / "runs/distill").glob("src5_s*.json"))]
for d in src5:
    print(f"  src5 seed {d['seed']}: " + "  ".join(f"{t.replace('-v1', '')}={v:.2f}" for t, v in d["success_once"].items()))
sources = ["PushCube-v1", "PullCube-v1", "PickCube-v1", "StackCube-v1", "PokeCube-v1"]
verdict[1] = all(expert.get(t, 0) >= 0.8 for t in sources) if expert else None

print("== 2. restore fidelity under the experts ==")
rj = ROOT / "data/restore_expert.json"
if rj.exists():
    r = json.load(open(rj))
    for t, per in r["per_task"].items():
        print(f"  {t:20s} " + "  ".join(f"t0={k}: agree={v['agree']:.3f} cons={v['consensus']:.3f}"
                                        for k, v in per.items() if v["n_states"]))
    print(f"  overall agreement {r['overall_agree']:.3f}")
    verdict[2] = r["overall_agree"] >= 0.95
else:
    verdict[2] = None

print(f"== 3. E0 and the hand gate: steps to {a.threshold:.0%} success (M), median over seeds ==")
runs = defaultdict(dict)
for d in sorted((ROOT / "runs/e0/runs").glob("*_s*")):
    m = re.match(r"(.+)_(scratch|finetune|kickstart|handgate)_s(\d+)$", d.name)
    rows = read_csv(d / "eval.csv")
    if m and rows:
        runs[(m.group(1), m.group(2))][int(m.group(3))] = rows
tasks = sorted({k[0] for k in runs})
arms = ["scratch", "finetune", "kickstart", "handgate"]
speedup = {}
print("  task            " + "".join(f"{x:>18s}" for x in arms))
for t in tasks:
    med = {}
    cells = []
    for arm in arms:
        v = [steps_to(rows, a.threshold) for rows in runs.get((t, arm), {}).values()]
        med[arm] = np.median(v) if v else np.nan
        cells.append("-" if not v else f"{med[arm] / 1e6:6.2f} (n={len(v)}, {sum(np.isinf(v))} cens)")
    print(f"  {t:16s}" + "".join(f"{c:>18s}" for c in cells))
    if np.isfinite(med["finetune"]) or np.isfinite(med["scratch"]):
        speedup[t] = med["scratch"] / med["finetune"]
for t, s in speedup.items():
    print(f"  {t}: scratch / finetune = {s:.2f}")
verdict[3] = (any(s >= 1.3 for s in speedup.values()) and any(s <= 1.0 for s in speedup.values())) if speedup else None

fig, axes = plt.subplots(1, max(len(tasks), 1), figsize=(6 * max(len(tasks), 1), 4.5), squeeze=False)
for ax, t in zip(axes[0], tasks):
    for i, arm in enumerate(arms):
        for j, rows in enumerate(runs.get((t, arm), {}).values()):
            x = [float(r["global_step"]) / 1e6 for r in rows]
            y = [float(r["success_once"]) for r in rows]
            ax.plot(x, y, color=f"C{i}", alpha=0.7, label=arm if j == 0 else None)
    ax.axhline(a.threshold, color="0.6", ls=":")
    ax.set_title(t)
    ax.set_xlabel("env steps (M)")
    ax.set_ylabel("eval success_once")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(fig_dir / "e0_handgate_curves.png", dpi=150)
plt.close(fig)

print("== 4. F8: KL per episode against the number of decisions m ==")
kl = defaultdict(list)
final = defaultdict(list)
for d in sorted((ROOT / "runs/klscale/runs").glob("*_cf*_s*")):
    m = re.match(r"(clip|cov)_cf(\d+)_s(\d+)$", d.name)
    tr, ev = read_csv(d / "train.csv"), read_csv(d / "eval.csv")
    if not (m and tr and ev):
        continue
    loss, cf = m.group(1), int(m.group(2))
    k = np.array([float(r["kl_per_episode"]) for r in tr])
    kl[(loss, int(tr[0]["decisions_per_episode"]))].append(np.median(k[len(k) // 10:]))
    final[(loss, cf)].append(np.mean([float(r["success_once"]) for r in ev[-3:]]))
for key in sorted(kl):
    print(f"  {key[0]:4s} m={key[1]:4d}  KL/episode median={np.mean(kl[key]):.3f}  (n={len(kl[key])})")
ms = sorted({m for _, m in kl})
cov = np.array([np.mean(kl[("cov", m)]) for m in ms if ("cov", m) in kl])
clip_m = [m for m in ms if ("clip", m) in kl]
clip = np.array([np.mean(kl[("clip", m)]) for m in clip_m])
ok_cov = len(cov) >= 2 and cov.max() <= 1.2 * cov.mean() and cov.min() >= 0.8 * cov.mean()
growth = 2 ** np.polyfit(np.log2(clip_m), np.log2(clip), 1)[0] if len(clip) >= 2 else np.nan
perf = (np.mean(final[("clip", 20)]) - np.mean(final[("cov", 20)])) if final[("clip", 20)] and final[("cov", 20)] else np.nan
for (loss, cf), v in sorted(final.items()):
    print(f"  final success {loss:4s} {cf:2d} Hz: {np.mean(v):.3f}")
print(f"  covariant KL/episode: mean {cov.mean() if len(cov) else np.nan:.3f} (Delta={a.delta}), flat within 20%: {ok_cov}")
print(f"  PPO-clip KL/episode growth per doubling of m: {growth:.2f}  (needs >= 1.8)")
print(f"  success gap clip - cov at 20 Hz: {perf:.3f}  (needs <= 0.10)")
verdict[4] = (bool(ok_cov) and growth >= 1.8 and perf <= 0.10) if len(cov) >= 2 and np.isfinite(perf) else None

fig, ax = plt.subplots(figsize=(6, 4.5))
for loss, c in (("clip", "C3"), ("cov", "C0")) if kl else ():
    xs = [m for m in ms if (loss, m) in kl]
    ax.plot(xs, [np.mean(kl[(loss, m)]) for m in xs], "o-", color=c, label={"clip": "PPO-clip", "cov": "covariant (F8)"}[loss])
ax.axhline(a.delta, color="C0", ls=":", label=f"Delta = {a.delta}")
if kl:
    ax.set_xscale("log")
    ax.set_yscale("log")
ax.set_xlabel("decisions per episode m (same physical task)")
ax.set_ylabel("KL between successive policies, per episode")
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "g1_kl_scaling.png", dpi=150)
plt.close(fig)

print("== G1 ==")
for k in (1, 2, 3, 4):
    print(f"  condition {k}: {'PASS' if verdict[k] else 'not yet run' if verdict[k] is None else 'FAIL'}")
