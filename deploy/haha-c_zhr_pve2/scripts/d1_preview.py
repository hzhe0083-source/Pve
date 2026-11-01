"""Quick D1 preview on the first labels: is a held-out task's criticality predictable from the other tasks?

Leave-one-task-out over the labelled tasks. For each feature set, a ridge regression and a gradient-boosted model are
fit on the other tasks' states to the within-task percentile of the label, then scored on the held-out task: Spearman
against the label, and AUROC for the label's top 10% of states. Feature sets:
    latent (one-hot)      trunk of the shared network distilled with the task one-hot
    latent (no task id)   trunk of the shared network distilled with the one-hot zeroed
    raw obs               the normalised unified observation, one-hot zeroed
    hand features         gripper-mover, mover-goal and tool-mover distances, grasp flags, gripper opening, mover
                          height, slot flags, time t/H
    time only             t/H
References (not predictors): p(1-p) from the labels' own rollouts, and the split-half noise ceiling.
Caveat: both latents were distilled on every task, the held-out one included, so this is not yet zero-shot.
"""
import argparse
from pathlib import Path

import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from mani_skill.utils.registration import REGISTERED_ENVS
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from wrwl.envs.unified import TASK_DIMS
from wrwl.shared import SharedPolicy

p = argparse.ArgumentParser()
p.add_argument("--labels", default=str(Path.home() / "pve/data/labels.pt"))
p.add_argument("--onehot", default=str(Path.home() / "pve/runs/distill/shared.pt"))
p.add_argument("--notask", default=str(Path.home() / "pve/runs/distill/shared_notask.pt"))
p.add_argument("--exclude", default="PullCubeTool-v1")
a = p.parse_args()

labels = {t: v for t, v in torch.load(a.labels, weights_only=False).items() if t not in a.exclude.split(",")}
tasks = list(labels)


def load(path):
    m = SharedPolicy()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def pct(y):
    return np.argsort(np.argsort(y)) / max(len(y) - 1, 1)


def top10(y):
    q = np.quantile(y, 0.9)
    return (y >= q) & (y > 0)


m1, m0 = load(a.onehot), load(a.notask)
names = ["latent (one-hot)", "latent (no task id)", "raw obs", "hand features", "time only"]
feats = {k: [] for k in names}
ys = {"c*": [], "KL": []}
ref = {"p(1-p)": []}
halves = []
group = []
for k, (t, v) in enumerate(labels.items()):
    obs = v["obs"].float()
    o0 = obs.clone()
    o0[:, TASK_DIMS] = 0
    tt = (v["t"].float() / REGISTERED_ENVS[t].max_episode_steps).numpy()[:, None]
    with torch.no_grad():
        feats["latent (one-hot)"].append(m1.latent(obs).numpy())
        feats["latent (no task id)"].append(m0.latent(o0).numpy())
        feats["raw obs"].append(((o0 - m1.obs_mean) / m1.obs_std).numpy())
    o = obs.numpy()
    hand = np.stack([np.linalg.norm(o[:, 63:66], axis=1), np.linalg.norm(o[:, 66:69], axis=1) * o[:, 62], o[:, 62],
                     o[:, 35], o[:, 46], o[:, 7] + o[:, 8], o[:, 27],
                     np.linalg.norm(o[:, 36:39] - o[:, 25:28], axis=1) * o[:, 47], o[:, 47], o[:, 58]], axis=1)
    feats["hand features"].append(np.concatenate([hand, tt], axis=1))
    feats["time only"].append(tt)
    ys["c*"].append(v["c_star"].numpy())
    ys["KL"].append(v["kl"].numpy())
    ps = v["p_state"].numpy()
    ref["p(1-p)"].append(ps * (1 - ps))
    halves.append((v["c_star_h1"].numpy(), v["c_star_h2"].numpy()))
    group.append(np.full(len(obs), k))
feats = {n: np.concatenate(x) for n, x in feats.items()}
ys = {n: np.concatenate(x) for n, x in ys.items()}
group = np.concatenate(group)
pct_y = {n: np.concatenate([pct(y[group == k]) for k in range(len(tasks))]) for n, y in ys.items()}

print(f"tasks held out in turn: {[t.replace('-v1', '') for t in tasks]}; states per task: {np.bincount(group).tolist()}")
for target in ["c*", "KL"]:
    print(f"\n=== target {target}: held-out Spearman / top-10% AUROC, mean over tasks (per-task AUROC) ===")
    for name in names:
        for mdl_name in ["ridge", "gbm"]:
            sp, auc = [], []
            for k in range(len(tasks)):
                tr, te = group != k, group == k
                mdl = Ridge(alpha=10.0) if mdl_name == "ridge" else HistGradientBoostingRegressor(
                    max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
                mdl.fit(feats[name][tr], pct_y[target][tr])
                pred = mdl.predict(feats[name][te])
                y = ys[target][te]
                sp.append(spearman(pred, y))
                auc.append(roc_auc_score(top10(y), pred) if 0 < top10(y).sum() < len(y) else np.nan)
            print(f"  {name:20s} {mdl_name:5s}  spearman {np.nanmean(sp):+.2f}  AUROC {np.nanmean(auc):.2f}   ("
                  + " ".join(f"{x:.2f}" for x in auc) + ")")
    sp, auc = [], []
    for k in range(len(tasks)):
        te = group == k
        y, r = ys[target][te], np.concatenate(ref["p(1-p)"])[te]
        sp.append(spearman(r, y))
        auc.append(roc_auc_score(top10(y), r) if 0 < top10(y).sum() < len(y) else np.nan)
    print(f"  {'p(1-p), gold p':20s} ref    spearman {np.nanmean(sp):+.2f}  AUROC {np.nanmean(auc):.2f}   ("
          + " ".join(f"{x:.2f}" for x in auc) + ")")
# Two follow-ups. (1) G2's own baseline: a probe trained to predict p(1-p) (not c*) on the other tasks, used to rank
# the held-out task's c*. (2) What is left after the bound c* <= p(1-p): the share rho = c* / p(1-p) of the outcome
# variance still open at s that this one decision decides.
pq = np.concatenate(ref["p(1-p)"])
pct_pq = np.concatenate([pct(pq[group == k]) for k in range(len(tasks))])
rho = np.where(pq > 0.01, ys["c*"] / np.maximum(pq, 1e-6), 0.0).clip(0, 1)
pct_rho = np.concatenate([pct(rho[group == k]) for k in range(len(tasks))])
print("\n=== G2 baseline: probe of p(1-p) (gbm) used to rank the held-out task's c* ===")
for name in names[:3]:
    sp, auc = [], []
    for k in range(len(tasks)):
        tr, te = group != k, group == k
        mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
        mdl.fit(feats[name][tr], pct_pq[tr])
        pred, y = mdl.predict(feats[name][te]), ys["c*"][te]
        sp.append(spearman(pred, y))
        auc.append(roc_auc_score(top10(y), pred))
    print(f"  {name:20s} spearman {np.mean(sp):+.2f}  AUROC {np.mean(auc):.2f}   (" + " ".join(f"{x:.2f}" for x in auc) + ")")
print("\n=== share rho = c*/p(1-p): held-out Spearman (gbm), and the split-half ceiling of rho ===")
for name in names:
    sp = []
    for k in range(len(tasks)):
        tr, te = group != k, group == k
        mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
        mdl.fit(feats[name][tr], pct_rho[tr])
        sp.append(spearman(mdl.predict(feats[name][te]), rho[te]))
    print(f"  {name:20s} spearman {np.mean(sp):+.2f}   (" + " ".join(f"{x:+.2f}" for x in sp) + ")")
r_h = []
for k, (h1, h2) in enumerate(halves):
    q = pq[group == k]
    r1 = np.where(q > 0.01, h1 / np.maximum(q, 1e-6), 0).clip(0, 1)
    r2 = np.where(q > 0.01, h2 / np.maximum(q, 1e-6), 0).clip(0, 1)
    r_h.append(spearman(r1, r2))
print(f"  {'split-half ceiling':20s} spearman {np.mean(r_h):+.2f}   (" + " ".join(f"{x:+.2f}" for x in r_h) + ")")
sp = [spearman(h1, h2) for h1, h2 in halves]
auc = [np.mean([roc_auc_score(top10(h2), h1), roc_auc_score(top10(h1), h2)]) for h1, h2 in halves]
print(f"\nc* split-half ceiling: spearman {np.mean(sp):.2f}  AUROC {np.mean(auc):.2f}   (" + " ".join(f"{x:.2f}" for x in auc) + ")")
