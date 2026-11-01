"""D1 with a few labels from the new task: does a handful of labelled states rescue the cross-task prediction?

For each held-out task, m of its labelled states ("shots") are added to the training set of the other tasks (weighted
so that the shots carry as much weight as half of the other tasks' data), and the model is scored on the held-out
task's remaining states: Spearman against c*, and AUROC for its top 10%. m = 0 is the zero-shot D1 of d1_preview.py.
"shots only" fits a ridge model on the m shots alone, to separate transfer from plain within-task supervision.
Averaged over 5 random draws of the shots. Caveat: here the shots' labels come from the held-out task's own expert;
on a truly new task they would come from the learner once it starts to succeed.
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
p.add_argument("--notask", default=str(Path.home() / "pve/runs/distill/shared_notask.pt"))
p.add_argument("--onehot", default=str(Path.home() / "pve/runs/distill/shared.pt"))
p.add_argument("--exclude", default="PullCubeTool-v1")
p.add_argument("--shots", default="0,8,16,32,64")
p.add_argument("--draws", type=int, default=5)
a = p.parse_args()

labels = {t: v for t, v in torch.load(a.labels, weights_only=False).items() if t not in a.exclude.split(",")}
tasks = list(labels)


def load(path):
    m = SharedPolicy()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def spearman(x, y):
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def pct(y):
    return np.argsort(np.argsort(y)) / max(len(y) - 1, 1)


def top10(y):
    return (y >= np.quantile(y, 0.9)) & (y > 0)


m0, m1 = load(a.notask), load(a.onehot)
feats = {"latent (no task id)": [], "raw obs": [], "hand features": []}
ys, group = [], []
for k, (t, v) in enumerate(labels.items()):
    obs = v["obs"].float()
    o0 = obs.clone()
    o0[:, TASK_DIMS] = 0
    tt = (v["t"].float() / REGISTERED_ENVS[t].max_episode_steps).numpy()[:, None]
    with torch.no_grad():
        feats["latent (no task id)"].append(m0.latent(o0).numpy())
        feats["raw obs"].append(((o0 - m1.obs_mean) / m1.obs_std).numpy())
    o = obs.numpy()
    hand = np.stack([np.linalg.norm(o[:, 63:66], axis=1), np.linalg.norm(o[:, 66:69], axis=1) * o[:, 62], o[:, 62],
                     o[:, 35], o[:, 46], o[:, 7] + o[:, 8], o[:, 27],
                     np.linalg.norm(o[:, 36:39] - o[:, 25:28], axis=1) * o[:, 47], o[:, 47], o[:, 58]], axis=1)
    feats["hand features"].append(np.concatenate([hand, tt], axis=1))
    ys.append(v["c_star"].numpy())
    group.append(np.full(len(obs), k))
feats = {n: np.concatenate(x) for n, x in feats.items()}
y_all = np.concatenate(ys)
group = np.concatenate(group)
pct_all = np.concatenate([pct(y_all[group == k]) for k in range(len(tasks))])
rng = np.random.default_rng(0)

print(f"held-out c*: Spearman / top-10% AUROC on the held-out task's remaining states, mean over "
      f"{len(tasks)} tasks x {a.draws} draws")
print(f"{'features':22s} {'shots':>5s}   {'other tasks + shots':>22s}   {'shots only (ridge)':>20s}")
for name, X in feats.items():
    for m in map(int, a.shots.split(",")):
        res_c, res_s = [], []
        for k in range(len(tasks)):
            idx_k = np.nonzero(group == k)[0]
            for d in range(a.draws if m else 1):
                shots = rng.choice(idx_k, size=m, replace=False) if m else np.array([], dtype=int)
                test = np.setdiff1d(idx_k, shots)
                other = np.nonzero(group != k)[0]
                tr = np.concatenate([other, shots])
                y_tr = np.concatenate([pct_all[other], pct(y_all[shots]) if m else np.array([])])
                w = np.concatenate([np.ones(len(other)), np.full(m, 0.5 * len(other) / max(m, 1))])
                mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
                mdl.fit(X[tr], y_tr, sample_weight=w)
                pred, yt = mdl.predict(X[test]), y_all[test]
                res_c.append((spearman(pred, yt), roc_auc_score(top10(yt), pred)))
                if m >= 8:
                    r = Ridge(alpha=10.0).fit(X[shots], pct(y_all[shots]))
                    ps = r.predict(X[test])
                    res_s.append((spearman(ps, yt), roc_auc_score(top10(yt), ps)))
        c = np.nanmean(np.array(res_c), axis=0)
        s = np.nanmean(np.array(res_s), axis=0) if res_s else (np.nan, np.nan)
        print(f"{name:22s} {m:5d}   {c[0]:+.2f} / {c[1]:.2f}{'':12s}   {s[0]:+.2f} / {s[1]:.2f}")
