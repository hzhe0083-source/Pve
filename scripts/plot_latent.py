"""Latent point clouds of the shared network: states of all tasks embedded with UMAP and PCA.

Panels are coloured by task, by episode phase, by grasp state and (if a labels file exists) by criticality.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import umap  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from wrwl.envs.unified import TASK_DIMS  # noqa: E402
from wrwl.shared import SharedPolicy  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--rollouts", default=str(Path.home() / "pve/data/rollouts.pt"))
p.add_argument("--model", default=str(Path.home() / "pve/runs/distill/shared.pt"))
p.add_argument("--labels", default=str(Path.home() / "pve/data/labels.pt"))
p.add_argument("--per-task", type=int, default=3000)
p.add_argument("--out-dir", default=str(Path.home() / "pve/figures"))
p.add_argument("--drop-task-id", action="store_true", help="zero the task one-hot (for a model distilled without it)")
p.add_argument("--exclude", default="", help="comma-separated tasks to leave out, e.g. one whose expert failed")
p.add_argument("--tag", default="", help="suffix for the output file names")
a = p.parse_args()

data = torch.load(a.rollouts, weights_only=False)
skip = set(filter(None, a.exclude.split(",")))
data = {t: d for t, d in data.items() if t not in skip}
model = SharedPolicy().cuda()
model.load_state_dict(torch.load(a.model))
model.eval()
g = torch.Generator().manual_seed(0)

rows = []
for task, d in data.items():
    n_env, horizon = d["obs"].shape[:2]
    succ = d["succ_time"].clamp_min(1).float()
    succ[d["succ_time"] < 0] = horizon
    obs = d["obs"].flatten(0, 1).clone()
    if a.drop_task_id:
        obs[:, TASK_DIMS] = 0
    t = d["t"].flatten(0, 1).float()
    phase = (t / succ.repeat_interleave(horizon)).clamp(0, 1.2)
    grasp = torch.maximum(obs[:, 35], obs[:, 46])   # mover grasped, tool grasped
    idx = torch.randperm(len(obs), generator=g)[:a.per_task]
    rows.append((task, obs[idx], phase[idx], grasp[idx]))

with torch.no_grad():
    z = torch.cat([model.latent(o.cuda()).cpu() for _, o, _, _ in rows]).numpy()
task_lab = np.concatenate([[i] * len(o) for i, (_, o, _, _) in enumerate(rows)])
phase = torch.cat([ph for *_, ph, _ in rows]).numpy()
grasp = torch.cat([gr for *_, gr in rows]).numpy()
names = [t.replace("-v1", "") for t, *_ in rows]

emb = {"umap": umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit_transform(z),
       "pca": PCA(n_components=2, random_state=0).fit_transform(z)}
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
cmap = plt.get_cmap("tab10")
for kind, e in emb.items():
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for i, name in enumerate(names):
        m = task_lab == i
        axes[0].scatter(e[m, 0], e[m, 1], s=2, color=cmap(i), label=name, alpha=0.6)
    axes[0].legend(markerscale=6, fontsize=9, loc="best")
    axes[0].set_title("Shared latent, coloured by task")
    sc = axes[1].scatter(e[:, 0], e[:, 1], s=2, c=phase, cmap="viridis", alpha=0.6)
    fig.colorbar(sc, ax=axes[1], label="episode phase  t / t_success")
    axes[1].set_title("Coloured by phase")
    axes[2].scatter(e[:, 0], e[:, 1], s=2, c=np.where(grasp > 0.5, 1.0, 0.0), cmap="coolwarm", alpha=0.6)
    axes[2].set_title("Coloured by grasp (red = object held)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / f"latent_{kind}{a.tag}.png", dpi=160)
    plt.close(fig)
    print("saved", out / f"latent_{kind}{a.tag}.png")
np.savez(out / f"latent_embedding{a.tag}.npz", umap=emb["umap"], pca=emb["pca"], task=task_lab, phase=phase, grasp=grasp,
         names=np.array(names))

if Path(a.labels).exists():
    labels = {t: v for t, v in torch.load(a.labels, weights_only=False).items() if t not in skip}
    lab_obs = torch.cat([v["obs"] for v in labels.values()]).clone()
    if a.drop_task_id:
        lab_obs[:, TASK_DIMS] = 0
    lab_task = np.concatenate([[names.index(t.replace("-v1", ""))] * len(v["obs"]) for t, v in labels.items()])
    c_star = torch.cat([v["c_star"] for v in labels.values()]).numpy()
    c_bar_nats = torch.cat([v["kl"] for v in labels.values()]).numpy()
    with torch.no_grad():
        z_lab = model.latent(lab_obs.cuda()).cpu().numpy()
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit(np.concatenate([z, z_lab]))
    e_all = reducer.embedding_
    e_bg, e_lab = e_all[:len(z)], e_all[len(z):]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax in axes:
        ax.scatter(e_bg[:, 0], e_bg[:, 1], s=1, color="0.85", alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    for i, name in enumerate(names):
        m = lab_task == i
        axes[0].scatter(e_lab[m, 0], e_lab[m, 1], s=10, color=cmap(i), label=name)
    axes[0].legend(markerscale=2, fontsize=9)
    axes[0].set_title("Labelled states by task (grey: all states)")
    sc = axes[1].scatter(e_lab[:, 0], e_lab[:, 1], s=12, c=np.log10(c_star + 1e-4), cmap="magma")
    fig.colorbar(sc, ax=axes[1], label="log10 c*  (copy price)")
    axes[1].set_title("Criticality c* = Var_a Q under the tempered expert")
    sc = axes[2].scatter(e_lab[:, 0], e_lab[:, 1], s=12, c=c_bar_nats, cmap="viridis")
    fig.colorbar(sc, ax=axes[2], label="KL(pi*_tau || pi_base)  nats  (learning amount)")
    axes[2].set_title("Integrated criticality c_bar / (2 tau^2)")
    fig.tight_layout()
    fig.savefig(out / f"latent_criticality{a.tag}.png", dpi=160)
    plt.close(fig)
    print("saved", out / f"latent_criticality{a.tag}.png")
