"""Does the new task fit into the space the old task occupies, and is the change it needs concentrated in a few
directions? Offline, on states already collected by scripts/conflict_concentration.py (--collect pick / stack).

For each layer of the frozen PickCube actor, the input space the layer's weights act on (raw state for layer 1, the
previous hidden layer after that, the last hidden layer for the readout):
  occupancy   the old task's occupied subspace = top principal directions (uncentered) of the inputs on PickCube states
              holding 99% of their energy; share of the Stack states' input energy inside it
  change      fine-tune a copy of the actor on the Stack expert's actions (behaviour cloning from the PickCube weights) and
              take each layer's weight change dW: share of its input-side energy inside the old subspace; how many
              directions hold 90% of it (effective rank) against the same for a control fine-tuned on random targets;
              and how many neurons (rows) hold 90% of it, the axis-aligned view gate 1 used
  picture     2-D PCA of the last hidden layer for both tasks, the target coloured by phase (for intuition only)
  python scripts/occupancy_test.py --pick=P --data=data/bc_pick_c1.pt,data/bc_stack_c1.pt --out=logs/occupancy_test
"""
import argparse
import copy
import json

import torch

p = argparse.ArgumentParser()
p.add_argument("--pick", required=True)
p.add_argument("--data", required=True)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--energy", type=float, default=0.99)
p.add_argument("--out", required=True)
a = p.parse_args()
torch.manual_seed(0)

from wrwl.experts import load_policy  # noqa: E402

pick = load_policy(a.pick, 42, device="cpu")
sp = pick.actor_logstd.detach().exp().flatten()
D = [torch.load(f, map_location="cpu") for f in a.data.split(",")]
X = [d["x"][d["keep"]].float() for d in D]
Y = [d["y"][d["keep"]].float() for d in D]
print(f"states kept: PickCube {len(X[0])}, Stack {len(X[1])}")
lin = [m for m in pick.actor_mean if isinstance(m, torch.nn.Linear)]


def layer_inputs(net, x):
    """Inputs to each Linear of the actor: the state, then each hidden layer's tanh output."""
    out, h = [], x
    for m in net.actor_mean:
        if isinstance(m, torch.nn.Linear):
            out.append(h)
        h = m(h)
    return out


with torch.no_grad():
    IN = [layer_inputs(pick, x) for x in X]


def subspace(z, energy):
    """Top principal directions (uncentered) holding `energy` of the second moment."""
    ev, V = torch.linalg.eigh(z.T @ z / len(z))
    ev, V = ev.flip(0).clamp_min(0), V.flip(1)
    k = int((ev.cumsum(0) / ev.sum() < energy).sum()) + 1
    return V[:, :k], k


def eff_rank(m, frac=0.9):
    s = torch.linalg.svdvals(m) ** 2
    return int((s.cumsum(0) / s.sum() < frac).sum()) + 1


def rows_for(m, frac=0.9):
    r = m.pow(2).sum(1).sort(descending=True).values
    return int((r.cumsum(0) / r.sum() < frac).sum()) + 1


def finetune(target):
    net = copy.deepcopy(pick)
    opt = torch.optim.Adam(net.actor_mean.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(1)
    for _ in range(a.steps):
        i = torch.randint(len(X[1]), (2048,), generator=g)
        loss = (((net.actor_mean(X[1][i]) - target[i]) / sp) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        err = (((net.actor_mean(X[1]) - target) / sp) ** 2).mean().item()
    return net, err


with torch.no_grad():
    err0 = (((pick.actor_mean(X[1]) - Y[1]) / sp) ** 2).mean().item()
ft, err_ft = finetune(Y[1])
rnd_target = Y[1][torch.randperm(len(Y[1]))]     # control: the same targets, shuffled across states
ctl, err_ctl = finetune(rnd_target)
print(f"Stack BC error (PickCube std units): PickCube weights {err0:.3f} -> fine-tuned {err_ft:.3f}; "
      f"shuffled-target control {err_ctl:.3f}")

res = {"err": [err0, err_ft, err_ctl], "layers": []}
names = ["layer 1 (state -> h1)", "layer 2 (h1 -> h2)", "layer 3 (h2 -> h3)", "readout (h3 -> action)"]
print(f"\n{'layer':24s} | old subspace k/dim | Stack energy inside | dW inside old | eff.rank dW (ctl) | rank of dW inside old (ctl) | rows for 90% of dW (ctl) / rows")
for i, l in enumerate(lin):
    U, k = subspace(IN[0][i], a.energy)
    zt = IN[1][i]
    occ = ((zt @ U).pow(2).sum() / zt.pow(2).sum()).item()
    dW = ft.actor_mean[2 * i].weight.detach() - l.weight.detach()
    dC = ctl.actor_mean[2 * i].weight.detach() - l.weight.detach()
    inside = lambda m: ((m @ U).pow(2).sum() / m.pow(2).sum()).item()
    row = {"layer": names[i], "dim": U.shape[0], "k": k, "stack_energy_inside": occ, "dW_inside": inside(dW),
           "ctl_inside": inside(dC), "eff_rank": eff_rank(dW), "eff_rank_ctl": eff_rank(dC),
           "eff_rank_inside": eff_rank(dW @ U), "eff_rank_inside_ctl": eff_rank(dC @ U),
           "rows90": rows_for(dW), "rows90_ctl": rows_for(dC), "rows": dW.shape[0]}
    res["layers"].append(row)
    print(f"{names[i]:24s} | {k:3d}/{U.shape[0]:3d}            | {occ:6.3f}             | {row['dW_inside']:5.3f} ({row['ctl_inside']:.3f}) "
          f"| {row['eff_rank']:3d} ({row['eff_rank_ctl']:3d})         | {row['eff_rank_inside']:3d} ({row['eff_rank_inside_ctl']:3d})                  "
          f"| {row['rows90']:3d} ({row['rows90_ctl']:3d}) / {dW.shape[0]}")
json.dump(res, open(a.out + ".json", "w"), indent=1)

try:      # the picture is for intuition only; the numbers above are the test
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs, ht = IN[0][3], IN[1][3]
    g = torch.Generator().manual_seed(2)
    si, ti = torch.randperm(len(hs), generator=g)[:4000], torch.randperm(len(ht), generator=g)[:4000]
    z = torch.cat([hs, ht])
    mu = z.mean(0)
    _, _, V = torch.linalg.svd(z - mu, full_matrices=False)
    P = lambda h: ((h - mu) @ V[:2].T).numpy()
    xt = X[1][ti]
    held, near = xt[:, 18] > 0.5, xt[:, 39:42].norm(dim=1) <= 0.025
    phase = torch.where(held & near, 2, torch.where(held, 1, torch.where(near, 3, 0)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    ps, pt = P(hs[si]), P(ht[ti])
    ax[0].scatter(ps[:, 0], ps[:, 1], s=2, alpha=0.35, c="#4C78A8", label="old task (PickCube) states")
    ax[0].scatter(pt[:, 0], pt[:, 1], s=2, alpha=0.35, c="#F58518", label="new task (Stack) states")
    ax[0].set_title("last hidden layer of the frozen PickCube policy, PCA to 2-D")
    ax[0].legend(markerscale=6)
    cols = ["#BAB0AC", "#54A24B", "#E45756", "#B279A2"]
    labs = ["approaching", "holding, away from goal", "holding at goal (conflict: Pick holds, Stack releases)",
            "released near goal"]
    for c in range(4):
        m = (phase == c).numpy()
        ax[1].scatter(pt[m, 0], pt[m, 1], s=3, alpha=0.5, c=cols[c], label=labs[c])
    ax[1].set_title("new-task states by phase (same projection)")
    ax[1].legend(markerscale=5, fontsize=8)
    for x_ in ax:
        x_.set_xlabel("PC1")
        x_.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=110)
    print("picture:", a.out + ".png")
except ImportError as e:
    print("no picture:", e)
