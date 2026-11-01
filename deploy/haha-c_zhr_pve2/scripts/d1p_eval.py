"""D1' evaluation: can cold-start signals, computed from the sources alone, tell where copying a source hurts?

Gold label per (target state s, copied source X), from label_copyloss.py: Delta(s) = V*(s) - Q_{X,1}(s), or the larger
of the 1- and 3-chunk losses where pi* still has >= 15 steps after the 3-chunk window (later, the 3-chunk loss mostly
measures the deadline). "Harmful" = Delta >= 0.1, over states with V*(s) > 0.1.
Cold-start signals (no target success, no target expert, no library policy trained on the target MDP):
    D_all     mean squared distance between X's and each other library policy's action means, clamped to the action
              range and scaled by one pooled std, i.e. the symmetric KL with a shared std on executed actions
    D_fam     the same over library policies familiar with s; NaN (reported as coverage) where none is familiar
    D_fam|unc D_fam with uncovered states scored as harmful, which is how a gate would treat them
    unfam     X's own unfamiliarity with s: mean distance to the 10 nearest of X's own-task pre-success states
    c_src     the source task's criticality c* at s, by kNN over the source's labels (data/labels.pt)
    cbar      |clamp(mu_X(s))|^2
    rank sum  average rank of D_fam|unc plus that of unfam
Distances use relative features (tcp->obj, obj->goal; the push goal height is dropped) in units of 5 cm, plus the grasp
flag. Familiar = within the 90th percentile of the policy's own held-out pre-success states.
Heuristics (direction chosen in their favour, marked *): gripper-mover distance, grasp flag, t/H, mover-goal distance,
goal progress of X's next chunk.
Slices: all valid states, early (t0 <= 20), contact (gripper-mover distance <= 5 cm). Ceiling: AUROC of one half's
Delta against the other half's label (halves over chunk samples in label files written after 03:40; earlier files
split over continuations, which overstates the ceiling).
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import wrwl.envs.push_dir  # noqa: F401
import wrwl.envs.stack_narrow  # noqa: F401
from wrwl.adapters import ADAPTERS
from wrwl.experts import flat_state, load_policy

ROOT = Path.home() / "pve"
p = argparse.ArgumentParser()
p.add_argument("--labels", nargs="+", required=True)
p.add_argument("--crit", default=str(ROOT / "data/labels.pt"), help="source-task criticality labels")
p.add_argument("--out", default=str(ROOT / "logs/d1p_eval.json"))
a = p.parse_args()
dev = "cuda"
SCALE = 0.05

# library: name -> (checkpoint, own task env, env kwargs, criticality labels of the closest source task, own theta)
LIB = {
    "push": {
        "S0": (ROOT / "runs/experts/runs/PushCube/final_ckpt.pt", "PushCube-v1", {}, "PushCube-v1", None),
        "S180": (ROOT / "runs/experts/runs/PullCube/final_ckpt.pt", "PullCube-v1", {}, "PullCube-v1", None),
        "t0_s2": (ROOT / "runs/m1/runs/t0_scratch_s2/final_ckpt.pt", "PushCubeDir-v1", {"theta_deg": 0.0}, "PushCube-v1", 0),
        "t0_s3": (ROOT / "runs/m1/runs/t0_scratch_s3/final_ckpt.pt", "PushCubeDir-v1", {"theta_deg": 0.0}, "PushCube-v1", 0),
        "t180_s2": (ROOT / "runs/m1/runs/t180_scratch_s2/final_ckpt.pt", "PushCubeDir-v1", {"theta_deg": 180.0}, "PullCube-v1", 180),
        "t180_s3": (ROOT / "runs/m1/runs/t180_scratch_s3/final_ckpt.pt", "PushCubeDir-v1", {"theta_deg": 180.0}, "PullCube-v1", 180),
    },
    "stack": {
        "Pick": (ROOT / "runs/experts_wc/runs/PickCube_wc_s1/final_ckpt.pt", "PickCube-v1",
                 {"robot_uids": "panda_wristcam"}, "PickCube-v1", None),
        "Pick_s2": (ROOT / "runs/experts_wc/runs/PickCube_wc_s2/final_ckpt.pt", "PickCube-v1",
                    {"robot_uids": "panda_wristcam"}, "PickCube-v1", None),
    },
}
OBS = {"push": 35, "stack": 42}


def feat(rel, fam):
    """Relative features in 5 cm units; the push goal sits on the table at a fixed height, so its z is dropped."""
    if fam == "push":
        return torch.cat([rel[:, 0:3] / SCALE, rel[:, 3:5] / SCALE, rel[:, 6:7]], 1)
    return torch.cat([rel[:, 0:6] / SCALE, rel[:, 6:7]], 1)


def own_rel(task, x, g):
    if task == "PickCube-v1":
        return torch.cat([x[:, 36:39], x[:, 39:42], g.reshape(-1, 1).float()], 1)
    return torch.cat([x[:, 28:31] - x[:, 18:21], x[:, 25:28] - x[:, 28:31], g.reshape(-1, 1).float()], 1)


def knn_dist(q, bank, k=10):
    return torch.cdist(q, bank).topk(k, largest=False).values.mean(1)


def own_bank(path, task, kw, obs_dim, fam, n=256, horizon=50):
    """Pre-success relative features of the policy on its own task: a fit half and the 90th-percentile familiarity
    radius calibrated on the held-out half of the episodes."""
    env = gym.make(task, num_envs=n, obs_mode="state_dict", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cuda", render_mode=None, **kw)
    base = env.unwrapped
    pol = load_policy(path, obs_dim)
    std = pol.actor_logstd.exp().detach()
    obs, _ = env.reset(seed=7)
    succ = torch.zeros(n, dtype=torch.bool, device=dev)
    rows, envs = [], []
    with torch.no_grad():
        for _ in range(horizon):
            x = flat_state(obs)
            obj = base.cube if task == "PickCube-v1" else base.obj
            f = feat(own_rel(task, x, base.agent.is_grasping(obj)), fam)
            rows.append(f[~succ])
            envs.append(torch.arange(n, device=dev)[~succ])
            m = pol.actor_mean(x)
            obs, _, _, _, info = env.step((m + std * torch.randn_like(m)).clamp(-1, 1))
            succ |= info["success"].bool()
    env.close()
    torch.cuda.empty_cache()
    rows, envs = torch.cat(rows), torch.cat(envs)
    fit, cal = rows[envs < n // 2], rows[envs >= n // 2]
    return fit, knn_dist(cal, fit).quantile(0.9)


def auroc(y, score):
    y, score = np.asarray(y, bool), np.asarray(score, float)
    ok = ~np.isnan(score)
    y, score = y[ok], score[ok]
    if np.isinf(score).any():                       # uncovered states rank above every finite score
        fin = score[np.isfinite(score)]
        score = np.where(np.isposinf(score), (fin.max() if len(fin) else 0.0) + 1.0, score)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y, score))


crit = torch.load(a.crit, weights_only=False)
results, pooled = {}, {}
banks, pols = {}, {}
for lf in a.labels:
    if not Path(lf).exists():
        print("missing", lf)
        continue
    L = torch.load(lf, weights_only=False)
    fam = L["family"]
    theta = float(L["target"][4:]) if fam == "push" else None
    lib = {k: v for k, v in LIB[fam].items() if Path(v[0]).exists() and (v[4] is None or v[4] != theta)}
    for name, (path, task, kw, _, _) in lib.items():
        if name not in banks:
            banks[name] = own_bank(path, task, kw, OBS[fam], fam)
            pol = load_policy(path, OBS[fam])
            pols[name] = (pol, pol.actor_logstd.exp().detach())
    sig_c = torch.stack([pols[n][1].flatten() for n in lib]).mean(0)          # pooled std per action dim
    x, g = L["x"].to(dev), L["grasp"].to(dev)
    f = feat(L["rel"].to(dev), fam)
    adapter = ADAPTERS["push" if fam == "push" else "stack_to_pick"]
    with torch.no_grad():
        mus = {n: pols[n][0].actor_mean(adapter(x, g)).clamp(-1, 1) for n in lib}
    unfam = {n: knn_dist(f, banks[n][0]) for n in lib}
    familiar = {n: unfam[n] <= banks[n][1] for n in lib}
    Q, Qh = L["Q"], L["Q_halves"]
    V = Q[:, 0]
    ok3 = L["t"] + 3 * L["chunk_len"] + 15 <= L["horizon"]
    valid = (V > 0.1).numpy()
    early = (L["t"] <= 20).numpy()
    tH = (L["t"].float() / L["horizon"]).numpy()
    if fam == "push":
        d_tcp = torch.linalg.norm(x[:, 28:31] - x[:, 18:21], dim=1)
        d_goal = torch.linalg.norm(x[:, 25:28] - x[:, 28:31], dim=1)
    else:
        d_tcp = torch.linalg.norm(x[:, 39:42], dim=1)
        dz = torch.zeros_like(x[:, :3])
        dz[:, 2] = 0.04
        d_goal = torch.linalg.norm(x[:, 45:48] + dz, dim=1)
    contact = (d_tcp <= 0.05).cpu().numpy()
    for bi, (src, w) in enumerate(L["branches"]):
        if src in ("expert", "base") or w != 1:
            continue
        cols = {w_: j for j, (s_, w_) in enumerate(L["branches"]) if s_ == src}

        def delta_of(q, v):
            d = v - q[:, cols[1]]
            if 3 in cols:
                d = torch.where(ok3, torch.maximum(d, v - q[:, cols[3]]), d)
            return d.numpy()
        delta = delta_of(Q, V)
        dh = [delta_of(Qh[:, h], Qh[:, h, 0]) for h in (0, 1)]
        harm = delta >= 0.1
        others = [n for n in lib if n != src]
        ms = mus[src]
        sig = {"unfam": unfam[src].cpu().numpy(), "cbar": (ms ** 2).sum(1).cpu().numpy()}
        coverage = np.zeros(len(V), bool)
        if others:
            dist = torch.stack([(((ms - mus[n]) / sig_c) ** 2).sum(1) for n in others])
            fm = torch.stack([familiar[n] for n in others]).float()
            cover = fm.sum(0) > 0
            d_fam = (dist * fm).sum(0) / fm.sum(0).clamp_min(1)
            sig["D_all"] = dist.mean(0).cpu().numpy()
            sig["D_fam"] = torch.where(cover, d_fam, torch.nan).cpu().numpy()
            sig["D_fam|unc"] = torch.where(cover, d_fam, torch.inf).cpu().numpy()
            sig["rank sum"] = rankdata(sig["D_fam|unc"]) + rankdata(sig["unfam"])
            coverage = cover.cpu().numpy()
        c_lab = crit.get(LIB[fam][src][3])
        if c_lab is not None:
            u = c_lab["obs"].to(dev)           # unified obs: tcp->mover 63:66, mover->goal 66:69, grasp 35
            c_rel = feat(torch.cat([u[:, 63:66], u[:, 66:69], u[:, 35:36]], 1), fam)
            nn_idx = torch.cdist(f, c_rel).topk(10, largest=False).indices
            sig["c_src"] = c_lab["c_star"].to(dev)[nn_idx].mean(1).cpu().numpy()
            if "D_all" in sig:
                sig["D_all*c_src"] = sig["D_all"] * sig["c_src"]
        heur = {"dist tcp-mover": -d_tcp.cpu().numpy(), "grasp": g.float().cpu().numpy(), "t/H": tH,
                "dist mover-goal": -d_goal.cpu().numpy(), "src chunk progress": -L["progress"][:, bi].numpy()}
        pair = f"{L['target']}<-{src}"
        dv = np.clip(delta[valid], 0, None)
        res = {"n_valid": int(valid.sum()), "n_early": int((valid & early).sum()),
               "n_contact": int((valid & contact).sum()), "coverage": float(coverage[valid].mean()),
               "P(Delta<0.1)": float((delta[valid] < 0.1).mean()),
               "P(Delta<0.1) early": float((delta[valid & early] < 0.1).mean()) if (valid & early).any() else float("nan"),
               "harm share of worst 20%": float(np.sort(dv)[::-1][:max(1, int(0.2 * len(dv)))].sum() / max(dv.sum(), 1e-9)),
               "base rate harmful": float(harm[valid].mean()), "library": others}
        for sl_name, sl in (("all", valid), ("early", valid & early), ("contact", valid & contact)):
            res[f"ceiling[{sl_name}]"] = float(np.nanmean([auroc(dh[1][sl] >= 0.1, dh[0][sl]),
                                                           auroc(dh[0][sl] >= 0.1, dh[1][sl])]))
            for k, v in sig.items():
                res[f"{k}[{sl_name}]"] = auroc(harm[sl], v[sl])
            for k, v in heur.items():
                au = auroc(harm[sl], v[sl])
                res[f"*{k}[{sl_name}]"] = float(max(au, 1 - au)) if au == au else au
        results[pair] = res
        if fam == "push":
            pooled.setdefault("harm", []).append(harm[valid])
            for k in ("D_fam|unc", "D_all", "unfam", "rank sum"):
                if k in sig:
                    pooled.setdefault(k, []).append(sig[k][valid])
        print(f"\n== {pair}: valid {res['n_valid']} (early {res['n_early']}, contact {res['n_contact']}), coverage "
              f"{res['coverage']:.2f}, base rate harmful {res['base rate harmful']:.2f}, P(Delta<0.1) "
              f"{res['P(Delta<0.1)']:.2f} (early {res['P(Delta<0.1) early']:.2f}), worst-20% harm share "
              f"{res['harm share of worst 20%']:.2f}; library {others}")
        for sl_name in ("all", "early", "contact"):
            line = "   ".join(f"{k.split('[')[0]} {v:.2f}" for k, v in res.items()
                              if isinstance(v, float) and k.endswith(f"[{sl_name}]"))
            print(f"  [{sl_name}] {line}")
if pooled.get("harm"):
    y = np.concatenate(pooled["harm"])
    print("\npooled over push pairs (source selection + gating): "
          + "   ".join(f"{k} {auroc(y, np.concatenate(v)):.2f}" for k, v in pooled.items() if k != "harm"))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(a.out, "w"), indent=1)
print("saved", a.out)
