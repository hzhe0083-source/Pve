"""Gate 1 for conflict-local unsharing: when one network learns two tasks, does the cross-task gradient conflict sit in
a few shared hidden units, and does giving only those units a per-task copy recover most of what two separate networks
gain, more than splitting random or magnitude-matched units? Offline behaviour cloning of two experts, a proxy for
multi-task RL (no reward).

Tasks: PickCube (the PickCube_wc expert on its own env) and Stack (the StackCube expert on StackCubeNarrow, seen through
the PickCube adapter), both in PickCube's 42-dim layout plus a 2-dim task one-hot. Targets are each expert's action
means clamped to [-1, 1]; the residual is zero where output and target are past the same bound (both execute the same
action), and errors are in units of the PickCube expert's std (the copy term's weighting).
Data: --rounds x --envs episodes of 50 sampled-action steps per task, each row with its episode id. States at or after an
episode's first success are dropped (keep mask): the experts were trained with partial resets and never acted there.
Train/test split by whole episode (20% of episodes held out). Conflict is measured on 8192 fixed training rows; test
rows are only scored. Under --init pick the PickCube loss must be ~0 at step 0 (asserted).
The readout is per-task in arms a, b, c, e and d; a0 is the one fully shared network. So b, c and e differ only in which
hidden units they split, and a0 vs a is the per-task-head effect.
  conflict   in the ranking run (arm a), at steps 0/100/--rank_step/1000/--steps and per hidden layer (256 output rows,
             a row = incoming weights + bias; layer 1 without its task one-hot columns, which are per-task):
             loss per task and hidden gradient norms; top 1/5/10% share of conflict max(0, -<g_pick, g_stack>) within the
             layer ('*' = above the null's 95th percentile) next to a permutation null (keep each row's |dot|, shuffle
             which rows conflict, --null_draws draws: mean and 95th percentile) and the same shares for alignment
             max(0, +dot); conflict mass C/(C+A); rows with dot < 0; share of rows with cosine < -0.3 and the 10/50/90%
             cosine quantiles; layer cosine; Jaccard overlap of the top --frac set with the ranking set used for b.
  ranking    which rows conflict changes fast (on stand-in data the per-row rank correlation fell to ~0 within 10-25
             steps and changed sign), so b is ranked on the per-row dot averaged over the --rank_window steps up to
             --rank_step (every --rank_every steps), not at one step. Printed: the window average's shares against its
             null, and the Jaccard overlap of the top sets of the window's two halves (stability; chance is about
             N/256). If the halves barely overlap, "the conflicting units" is not a stable property of units here.
             A step where a task's loss is at rounding level (step 0 under --init pick) prints only its losses, and a
             layer whose conflict is at rounding level gets no shares (-). Do not read step 0 under either init (pick:
             no PickCube signal; random: tiny readout gain).
             The 8 readout rows come from a0 (the only shared head), as signed dot and cosine, never ranked with hidden
             rows. Under --init pick, PickCube's gradient is a restoring force against the drift Stack causes.
  unsharing  from the same init and batches: (a0) all shared, (a) shared hidden layers + per-task heads, (b) the top
             --frac conflict rows of each hidden layer by the window-averaged dot of run a get a per-task copy of their incoming
             weights, (c) random rows with the same per-layer count (--rand_draws masks), (e) magnitude-matched: same
             count, rows with the nearest |g_pick||g_stack| among rows with dot >= 0 (or, if too few, the highest
             cosine), (d) every unit split, i.e. two separate networks. Per arm: test error per task (arm mean / gripper),
             in the held-at-goal states (PickCube keeps holding, Stack must release), in Stack states released near the
             goal, on as many training rows (fit), the test loss (the training objective, summed over tasks) and the
             recovery (L_a - L_arm) / (L_a - L_d). Repeated for --seeds (init for --init random, batches, masks);
             the last table gives mean and sd per arm over seeds (and over draws for c).
  reading    conflict sits in a few units only for a layer whose share beats the null's 95th percentile and whose
             C/(C+A) is well above zero; b supports conflict-local unsharing only if its recovery is large and beats
             both c and e by more than their spread.
Simulator envs cannot be rebuilt in one process, so each task is collected by its own call:
  python scripts/conflict_concentration.py --collect=pick --pick=P --stack=S --data=data/bc_pick.pt
  python scripts/conflict_concentration.py --collect=stack --pick=P --stack=S --data=data/bc_stack.pt
  python scripts/conflict_concentration.py --pick=P --stack=S --data=data/bc_pick.pt,data/bc_stack.pt --out=logs/x.json
"""
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--pick", required=True)
p.add_argument("--stack", required=True)
p.add_argument("--collect", default="", choices=["", "pick", "stack"])
p.add_argument("--data", required=True, help="output path when collecting; pick,stack paths when analysing")
p.add_argument("--envs", type=int, default=512)
p.add_argument("--rounds", type=int, default=4, help="collection resets; episodes per task = rounds * envs")
p.add_argument("--init", default="pick", choices=["pick", "random"])
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch", type=int, default=2048, help="states per task per step")
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--rank_step", type=int, default=300)
p.add_argument("--rank_window", type=int, default=100, help="average the per-row dot over this many steps up to --rank_step")
p.add_argument("--rank_every", type=int, default=5, help="record the conflict every this many steps inside the window")
p.add_argument("--frac", type=float, default=0.05, help="share of each hidden layer's rows split in b, c and e")
p.add_argument("--rand_draws", type=int, default=5, help="random masks in arm c")
p.add_argument("--null_draws", type=int, default=1000)
p.add_argument("--seed", type=int, default=0, help="train/test episode split")
p.add_argument("--seeds", default="0,1,2", help="training seeds: init (random), batches, masks of c")
p.add_argument("--device", default="cuda")
p.add_argument("--out", default="")
a = p.parse_args()
assert a.rank_step <= a.steps, "--rank_step must be <= --steps"
dev = a.device
H = 50                                # steps per episode (max_episode_steps of both envs)


def held_goal_mask(x):
    return (x[:, 18] > 0.5) & (torch.linalg.norm(x[:, 39:42], dim=1) <= 0.025)


def released_mask(x):
    return (x[:, 18] < 0.5) & (torch.linalg.norm(x[:, 39:42], dim=1) <= 0.025)


from wrwl.experts import load_policy  # noqa: E402

pick, stack = load_policy(a.pick, 42, device=dev), load_policy(a.stack, 48, device=dev)
sp = pick.actor_logstd.exp().flatten().detach()

if a.collect:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    import wrwl.envs.stack_narrow  # noqa: F401
    from wrwl.adapters import stack_to_pick
    from wrwl.experts import flat_state

    if a.collect == "pick":
        env = gym.make("PickCube-v1", num_envs=a.envs, obs_mode="state", robot_uids="panda_wristcam",
                       control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None)
    else:
        env = gym.make("StackCubeNarrow-v1", num_envs=a.envs, obs_mode="state_dict", reward_mode="sparse",
                       control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None)
    base = env.unwrapped
    xs, ys, yr, ks, eps, done = [], [], [], [], [], []
    with torch.no_grad():
        for r in range(a.rounds):     # row = (r * H + t) * envs + env, episode id = r * envs + env
            obs, _ = env.reset(seed=(123 if a.collect == "pick" else 456) + 1000 * r)
            succ = torch.zeros(a.envs, dtype=torch.bool, device=dev)
            for _ in range(H):        # sampled actions, as in training
                if a.collect == "pick":
                    x, net, inp = obs, pick, obs
                else:
                    x48 = flat_state(obs)
                    x, net, inp = stack_to_pick(x48, base.agent.is_grasping(base.cubeA)), stack, x48
                mean = net.actor_mean(inp)
                xs.append(x.float())
                ys.append(mean.clamp(-1, 1))
                yr.append(mean)
                ks.append(~succ)      # drops the first success state and all later ones, as partial resets did
                eps.append(r * a.envs + torch.arange(a.envs, device=dev))
                obs, _, _, _, info = env.step((mean + net.actor_logstd.exp() * torch.randn_like(mean)).clamp(-1, 1))
                succ |= info["success"]
            done.append(succ)
    env.close()
    x, yr, keep = torch.cat(xs), torch.cat(yr), torch.cat(ks)
    succ = torch.cat(done).float().mean().item()
    torch.save({"x": x.cpu(), "y": torch.cat(ys).cpu(), "y_raw": yr.cpu(), "ep": torch.cat(eps).cpu(),
                "keep": keep.cpu(), "success": succ}, a.data)
    n_ep = a.rounds * a.envs
    hg, rl = held_goal_mask(x[keep]), released_mask(x[keep])
    past = yr.abs() > 1
    print(f"{a.collect}: {len(x)} states in {n_ep} episodes, expert success {succ:.3f}, kept {keep.float().mean():.3f} "
          f"(before the first success) -> {a.data}")
    print(f"  kept held@goal {int(hg.sum())} ({hg.sum() / n_ep:.2f}/episode), released near goal {int(rl.sum())} "
          f"({rl.sum() / n_ep:.2f}/episode); ~20% become test states: raise --rounds if that is under ~200")
    print(f"  raw means past +-1 per dim {' '.join(f'{v:.2f}' for v in past.float().mean(0))} | mean overshoot there "
          f"{' '.join(f'{v:.2f}' for v in ((yr.abs() - 1) * past).sum(0) / past.sum(0).clamp_min(1))}")
    raise SystemExit

D = [torch.load(f, map_location="cpu") for f in a.data.split(",")]
X, Y, EP = [], [], []
for d, name in zip(D, ("PickCube", "Stack")):
    k = d["keep"].bool()
    X.append(d["x"][k].float().to(dev))
    Y.append(d["y"][k].float().to(dev))
    EP.append(d["ep"][k])
    print(f"{name:8s} expert success {d['success']:.3f} | {int(d['ep'].max()) + 1} episodes, kept {int(k.sum())} of "
          f"{len(k)} states ({k.float().mean():.3f}; dropped from the first success on)")
HG = [held_goal_mask(x) for x in X]
REL = released_mask(X[1])
gsplit = torch.Generator().manual_seed(a.seed)
tr, te, rk, tr_eval = [], [], [], []
for ep in EP:                         # whole episodes held out; te shuffled so any prefix spans all phases
    ids = ep.unique()
    is_te = torch.isin(ep, ids[torch.randperm(len(ids), generator=gsplit)[: max(1, round(0.2 * len(ids)))]])
    rows_te, rows_tr = is_te.nonzero().squeeze(1), (~is_te).nonzero().squeeze(1)
    rows_te = rows_te[torch.randperm(len(rows_te), generator=gsplit)]
    perm_tr = rows_tr[torch.randperm(len(rows_tr), generator=gsplit)]
    tr.append(rows_tr.to(dev))
    te.append(rows_te.to(dev))
    rk.append(perm_tr[:8192].to(dev))                 # fixed training rows for the conflict gradients
    tr_eval.append(perm_tr[: len(rows_te)].to(dev))   # training rows scored next to the test rows (fit)
data_info = {"kept": [len(x) for x in X], "train": [len(t) for t in tr], "test": [len(t) for t in te],
             "test_held_goal": [int(HG[t][te[t]].sum()) for t in (0, 1)], "test_released": int(REL[te[1]].sum())}
for task, name in ((0, "PickCube"), (1, "Stack")):
    extra = f", released near goal {data_info['test_released']}" if task == 1 else ""
    print(f"{name:8s} split by episode: train {len(tr[task])} rows, test {len(te[task])} rows | test held@goal "
          f"{data_info['test_held_goal'][task]}{extra}")
if min(data_info["test_held_goal"] + [data_info["test_released"]]) < 200:
    print("WARNING: a goal subset has under 200 test states; collect more episodes (--rounds)")


class MTNet(nn.Module):
    """3x256 Tanh MLP on [x, task one-hot]; units in the split masks use a per-task copy of their incoming weights."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(44, 256), nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 8)])
        if a.init == "pick":          # reuse the PickCube actor; the task columns start at 0
            src = [m for m in pick.actor_mean if isinstance(m, nn.Linear)]
            with torch.no_grad():
                for l, s in zip(self.layers, src):
                    l.weight.zero_()
                    l.weight[:, : s.weight.shape[1]].copy_(s.weight)
                    l.bias.copy_(s.bias)
        else:
            for k, l in enumerate(self.layers):
                nn.init.orthogonal_(l.weight, 0.01 * 2 ** 0.5 if k == 3 else 2 ** 0.5)
                nn.init.zeros_(l.bias)
        self.masks = None

    def split(self, masks):
        self.masks = [m.float() for m in masks]
        self.sw = nn.ParameterList([nn.Parameter(l.weight.detach().clone().expand(2, -1, -1).clone()) for l in self.layers])
        self.sb = nn.ParameterList([nn.Parameter(l.bias.detach().clone().expand(2, -1).clone()) for l in self.layers])

    def forward(self, x, task):
        h = torch.cat([x, F.one_hot(torch.full((len(x),), task, device=x.device), 2).float()], 1)
        for k, l in enumerate(self.layers):
            if self.masks is None:
                z = l(h)
            else:
                m = self.masks[k]
                z = h @ (l.weight * (1 - m[:, None]) + self.sw[k][task] * m[:, None]).T \
                    + l.bias * (1 - m) + self.sb[k][task] * m
            h = torch.tanh(z) if k < 3 else z
        return h


def resid(out, y):
    """out - y, but 0 where both are past the same bound (the same executed action); y is clamped, so |y| = 1 there."""
    same = ((out >= 1) & (y >= 1)) | ((out <= -1) & (y <= -1))
    return torch.where(same, torch.zeros_like(out), out - y)


def loss_on(net, task, rows):
    return ((resid(net(X[task][rows], task), Y[task][rows]) / sp) ** 2).mean()


def unit_conflict(net):
    """Per layer, per output row (incoming weights and bias; layer 1 without the per-task one-hot columns): the signed
    <g_pick, g_stack>, |g_pick| and |g_stack| of the full-batch gradients on the fixed training rows rk, plus each
    task's loss and the layer cosine. Only the hidden layers are shared when the heads are per-task."""
    grads, loss = [], []
    for task in (0, 1):
        net.zero_grad()
        l_ = loss_on(net, task, rk[task])
        l_.backward()
        loss.append(l_.item())
        g = []
        for k, l in enumerate(net.layers):
            w = l.weight.grad[:, :42] if k == 0 else l.weight.grad
            g.append(torch.cat([w, l.bias.grad[:, None]], 1).double())
        grads.append(g)
    net.zero_grad()
    out = []
    for g0, g1 in zip(*grads):
        dot = (g0 * g1).sum(1)
        out.append({"dot": dot.cpu(), "n0": g0.norm(dim=1).cpu(), "n1": g1.norm(dim=1).cpu(),
                    "layer_cos": (dot.sum() / (g0.norm() * g1.norm()).clamp_min(1e-30)).item()})
    gn = [sum(g.pow(2).sum() for g in gt[:3]).sqrt().item() for gt in grads]
    return {"loss": loss, "hidden_grad_norm": gn, "layers": out}


def window_avg(recs):
    """Per hidden layer: the per-row dot, |g_pick| and |g_stack| averaged over the recorded window steps."""
    return [{key: torch.stack([r["layers"][k][key] for r in recs]).mean(0) for key in ("dot", "n0", "n1")}
            | {"layer_cos": sum(r["layers"][k]["layer_cos"] for r in recs) / len(recs)} for k in range(3)]


QS = (0.01, 0.05, 0.10)
N_SPLIT = max(1, int(a.frac * 256))


def top_shares(v):
    """Share of the total held by the top 1/5/10% rows; v is (..., n), nonnegative."""
    n = v.shape[-1]
    c = v.sort(-1, descending=True).values.cumsum(-1)
    return torch.stack([c[..., max(1, int(q * n)) - 1] for q in QS], -1) / c[..., -1:].clamp_min(1e-300)


def layer_summary(u, ok, ref, rank_top, gnull):
    """Within-layer conflict statistics of one hidden layer; None when a loss is at rounding level, and no shares
    when the layer's conflict is."""
    if not ok:
        return None
    dot, mag = u["dot"], u["n0"] * u["n1"]
    conf, aln = (-dot).clamp_min(0), dot.clamp_min(0)
    cos = dot / mag.clamp_min(1e-300)
    neg = dot < 0
    C, A = conf.sum().item(), aln.sum().item()
    s = {"C": C, "A": A, "C_frac": C / (C + A) if C + A > 0 else None, "n_neg": int(neg.sum()),
         "frac_cos_lt_-0.3": (cos < -0.3).double().mean().item(),
         "cos_q10_50_90": cos.quantile(torch.tensor([0.1, 0.5, 0.9], dtype=cos.dtype)).tolist(),
         "layer_cos": u["layer_cos"], "share": None, "null_mean": None, "null_p95": None, "align_share": None,
         "jaccard_rank": None}
    if C <= 1e-6 * ref:
        return s
    perm = torch.rand(a.null_draws, len(dot), generator=gnull).argsort(1)
    null = top_shares(dot.abs()[None] * neg[perm])
    top = set(conf.topk(N_SPLIT).indices.tolist())
    s.update(share=top_shares(conf).tolist(), null_mean=null.mean(0).tolist(), null_p95=null.quantile(0.95, 0).tolist(),
             align_share=top_shares(aln).tolist() if A > 0 else None,
             jaccard_rank=len(top & rank_top) / len(top | rank_top) if rank_top else None)
    return s


def summarize(rec, seed, shared_head, rank_layers=None):
    """Per recorded step: losses, gradient norms and the hidden-layer summaries (plus the readout rows when the head
    is shared); guards against rounding-level losses (< 1e-8) and conflict (< 1e-6 of the layer's largest sum |dot|
    over the recorded steps). Jaccard is against the top set of rank_layers (the window average used for b)."""
    gnull = torch.Generator().manual_seed(seed + 3)
    ref = [max(r["layers"][k]["dot"].abs().sum().item() for r in rec.values()) for k in range(3)]
    rank_top = [set((-L["dot"]).clamp_min(0).topk(N_SPLIT).indices.tolist()) for L in rank_layers] \
        if rank_layers is not None else [None] * 3
    out = {}
    for s, r in rec.items():
        ok = min(r["loss"]) >= 1e-8
        out[s] = {"loss": r["loss"], "hidden_grad_norm": r["hidden_grad_norm"], "rounding": not ok,
                  "layers": [layer_summary(r["layers"][k], ok, ref[k], rank_top[k], gnull) for k in range(3)]}
        if shared_head and ok:
            ro = r["layers"][3]
            out[s]["readout"] = {"dot": ro["dot"].tolist(),
                                 "cos": (ro["dot"] / (ro["n0"] * ro["n1"]).clamp_min(1e-300)).tolist()}
    return out


def print_conflict(sm, sm0):
    f3 = lambda v: "  -  /  -  /  -  " if v is None else "/".join(f"{x:.2f}" for x in v)
    for s, r in sm.items():
        flag = " (a loss at rounding level: nothing to measure)" if r["rounding"] else \
            (" (init: do not read)" if s == 0 else "")
        print(f"  step {s:5d} loss pick {r['loss'][0]:.3e} stack {r['loss'][1]:.3e} | hidden |g| pick "
              f"{r['hidden_grad_norm'][0]:.2e} stack {r['hidden_grad_norm'][1]:.2e}{flag}")
        if r["rounding"]:
            continue
        for k, L in enumerate(r["layers"]):
            star = "" if L["share"] is None else "".join("*" if o > n else " " for o, n in zip(L["share"], L["null_p95"]))
            cf = "  -  " if L["C_frac"] is None else f"{L['C_frac']:.3f}"
            jc = "  - " if L["jaccard_rank"] is None else f"{L['jaccard_rank']:.2f}"
            print(f"    L{k + 1} conflict top1/5/10% {f3(L['share'])}{star:3s} null {f3(L['null_mean'])} p95 "
                  f"{f3(L['null_p95'])} | align {f3(L['align_share'])} | C/(C+A) {cf} neg {L['n_neg']:3d}/256 "
                  f"cos<-0.3 {L['frac_cos_lt_-0.3']:.2f} cos q10/50/90 {'/'.join(f'{v:+.2f}' for v in L['cos_q10_50_90'])}"
                  f" layer cos {L['layer_cos']:+.2f} | Jaccard vs rank step {jc}")
        ro = sm0[s].get("readout")
        if ro is not None:
            print(f"    readout rows (a0, the shared head) dot {' '.join(f'{v:+.1e}' for v in ro['dot'])} | cos "
                  f"{' '.join(f'{v:+.2f}' for v in ro['cos'])}")


def arm_masks(st, gr):
    """b, c (list) and e hidden masks from the rank-step rows of run a; every arm adds per-task heads."""
    b, e, c = [], [], [[] for _ in range(a.rand_draws)]
    info = []
    for u in st:
        dot, mag = u["dot"], u["n0"] * u["n1"]
        cos, lm = dot / mag.clamp_min(1e-300), mag.clamp_min(1e-300).log10()
        top = (-dot).clamp_min(0).topk(N_SPLIT).indices
        free = torch.ones(256, dtype=torch.bool)
        free[top] = False
        pool = free & (dot >= 0)
        n_al = int(pool.sum())
        if n_al < N_SPLIT:            # too few aligned rows: add the least conflicting (highest cosine) others
            pool[cos.masked_fill(~free | pool, -2).topk(N_SPLIT - n_al).indices] = True
        em = torch.zeros(256, dtype=torch.bool)
        for i in top[lm[top].argsort(descending=True)]:
            cand = pool.nonzero().squeeze(1)
            j = cand[(lm[cand] - lm[i]).abs().argmin()]
            em[j], pool[j] = True, False
        bm = ~free
        b.append(bm)
        e.append(em)
        for cm in c:
            m = torch.zeros(256, dtype=torch.bool)
            m[torch.randperm(256, generator=gr)[:N_SPLIT]] = True
            cm.append(m)
        mean = lambda v, ms: sum(v[m].mean().item() for m in ms) / len(ms)
        info.append({"b_conflicting": int((dot[bm] < 0).sum()), "e_aligned": int((dot[em] >= 0).sum()),
                     "log10_mag": [mean(lm, [bm]), mean(lm, [em]), mean(lm, [cm[-1] for cm in c])],
                     "cos": [mean(cos, [bm]), mean(cos, [em]), mean(cos, [cm[-1] for cm in c])]})
    head = [torch.ones(8, dtype=torch.bool)]
    return b + head, [m + head for m in c], e + head, info


def evaluate(net):
    res = {}
    with torch.no_grad():
        for task, name in ((0, "pick"), (1, "stack")):
            err = resid(net(X[task][te[task]], task), Y[task][te[task]]).abs() / sp
            am = lambda m: [err[m, :7].mean().item(), err[m, 7].mean().item()] if m.any() else None
            res[name] = am(torch.ones(len(err), dtype=torch.bool, device=dev))
            res[f"{name}_held_goal"] = am(HG[task][te[task]])
            if task == 1:
                res["stack_released"] = am(REL[te[1]])
            fit = resid(net(X[task][tr_eval[task]], task), Y[task][tr_eval[task]]).abs() / sp
            res[f"{name}_train"] = [fit[:, :7].mean().item(), fit[:, 7].mean().item()]
            res[f"{name}_loss"] = loss_on(net, task, te[task]).item()
    res["loss"] = res["pick_loss"] + res["stack_loss"]
    return res


def train(seed, masks=None, record=()):
    torch.manual_seed(seed)
    net = MTNet().to(dev)
    if masks is not None:
        net.split([m.to(dev) for m in masks])
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    gb = torch.Generator().manual_seed(seed + 1)
    rec = {}
    for step in range(a.steps + 1):
        if step in record:
            rec[step] = unit_conflict(net)
        if step == a.steps:
            break
        loss = 0
        for task in (0, 1):
            rows = tr[task][torch.randint(len(tr[task]), (a.batch,), generator=gb).to(dev)]
            loss = loss + loss_on(net, task, rows)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net, rec


if a.init == "pick":
    with torch.no_grad():
        l0 = loss_on(MTNet().to(dev), 0, torch.arange(len(X[0]), device=dev)).item()
    print(f"--init pick: PickCube loss at step 0 {l0:.2e} (the net is the PickCube expert, so ~0)")
    assert l0 < 1e-6, "PickCube targets do not match --pick: wrong checkpoint, adapter or clamp"

seeds = [int(s) for s in a.seeds.split(",")]
steps = sorted(s for s in {0, 100, a.rank_step, 1000, a.steps} if s <= a.steps)
wsteps = sorted(set(range(max(1, a.rank_step - a.rank_window), a.rank_step + 1, a.rank_every)) | {a.rank_step})
assert len(wsteps) >= 4, "the ranking window needs at least 4 recorded steps (--rank_window / --rank_every)"
heads = [torch.zeros(256, dtype=torch.bool)] * 3 + [torch.ones(8, dtype=torch.bool)]
fmt = lambda v: "   -   " if v is None else f"{v[0]:.2f}/{v[1]:.2f}"
res = {"args": vars(a), "data": data_info, "seeds": {}}
for seed in seeds:
    S = res["seeds"][seed] = {}
    net0, rec0 = train(seed, None, steps)
    net_a, rec_all = train(seed, heads, sorted(set(steps) | set(wsteps)))
    rec, recw = {s: rec_all[s] for s in steps}, [rec_all[s] for s in wsteps]
    assert min(min(r["loss"]) for r in recw) >= 1e-8, "a task's loss is at rounding level in the ranking window: rank later"
    avg = window_avg(recw)
    S["conflict"], S["conflict_a0"] = summarize(rec, seed, False, avg), summarize(rec0, seed, True)
    print(f"\nseed {seed}, init={a.init}: cross-task conflict in run a (shared hidden layers, per-task heads); rows of "
          f"each hidden layer ranked within the layer, top 1/5/10% = {', '.join(str(max(1, int(q * 256))) for q in QS)} "
          f"of 256; '*' = above the null's 95th percentile")
    print_conflict(S["conflict"], S["conflict_a0"])
    half = len(recw) // 2
    h1, h2 = window_avg(recw[:half]), window_avg(recw[half:])
    gnull = torch.Generator().manual_seed(seed + 4)
    S["window"] = {"steps": [wsteps[0], wsteps[-1], a.rank_every], "layers": []}
    print(f"ranking window: steps {wsteps[0]}-{wsteps[-1]} every {a.rank_every} ({len(wsteps)} records); stability = "
          f"Jaccard of the top {N_SPLIT} between the window's halves (chance ~{N_SPLIT / 256:.2f}) and the correlation "
          f"of their per-row dots")
    for k in range(3):
        t1, t2 = [set((-h[k]["dot"]).clamp_min(0).topk(N_SPLIT).indices.tolist()) for h in (h1, h2)]
        jac = len(t1 & t2) / len(t1 | t2)
        corr = torch.corrcoef(torch.stack([h1[k]["dot"], h2[k]["dot"]]))[0, 1].item()
        L = layer_summary(avg[k], True, avg[k]["dot"].abs().sum().item(), None, gnull)
        S["window"]["layers"].append({"stability_jaccard": jac, "halves_dot_corr": corr, "summary": L})
        f3 = lambda v: "  -  /  -  /  -  " if v is None else "/".join(f"{x:.2f}" for x in v)
        star = "" if L["share"] is None else "".join("*" if o > n else " " for o, n in zip(L["share"], L["null_p95"]))
        print(f"  L{k + 1} window-avg conflict top1/5/10% {f3(L['share'])}{star:3s} null p95 {f3(L['null_p95'])} | "
              f"C/(C+A) {'  -  ' if L['C_frac'] is None else format(L['C_frac'], '.3f')} | halves Jaccard {jac:.2f} "
              f"dot corr {corr:+.2f}")
    b, c, e, info = arm_masks(avg, torch.Generator().manual_seed(seed + 2))
    S["split"] = {"per_layer": N_SPLIT, "layers": info}
    print(f"split by the window average: {N_SPLIT} rows per hidden layer (readout per-task in every arm but a0)")
    for k, i in enumerate(info):
        warn = "  WARNING: b includes rows without conflict" if i["b_conflicting"] < N_SPLIT else ""
        print(f"  L{k + 1} b rows with dot<0 {i['b_conflicting']:2d}, e rows with dot>=0 {i['e_aligned']:2d} | mean "
              f"log10|g_p||g_s| b/e/c {'/'.join(f'{v:.2f}' for v in i['log10_mag'])} | mean cos b/e/c "
              f"{'/'.join(f'{v:+.2f}' for v in i['cos'])}{warn}")
    arms = [("a0 all shared", None), ("a  task heads", heads), ("b  top conflict", b)] + \
           [(f"c  random #{i}", m) for i, m in enumerate(c)] + [("e  magnitude-matched", e),
                                                                  ("d  separate", [torch.ones_like(m) for m in heads])]
    S["eval"] = {}
    print(f"held-out (test-episode) error after {a.steps} steps, PickCube std units, arm mean / gripper; recovery = "
          f"(L_a - L) / (L_a - L_d) on the test loss")
    for name, masks in arms:
        net = net0 if masks is None else net_a if masks is heads else train(seed, masks)[0]
        S["eval"][name] = evaluate(net)
    gap = S["eval"]["a  task heads"]["loss"] - S["eval"]["d  separate"]["loss"]
    for name, ev in S["eval"].items():
        ev["recovery"] = (S["eval"]["a  task heads"]["loss"] - ev["loss"]) / gap if gap > 1e-12 else None
        rc = "  -  " if ev["recovery"] is None else f"{ev['recovery']:+.2f}"
        print(f"  {name:20s} pick {fmt(ev['pick'])} held@goal {fmt(ev['pick_held_goal'])} fit {fmt(ev['pick_train'])} | "
              f"stack {fmt(ev['stack'])} held@goal {fmt(ev['stack_held_goal'])} released {fmt(ev['stack_released'])} "
              f"fit {fmt(ev['stack_train'])} | test loss {ev['loss']:.4f} recovery {rc}")

groups = {}
for S in res["seeds"].values():
    for name, ev in S["eval"].items():
        groups.setdefault(name.split()[0], []).append(ev)
res["aggregate"] = {}


def msd(v):
    v = torch.tensor([x for x in v if x is not None], dtype=torch.float64)
    return (v.mean().item(), v.std().item() if len(v) > 1 else 0.0) if len(v) else (None, None)


print(f"\nover seeds {seeds} (c: over seeds x {a.rand_draws} masks): mean +- sd")
for g, evs in groups.items():
    agg = {key: msd([ev[key] for ev in evs]) for key in ("loss", "pick_loss", "stack_loss", "recovery")}
    agg["n"] = len(evs)
    res["aggregate"][g] = agg
    rc = "   -   " if agg["recovery"][0] is None else f"{agg['recovery'][0]:+.2f} +- {agg['recovery'][1]:.2f}"
    print(f"  {g:3s} (n={len(evs):2d}) test loss {agg['loss'][0]:.4f} +- {agg['loss'][1]:.4f} (pick "
          f"{agg['pick_loss'][0]:.4f}, stack {agg['stack_loss'][0]:.4f}) | recovery {rc}")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
