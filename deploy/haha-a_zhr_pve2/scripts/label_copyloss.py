"""D1' labels: the signed copy loss of each source at states of a target task, by branch rollouts (pivot memo step 1).

Target T has an expert pi*_T. Copy candidates X are source experts acting through an observation adapter.
  pool      states from roll-ins by pi*_T, by each X, by X for k in {5,10,15,20} steps and then pi*_T, and by X with
            twice its std (a stand-in for an early learner). Snapshots every 5 steps, before success, with room for
            the longest window.
  branches  at each state s and for each branch (pi*_T, 1 chunk), (X, 1 chunk), (X, 3 chunks), (pi_base, 1 chunk):
            the branch policy acts for its window (chunks of K steps; one sample per chunk, shared by that chunk's
            continuations), then pi*_T finishes the episode. M chunk samples x N continuations per branch; success
            counts only within the remaining horizon.
  labels    Q_b(s) = mean success; V*(s) = Q_{pi*}(s); copy loss Delta_X^w(s) = V*(s) - Q_{X,w}(s). Halves over the
            chunk samples (M/2 + M/2, independent chunk draws and continuations) give a noise ceiling.
Right after a restore the grasp flag is stale, so the first action of every branch uses the observation and grasp flag
cached at roll-in.
"""
import argparse
import math
import os
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

import wrwl.envs.push_dir  # noqa: F401
import wrwl.envs.stack_narrow  # noqa: F401
from wrwl.adapters import ADAPTERS, REL, STACK_GOAL_DZ
from wrwl.experts import flat_state, load_policy

p = argparse.ArgumentParser()
p.add_argument("--target", required=True, help="push<theta> (PushCubeDir) or stack (StackCube)")
p.add_argument("--expert", required=True, help="checkpoint of the target expert")
p.add_argument("--sources", required=True, help="name=checkpoint,... copy candidates")
p.add_argument("--states", type=int, default=1024)
p.add_argument("--envs-per-batch", type=int, default=4096)
p.add_argument("--chunks", type=int, default=8)
p.add_argument("--conts", type=int, default=8)
p.add_argument("--chunk-len", type=int, default=5)
p.add_argument("--windows", default="1,3")
p.add_argument("--rollin-envs", type=int, default=256)
p.add_argument("--base-std", type=float, default=float(math.exp(-0.5)))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()
dev = "cuda"
g_cpu = torch.Generator().manual_seed(a.seed)
torch.manual_seed(a.seed)
M, N, K = a.chunks, a.conts, a.chunk_len
windows = [int(w) for w in a.windows.split(",")]
W = max(windows)

if a.target.startswith("push"):
    env_id, env_kw, family, src_obs, adapter = "PushCubeDir-v1", dict(theta_deg=float(a.target[4:])), "push", 35, ADAPTERS["push"]
else:
    env_id, env_kw, family, src_obs, adapter = "StackCubeNarrow-v1", {}, "stack", 42, ADAPTERS["stack_to_pick"]
H = 50
tgt_obs = 35 if family == "push" else 48
rel_fn = REL[family]


def make(n):
    return gym.make(env_id, num_envs=n, obs_mode="state_dict", reward_mode="sparse", control_mode="pd_joint_delta_pos",
                    sim_backend="physx_cuda", render_mode=None, **env_kw)


def grasp(base):
    return base.agent.is_grasping(base.obj if family == "push" else base.cubeA)


def mover_goal_dist(x):
    if family == "push":
        return torch.linalg.norm(x[:, 25:28] - x[:, 28:31], dim=1)
    dz = torch.zeros_like(x[:, :3])
    dz[:, 2] = STACK_GOAL_DZ
    return torch.linalg.norm(x[:, 45:48] + dz, dim=1)


expert = load_policy(a.expert, tgt_obs)
e_std = expert.actor_logstd.exp().detach()
sources = {}
for item in a.sources.split(","):
    name, path = item.split("=")
    pol = load_policy(path, src_obs)
    sources[name] = (pol, pol.actor_logstd.exp().detach())


def src_act(name, x, g, noise, scale=1.0):
    pol, std = sources[name]
    return pol.actor_mean(adapter(x, g)) + scale * std * noise


# ---------------- roll-ins: the state pool ----------------
kinds = ["expert"] + [f"{k}:{s}" for s in sources for k in ("src", "switch", "noisy")]
share = {"expert": 0.25}
for k in kinds[1:]:
    share[k] = 0.75 / (len(kinds) - 1)
R = a.rollin_envs
pool = []          # (kind, t, state_dict row, x row, g row)
for ki, kind in enumerate(kinds):
    env = make(R)
    base = env.unwrapped
    obs, _ = env.reset(seed=1000 * a.seed + ki)
    succ = torch.zeros(R, dtype=torch.bool, device=dev)
    switch_at = torch.tensor([5, 10, 15, 20])[torch.randint(0, 4, (R,), generator=g_cpu)].to(dev)
    cand = []
    with torch.no_grad():
        for t in range(H):
            x, g = flat_state(obs), grasp(base)
            if t % 5 == 0 and t + W * K < H:
                ok = (~succ).nonzero().flatten()
                if len(ok):
                    sd = base.get_state_dict()
                    cand.append((t, ok.cpu(), {k_: {n_: v[ok].cpu() for n_, v in d.items()} for k_, d in sd.items()},
                                 x[ok].cpu(), g[ok].cpu()))
            noise = torch.randn(R, 8, device=dev)
            a_exp = expert.actor_mean(x) + e_std * noise
            if kind == "expert":
                act = a_exp
            else:
                how, name = kind.split(":")
                a_src = src_act(name, x, g, noise, 2.0 if how == "noisy" else 1.0)
                act = torch.where((t < switch_at)[:, None], a_src, a_exp) if how == "switch" else a_src
            obs, _, _, _, info = env.step(act.clamp(-1, 1))
            succ |= info["success"].bool()
    env.close()
    torch.cuda.empty_cache()
    rows = [(t, j) for t, ok, *_ in cand for j in range(len(ok))]
    by_t = {t: (sd, x, g) for t, _, sd, x, g in cand}
    want = min(len(rows), round(share[kind] * a.states))
    for i in torch.randperm(len(rows), generator=g_cpu)[:want].tolist():
        t, j = rows[i]
        sd, x, g = by_t[t]
        pool.append((kind, t, {k_: {n_: v[j:j + 1] for n_, v in d.items()} for k_, d in sd.items()}, x[j], g[j]))
    print(f"roll-in {kind:20s} candidates={len(rows)} kept={want}", flush=True)

# ---------------- branches ----------------
branches = [("expert", 1)] + [(s, w) for s in sources for w in windows] + [("base", 1)]
nb = len(branches)
per_state = nb * M * N
SB = max(1, a.envs_per_batch // per_state)
E = SB * per_state
S = len(pool)
env = make(E)
base = env.unwrapped
env.reset(seed=a.seed)
Q = torch.zeros(S, nb)
Qh = torch.zeros(S, 2, nb)
progress = torch.zeros(S, nb)
for b0 in range(0, S, SB):
    idx = list(range(b0, min(S, b0 + SB)))
    idx_pad = idx + [idx[0]] * (SB - len(idx))
    batch = {k_: {n_: torch.cat([pool[i][2][k_][n_] for i in idx_pad]).repeat_interleave(per_state, 0).to(dev)
                  for n_ in pool[idx_pad[0]][2][k_]} for k_ in pool[idx_pad[0]][2]}
    obs, _ = env.reset(options=dict(reset_to_env_states=dict(env_states=batch)))
    x0 = torch.stack([pool[i][3] for i in idx_pad]).to(dev).repeat_interleave(per_state, 0)
    g0 = torch.stack([pool[i][4] for i in idx_pad]).to(dev).repeat_interleave(per_state, 0)
    t0 = torch.tensor([pool[i][1] for i in idx_pad], device=dev).repeat_interleave(per_state)
    remaining = H - t0
    noise_c = torch.randn(SB, nb, M, 1, W * K, 8, device=dev)        # one chunk sample, shared by its continuations
    d0 = mover_goal_dist(x0).view(SB, nb, M, N)[..., 0]
    dK = torch.zeros(SB, nb, M, device=dev)
    success = torch.zeros(E, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for t in range(H):
            if t == 0:
                x, g = x0, g0
            else:
                x, g = flat_state(obs), grasp(base)
            if t == K:
                dK = mover_goal_dist(x).view(SB, nb, M, N)[..., 0]
            mu_e = expert.actor_mean(x)
            act = mu_e + e_std * torch.randn_like(mu_e)
            act5 = act.view(SB, nb, M, N, 8)
            x5, g5, mu5 = x.view(SB, nb, M, N, -1), g.view(SB, nb, M, N), mu_e.view(SB, nb, M, N, 8)
            for bi, (pol, w) in enumerate(branches):
                if t >= w * K:
                    continue
                nz = noise_c[:, bi, :, :, t].expand(SB, M, N, 8)
                if pol == "expert":
                    act5[:, bi] = mu5[:, bi] + e_std * nz
                elif pol == "base":
                    act5[:, bi] = a.base_std * nz
                else:
                    xb, gb = x5[:, bi].reshape(-1, x.shape[1]), g5[:, bi].reshape(-1)
                    act5[:, bi] = src_act(pol, xb, gb, nz.reshape(-1, 8)).view(SB, M, N, 8)
            obs, _, _, _, info = env.step(act5.view(E, 8).clamp(-1, 1))
            success |= info["success"].bool() & (t < remaining)
    s5 = success.float().view(SB, nb, M, N)
    n = len(idx)
    Q[idx] = s5.mean((2, 3))[:n].cpu()
    Qh[idx, 0] = s5[:, :, : M // 2].mean((2, 3))[:n].cpu()     # halves over independent chunk samples
    Qh[idx, 1] = s5[:, :, M // 2:].mean((2, 3))[:n].cpu()
    progress[idx] = (d0 - dK).mean(2)[:n].cpu()
    if (b0 // SB) % 10 == 0:
        print(f"  batch {b0 // SB + 1}/{(S + SB - 1) // SB}: mean Q = "
              + " ".join(f"{p_}{w}={Q[idx, bi].mean():.2f}" for bi, (p_, w) in enumerate(branches)), flush=True)
env.close()

xs = torch.stack([r[3] for r in pool])
gs = torch.stack([r[4] for r in pool])
out = dict(target=a.target, family=family, branches=branches, kinds=[r[0] for r in pool],
           t=torch.tensor([r[1] for r in pool]), x=xs, grasp=gs, rel=rel_fn(xs, gs), Q=Q, Q_halves=Qh,
           progress=progress, horizon=H, chunk_len=K, M=M, N=N, expert=a.expert, sources=a.sources)
v = Q[:, 0]
print(f"{a.target}: {S} states, V* mean {v.mean():.2f}, V*>0.1: {(v > 0.1).float().mean():.2f}")
for bi, (p_, w) in enumerate(branches[1:-1], 1):
    d = (v - Q[:, bi])[v > 0.1]
    print(f"  copy {p_} for {w} chunk(s): Delta mean {d.mean():+.3f}, P(Delta<0.1) {(d < 0.1).float().mean():.2f}, "
          f"P(Delta>=0.3) {(d >= 0.3).float().mean():.2f}")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(out, a.out + ".tmp")
os.replace(a.out + ".tmp", a.out)           # atomic, since queued jobs wait for this file to appear
print("saved", a.out)
