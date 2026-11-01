"""Is the part of the update that hurts the old skill also needed by the new one? No retraining: take a checkpoint
fine-tuned on the new skill (Stack) from the PickCube expert, remove from each hidden layer's update either the block
that hurts the old skill to first order (ours: dW - P_out dW P_in, as in scripts/forgetting_geometry.py) or every
direction the old skill's inputs occupy (GPM: dW (I - P_in)), at several energies, and evaluate in simulation:
  old skill  PickCube-v1 through the edited hidden layers with the OLD output layer
  new skill  StackCubeNarrow-v1 (through the PickCube adapter) with the checkpoint's own output layer
Deterministic actions, the same starts for every variant, first episode per env.
  python scripts/posthoc_projection_eval.py --old=P --data=data/bc_pick_c1.pt --ckpts=a.pt,b.pt --out=logs/x.json
"""
import argparse
import json

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch
from torch.func import jacrev, vmap

import wrwl.envs.stack_narrow  # noqa: F401
from wrwl.adapters import stack_to_pick
from wrwl.experts import flat_state, load_policy

p = argparse.ArgumentParser()
p.add_argument("--old", required=True)
p.add_argument("--data", required=True)
p.add_argument("--ckpts", required=True)
p.add_argument("--n", type=int, default=20000)
p.add_argument("--energy", default="0.9,0.95,0.99")
p.add_argument("--envs", type=int, default=256)
p.add_argument("--out", default="")
a = p.parse_args()
torch.manual_seed(0)
dev = "cuda"

old = load_policy(a.old, 42, device=dev)
A = list(old.actor_mean)
LIN = [0, 2, 4]
d = torch.load(a.data, map_location="cpu")
S = d["x"][d["keep"]].float()
S = S[torch.randperm(len(S))[:a.n]].to(dev)
EN = [float(e) for e in a.energy.split(",")]


def aug(x):
    return torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1)


def downstream(i):
    mods = A[i + 1:]

    def f(z):
        h = z
        for m in mods:
            h = m(h)
        return h
    return f


with torch.no_grad():
    X, Z, h = {}, {}, S
    for i, m in enumerate(A):
        if i in LIN:
            X[i], Z[i] = h, m(h)
        h = m(h)
C, Q = {}, {}
for i in LIN:
    xa = aug(X[i])
    C[i] = (xa.T @ xa / len(xa)).detach()
    Ji = torch.cat([vmap(jacrev(downstream(i)))(Z[i][k:k + 2000]).detach() for k in range(0, len(S), 2000)])
    Q[i] = torch.einsum("nao,nap->op", Ji, Ji) / len(Ji)


def top(M, e):
    ev, V = torch.linalg.eigh(M)
    ev, V = ev.flip(0).clamp_min(0), V.flip(1)
    k = int((ev.cumsum(0) / ev.sum() < e).sum()) + 1
    return V[:, :k] @ V[:, :k].T


PRJ = {e: {i: (top(C[i], e), top(Q[i], e)) for i in LIN} for e in EN}
torch.set_grad_enabled(False)

pick_env = gym.make("PickCube-v1", num_envs=a.envs, obs_mode="state", robot_uids="panda_wristcam",
                    control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None)
stack_env = gym.make("StackCubeNarrow-v1", num_envs=a.envs, obs_mode="state_dict", reward_mode="sparse",
                     control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None)


def net(Ws, head):
    def f(x):
        h = x
        for i, m in enumerate(A[:-1]):
            h = h @ Ws[i][0].T + Ws[i][1] if i in LIN else m(h)
        return head(h)
    return f


def success(env, act, stack):
    obs, _ = env.reset(seed=0)
    base = env.unwrapped
    s = None
    for _ in range(50):
        x = stack_to_pick(flat_state(obs), base.agent.is_grasping(base.cubeA)) if stack else obs
        obs, _, _, _, info = env.step(act(x).clamp(-1, 1))
        s = info["success"] if s is None else s | info["success"]
    return s.float().mean().item()


res = {"old_on_pick": success(pick_env, old.actor_mean, False), "runs": {}}
print(f"old PickCube expert: PickCube {res['old_on_pick']:.3f}")
for c in a.ckpts.split(","):
    new = load_policy(c, 42, device=dev)
    Nm = list(new.actor_mean)
    dW = {i: torch.cat([Nm[i].weight - A[i].weight, (Nm[i].bias - A[i].bias)[:, None]], 1) for i in LIN}
    variants = {"none (as trained)": dW}
    for e in EN:
        variants[f"ours {e}"] = {i: dW[i] - PRJ[e][i][1] @ dW[i] @ PRJ[e][i][0] for i in LIN}
        variants[f"gpm {e}"] = {i: dW[i] - dW[i] @ PRJ[e][i][0] for i in LIN}
    variants["all removed (old trunk)"] = {i: torch.zeros_like(dW[i]) for i in LIN}
    rows = {}
    name = "/".join(c.split("/")[-3:-1])
    print(f"\n{name}\n  {'variant':24s} | share of update removed | old skill (PickCube) | new skill (Stack)")
    for v, U in variants.items():
        Ws = {i: (A[i].weight + U[i][:, :-1], A[i].bias + U[i][:, -1]) for i in LIN}
        kept = sum(U[i].pow(2).sum() for i in LIN) / sum(dW[i].pow(2).sum() for i in LIN)
        r = {"removed": 1 - kept.item(), "pick_old_head": success(pick_env, net(Ws, A[-1]), False),
             "stack_new_head": success(stack_env, net(Ws, Nm[-1]), True)}
        rows[v] = r
        print(f"  {v:24s} | {r['removed']:22.3f} | {r['pick_old_head']:19.3f} | {r['stack_new_head']:.3f}")
    res["runs"][c] = rows
pick_env.close()
stack_env.close()
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
