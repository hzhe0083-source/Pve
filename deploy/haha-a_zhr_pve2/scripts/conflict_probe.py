"""Is the conflict between a reused PickCube expert and the StackCube task local? Offline, no training.

On StackCubeNarrow (seen through the PickCube adapter) we roll out the StackCube expert and the PickCube expert and,
on the states they visit, compare the two experts by task phase:
  conflict   the gap between the two experts' action means in units of the PickCube expert's std (the copy term's own
             weighting), averaged over the 7 arm joints and for the gripper; the gripper command of each expert
             (+1 open, -1 close) and how often the two disagree in sign
  probe      can the StackCube expert's actions be read linearly off the PickCube actor's last hidden layer (reuse the
             'impressions', learn only a new readout)? Ridge fit on 80% of the states, error on the rest, by phase and
             in the same std units, against two controls: a randomly initialised network of the same shape and the raw
             adapted state
  python scripts/conflict_probe.py --pick=runs/experts_wc/runs/PickCube_wc_s1/final_ckpt.pt \
      --stack=runs/experts/runs/StackCube/final_ckpt.pt --out=logs/conflict_probe.json
"""
import argparse
import json
import types

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

import wrwl.envs.stack_narrow  # noqa: F401
from wrwl.adapters import STACK_GOAL_DZ, stack_to_pick
from wrwl.experts import Agent, flat_state, load_policy

p = argparse.ArgumentParser()
p.add_argument("--pick", required=True)
p.add_argument("--stack", required=True)
p.add_argument("--envs", type=int, default=512)
p.add_argument("--ridge", type=float, default=1e-2)
p.add_argument("--out", default="")
a = p.parse_args()
dev = "cuda"
torch.manual_seed(0)

pick, stack = load_policy(a.pick, 42), load_policy(a.stack, 48)
spaces = types.SimpleNamespace(single_observation_space=types.SimpleNamespace(shape=(42,)),
                               single_action_space=types.SimpleNamespace(shape=(8,)))
rand = Agent(spaces).to(dev).eval()     # same 3x256 Tanh actor, orthogonal init, never trained
sp = pick.actor_logstd.exp().flatten()
print(f"action std  PickCube arm {sp[:7].mean():.3f} gripper {sp[7]:.3f} | StackCube arm "
      f"{stack.actor_logstd.exp().flatten()[:7].mean():.3f} gripper {stack.actor_logstd.exp().flatten()[7]:.3f}")

env = gym.make("StackCubeNarrow-v1", num_envs=a.envs, obs_mode="state_dict", reward_mode="sparse",
               control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None)
base = env.unwrapped
PHASES = ["free, tcp > 5 cm", "free, tcp <= 5 cm", "held, goal > 5 cm", "held, goal 2.5-5 cm", "held, goal <= 2.5 cm",
          "released near goal"]


def rollout(who, seed):
    """Sampled actions, as in training; returns the raw and adapted states, their phases and the success rate."""
    obs, _ = env.reset(seed=seed)
    x48s, x42s, phs = [], [], []
    succ = torch.zeros(a.envs, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for _ in range(50):
            g = base.agent.is_grasping(base.cubeA)
            x48 = flat_state(obs)
            x42 = stack_to_pick(x48, g)
            pa, pb = base.cubeA.pose.p, base.cubeB.pose.p
            dg = torch.linalg.norm(pa - (pb + torch.tensor([0, 0, STACK_GOAL_DZ], device=dev)), dim=1)
            dt = torch.linalg.norm(base.agent.tcp.pose.p - pa, dim=1)
            ph = torch.where(g, torch.where(dg > 0.05, 2, torch.where(dg > 0.025, 3, 4)),
                             torch.where(dg <= 0.025, 5, torch.where(dt > 0.05, 0, 1)))
            x48s.append(x48)
            x42s.append(x42)
            phs.append(ph)
            net, x = (stack, x48) if who == "stack" else (pick, x42)
            mean = net.actor_mean(x)
            obs, _, _, _, info = env.step((mean + net.actor_logstd.exp() * torch.randn_like(mean)).clamp(-1, 1))
            succ |= info["success"]
    return torch.cat(x48s), torch.cat(x42s), torch.cat(phs), succ.float().mean().item()


res = {"phases": PHASES}
X48, X42, PH, SRC = [], [], [], []
for who, seed in (("stack", 123), ("pick", 456)):
    x48, x42, ph, sr = rollout(who, seed)
    print(f"{who} expert rollout: success {sr:.3f}, {len(ph)} states")
    res[f"{who}_success"] = sr
    X48.append(x48), X42.append(x42), PH.append(ph), SRC.append(torch.full_like(ph, who == "stack"))
X48, X42, PH, SRC = torch.cat(X48), torch.cat(X42), torch.cat(PH), torch.cat(SRC).bool()
env.close()

with torch.no_grad():
    ys = stack.actor_mean(X48).clamp(-1, 1)          # what the new task wants
    yp = pick.actor_mean(X42).clamp(-1, 1)           # what the reused policy does
    z = (ys - yp).abs() / sp
    feats = {"pick features": pick.actor_mean[:-1](X42), "random features": rand.actor_mean[:-1](X42), "raw state": X42}

n = len(PH)
perm = torch.randperm(n, device=dev)
tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
probe_z = {}
for name, h in feats.items():
    h = h.double()
    mu, sd = h[tr].mean(0), h[tr].std(0).clamp_min(1e-6)
    hb = torch.cat([(h - mu) / sd, torch.ones(n, 1, device=dev, dtype=h.dtype)], 1)
    A = hb[tr].T @ hb[tr] + a.ridge * len(tr) * torch.eye(hb.shape[1], device=dev, dtype=h.dtype)
    W = torch.linalg.solve(A, hb[tr].T @ ys[tr].double())
    probe_z[name] = ((hb @ W).float() - ys).abs() / sp
    r2 = 1 - ((hb[te] @ W).float() - ys[te]).pow(2).sum(0) / (ys[te] - ys[te].mean(0)).pow(2).sum(0)
    print(f"probe {name:16s} held-out R^2  arm {r2[:7].mean():.3f}  gripper {r2[7]:.3f}")
    res[f"r2_{name}"] = {"arm": r2[:7].mean().item(), "gripper": r2[7].item()}

test = torch.zeros(n, dtype=torch.bool, device=dev)
test[te] = True
print("\ngap to the StackCube expert's action mean, in PickCube std units (arm = mean over 7 joints); "
      "'as is' = PickCube head unchanged, the probes on held-out states only")
print(f"  {'phase':22s} {'share':>6s} {'grip Stack':>10s} {'grip Pick':>9s} {'sign diff':>9s} "
      f"{'as is arm':>9s} {'grip':>6s} | {'probe pick arm':>14s} {'grip':>6s} | {'random arm':>10s} {'grip':>6s} | "
      f"{'raw arm':>7s} {'grip':>6s}")
res["by_phase"] = {}
for k, nm in enumerate(PHASES):
    m = PH == k
    if m.sum() == 0:
        continue
    mt = m & test
    row = {"share": m.float().mean().item(), "share_from_stack_rollout": SRC[m].float().mean().item(),
           "grip_stack": ys[m, 7].mean().item(), "grip_pick": yp[m, 7].mean().item(),
           "grip_sign_diff": ((ys[m, 7] > 0) != (yp[m, 7] > 0)).float().mean().item(),
           "as_is": [z[m, :7].mean().item(), z[m, 7].mean().item()]}
    for name in feats:
        row[name] = [probe_z[name][mt, :7].mean().item(), probe_z[name][mt, 7].mean().item()]
    res["by_phase"][nm] = row
    print(f"  {nm:22s} {row['share']:6.3f} {row['grip_stack']:10.2f} {row['grip_pick']:9.2f} "
          f"{row['grip_sign_diff']:9.2f} {row['as_is'][0]:9.2f} {row['as_is'][1]:6.2f} | "
          f"{row['pick features'][0]:14.2f} {row['pick features'][1]:6.2f} | {row['random features'][0]:10.2f} "
          f"{row['random features'][1]:6.2f} | {row['raw state'][0]:7.2f} {row['raw state'][1]:6.2f}")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
