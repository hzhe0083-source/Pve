"""Which visited states does each oracle gate anchor on Stack? The true-label gate vs the --oracle_shuffle control.

Rolls a policy out on StackCubeNarrow (sampled actions, as in training), rebuilds both gates exactly as wrwl/ppo.py
does (kNN k=5 on the adapted 42-dim obs scaled by the labelled std, majority vote, fallback to copy beyond the radius),
and reports each gate's copy fraction by phase over the same visited states. Also prints each checkpoint's action std.
  python scripts/gate_phase.py --policies oracle10=runs/diag/runs/stack_oracle_ck_s1/ckpt_10240000.pt,...
"""
import argparse
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

import wrwl.envs.stack_narrow  # noqa: F401
from wrwl.adapters import STACK_GOAL_DZ, stack_to_pick
from wrwl.experts import flat_state, load_policy

p = argparse.ArgumentParser()
p.add_argument("--policies", required=True, help="name=path,...")
p.add_argument("--std-only", default="", help="name=path,... checkpoints whose action std to print, no rollout")
p.add_argument("--labels", default=str(Path.home() / "pve/data/copyloss_stack.pt"))
p.add_argument("--envs", type=int, default=512)
p.add_argument("--seed", type=int, default=1, help="the training seed, which fixes the --oracle_shuffle permutation")
a = p.parse_args()
dev = "cuda"

# the gates, as in wrwl/ppo.py
L = torch.load(a.labels, weights_only=False)
cols = {w: j for j, (s, w) in enumerate(L["branches"]) if s == "Pick"}
v_star = L["Q"][:, 0]
delta = v_star - L["Q"][:, cols[1]]
ok3 = L["t"] + 3 * L["chunk_len"] + 15 <= L["horizon"]
delta = torch.where(ok3, torch.maximum(delta, v_star - L["Q"][:, cols[3]]), delta)
keep = v_star > 0.1
lx = stack_to_pick(L["x"][keep].to(dev), L["grasp"][keep].to(dev))
scale = lx.std(0).clamp_min(1e-2)
lxn = lx / scale
safe_true = (delta[keep] <= 0.1).float().to(dev)
safe_shuf = safe_true[torch.randperm(len(safe_true), generator=torch.Generator().manual_seed(a.seed)).to(dev)]
dd, _ = torch.cdist(lxn, lxn).topk(6, largest=False)
radius = 1.5 * dd[:, 1].quantile(0.95)


def gate(x, safe):
    d, i = torch.cdist(x / scale, lxn).topk(5, largest=False)
    far = d[:, 0] > radius
    return torch.where(far, 1.0, (safe[i].mean(1) >= 0.5).float()), far.float()


def show_std(name, path):
    sd = torch.load(path, map_location="cpu")
    std = sd["actor_logstd"].exp().flatten()
    print(f"  std {name:22s} arm {std[:7].mean():.3f}  gripper {std[7]:.3f}")


print("action std (mean over the 7 arm joints, and the gripper):")
for spec in [s for s in (a.policies + "," + a.std_only).split(",") if s]:
    show_std(*spec.split("=", 1))

env = gym.make("StackCubeNarrow-v1", num_envs=a.envs, obs_mode="state_dict", reward_mode="sparse",
               control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None)
base = env.unwrapped
PHASES = ["free, tcp > 5 cm", "free, tcp <= 5 cm", "held, goal > 5 cm", "held, goal 2.5-5 cm", "held, goal <= 2.5 cm",
          "released near goal"]
for spec in a.policies.split(","):
    name, path = spec.split("=", 1)
    pol = load_policy(path, 42)
    obs, _ = env.reset(seed=123)
    rec = {k: [] for k in ("x", "phase")}
    with torch.no_grad():
        for t in range(50):
            g = base.agent.is_grasping(base.cubeA)
            x = stack_to_pick(flat_state(obs), g)
            pa, pb = base.cubeA.pose.p, base.cubeB.pose.p
            dg = torch.linalg.norm(pa - (pb + torch.tensor([0, 0, STACK_GOAL_DZ], device=dev)), dim=1)
            dt = torch.linalg.norm(base.agent.tcp.pose.p - pa, dim=1)
            ph = torch.where(g, torch.where(dg > 0.05, 2, torch.where(dg > 0.025, 3, 4)),
                             torch.where(dg <= 0.025, 5, torch.where(dt > 0.05, 0, 1)))
            rec["x"].append(x)
            rec["phase"].append(ph)
            mean = pol.actor_mean(x)
            obs, *_ = env.step((mean + pol.actor_logstd.exp() * torch.randn_like(mean)).clamp(-1, 1))
    x, ph = torch.cat(rec["x"]), torch.cat(rec["phase"])
    gt, ft = gate(x, safe_true)
    gs, fs = gate(x, safe_shuf)
    print(f"\n{name}: {len(x)} visited states; copy fraction true {gt.mean():.3f} (fallback {ft.mean():.3f}), "
          f"shuffled {gs.mean():.3f} (fallback {fs.mean():.3f})")
    print(f"  {'phase':22s} {'share':>6s} {'true gate copies':>17s} {'shuffled copies':>16s} {'fallback':>9s}")
    for k, nm in enumerate(PHASES):
        m = ph == k
        if m.sum() == 0:
            continue
        print(f"  {nm:22s} {m.float().mean():6.3f} {gt[m].mean():17.3f} {gs[m].mean():16.3f} {ft[m].mean():9.3f}")
env.close()
