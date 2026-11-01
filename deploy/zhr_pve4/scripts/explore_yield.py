"""How often does each noise layout on the frozen source policy stumble into a target success, before any learning?

The 'exploration yield' check suggested by the feasibility review: the frozen source actor acts in the target task with
its own action means, and only the action noise differs between layouts. There is no training, hence no seed coin
flip: each layout's first-success yield is measured directly over many episodes. The noise changes only --dims
(gripper, arm or all); the other dims keep the source's std everywhere.
  source       the source expert's own std
  global:S     max(source, S) everywhere (the lab's a2r recipe uses S = 0.6 for the gripper on release tasks)
  seam:S       max(source, S) only in the hand-written region (--seam_rule pick: cube held within --radius of the
               goal; push: cube within --radius of the goal in xy)
  boost:B      source x e^B in that region (the pilot's --explore_boost=B)
  pseam:S / pboost:B   the same where the source's done-probe fires (--probe, scripts/train_done_probe.py)
  lseam:S / lpseam:S   seam / pseam latched: from the first step the region fires to the episode's end (the pick rule
               and the probe need the cube held, so unlatched the floor stops as soon as the gripper opens)
Per layout: success (any step) and the share of steps spent in the hand-written region; for the pick rule also how
often the cube was let go within --radius of the goal.
  python scripts/explore_yield.py --pick=runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt --probe=data/done_probe_pick_c1.pt
  python scripts/explore_yield.py --pick=runs/experts/runs/PushCube_c1/final_ckpt.pt --env=PushRetreat-v1 --adapter=none \
      --obs_dim=35 --seam_rule=push --radius=0.1 --dims=arm --layouts=source,global:0.3,seam:0.3,pseam:0.3 --probe=...
"""
import argparse
import json
import math

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

import wrwl.envs.place_cube  # noqa: F401
import wrwl.envs.push_retreat  # noqa: F401
import wrwl.envs.stack_narrow  # noqa: F401
from wrwl.adapters import stack_to_pick
from wrwl.experts import flat_state, load_policy

p = argparse.ArgumentParser()
p.add_argument("--pick", required=True, help="the source expert checkpoint")
p.add_argument("--env", default="StackCubeNarrow-v1")
p.add_argument("--env_kwargs", default="{}")
p.add_argument("--adapter", default="stack_to_pick", choices=["stack_to_pick", "none"])
p.add_argument("--obs_dim", type=int, default=42)
p.add_argument("--seam_rule", default="pick", choices=["pick", "push"])
p.add_argument("--dims", default="gripper", choices=["gripper", "arm", "all"])
p.add_argument("--layouts", default="source,global:0.6,seam:0.6,seam:1.0,boost:1.2,global:1.0")
p.add_argument("--envs", type=int, default=512)
p.add_argument("--batches", type=int, default=8, help="episodes per layout = envs x batches")
p.add_argument("--radius", type=float, default=0.025)
p.add_argument("--probe", default="", help="done-probe for the pseam / pboost layouts")
p.add_argument("--out", default="")
a = p.parse_args()
dev = "cuda"

src = load_policy(a.pick, a.obs_dim)
log_src = src.actor_logstd.detach().flatten()
n_act = len(log_src)
dmask = torch.zeros(n_act, dtype=torch.bool, device=dev)
if a.dims == "gripper":
    dmask[-1] = True
elif a.dims == "arm":
    dmask[:-1] = True
else:
    dmask[:] = True
if a.probe:
    pr = torch.load(a.probe)
    pw, pb = pr["w"].to(dev), pr["b"].to(dev)
stack = a.adapter == "stack_to_pick"
env = gym.make(a.env, num_envs=a.envs, obs_mode="state_dict" if stack else "state", reward_mode="sparse",
               control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None, **json.loads(a.env_kwargs))
base = env.unwrapped


def region(x):
    if a.seam_rule == "push":
        return torch.linalg.norm(x[:, 28:30] - x[:, 25:27], dim=1) < a.radius
    return (x[:, 18] > 0.5) & (torch.linalg.norm(x[:, 39:42], dim=1) <= a.radius)


def layout_logstd(kind, val, seam):
    """Per-state log-std for one layout: the source's, changed on --dims (everywhere or where `seam` holds)."""
    ls = log_src.expand(len(seam), -1).clone()
    sel = dmask.expand_as(ls).clone()
    if kind in ("seam", "pseam", "boost", "pboost", "lseam", "lpseam"):
        sel &= seam.unsqueeze(1)
    if kind in ("global", "seam", "pseam", "lseam", "lpseam"):
        return torch.where(sel, ls.clamp(min=math.log(val)), ls)
    if kind in ("boost", "pboost"):
        return ls + val * sel.float()
    return ls


res = {}
print(f"source std: arm {log_src[:-1].exp().mean():.3f}, gripper {log_src[-1].exp():.3f}; {a.env}, noise on {a.dims}, "
      f"{a.envs * a.batches} episodes per layout")
print(f"  {'layout':12s} {'success':>8s} {'let go at goal':>14s} {'steps in region':>15s}")
for spec in a.layouts.split(","):
    kind, _, v = spec.partition(":")
    val = float(v) if v else 0.0
    succ, rel, in_reg, n = 0.0, 0.0, 0.0, 0
    for b in range(a.batches):
        obs, _ = env.reset(seed=1000 + b)       # the same initial states for every layout
        s_any = torch.zeros(a.envs, dtype=torch.bool, device=dev)
        rel_goal = torch.zeros_like(s_any)
        prev_g = torch.zeros_like(s_any)
        lat = torch.zeros_like(s_any)
        with torch.no_grad():
            for _ in range(50):
                if stack:
                    g = base.agent.is_grasping(base.cubeA)
                    x = stack_to_pick(flat_state(obs), g)
                else:
                    x = obs
                    g = x[:, 18] > 0.5 if a.seam_rule == "pick" else torch.zeros_like(s_any)
                reg = region(x)
                if a.seam_rule == "pick":
                    rel_goal |= prev_g & ~g & (torch.linalg.norm(x[:, 39:42], dim=1) <= a.radius)
                prev_g = g
                in_reg += reg.float().sum().item()
                seam = (src.actor_mean[:-1](x) @ pw + pb) > 0 if kind in ("pseam", "pboost", "lpseam") else reg
                if kind in ("lseam", "lpseam"):
                    lat |= seam
                    seam = lat
                mean = src.actor_mean(x)
                std = layout_logstd(kind, val, seam).exp()
                obs, _, _, _, info = env.step((mean + std * torch.randn_like(mean)).clamp(-1, 1))
                s_any |= info["success"]
        succ += s_any.float().sum().item()
        rel += rel_goal.float().sum().item()
        n += a.envs
    row = {"success": succ / n, "let_go_at_goal": rel / n, "steps_in_region_per_episode": in_reg / n}
    res[spec] = row
    print(f"  {spec:12s} {row['success']:8.4f} {row['let_go_at_goal']:14.4f} {row['steps_in_region_per_episode']:15.2f}")
env.close()
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
