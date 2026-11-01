"""A matched-coverage random region: the locality control for the done-probe seam.

Rolls the source out on StackCubeNarrow (through the PickCube adapter), takes its frozen hidden layer, draws a random
direction and sets the threshold so the random half-space fires on the same share of these states as the done-probe
(--probe). The file has the probe's format, so --seam_probe accepts it: same amount of state space, random location.
Also prints how much of the hand-written region (cube held within 2.5 cm of the goal) each region covers.
  python scripts/make_random_probe.py --src=... --probe=data/done_probe_pick_c1.pt --out=data/random_probe_c1.pt
"""
import argparse

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

import wrwl.envs.stack_narrow  # noqa: F401
from wrwl.adapters import stack_to_pick
from wrwl.experts import flat_state, load_policy

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--probe", required=True)
p.add_argument("--envs", type=int, default=512)
p.add_argument("--batches", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()
dev = "cuda"
src = load_policy(a.src, 42)
trunk = src.actor_mean[:-1]
pr = torch.load(a.probe)
pw, pb = pr["w"].to(dev), pr["b"].to(dev)
env = gym.make("StackCubeNarrow-v1", num_envs=a.envs, obs_mode="state_dict", reward_mode="sparse",
               control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None)
base = env.unwrapped
H, R = [], []
with torch.no_grad():
    for b in range(a.batches):
        obs, _ = env.reset(seed=4000 + b)
        for _ in range(50):
            x = stack_to_pick(flat_state(obs), base.agent.is_grasping(base.cubeA))
            H.append(trunk(x))
            R.append((x[:, 18] > 0.5) & (torch.linalg.norm(x[:, 39:42], dim=1) <= 0.025))
            mean = src.actor_mean(x)
            obs, *_ = env.step((mean + src.actor_logstd.exp() * torch.randn_like(mean)).clamp(-1, 1))
env.close()
H, R = torch.cat(H), torch.cat(R)
fires = (H @ pw + pb) > 0
cover = fires.float().mean()
g = torch.Generator(device=dev).manual_seed(a.seed)
w = torch.randn(H.shape[1], device=dev, generator=g) / H.shape[1] ** 0.5
proj = H @ w
b = -torch.quantile(proj, 1 - cover.item())       # fires on the same share of states as the probe
rnd = (proj + b) > 0
print(f"{len(H)} states: done-probe fires on {cover:.3f}, random region on {rnd.float().mean():.3f}; share of the "
      f"hand-written region covered: probe {(fires & R).sum() / R.sum():.3f}, random {(rnd & R).sum() / R.sum():.3f}")
torch.save({"w": w.cpu(), "b": b.reshape(1).cpu(), "src": a.src, "kind": "random", "seed": a.seed}, a.out)
print("saved", a.out)
