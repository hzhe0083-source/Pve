"""Where does each policy's chain break on StackCubeNarrow? Reach, grasp, lift, carry to the goal, release, success.

Per policy (deterministic mean action, and sampled with the policy's own std): the fraction of episodes that ever
reach cubeA (tcp within 2.5 cm), grasp it, lift it 3 cm, bring it within 2.5 cm of the goal (cubeB + 4 cm), let go
there after a grasp, succeed; and the fraction that knock cubeB more than 2 cm. Policies see StackCube through the
PickCube layout (stack_to_pick), like the step 3 runs.
  python scripts/stack_diag.py --ckpts pick=runs/experts_wc/runs/PickCube_wc_s1/final_ckpt.pt,oracle=...
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

import wrwl.envs.stack_narrow  # noqa: F401
from wrwl.adapters import STACK_GOAL_DZ, stack_to_pick
from wrwl.experts import flat_state, load_policy

p = argparse.ArgumentParser()
p.add_argument("--ckpts", required=True, help="name=path,...")
p.add_argument("--envs", type=int, default=512)
p.add_argument("--out", default=str(Path.home() / "pve/logs/stack_diag.json"))
a = p.parse_args()
dev = "cuda"
env = gym.make("StackCubeNarrow-v1", num_envs=a.envs, obs_mode="state_dict", reward_mode="sparse",
               control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None)
base = env.unwrapped
STAGES = ["reach", "grasp", "lift", "at_goal", "release_at_goal", "success", "knock_B"]


def rollout(pol, sample):
    obs, _ = env.reset(seed=123)
    n = a.envs
    f = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in STAGES}
    b0 = base.cubeB.pose.p.clone()
    grasped_before = torch.zeros(n, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for _ in range(50):
            g = base.agent.is_grasping(base.cubeA)
            mean = pol.actor_mean(stack_to_pick(flat_state(obs), g))
            act = mean + pol.actor_logstd.exp() * torch.randn_like(mean) if sample else mean
            obs, _, _, _, info = env.step(act.clamp(-1, 1))
            pa, pb = base.cubeA.pose.p, base.cubeB.pose.p
            g = base.agent.is_grasping(base.cubeA)
            goal = pb + torch.tensor([0, 0, STACK_GOAL_DZ], device=dev)
            near = torch.linalg.norm(pa - goal, dim=1) < 0.025
            f["reach"] |= torch.linalg.norm(base.agent.tcp.pose.p - pa, dim=1) < 0.025
            f["grasp"] |= g
            f["lift"] |= pa[:, 2] > 0.05
            f["at_goal"] |= near & g
            f["release_at_goal"] |= near & ~g & grasped_before
            f["success"] |= info["success"].bool()
            grasped_before |= g
        f["knock_B"] = torch.linalg.norm(base.cubeB.pose.p[:, :2] - b0[:, :2], dim=1) > 0.02
    return {k: round(v.float().mean().item(), 3) for k, v in f.items()}


res = {}
print(f"{'policy':26s} {'mode':6s}" + "".join(f"{k:>16s}" for k in STAGES))
for spec in a.ckpts.split(","):
    name, path = spec.split("=", 1)
    if not Path(path).exists():
        print("missing", path)
        continue
    pol = load_policy(path, 42)
    for mode in ("mean", "sample"):
        r = rollout(pol, mode == "sample")
        res.setdefault(name, {})[mode] = r
        print(f"{name:26s} {mode:6s}" + "".join(f"{r[k]:16.3f}" for k in STAGES), flush=True)
env.close()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print("saved", a.out)
