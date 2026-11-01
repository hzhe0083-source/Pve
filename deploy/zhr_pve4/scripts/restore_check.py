"""Restore-fidelity check for ManiSkill3 GPU sim (week-1 gate G1, item 2).

Runs every env for --prefix random steps, snapshots the full state, plays a fixed random continuation (run A),
restores the snapshot with reset_to_env_states, replays the same continuation (run B) and compares outcomes.
"""
import argparse

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

p = argparse.ArgumentParser()
p.add_argument("--env", default="PushCube-v1")
p.add_argument("--num-envs", type=int, default=1024)
p.add_argument("--prefix", type=int, default=10)
p.add_argument("--horizon", type=int, default=30)
a = p.parse_args()

env = gym.make(a.env, num_envs=a.num_envs, obs_mode="state", control_mode="pd_joint_delta_pos",
               sim_backend="physx_cuda", render_mode=None)
base = env.unwrapped
env.reset(seed=0)
g = torch.Generator(device="cuda").manual_seed(0)
act_dim = env.action_space.shape[-1]


def rand_actions(n):
    return torch.rand(n, a.num_envs, act_dim, device="cuda", generator=g) * 2 - 1


for act in rand_actions(a.prefix):
    env.step(act)
snapshot = base.get_state_dict()
elapsed = base._elapsed_steps.clone()
continuation = rand_actions(a.horizon)


def rollout():
    success = torch.zeros(a.num_envs, dtype=torch.bool, device="cuda")
    for act in continuation:
        _, _, _, _, info = env.step(act)
        success |= info["success"].bool()
    final = base.get_state_dict()
    return success, final


succ_a, final_a = rollout()
env.reset(options=dict(reset_to_env_states=dict(env_states=snapshot)))
base._elapsed_steps[:] = elapsed
succ_b, final_b = rollout()

print(f"{a.env} envs={a.num_envs} prefix={a.prefix} horizon={a.horizon}")
print(f"  success rate A={succ_a.float().mean():.3f} B={succ_b.float().mean():.3f} "
      f"agreement={(succ_a == succ_b).float().mean():.4f}")
for name, pose_a in final_a["actors"].items():
    d = (pose_a[:, :3] - final_b["actors"][name][:, :3]).norm(dim=-1)
    print(f"  actor {name}: pos diff mean={d.mean() * 1e3:.3f}mm max={d.max() * 1e3:.3f}mm "
          f"share<1mm={(d < 1e-3).float().mean():.4f}")
for name, q_a in final_a["articulations"].items():
    d = (q_a - final_b["articulations"][name]).abs().max(dim=-1).values
    print(f"  articulation {name}: max abs diff mean={d.mean():.2e} max={d.max():.2e}")
env.close()
