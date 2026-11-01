"""ManiSkill3 GPU-sim smoke test: env-step throughput and GPU memory with a small policy in the loop.

Week-1 calibration for the WRWL experiments; prints one summary line per run.
"""
import argparse
import time

import gymnasium as gym
import mani_skill.envs  # noqa: F401  registers the tasks
import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

p = argparse.ArgumentParser()
p.add_argument("--env", default="PushCube-v1")
p.add_argument("--num-envs", type=int, default=2048)
p.add_argument("--steps", type=int, default=200)
a = p.parse_args()

env = gym.make(a.env, num_envs=a.num_envs, obs_mode="state", control_mode="pd_joint_delta_pos",
               sim_backend="physx_cuda", render_mode=None)
env = ManiSkillVectorEnv(env, a.num_envs, ignore_terminations=False, record_metrics=True)
obs, _ = env.reset(seed=0)
act_dim = env.single_action_space.shape[0]
policy = torch.nn.Sequential(
    torch.nn.Linear(obs.shape[-1], 256), torch.nn.Tanh(),
    torch.nn.Linear(256, 256), torch.nn.Tanh(),
    torch.nn.Linear(256, act_dim),
).cuda()

with torch.no_grad():
    for _ in range(10):
        obs, *_ = env.step(torch.tanh(policy(obs)))
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(a.steps):
        obs, rew, term, trunc, info = env.step(torch.tanh(policy(obs)))
    torch.cuda.synchronize()
    dt = time.time() - t0

free, total = torch.cuda.mem_get_info()
print(f"{a.env} envs={a.num_envs} steps={a.steps} obs_dim={obs.shape[-1]} act_dim={act_dim} "
      f"SPS={a.num_envs * a.steps / dt:.0f} gpu_used={(total - free) / 2**30:.2f}GiB/{total / 2**30:.1f}GiB")
env.close()
