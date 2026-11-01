"""Smoke test for the unified multi-task env: shapes, finiteness, slot masks and throughput with random actions."""
import argparse
import time

import torch

from wrwl.envs import OBS_DIM, TASKS, MultiTaskEnv

p = argparse.ArgumentParser()
p.add_argument("--tasks", default=",".join(TASKS))
p.add_argument("--num-envs-per-task", type=int, default=256)
p.add_argument("--steps", type=int, default=120)
a = p.parse_args()
tasks = a.tasks.split(",")

env = MultiTaskEnv(tasks, a.num_envs_per_task)
obs = env.reset()
assert obs.shape == (env.num_envs, OBS_DIM) and torch.isfinite(obs).all(), obs.shape
for i, t in enumerate(tasks):
    o = obs[i * env.n:(i + 1) * env.n]
    print(f"{t:22s} tool_present={o[0, 25 + 11 + 11].item():.0f} target_present={o[0, 25 + 11 + 12 + 10].item():.0f} "
          f"goal_present={o[0, 25 + 11 + 12 + 11 + 3].item():.0f} mover_half={o[0, 32:35].tolist()}")

torch.cuda.synchronize()
t0 = time.time()
succ_any = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
for _ in range(a.steps):
    act = torch.rand(env.num_envs, env.act_dim, device=env.device) * 2 - 1
    obs, rew, term, trunc, succ, final = env.step(act)
    assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    succ_any |= succ
torch.cuda.synchronize()
dt = time.time() - t0
free, total = torch.cuda.mem_get_info()
print(f"tasks={len(tasks)} envs={env.num_envs} obs_dim={OBS_DIM} SPS={env.num_envs * a.steps / dt:.0f} "
      f"gpu_used={(total - free) / 2**30:.2f}GiB random_success={succ_any.float().mean():.3f}")
env.close()
