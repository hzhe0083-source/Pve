"""Distil the single-task experts into one shared network (BC + DAgger) and report closed-loop success per task."""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch
from mani_skill.utils.registration import REGISTERED_ENVS

from wrwl.envs.unified import TASK_DIMS, unify
from wrwl.experts import flat_state, load_expert
from wrwl.shared import SharedPolicy

p = argparse.ArgumentParser()
p.add_argument("--rollouts", default=str(Path.home() / "pve/data/rollouts.pt"))
p.add_argument("--out", default=str(Path.home() / "pve/runs/distill/shared.pt"))
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--dagger-iters", type=int, default=3)
p.add_argument("--dagger-envs", type=int, default=256)
p.add_argument("--eval-envs", type=int, default=256)
p.add_argument("--tasks", default="", help="comma-separated subset of the rollout tasks, e.g. the source library")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--drop-task-id", action="store_true", help="zero the task one-hot, so tasks are told apart by their slots")
a = p.parse_args()
dev = "cuda"
torch.manual_seed(a.seed)
data = torch.load(a.rollouts, weights_only=False)
if a.tasks:
    data = {t: data[t] for t in a.tasks.split(",")}
tasks = list(data)

X = torch.cat([d["obs"].flatten(0, 1) for d in data.values()]).to(dev)
if a.drop_task_id:
    X[:, TASK_DIMS] = 0
Y = torch.cat([d["act_mean"].flatten(0, 1) for d in data.values()]).to(dev)
model = SharedPolicy().to(dev)
model.set_normalizer(X)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def fit(X, Y, epochs):
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=dev)
        tot = 0.0
        for i in range(0, len(X), 4096):
            idx = perm[i:i + 4096]
            loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
    return tot / len(X)


def rollout(task, n, collect, seed=12345):
    """Run the shared policy deterministically; optionally return (unified obs, expert labels) for DAgger."""
    env = gym.make(task, num_envs=n, obs_mode="state_dict", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cuda", render_mode=None)
    base = env.unwrapped
    obs, _ = env.reset(seed=seed)
    expert = load_expert(task, flat_state(obs).shape[1]) if collect else None
    xs, ys = [], []
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for _ in range(REGISTERED_ENVS[task].max_episode_steps):
            u = unify(task, obs, base)
            if a.drop_task_id:
                u[:, TASK_DIMS] = 0
            if collect:
                xs.append(u)
                ys.append(expert.actor_mean(flat_state(obs)))
            obs, _, _, _, info = env.step(model(u).clamp(-1, 1))
            done |= info["success"].bool()
    env.close()
    torch.cuda.empty_cache()
    return done.float().mean().item(), (torch.cat(xs), torch.cat(ys)) if collect else None


print(f"BC on {len(X)} samples from {len(tasks)} tasks; mse={fit(X, Y, a.epochs):.4f}")
for it in range(a.dagger_iters):
    new_x, new_y = [], []
    for task in tasks:
        _, (x, y) = rollout(task, a.dagger_envs, collect=True, seed=1000 * a.seed + it)
        new_x.append(x)
        new_y.append(y)
    X, Y = torch.cat([X, *new_x]), torch.cat([Y, *new_y])
    print(f"DAgger iter {it + 1}: {len(X)} samples, mse={fit(X, Y, a.epochs // 2):.4f}")

success = {}
for task in tasks:
    success[task], _ = rollout(task, a.eval_envs, collect=False)
    print(f"  shared policy on {task:20s} success_once={success[task]:.3f}")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), a.out)
json.dump(dict(tasks=tasks, seed=a.seed, success_once=success), open(Path(a.out).with_suffix(".json"), "w"), indent=1)
print("saved", a.out)
