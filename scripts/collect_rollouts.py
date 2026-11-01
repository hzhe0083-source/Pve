"""Roll out each trained expert in its own task and record unified observations.

The output feeds the shared-network distillation, the latent plots and (via saved env states) the criticality labels.
"""
import argparse
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch
from mani_skill.utils.registration import REGISTERED_ENVS

from wrwl.envs.unified import TASKS, unify
from wrwl.experts import EXPERT_NAMES, flat_state, load_expert

p = argparse.ArgumentParser()
p.add_argument("--tasks", default=",".join(EXPERT_NAMES))
p.add_argument("--num-envs", type=int, default=512)
p.add_argument("--state-every", type=int, default=5, help="save full env states every k steps (for labels)")
p.add_argument("--out", default=str(Path.home() / "pve/data/rollouts.pt"))
p.add_argument("--seed", type=int, default=100)
a = p.parse_args()

data = {}
for task in a.tasks.split(","):
    horizon = REGISTERED_ENVS[task].max_episode_steps
    env = gym.make(task, num_envs=a.num_envs, obs_mode="state_dict", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cuda", render_mode=None)
    base = env.unwrapped
    obs, _ = env.reset(seed=a.seed)
    expert = load_expert(task, flat_state(obs).shape[1])
    rec = {k: [] for k in ["obs", "act_mean", "action", "t"]}
    states, state_t = [], []
    succ_time = torch.full((a.num_envs,), -1, device=base.device)
    with torch.no_grad():
        for t in range(horizon):
            x = flat_state(obs)
            mean = expert.actor_mean(x)
            std = expert.actor_logstd.exp().expand_as(mean)
            act = mean + std * torch.randn_like(mean)
            rec["obs"].append(unify(task, obs, base))
            rec["act_mean"].append(mean)
            rec["action"].append(act)
            rec["t"].append(torch.full((a.num_envs,), t, device=base.device))
            if t % a.state_every == 0:
                states.append({k: {n: v.clone().cpu() for n, v in d.items()}
                               for k, d in base.get_state_dict().items()})
                state_t.append(t)
            obs, rew, term, trunc, info = env.step(act)
            newly = info["success"].bool() & (succ_time < 0)
            succ_time[newly] = t + 1
    stacked = {k: torch.stack(v, dim=1).cpu() for k, v in rec.items()}  # (envs, horizon, ...)
    stacked["succ_time"] = succ_time.cpu()
    stacked["states"] = states
    stacked["state_t"] = state_t
    stacked["horizon"] = horizon
    stacked["task_id"] = TASKS.index(task)
    data[task] = stacked
    print(f"{task:20s} horizon={horizon} expert success_once={(succ_time >= 0).float().mean():.3f} "
          f"median success step={succ_time[succ_time >= 0].float().median().item() if (succ_time >= 0).any() else -1}")
    env.close()
    torch.cuda.empty_cache()

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(data, a.out)
print("saved", a.out)
