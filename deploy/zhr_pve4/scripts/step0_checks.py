"""Step 0 of the pivot memo: under sparse reward, do the sources solve their own task and fail where they should?

  push   zero-shot success_once (deterministic) of every push-family policy on PushCubeDir theta in {0,45,90,135,180}
  stack  a PickCube expert on panda_wristcam acting in StackCubeNarrow (both cubes inside PickCube's support, same robot
         as StackCube) through the adapter, with and without a scripted release (open the gripper, and keep it open,
         once cubeA is within PickCube's 2.5 cm success radius of the adapted goal), and the StackCube expert
Go/no-go (memo G0): each source >= 0.85 on its own theta; S0 <= 0.05 on theta 90 and 180; Pick alone <= 0.3 on Stack.
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

import wrwl.envs.push_dir  # noqa: F401
import wrwl.envs.stack_narrow  # noqa: F401
from wrwl.adapters import stack_to_pick
from wrwl.experts import flat_state, load_policy

ROOT = Path.home() / "pve/runs"
p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=256)
p.add_argument("--out", default=str(Path.home() / "pve/logs/step0.json"))
p.add_argument("--stack-only", action="store_true")
p.add_argument("--pick", default=str(ROOT / "experts_wc/runs/PickCube_wc_s1/final_ckpt.pt"))
a = p.parse_args()
dev = "cuda"
THETAS = [0, 45, 90, 135, 180]
PUSH = {"S0 PushCube expert": ROOT / "experts/runs/PushCube/final_ckpt.pt",
        "S180 PullCube expert": ROOT / "experts/runs/PullCube/final_ckpt.pt",
        "M1 source_t0": ROOT / "m1/runs/source_t0/final_ckpt.pt"}
for th in THETAS:
    for s in (1, 2, 3):
        PUSH[f"M1 t{th}_scratch_s{s}"] = ROOT / f"m1/runs/t{th}_scratch_s{s}/final_ckpt.pt"


def make(env_id, **kw):
    return gym.make(env_id, num_envs=a.envs, obs_mode="state_dict", reward_mode="sparse",
                    control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None, **kw)


def run(env, act_fn, horizon):
    obs, _ = env.reset(seed=123)
    succ = torch.zeros(a.envs, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for _ in range(horizon):
            obs, _, _, _, info = env.step(act_fn(obs).clamp(-1, 1))
            succ |= info["success"].bool()
    return succ.float().mean().item()


res = {"push": {}, "stack": {}}
for th in ([] if a.stack_only else THETAS):
    env = make("PushCubeDir-v1", theta_deg=th)
    horizon = 50
    for name, path in PUSH.items():
        if not path.exists():
            continue
        pol = load_policy(path, 35)
        res["push"].setdefault(name, {})[th] = run(env, lambda o: pol.actor_mean(flat_state(o)), horizon)
    env.close()
    torch.cuda.empty_cache()
if res["push"]:
    print("zero-shot success_once on PushCubeDir (sparse, deterministic)")
    print(f"{'policy':24s}" + "".join(f"{f'theta {th}':>10s}" for th in THETAS))
    for name, row in res["push"].items():
        print(f"{name:24s}" + "".join(f"{row.get(th, float('nan')):10.2f}" for th in THETAS))

env = make("StackCubeNarrow-v1")
base = env.unwrapped
pick = load_policy(a.pick, 42)
released = torch.zeros(a.envs, dtype=torch.bool, device=dev)
min_goal = torch.full((a.envs,), 9.0, device=dev)
stack = load_policy(ROOT / "experts/runs/StackCube/final_ckpt.pt", 48)


def pick_act(obs, release):
    global released, min_goal
    x = flat_state(obs)
    xp = stack_to_pick(x, base.agent.is_grasping(base.cubeA))
    act = pick.actor_mean(xp)
    d = torch.linalg.norm(xp[:, 39:42], dim=1)
    min_goal = torch.minimum(min_goal, d)
    if release:
        released |= d < 0.025
        act[released, 7] = 1.0      # open the gripper and keep it open (Panda gripper action +1 = open)
    return act


res["stack"]["Pick via adapter"] = run(env, lambda o: pick_act(o, False), 50)
res["stack"]["Pick: min distance to adapted goal (median, m)"] = min_goal.median().item()
released[:] = False
min_goal[:] = 9.0
res["stack"]["Pick via adapter + scripted release"] = run(env, lambda o: pick_act(o, True), 50)
res["stack"]["release fired (fraction of episodes)"] = released.float().mean().item()
res["stack"]["StackCube expert"] = run(env, lambda o: stack.actor_mean(flat_state(o)), 50)
env.close()
print("StackCube (sparse, deterministic):", {k: round(v, 3) for k, v in res["stack"].items()})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print("saved", a.out)
