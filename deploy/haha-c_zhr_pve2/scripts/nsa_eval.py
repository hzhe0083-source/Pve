"""Every skill of a null-space adapter chain on its own task, through the chain's LAST level (wrwl/nsa.py).
For each skill: its head (from the checkpoint that trained it) reads the deepest level's last hidden layer; it acts
deterministically on its task (256 envs, seed 0 starts as retention_eval.py, first episode per env), and on the
states it visits the script also reports the largest change of its actions against reading its own level (0 = the
plain trunk), which is float noise if the chain keeps earlier skills exactly. --read own: every skill acts through its
own level instead (the prefix-sum ablation, --nsa_project False, where only that is exact); the gap is still reported.
  python scripts/nsa_eval.py --ckpt=CHAIN.pt --skills=pick:PickCube-v1:EXPERT.pt:0,stack:StackCubeNarrow-v1:S.pt:1 \
      --out=x.json
"""
import argparse
import json

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

import wrwl.envs.place_cube  # noqa: F401
import wrwl.envs.stack_narrow  # noqa: F401
from wrwl.adapters import stack_to_pick
from wrwl.nsa import NullSpaceTrunk, build_actor, depth

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True, help="the chain's checkpoint (its last level is used for every skill)")
p.add_argument("--skills", required=True, help="name:env_id:ckpt_with_its_head:level, comma-separated")
p.add_argument("--envs", type=int, default=256)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--read", default="last", choices=["last", "own"])
p.add_argument("--out", default="")
a = p.parse_args()
torch.set_grad_enabled(False)
dev = "cuda"

chain = build_actor(torch.load(a.ckpt, map_location="cpu"), 42, 8).to(dev)
feat = chain[0] if isinstance(chain[0], NullSpaceTrunk) else chain[:-1]
n = depth(feat) if isinstance(feat, NullSpaceTrunk) else 0


def own(level):
    """The features a skill trained at `level` read when it was trained."""
    return feat.level(level) if level else (feat.level(1).trunk if n else feat)


res = {"ckpt": a.ckpt, "levels": n, "read": a.read, "skills": {}}
print(f"chain {a.ckpt}: {n} levels")
for spec in a.skills.split(","):
    name, env_id, src, level = spec.split(":")
    level = int(level)
    src_actor = build_actor(torch.load(src, map_location="cpu"), 42, 8).to(dev)     # the network that trained the head
    head = src_actor[-1]
    kw = dict(num_envs=a.envs, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda",
              render_mode=None, reward_mode="sparse", reconfiguration_freq=1)   # every reset rebuilds the scene:
    # no grasp flag carried over from the previous rollout (reset steps no physics)
    if env_id != "StackCubeNarrow-v1":
        kw["robot_uids"] = "panda_wristcam"
    env = gym.make(env_id, **kw)
    base = env.unwrapped
    obs, _ = env.reset(seed=a.seed)
    done = torch.zeros(a.envs, dtype=torch.bool, device=dev)
    gap = 0.0
    for _ in range(50):
        x = stack_to_pick(obs, base.agent.is_grasping(base.cubeA)) if env_id == "StackCubeNarrow-v1" else obs
        act, act_own = head(feat(x)), head(own(level)(x))
        gap = max(gap, (act - src_actor(x)).abs().max().item())     # 0 for the newest skill; FT: its drift
        if a.read == "own":
            act = act_own
        obs, _, _, _, info = env.step(act.clamp(-1, 1))
        done |= info["success"]
    obs, _ = env.reset(seed=a.seed)     # reference: the source network itself on the same starts
    ref = torch.zeros(a.envs, dtype=torch.bool, device=dev)
    for _ in range(50):
        x = stack_to_pick(obs, base.agent.is_grasping(base.cubeA)) if env_id == "StackCubeNarrow-v1" else obs
        obs, _, _, _, info = env.step(src_actor(x).clamp(-1, 1))
        ref |= info["success"]
    env.close()
    res["skills"][name] = {"env": env_id, "level": level, "success": done.float().mean().item(), "max_gap": gap,
                          "ref_success": ref.float().mean().item()}
    print(f"  {name:8s} {env_id:20s} level {level}: success {done.float().mean().item():.3f}, "
          f"largest action change vs its own level {gap:.1e}")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
