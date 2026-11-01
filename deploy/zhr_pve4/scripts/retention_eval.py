"""Does learning the new skill break the old one? For each new-task checkpoint (warm-started from the PickCube expert),
evaluate the old skill on PickCube-v1 two ways, deterministic actions, first episode per env:
  old head   the new checkpoint's hidden layers with the OLD output layer: is the shared part of the network still good
             for the old skill (the view where every skill keeps its own small head)?
  new head   the new checkpoint as it is: does one network with one head still do the old skill?
plus how far each hidden layer's weights moved, ||W_new - W_old|| / ||W_old||. A null-space adapter checkpoint
(wrwl/nsa.py, any number of levels) counts as its last level's features: the old head reads them, and the weights
moved are the base trunk's (frozen there, so 0); a prefix-sum one (--nsa_project False, no projection) is read through
Pick's own level, the base trunk, with the last-level number kept as old_head_last. Every reset rebuilds the scene, so
no rollout starts from the previous one's contacts (the grasp flag in the observation).
  python scripts/retention_eval.py --old=runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt --ckpts=a.pt,b.pt --out=x.json
"""
import argparse
import json

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

from wrwl.experts import load_policy
from wrwl.nsa import NullSpaceTrunk, build_actor

p = argparse.ArgumentParser()
p.add_argument("--old", required=True)
p.add_argument("--ckpts", required=True, help="comma-separated new-task checkpoints")
p.add_argument("--envs", type=int, default=256)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="")
a = p.parse_args()

old = load_policy(a.old, 42)
env = gym.make("PickCube-v1", num_envs=a.envs, obs_mode="state", robot_uids="panda_wristcam",
               control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None,
               reconfiguration_freq=1)    # a new PhysX system at every reset: no contacts (grasp flag) carried over


def success(act):
    obs, _ = env.reset(seed=a.seed)       # the same starts for every network
    s = torch.zeros(a.envs, dtype=torch.bool, device=obs.device)
    with torch.no_grad():
        for _ in range(50):
            obs, _, _, _, info = env.step(act(obs).clamp(-1, 1))
            s |= info["success"]
    return s.float().mean().item()


res = {"old_expert": success(old.actor_mean), "runs": {}}
print(f"old PickCube expert on PickCube: {res['old_expert']:.3f}")
print(f"{'checkpoint':48s} | old head | new head | weight change per hidden layer")
for c in a.ckpts.split(","):
    new = build_actor(torch.load(c, map_location="cpu"), 42, 8).to("cuda")
    trunk, head = new[:-1], old.actor_mean[-1]
    lv = []                                    # adapter levels, outermost first
    base = new[0] if isinstance(new[0], NullSpaceTrunk) else new
    while isinstance(base, NullSpaceTrunk):
        lv.append(base)
        base = base.trunk
    moved = [((base[i].weight - old.actor_mean[i].weight).norm() / old.actor_mean[i].weight.norm()).item()
             for i in (0, 2, 4)]
    r = {"old_head": success(lambda x: head(trunk(x))), "new_head": success(new), "moved": moved}
    # --nsa_project False (some level's null is exactly I, wrwl/nsa.py:41-42): Pick is exact only through its own
    # level, the base trunk (old_eval.csv reads that too, ppo.py:790); keep the last-level number apart
    if lv and any(torch.equal(m.null, torch.eye(len(m.null), device=m.null.device)) for m in lv):
        r["old_head_last"], r["read"] = r["old_head"], "own"
        r["old_head"] = success(lambda x: head(base(x)))
    res["runs"][c] = r
    name = "/".join(c.split("/")[-3:-1])
    print(f"{name:48s} | {r['old_head']:8.3f} | {r['new_head']:8.3f} | {' '.join(f'{m:.3f}' for m in moved)}")
env.close()
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
