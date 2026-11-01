"""Why does the old skill fail after the new one is trained, even when its mean actions on its own states barely move?
Student = the checkpoint's hidden layers with the OLD output layer (the view retention_eval.py scores); teacher = the
PickCube expert. Both are rolled out on PickCube-v1 from the same starts (deterministic, first episode per env). On the
states each rollout visits, before that env's first success, record the student-teacher gap of the clamped actions per
dim in the teacher's std units (squared), split by whether the cube is grasped, plus success and grasp rates:
  on teacher states   the distribution the protection (and protect.csv's drift) sees
  on student states   where the student actually goes; a much larger gap there means its small errors compound
  python scripts/forgetting_closedloop.py --old=P --ckpts=a.pt,b.pt --out=x.json
"""
import argparse
import json

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

from wrwl.experts import load_policy

p = argparse.ArgumentParser()
p.add_argument("--old", required=True)
p.add_argument("--ckpts", required=True)
p.add_argument("--envs", type=int, default=256)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="")
a = p.parse_args()
torch.set_grad_enabled(False)

old = load_policy(a.old, 42)
sig = old.actor_logstd.exp().flatten()
env = gym.make("PickCube-v1", num_envs=a.envs, obs_mode="state", robot_uids="panda_wristcam",
               control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None)
base = env.unwrapped


def rollout(drive, student, teacher):
    """Roll out `drive`; on its states before each env's first success, the per-dim gap, split by grasp."""
    obs, _ = env.reset(seed=a.seed)
    done = torch.zeros(a.envs, dtype=torch.bool, device=obs.device)
    grasped = torch.zeros_like(done)
    gap = {k: torch.zeros(len(sig), device=obs.device) for k in ("free", "held")}
    cnt = {k: 0 for k in gap}
    for _ in range(50):
        s, t = student(obs).clamp(-1, 1), teacher(obs).clamp(-1, 1)
        g = ((s - t) / sig) ** 2
        held = base.agent.is_grasping(base.cube)
        for k, m in (("free", ~done & ~held), ("held", ~done & held)):
            gap[k] += g[m].sum(0)
            cnt[k] += int(m.sum())
        grasped |= held
        obs, _, _, _, info = env.step((s if drive == "student" else t))
        done |= info["success"]
    out = {"success": done.float().mean().item(), "grasped": grasped.float().mean().item()}
    for k in gap:
        out[f"gap_{k}"] = (gap[k] / max(cnt[k], 1)).tolist()
        out[f"gap_{k}_sum"] = (gap[k].sum() / max(cnt[k], 1)).item()
        out[f"n_{k}"] = cnt[k]
    return out


res = {"runs": {}}
print(f"{'checkpoint':34s} | drive   | success | grasped | gap free (sum, gripper) | gap held (sum, gripper)")
for c in a.ckpts.split(","):
    new = load_policy(c, 42)
    trunk, head = new.actor_mean[:-1], old.actor_mean[-1]
    student = lambda x: head(trunk(x))
    r = {d: rollout(d, student, old.actor_mean) for d in ("teacher", "student")}
    res["runs"][c] = r
    name = "/".join(c.split("/")[-2:-1])
    for d, o in r.items():
        print(f"{name:34s} | {d:7s} | {o['success']:7.3f} | {o['grasped']:7.3f} | "
              f"{o['gap_free_sum']:8.3f} {o['gap_free'][-1]:8.3f}      | {o['gap_held_sum']:8.3f} {o['gap_held'][-1]:8.3f}")
env.close()
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
