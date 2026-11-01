"""Every skill of a null-space adapter chain on its own task, through the chain's LAST level (wrwl/nsa.py).
For each skill: its head (from the checkpoint that trained it) reads the deepest level's last hidden layer; it acts
deterministically on its task (256 envs, seed 0 starts as retention_eval.py, first episode per env), and on the
states it visits the script also reports the largest change of its actions against the network that trained it
(max_gap; 0 for the newest skill), which is float noise if the chain keeps earlier skills exactly. ref_success is that
network's own success on the same starts (>= 0.9: the skill was learned). --read own: every skill acts through its
own level instead (the prefix-sum ablation, --nsa_project False, where only that is exact); max_gap then refers to the
path used, and max_gap_last / max_gap_own report both.
Each env is read through its map in wrwl.adapters.ENV_ADAPTER (StackCubeNarrow-v1: stack_to_pick, PlaceSphere-v1:
sphere_to_pick, ...), the grasp flag from the env or observed.
  a chain (scripts/run_chain.sh writes these):
    python scripts/nsa_eval.py --ckpt=CHAIN.pt --skills=pick:PickCube-v1:EXPERT.pt:0,stack:StackCubeNarrow-v1:S.pt:1 \
        --out=x.json
  the zero-shot gate (scripts/suite_gate.py reads skills[<key>].success): the Pick expert on a suite task
    python scripts/nsa_eval.py --ckpt=$PICK --skills=sphere:PlaceSphere-v1:$PICK:0 --out=runs/suite_zs/sphere.json
"""
import argparse
import json

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

import wrwl.envs.lift_peg  # noqa: F401
import wrwl.envs.place_cube  # noqa: F401
import wrwl.envs.push_nograsp  # noqa: F401
import wrwl.envs.pyramid_steps  # noqa: F401
import wrwl.envs.stack_narrow  # noqa: F401
import wrwl.envs.ycb_subset  # noqa: F401
from wrwl.adapters import ENV_ADAPTER, TO_PICK
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
    assert env_id in ENV_ADAPTER, f"{env_id}: no PickCube-layout map in wrwl.adapters.ENV_ADAPTER"
    ad = TO_PICK[ENV_ADAPTER[env_id]] if ENV_ADAPTER[env_id] else None
    src_actor = build_actor(torch.load(src, map_location="cpu"), 42, 8).to(dev)     # the network that trained the head
    head = src_actor[-1]
    kw = dict(num_envs=a.envs, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda",
              render_mode=None, reward_mode="sparse", reconfiguration_freq=1)   # every reset rebuilds the scene:
    # no grasp flag carried over from the previous rollout (reset steps no physics)
    if env_id != "StackCubeNarrow-v1":
        kw["robot_uids"] = "panda_wristcam"
    env = gym.make(env_id, **kw)
    base = env.unwrapped

    def view(o):
        """o in PickCube's layout: the env's map, with the grasp flag from the env or observed."""
        if ad is None:
            return o
        return ad.fn(o, base.agent.is_grasping(getattr(base, ad.obj)) if ad.obj else None)

    obs, _ = env.reset(seed=a.seed)
    done = torch.zeros(a.envs, dtype=torch.bool, device=dev)
    gap_last = gap_own = 0.0
    for _ in range(50):
        x = view(obs)
        act_last, act_own, act_src = head(feat(x)), head(own(level)(x)), src_actor(x)
        gap_last = max(gap_last, (act_last - act_src).abs().max().item())    # 0 for the newest skill; FT: its drift
        gap_own = max(gap_own, (act_own - act_src).abs().max().item())
        act = act_own if a.read == "own" else act_last
        obs, _, _, _, info = env.step(act.clamp(-1, 1))
        done |= info["success"]
    obs, _ = env.reset(seed=a.seed)     # reference: the source network itself on the same starts
    ref = torch.zeros(a.envs, dtype=torch.bool, device=dev)
    for _ in range(50):
        x = view(obs)
        obs, _, _, _, info = env.step(src_actor(x).clamp(-1, 1))
        ref |= info["success"]
    env.close()
    gap = gap_own if a.read == "own" else gap_last
    res["skills"][name] = {"env": env_id, "level": level, "success": done.float().mean().item(), "max_gap": gap,
                          "max_gap_last": gap_last, "max_gap_own": gap_own, "ref_success": ref.float().mean().item()}
    print(f"  {name:8s} {env_id:20s} level {level}: success {done.float().mean().item():.3f} (its own network "
          f"{ref.float().mean().item():.3f}), largest action change vs its own network {gap:.1e}")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
