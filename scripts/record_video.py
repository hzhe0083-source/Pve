"""Record rollout videos of a policy checkpoint (deterministic actions), several envs tiled in one video.

The same rollout loop as the evaluation: the policy acts on the flat state (through the PickCube adapter for Stack)
with its action mean; ManiSkill's RecordEpisode renders the scene camera. Prints the success rate of the recorded
episodes so each video can be captioned with what it shows.
  python scripts/record_video.py --ckpt=runs/pilot2/runs/stack_floor_probe10_s1_c1/final_ckpt.pt \
      --env=StackCubeNarrow-v1 --adapter=stack_to_pick --out=videos/stack_method
"""
import argparse
import json

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch
from mani_skill.utils.wrappers.record import RecordEpisode

import wrwl.envs.place_cube  # noqa: F401
import wrwl.envs.push_retreat  # noqa: F401
import wrwl.envs.stack_narrow  # noqa: F401
from wrwl.adapters import stack_to_pick
from wrwl.experts import flat_state, load_policy

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--env", default="StackCubeNarrow-v1")
p.add_argument("--env_kwargs", default="{}")
p.add_argument("--adapter", default="stack_to_pick", choices=["stack_to_pick", "none"])
p.add_argument("--obs_dim", type=int, default=42)
p.add_argument("--envs", type=int, default=6)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()

pol = load_policy(a.ckpt, a.obs_dim)
stack = a.adapter == "stack_to_pick"
env = gym.make(a.env, num_envs=a.envs, obs_mode="state_dict" if stack else "state", render_mode="rgb_array",
               reward_mode="sparse", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda",
               **json.loads(a.env_kwargs))
env = RecordEpisode(env, output_dir=a.out, save_trajectory=False, save_video=True, video_fps=20,
                    max_steps_per_video=a.steps)
base = env.unwrapped
obs, _ = env.reset(seed=a.seed)
succ = None
with torch.no_grad():
    for _ in range(a.steps):
        x = stack_to_pick(flat_state(obs), base.agent.is_grasping(base.cubeA)) if stack else obs
        obs, _, _, _, info = env.step(pol.actor_mean(x).clamp(-1, 1))
        succ = info["success"] if succ is None else succ | info["success"]
env.close()
print(f"{a.out}: success in {int(succ.sum())}/{len(succ)} recorded episodes "
      f"({', '.join(str(int(s)) for s in succ.tolist())})")
