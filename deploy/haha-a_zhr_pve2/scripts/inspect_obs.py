"""List the state-observation fields of each pool task, to design the unified multi-task observation."""
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

TASKS = ["PushCube-v1", "PullCube-v1", "PickCube-v1", "StackCube-v1", "PlaceSphere-v1", "PokeCube-v1",
         "LiftPegUpright-v1", "PullCubeTool-v1", "PegInsertionSide-v1"]


def walk(prefix, x):
    if isinstance(x, dict):
        for k, v in x.items():
            walk(f"{prefix}/{k}" if prefix else k, v)
    else:
        print(f"    {prefix}: {tuple(x.shape)}")


for t in TASKS:
    env = gym.make(t, num_envs=4, obs_mode="state_dict", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cuda", render_mode=None)
    obs, _ = env.reset(seed=0)
    base = env.unwrapped
    print(f"{t}: max_episode_steps={env.spec.max_episode_steps} actors={list(base.get_state_dict()['actors'].keys())}")
    walk("", obs)
    env.close()
    torch.cuda.empty_cache()
