"""PushCubeNoGrasp-v1 / PullCubeNoGrasp-v1: PushCube-v1 / PullCube-v1 where the cube must not be grasped. A grasp of
the cube at any step after the reset fails the episode (info["fail"]: the sparse reward is success - fail = -1 and the
episode ends; with terminations ignored, as in evaluation, it stays failed), and success is the parent's while the cube
has not been grasped. The parents' success tests accept a cube carried to the target, which is what the PickCube
expert does; here only the non-prehensile push / pull their dense rewards describe (push_cube.py:209-242,
pull_cube.py:128-147) counts. info["is_grasped"] is the current flag, which adapters.push_to_pick reads for final
observations; nograsp=False keeps the parent's success and only adds it. Spawn, goal and observation (35-dim) are the
parent's; the robot defaults to panda_wristcam, the PickCube expert's.
A reset steps no physics, so right after it is_grasping still sees the previous episode's contacts: at elapsed_steps 0
the latch is cleared, not set.
"""
import torch
from mani_skill.envs.tasks.tabletop.pull_cube import PullCubeEnv
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env


class NoGraspMixin:
    def __init__(self, *args, robot_uids="panda_wristcam", nograsp=True, **kwargs):
        self.nograsp = nograsp
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.grasped_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def evaluate(self):
        info = super().evaluate()
        info["is_grasped"] = self.agent.is_grasping(self.obj)
        if self.nograsp:
            self.grasped_once = (self.grasped_once | info["is_grasped"]) & (self.elapsed_steps > 0)
            info["fail"] = self.grasped_once.clone()
            info["success"] = info["success"] & ~self.grasped_once
        return info


@register_env("PushCubeNoGrasp-v1", max_episode_steps=50)
class PushCubeNoGraspEnv(NoGraspMixin, PushCubeEnv):
    pass


@register_env("PullCubeNoGrasp-v1", max_episode_steps=50)
class PullCubeNoGraspEnv(NoGraspMixin, PullCubeEnv):
    pass
