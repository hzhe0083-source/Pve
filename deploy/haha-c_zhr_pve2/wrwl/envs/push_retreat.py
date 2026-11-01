"""PushRetreat-v1: PushCube where success also needs the gripper backed away from the cube (at least retreat_dist) once
the cube is in the goal region: PushCube extended by one step that moves the arm, not the gripper. Same observation
(35-dim), spawn, goal and dense reward as PushCube, so a PushCube expert acts in it unchanged; it pushes the cube in and
then does whatever it learned to do after success, which PushCube's early termination never trained.
With env_kwargs {"retreat_dist": 0.15} the source almost never backs away far enough, so the arm has to explore after
the push, while arm noise during the push still spoils it: the case where seam-only exploration should beat global.
"""
import torch
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env


@register_env("PushRetreat-v1", max_episode_steps=50)
class PushRetreatEnv(PushCubeEnv):
    def __init__(self, *args, retreat_dist=0.06, **kwargs):
        self.retreat_dist = retreat_dist
        super().__init__(*args, **kwargs)

    def evaluate(self):
        info = super().evaluate()
        far = torch.linalg.norm(self.agent.tcp.pose.p - self.obj.pose.p, axis=1) >= self.retreat_dist
        info["is_obj_placed"] = info["success"]
        info["success"] = info["success"] & far
        return info
