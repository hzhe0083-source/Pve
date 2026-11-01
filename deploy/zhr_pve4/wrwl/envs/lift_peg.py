"""LiftPegUprightG-v1: LiftPegUpright-v1 that also reports the peg's grasp flag (info["is_grasped"]), which its own
evaluate() does not and adapters.peg_to_pick reads for final observations; the robot defaults to panda_wristcam, the
PickCube expert's. Spawn (the peg lying along x, lift_peg_upright.py:76-87), success (lift_peg_upright.py:89-99; no
release needed) and the observation (32-dim) are the parent's.
"""
from mani_skill.envs.tasks.tabletop.lift_peg_upright import LiftPegUprightEnv
from mani_skill.utils.registration import register_env


@register_env("LiftPegUprightG-v1", max_episode_steps=50)
class LiftPegUprightGEnv(LiftPegUprightEnv):
    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def evaluate(self):
        info = super().evaluate()
        info["is_grasped"] = self.agent.is_grasping(self.peg)
        return info
