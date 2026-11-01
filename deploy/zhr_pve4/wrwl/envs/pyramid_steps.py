"""StackPyramid-v1 in two steps, each a 50-step task with every cube centre in the PickCube expert's spawn square
[-0.1, 0.1]^2 (as wrwl/envs/stack_narrow.py does for StackCube; StackPyramid's own spawn reaches y = +-0.2):
  PyramidBeside-v1   put the red cube A next to the green cube B on the table and let go. Success: StackPyramid's
                     "next to" test for A (centres within |2 half sizes|_xy + 5 mm = 6.16 cm in xy, A static, A not
                     grasped; stack_pyramid.py:149-168) with A on the table (centre within 5 mm of B's height). A
                     starts at least min_ab = 0.10 from B, so it has to be carried; the blue cube C starts clear of A,
                     B and the spot 5 cm from B towards A (adapters.beside_to_pick's goal).
  PyramidOnPair-v1   A and B start side by side (centres 4.5 cm apart, faces parallel); put C on both and let go.
                     Success: StackPyramid-v1's own test (stack_pyramid.py:140-180).
  StackPyramidNarrow-v1  the whole task (250 steps, StackPyramid's success) from PyramidBeside's starts, to run the
                     two skills one after the other.
The observation is StackPyramid's (64-dim); adapters.beside_to_pick and onpair_to_pick read it in PickCube's layout.
evaluate() adds is_cubeA_grasped and is_cubeC_grasped, the adapters' final-observation flags. The robot is
StackPyramid's default panda_wristcam; its rewards are sparse or none only (stack_pyramid.py:43).
"""
import math

import torch
from mani_skill.envs.tasks.tabletop.stack_pyramid import StackPyramidEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

HALF = 0.02
NEXT_TO = math.hypot(2 * HALF, 2 * HALF) + 0.005     # stack_pyramid.py:150-153
CLEAR = 0.06    # centres this far apart: two cubes of half size 0.02 cannot touch at any yaw (2 x 0.0283)


def _square(b, k):
    """k candidate xy per env, uniform in [-0.1, 0.1]^2."""
    return torch.rand((b, k, 2)) * 0.2 - 0.1


def _first_clear(cand, others, dist):
    """Per env the first of the (b, k, 2) candidates at least dist from each (b, 2) point in others; where there is
    none, the candidate farthest from its nearest point."""
    gap = torch.stack([torch.linalg.norm(cand - o[:, None], dim=2) for o in others]).amin(0)
    ok = gap >= dist
    idx = torch.where(ok.any(1), ok.float().argmax(1), gap.argmax(1))
    return cand[torch.arange(len(cand)), idx]


class _PyramidNarrow(StackPyramidEnv):
    min_ab = 0.10

    def _place(self, actor, xy, q=None):
        xyz = torch.zeros((len(xy), 3))
        xyz[:, :2] = xy
        xyz[:, 2] = HALF
        if q is None:
            q = randomization.random_quaternions(len(xy), lock_x=True, lock_y=True)
        actor.set_pose(Pose.create_from_pq(p=xyz, q=q))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xy_b = _square(b, 1)[:, 0]
            xy_a = _first_clear(_square(b, 64), [xy_b], self.min_ab)
            d = xy_a - xy_b
            spot = xy_b + 0.05 * d / torch.linalg.norm(d, dim=1, keepdim=True).clamp_min(1e-6)
            xy_c = _first_clear(_square(b, 64), [xy_a, xy_b, spot], CLEAR)
            for actor, xy in ((self.cubeA, xy_a), (self.cubeB, xy_b), (self.cubeC, xy_c)):
                self._place(actor, xy)

    def evaluate(self):
        info = super().evaluate()
        info["is_cubeA_grasped"] = self.agent.is_grasping(self.cubeA)
        info["is_cubeC_grasped"] = self.agent.is_grasping(self.cubeC)
        return info


@register_env("StackPyramidNarrow-v1", max_episode_steps=250)
class StackPyramidNarrowEnv(_PyramidNarrow):
    pass


@register_env("PyramidBeside-v1", max_episode_steps=50)
class PyramidBesideEnv(_PyramidNarrow):
    def evaluate(self):
        info = super().evaluate()
        a, b = self.cubeA.pose.p, self.cubeB.pose.p
        beside = (torch.linalg.norm(a[:, :2] - b[:, :2], dim=1) <= NEXT_TO) & (torch.abs(a[:, 2] - b[:, 2]) <= 0.005)
        info["is_pyramid"] = info["success"]
        info["success"] = beside & self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5) & ~info["is_cubeA_grasped"]
        return info


@register_env("PyramidOnPair-v1", max_episode_steps=50)
class PyramidOnPairEnv(_PyramidNarrow):
    pair_gap = 0.045    # A and B centres: their facing sides 5 mm apart

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            mid = torch.rand((b, 2)) * 0.15 - 0.075
            phi = torch.rand(b) * 2 * math.pi
            d = torch.stack([torch.cos(phi), torch.sin(phi)], dim=1)
            q = torch.zeros((b, 4))
            q[:, 0], q[:, 3] = torch.cos(phi / 2), torch.sin(phi / 2)      # yaw phi: faces normal to d
            xy_a, xy_b = mid - self.pair_gap / 2 * d, mid + self.pair_gap / 2 * d
            xy_c = _first_clear(_square(b, 64), [xy_a, xy_b], CLEAR)
            self._place(self.cubeA, xy_a, q)
            self._place(self.cubeB, xy_b, q.clone())
            self._place(self.cubeC, xy_c)
