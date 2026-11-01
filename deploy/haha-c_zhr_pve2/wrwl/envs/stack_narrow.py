"""StackCube with both cubes spawned inside PickCube's support (xy in [-0.1, 0.1]^2), so a PickCube expert acting in
StackCube through the adapter sees cube and goal positions it was trained on. Everything else is StackCube-v1: the same
robot (panda_wristcam by default), rewards and success test. The narrow region is a subset of StackCube's own spawn, so
the StackCube expert stays in distribution too.
"""
import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("StackCubeNarrow-v1", max_episode_steps=50)
class StackCubeNarrowEnv(StackCubeEnv):
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            sampler = randomization.UniformPlacementSampler(bounds=[[-0.1, -0.1], [0.1, 0.1]], batch_size=b,
                                                            device=self.device)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            cube_a = sampler.sample(radius, 100)
            cube_b = sampler.sample(radius, 100, verbose=False)
            for actor, xy in ((self.cubeA, cube_a), (self.cubeB, cube_b)):
                xyz[:, :2] = xy
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
                actor.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
