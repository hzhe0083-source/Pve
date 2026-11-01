"""PlaceCube-v1: PickCube with the goal on the table, where success needs the cube released: PickCube extended by one
step (release), like StackCube without the second cube. The observation is PickCube's (42-dim, with the grasp flag), so
a PickCube expert acts in it without an adapter: it carries the cube to the goal and holds it there. The goal is drawn
from PickCube's spawn square at least min_goal_dist from the cube, so the cube has to be carried. Same robot, spawn and
dense reward as PickCube (the dense reward does not ask for the release; the experiments use the sparse reward).
"""
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PlaceCube-v1", max_episode_steps=50)
class PlaceCubeEnv(PickCubeEnv):
    min_goal_dist = 0.06

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            h, c = self.cube_spawn_half_size, self.cube_spawn_center
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * h * 2 - h
            xyz[:, 0] += c[0]
            xyz[:, 1] += c[1]
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            # 16 candidate goals per env; take the first far enough from the cube (the square is 0.2 m wide, so with
            # min_goal_dist 0.06 almost every env has one; the rare rest keep a candidate pushed away from the cube)
            cand = torch.rand((b, 16, 2)) * h * 2 - h + torch.tensor(c[:2])
            ok = torch.linalg.norm(cand - xyz[:, None, :2], dim=2) >= self.min_goal_dist
            first = torch.where(ok.any(1), ok.float().argmax(1), torch.zeros(b, dtype=torch.long))
            goal_xy = cand[torch.arange(b), first]
            away = goal_xy - xyz[:, :2]
            short = torch.linalg.norm(away, dim=1) < self.min_goal_dist
            away = away / torch.linalg.norm(away, dim=1, keepdim=True).clamp_min(1e-6)
            goal_xy[short] = xyz[short, :2] + self.min_goal_dist * away[short]
            goal = torch.zeros((b, 3))
            goal[:, :2] = goal_xy
            goal[:, 2] = self.cube_half_size
            self.goal_site.set_pose(Pose.create_from_pq(goal))

    def evaluate(self):
        info = super().evaluate()
        info["success"] = info["is_obj_placed"] & ~info["is_grasped"]
        return info
