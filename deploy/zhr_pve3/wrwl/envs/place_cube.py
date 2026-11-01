"""PlaceCube-v1: PickCube with the goal on the table, where success needs the cube released: PickCube extended by one
step (release), like StackCube without the second cube. The observation is PickCube's (42-dim, with the grasp flag), so
a PickCube expert acts in it without an adapter: it carries the cube to the goal and holds it there. The goal is drawn
from PickCube's spawn square at least min_goal_dist from the cube, so the cube has to be carried. Same robot, spawn and
dense reward as PickCube (the dense reward does not ask for the release; the experiments use the sparse reward).
stage1_cost=beta (env kwarg, with reward_mode "sparse"): the sparse reward minus beta * (1 - stage1_progress) per step,
a dense cost on the first stage only (reach, grasp, carry to the goal). It is 0 once the cube is held at the goal or
placed, so it neither pays for holding nor charges for the release; success still ends the episode with reward 1.
"""
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


def stage1_progress(tcp_p, cube_p, goal_p, grasped, placed):
    """In [0, 1], 1 when the cube is held at the goal or placed: (reach + held + held * carry) / 3 with PickCube's
    dense terms, held = grasped or placed, reach = 1 while held, carry = 1 once placed (within goal_thresh)."""
    held = grasped | placed
    reach = 1 - torch.tanh(5 * torch.linalg.norm(cube_p - tcp_p, dim=1))
    carry = 1 - torch.tanh(5 * torch.linalg.norm(goal_p - cube_p, dim=1))
    reach = torch.where(held, torch.ones_like(reach), reach)
    carry = torch.where(placed, torch.ones_like(carry), carry)
    return (reach + held.float() * (1 + carry)) / 3


@register_env("PlaceCube-v1", max_episode_steps=50)
class PlaceCubeEnv(PickCubeEnv):
    min_goal_dist = 0.06

    def __init__(self, *args, stage1_cost=0.0, **kwargs):
        self.stage1_cost = float(stage1_cost)
        super().__init__(*args, **kwargs)

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

    def compute_sparse_reward(self, obs, action, info):
        reward = super().compute_sparse_reward(obs=obs, action=action, info=info)
        if not self.stage1_cost:    # the default: exactly the sparse reward as before
            return reward
        phi = stage1_progress(self.agent.tcp_pose.p, self.cube.pose.p, self.goal_site.pose.p, info["is_grasped"],
                              info["is_obj_placed"])
        return reward.float() - self.stage1_cost * (1 - phi)
