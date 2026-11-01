"""PushCubeDir-v1: PushCube with the push direction rotated by theta, a controlled action-conflict knob (mechanism M1).

theta = 0 pushes towards +x like PushCube. Reaching the cube is the same for every theta; the face of the cube the gripper
must contact, and the push direction, rotate with theta. Reward terms and the success test are PushCube's, rotated.
The spawn square is shifted to x = -0.05 for every theta so all goals stay comfortably inside the Panda's reach.
"""
import numpy as np
import torch
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat


@register_env("PushCubeDir-v1", max_episode_steps=50)
class PushCubeDirEnv(PushCubeEnv):
    def __init__(self, *args, theta_deg=0.0, spawn_center=(-0.05, 0.0), **kwargs):
        self.theta = float(np.deg2rad(theta_deg))
        self.spawn_center = tuple(spawn_center)
        super().__init__(*args, **kwargs)

    def _direction(self):
        return torch.tensor([np.cos(self.theta), np.sin(self.theta), 0.0], dtype=torch.float32, device=self.device)

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = torch.rand(b) * 0.2 - 0.1 + self.spawn_center[0]
            xyz[..., 1] = torch.rand(b) * 0.2 - 0.1 + self.spawn_center[1]
            xyz[..., 2] = self.cube_half_size
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))
            goal = xyz + (0.1 + self.goal_radius) * self._direction()
            goal[..., 2] = 1e-3
            self.goal_region.set_pose(Pose.create_from_pq(p=goal, q=euler2quat(0, np.pi / 2, 0)))

    def compute_dense_reward(self, obs, action, info):
        u = self._direction()
        # distance from the cube centre to the face being pushed, so the gap to the face is 5 mm for every theta
        face = self.cube_half_size / max(abs(np.cos(self.theta)), abs(np.sin(self.theta)))
        push_p = self.obj.pose.p - (face + 0.005) * u
        tcp_to_push = torch.linalg.norm(push_p - self.agent.tcp.pose.p, axis=1)
        reward = 1 - torch.tanh(5 * tcp_to_push)
        reached = tcp_to_push < 0.01
        obj_to_goal = torch.linalg.norm(self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1)
        place = 1 - torch.tanh(5 * obj_to_goal)
        reward += place * reached
        z_reward = 1 - torch.tanh(5 * torch.abs(self.obj.pose.p[..., 2] - self.cube_half_size))
        reward += place * z_reward * reached
        reward[info["success"]] = 4
        return reward
