"""Map a target task's flat state observation onto a source task's layout, so a source expert can act in the target.

Flat layouts (ManiSkill 3.0.1, obs_mode="state"):
    PushCube / PullCube / PushCubeDir (35): qpos 0:9, qvel 9:18, tcp_pose 18:25, goal_pos 25:28, obj_pose 28:35
    PickCube (42): qpos 0:9, qvel 9:18, is_grasped 18, tcp_pose 19:26, goal_pos 26:29, obj_pose 29:36,
                   tcp_to_obj 36:39, obj_to_goal 39:42
    StackCube (48): qpos 0:9, qvel 9:18, tcp_pose 18:25, cubeA_pose 25:32, cubeB_pose 32:39, tcp_to_cubeA 39:42,
                    tcp_to_cubeB 42:45, cubeA_to_cubeB 45:48
"""
import torch

STACK_GOAL_DZ = 0.04   # cubeA resting on cubeB: centres two half sizes (0.02) apart


def identity(x, grasped=None):
    return x


def stack_to_pick(x, grasped):
    """StackCube -> PickCube: the cube is cubeA, the goal is 4 cm above cubeB, is_grasped comes from the env."""
    dz = torch.zeros_like(x[:, :3])
    dz[:, 2] = STACK_GOAL_DZ
    goal = x[:, 32:35] + dz
    return torch.cat([x[:, 0:18], grasped.reshape(-1, 1).float(), x[:, 18:25], goal, x[:, 25:32], x[:, 39:42],
                      x[:, 45:48] + dz], dim=1)


def push_rel(x, grasped):
    """Relative features of a push-family state: tcp->obj, obj->goal, grasp flag."""
    return torch.cat([x[:, 28:31] - x[:, 18:21], x[:, 25:28] - x[:, 28:31], grasped.reshape(-1, 1).float()], dim=1)


def stack_rel(x, grasped):
    """The same relative features for StackCube read through the PickCube adapter (goal = 4 cm above cubeB)."""
    dz = torch.zeros_like(x[:, :3])
    dz[:, 2] = STACK_GOAL_DZ
    return torch.cat([x[:, 39:42], x[:, 45:48] + dz, grasped.reshape(-1, 1).float()], dim=1)


ADAPTERS = {"push": identity, "stack_to_pick": stack_to_pick}
REL = {"push": push_rel, "stack": stack_rel}
