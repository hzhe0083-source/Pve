"""Map a target task's flat state observation onto a source task's layout, so a source expert can act in the target.

Flat layouts (ManiSkill 3.0.1, obs_mode="state": qpos 0:9, qvel 9:18, then the task's extras in the insertion order
of its _get_obs_extra; a grasp flag is one column; a pose is p xyz then q wxyz):
    PushCube / PullCube / PushCubeDir (35): qpos 0:9, qvel 9:18, tcp_pose 18:25, goal_pos 25:28, obj_pose 28:35
    PickCube (42): qpos 0:9, qvel 9:18, is_grasped 18, tcp_pose 19:26, goal_pos 26:29, obj_pose 29:36,
                   tcp_to_obj 36:39, obj_to_goal 39:42
    StackCube (48): qpos 0:9, qvel 9:18, tcp_pose 18:25, cubeA_pose 25:32, cubeB_pose 32:39, tcp_to_cubeA 39:42,
                    tcp_to_cubeB 42:45, cubeA_to_cubeB 45:48
    PlaceSphere (39): is_grasped 18, tcp_pose 19:26, bin_pos 26:29, obj_pose 29:36, tcp_to_obj 36:39
    PickSingleYCB (45): tcp_pose 18:25, goal_pos 25:28, is_grasped 28, tcp_to_goal 29:32, obj_pose 32:39,
                        tcp_to_obj 39:42, obj_to_goal 42:45
    LiftPegUpright (32): tcp_pose 18:25, obj_pose 25:32 (the peg)
    StackPyramid (64): tcp_pose 18:25, cubeA_pose 25:32, cubeB_pose 32:39, cubeC_pose 39:46, tcp_to_cubeA 46:49,
                       tcp_to_cubeB 49:52, tcp_to_cubeC 52:55, cubeA_to_cubeB 55:58, cubeB_to_cubeC 58:61,
                       cubeA_to_cubeC 61:64
The *_to_pick maps (wrwl/ppo.py --obs_adapter) return PickCube's 42 dims; the new ones compute tcp_to_obj and
obj_to_goal from the mapped poses, as pick_cube.py:142-143 does. TO_PICK also says where each map's grasp flag comes
from: obj, the env attribute the flag is about (agent.is_grasping(getattr(env, obj))), or None when the observation
has the flag; and final_key, the info key with that flag for a final observation (after an auto-reset the env already
holds the next episode, so the flag cannot be read from it).
"""
from typing import Callable, NamedTuple, Optional

import torch

STACK_GOAL_DZ = 0.04   # cubeA resting on cubeB: centres two half sizes (0.02) apart
CUBE_HALF = 0.02       # a cube resting on the table has its centre at 0.02 (push_cube.py:60, stack_pyramid.py:97)
SPHERE_GOAL_DZ = 0.0225     # sphere radius 0.02 + bin floor half thickness 0.0025: the sphere's centre resting in the
                            # bin, above the bin's pose (place_sphere.py:44-46, 179, 188-191)
PEG_HALF_LENGTH = 0.12      # an upright peg standing on the table has its centre at 0.12 (lift_peg_upright.py:39, 96)
PEG_GRIP = 0.09             # peg_to_pick's object: the point on the axis 0.09 from the centre towards +x
BESIDE_DIST = 0.05          # beside_to_pick's goal: cubeA's centre this far from cubeB's (StackPyramid accepts 6.16 cm)


def identity(x, grasped=None):
    return x


def stack_to_pick(x, grasped):
    """StackCube -> PickCube: the cube is cubeA, the goal is 4 cm above cubeB, is_grasped comes from the env."""
    dz = torch.zeros_like(x[:, :3])
    dz[:, 2] = STACK_GOAL_DZ
    goal = x[:, 32:35] + dz
    return torch.cat([x[:, 0:18], grasped.reshape(-1, 1).float(), x[:, 18:25], goal, x[:, 25:32], x[:, 39:42],
                      x[:, 45:48] + dz], dim=1)


def _pick(x, grasped, tcp_pose, goal, obj_pose):
    """PickCube's 42 dims from x's qpos and qvel, a grasp flag, the TCP pose, the goal and the object pose."""
    obj = obj_pose[:, 0:3]
    return torch.cat([x[:, 0:18], grasped.reshape(-1, 1).to(x.dtype), tcp_pose, goal, obj_pose, obj - tcp_pose[:, 0:3],
                      goal - obj], dim=1)


def _at_z(p, z):
    """(n, 3) positions with z replaced by the constant z."""
    return torch.cat([p[:, 0:2], torch.full_like(p[:, 2:3], z)], dim=1)


def quat_mul(a, b):
    """Hamilton product of (n, 4) wxyz quaternions."""
    aw, ax, ay, az = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bw, bx, by, bz = b[:, 0:1], b[:, 1:2], b[:, 2:3], b[:, 3:4]
    return torch.cat([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], dim=1)


def quat_x_axis(q):
    """The body x-axis in the world, R(q) e_x, for (n, 4) unit wxyz quaternions."""
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    return torch.cat([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=1)


def peg_swing(q):
    """The peg's orientation without its roll about the long axis: the shortest rotation taking e_x to the peg's axis
    u = R(q) e_x, i.e. (1 + u_x, 0, -u_z, u_y) normalised (wxyz). The peg (half sizes 0.12, 0.025, 0.025 along its
    body x, y, z) has a square cross-section, so its roll does not change where it can be grasped. The lying peg
    reads as the identity at every resting roll, like an unrotated cube (spawn euler2quat(pi/2, 0, 0),
    lift_peg_upright.py:84, raw (0.71, 0.71, 0, 0)); turned on the table it reads as a yaw-only quaternion with
    w > 0, inside the range of cube orientations the Pick expert saw; tipping it up about its -x end reads as a pitch
    that passes continuously through upright, inside LiftPegUpright's success window (lift_peg_upright.py:89-99).
    Singular only at u = -e_x (the peg turned end over end), where it returns zeros instead of NaN."""
    u = quat_x_axis(q)
    s = torch.cat([1 + u[:, 0:1], torch.zeros_like(u[:, 0:1]), -u[:, 2:3], u[:, 1:2]], dim=1)
    return s / torch.linalg.norm(s, dim=1, keepdim=True).clamp_min(1e-6)


def push_to_pick(x, grasped):
    """PushCube / PullCube / PushCubeDir / PushRetreat and the NoGrasp variants (35): the cube is the object, the goal
    is the target's xy at the height of a resting cube's centre (the target disc lies at z = 1e-3,
    push_cube.py:168-170). The flag comes from the env."""
    return _pick(x, grasped, x[:, 18:25], _at_z(x[:, 25:28], CUBE_HALF), x[:, 28:35])


def sphere_to_pick(x, grasped=None):
    """PlaceSphere (39): the sphere is the object, the goal is its centre resting in the bin (the bin's pose + 0.0225
    in z, what evaluate() tests to 5 mm), and its orientation reads as the identity: a sphere's does not matter, and a
    rolling one's would leave the yaw-only cube orientations the Pick expert saw. The flag is observed (column 18)
    unless grasped is given."""
    g = x[:, 18] if grasped is None else grasped
    goal = x[:, 26:29] + x.new_tensor([0.0, 0.0, SPHERE_GOAL_DZ])
    obj = torch.cat([x[:, 29:32], torch.zeros_like(x[:, 0:4]) + x.new_tensor([1.0, 0.0, 0.0, 0.0])], dim=1)
    return _pick(x, g, x[:, 19:26], goal, obj)


def ycb_to_pick(x, grasped=None):
    """PickSingleYCB / PickYCBSubset (45): 42 of its columns in PickCube's order (tcp_to_goal 29:32 dropped). The flag
    is observed (column 28) unless grasped is given."""
    g = x[:, 28] if grasped is None else grasped
    return torch.cat([x[:, 0:18], g.reshape(-1, 1).to(x.dtype), x[:, 18:28], x[:, 32:45]], dim=1)


def peg_to_pick(x, grasped):
    """LiftPegUpright (32): the object is the point on the peg's axis 0.09 from its centre towards the +x end (3 cm
    from that end), the goal is where that point is once the peg stands upright on its -x end where that end now is
    (0.21 above it). The task observes no goal; with this one, carrying the object to the goal tips the peg up about
    its -x end, as ManiSkill's motion-planning solution grasps off centre and lets the peg swing
    (examples/motionplanning/panda/solutions/lift_peg_upright.py:56-97). With the goal at the peg's own centre lifted
    to 0.12 the Pick expert lifts the peg flat. The orientation is peg_swing's. The flag comes from the env."""
    p, q = x[:, 25:28], x[:, 28:32]
    u = quat_x_axis(q)
    foot = p - PEG_HALF_LENGTH * u
    goal = _at_z(foot, PEG_HALF_LENGTH + PEG_GRIP)
    return _pick(x, grasped, x[:, 18:25], goal, torch.cat([p + PEG_GRIP * u, peg_swing(q)], dim=1))


def beside_to_pick(x, grasped):
    """StackPyramid layout (64), PyramidBeside: the object is cubeA, the goal is cubeA's centre on the table 5 cm from
    cubeB's, on cubeA's side. The goal follows the cubes, as stack_to_pick's follows cubeB. The flag (cubeA) comes
    from the env."""
    a, b = x[:, 25:28], x[:, 32:35]
    d = a[:, 0:2] - b[:, 0:2]
    d = d / torch.linalg.norm(d, dim=1, keepdim=True).clamp_min(1e-6)
    goal = torch.cat([b[:, 0:2] + BESIDE_DIST * d, torch.full_like(b[:, 2:3], CUBE_HALF)], dim=1)
    return _pick(x, grasped, x[:, 18:25], goal, x[:, 25:32])


def onpair_to_pick(x, grasped):
    """StackPyramid layout (64), PyramidOnPair: the object is cubeC, the goal is 4 cm above the middle of cubeA and
    cubeB (cubeC resting on both). The flag (cubeC) comes from the env."""
    goal = (x[:, 25:28] + x[:, 32:35]) / 2 + x.new_tensor([0.0, 0.0, STACK_GOAL_DZ])
    return _pick(x, grasped, x[:, 18:25], goal, x[:, 39:46])


def push_rel(x, grasped):
    """Relative features of a push-family state: tcp->obj, obj->goal, grasp flag."""
    return torch.cat([x[:, 28:31] - x[:, 18:21], x[:, 25:28] - x[:, 28:31], grasped.reshape(-1, 1).float()], dim=1)


def stack_rel(x, grasped):
    """The same relative features for StackCube read through the PickCube adapter (goal = 4 cm above cubeB)."""
    dz = torch.zeros_like(x[:, :3])
    dz[:, 2] = STACK_GOAL_DZ
    return torch.cat([x[:, 39:42], x[:, 45:48] + dz, grasped.reshape(-1, 1).float()], dim=1)


class ToPick(NamedTuple):
    fn: Callable                 # (x, grasped) -> (n, 42)
    obj: Optional[str]           # env attribute the grasp flag is about; None: the observation has the flag
    final_key: Optional[str]     # info key with that flag, read for final observations (None with obj None)


TO_PICK = {
    "stack_to_pick": ToPick(stack_to_pick, "cubeA", "is_cubeA_grasped"),
    "push_to_pick": ToPick(push_to_pick, "obj", "is_grasped"),             # wrwl/envs/push_nograsp.py adds the key
    "sphere_to_pick": ToPick(sphere_to_pick, None, None),
    "ycb_to_pick": ToPick(ycb_to_pick, None, None),
    "peg_to_pick": ToPick(peg_to_pick, "peg", "is_grasped"),               # wrwl/envs/lift_peg.py adds the key
    "beside_to_pick": ToPick(beside_to_pick, "cubeA", "is_cubeA_grasped"),  # wrwl/envs/pyramid_steps.py adds both
    "onpair_to_pick": ToPick(onpair_to_pick, "cubeC", "is_cubeC_grasped"),
}
ADAPTERS = {"push": identity, **{k: v.fn for k, v in TO_PICK.items()}}
REL = {"push": push_rel, "stack": stack_rel}

# the map a PickCube-layout network reads each env through (None: the env has PickCube's layout). ppo.py also needs
# the map's final_key in the env's info, which the stock push / pull / peg tasks do not report.
ENV_ADAPTER = {
    "PickCube-v1": None, "PlaceCube-v1": None,
    "StackCube-v1": "stack_to_pick", "StackCubeNarrow-v1": "stack_to_pick",
    "PlaceSphere-v1": "sphere_to_pick",
    "PickSingleYCB-v1": "ycb_to_pick", "PickYCBSubset-v1": "ycb_to_pick",
    "PushCube-v1": "push_to_pick", "PullCube-v1": "push_to_pick", "PushCubeDir-v1": "push_to_pick",
    "PushRetreat-v1": "push_to_pick", "PushCubeNoGrasp-v1": "push_to_pick", "PullCubeNoGrasp-v1": "push_to_pick",
    "LiftPegUpright-v1": "peg_to_pick", "LiftPegUprightG-v1": "peg_to_pick",
    "PyramidBeside-v1": "beside_to_pick", "PyramidOnPair-v1": "onpair_to_pick",
}
