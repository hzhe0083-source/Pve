"""CPU checks of wrwl/adapters.py against the ManiSkill 3.0.1 layouts, on synthetic tensors (no simulator, no GPU).

  layouts   each source layout transcribed below (the extras' names, in the insertion order of the task's
            _get_obs_extra) against the installed ManiSkill source, parsed with ast (found with importlib, or
            --ms_src=<mani_skill dir>; skipped when neither exists), each field's width taken from its expression
            (raw_pose 7, .p 3, info[...] 1); the dims 42 / 35 / 48 / 39 / 45 / 32 / 64
  maps      every *_to_pick map on observations built field by field in its source layout from random poses (every
            z random, so a z read from the wrong column cannot hide behind resting values), the difference vectors
            computed as the task does: each of PickCube's slots against what it should hold (qpos, qvel, flag,
            tcp_pose, goal, obj_pose, obj - tcp, goal - obj), the goals written out here from the task files,
            independently of adapters.py; plus exact cases (object at its goal -> obj_to_goal 0)
  quats     quat_mul, quat_x_axis and peg_swing against rotation matrices; peg_swing continuous through upright
  registry  TO_PICK / ENV_ADAPTER / ADAPTERS; each env's flag object against a table written from the tasks' success
            tests; the info keys the wrwl env modules add, each assigned is_grasping of the map's own object, an
            actor the parent task builds; scripts/suite_tasks.sh
  ycb       the subset rule of wrwl/envs/ycb_subset.py on synthetic boxes, and the list it gives on this host's YCB
            assets (skipped without ManiSkill or the assets)
  python scripts/test_adapters_cpu.py [--ms_src=.../site-packages/mani_skill]
"""
import argparse
import ast
import importlib.util
import math
import re
import sys
from pathlib import Path

import torch

from wrwl import adapters as A

N = 256
MS = None       # the mani_skill package directory, if found
WRWL = Path(A.__file__).resolve().parent
# extras after qpos (9) and qvel (9), in _get_obs_extra's insertion order: env id -> (file, class, names)
LAYOUTS = {
    "PickCube-v1": ("pick_cube.py", "PickCubeEnv",            # pick_cube.py:132-145
                    ["is_grasped", "tcp_pose", "goal_pos", "obj_pose", "tcp_to_obj_pos", "obj_to_goal_pos"]),
    "PushCube-v1": ("push_cube.py", "PushCubeEnv", ["tcp_pose", "goal_pos", "obj_pose"]),        # push_cube.py:194-207
    "PullCube-v1": ("pull_cube.py", "PullCubeEnv", ["tcp_pose", "goal_pos", "obj_pose"]),        # pull_cube.py:117-126
    "StackCube-v1": ("stack_cube.py", "StackCubeEnv",          # stack_cube.py:133-143
                     ["tcp_pose", "cubeA_pose", "cubeB_pose", "tcp_to_cubeA_pos", "tcp_to_cubeB_pos",
                      "cubeA_to_cubeB_pos"]),
    "PlaceSphere-v1": ("place_sphere.py", "PlaceSphereEnv",    # place_sphere.py:203-214
                       ["is_grasped", "tcp_pose", "bin_pos", "obj_pose", "tcp_to_obj_pos"]),
    "PickSingleYCB-v1": ("pick_single_ycb.py", "PickSingleYCBEnv",     # pick_single_ycb.py:215-228
                         ["tcp_pose", "goal_pos", "is_grasped", "tcp_to_goal_pos", "obj_pose", "tcp_to_obj_pos",
                          "obj_to_goal_pos"]),
    "LiftPegUpright-v1": ("lift_peg_upright.py", "LiftPegUprightEnv", ["tcp_pose", "obj_pose"]),  # :101-109
    "StackPyramid-v1": ("stack_pyramid.py", "StackPyramidEnv",     # stack_pyramid.py:182-196
                        ["tcp_pose", "cubeA_pose", "cubeB_pose", "cubeC_pose", "tcp_to_cubeA_pos", "tcp_to_cubeB_pos",
                         "tcp_to_cubeC_pos", "cubeA_to_cubeB_pos", "cubeB_to_cubeC_pos", "cubeA_to_cubeC_pos"]),
}
DIMS = {"PickCube-v1": 42, "PushCube-v1": 35, "PullCube-v1": 35, "StackCube-v1": 48, "PlaceSphere-v1": 39,
        "PickSingleYCB-v1": 45, "LiftPegUpright-v1": 32, "StackPyramid-v1": 64}
# the wrwl env modules: registered id -> (module, parent task); none overrides _get_obs_extra
VARIANTS = {"PushCubeNoGrasp-v1": ("push_nograsp.py", "PushCube-v1"),
            "PullCubeNoGrasp-v1": ("push_nograsp.py", "PullCube-v1"),
            "PickYCBSubset-v1": ("ycb_subset.py", "PickSingleYCB-v1"),
            "LiftPegUprightG-v1": ("lift_peg.py", "LiftPegUpright-v1"),
            "PyramidBeside-v1": ("pyramid_steps.py", "StackPyramid-v1"),
            "PyramidOnPair-v1": ("pyramid_steps.py", "StackPyramid-v1"),
            "StackPyramidNarrow-v1": ("pyramid_steps.py", "StackPyramid-v1")}


def width(name):
    return 1 if name == "is_grasped" else 7 if name.endswith("_pose") else 3


def close(a, b, what, atol=1e-6):
    err = (a.float() - b.float()).abs().max().item()
    assert err <= atol, f"{what}: max error {err:.2e}"


def unit(v):
    return v / torch.linalg.norm(v, dim=1, keepdim=True)


def rot(q):
    """(n, 3, 3) rotation matrices of (n, 4) unit wxyz quaternions."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rows = [torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=1)]
    return torch.stack(rows, dim=1)


def axis_quat(axis, angle, n):
    s, c = math.sin(angle / 2), math.cos(angle / 2)
    q = [c, 0.0, 0.0, 0.0]
    q[1 + "xyz".index(axis)] = s
    return torch.tensor([q] * n)


def rand_pose(n, z=None):
    p = torch.rand(n, 3) * 0.3 - 0.15
    if z is not None:
        p[:, 2] = z
    return torch.cat([p, unit(torch.randn(n, 4))], dim=1)


def build(env_id, f):
    """The flat observation of env_id from primary fields f (qpos, qvel, poses, goal / bin, flag), each difference
    vector computed as the task's _get_obs_extra does (pick_cube.py:142-143, stack_cube.py:139-141,
    pick_single_ycb.py:223-226, place_sphere.py:212, stack_pyramid.py:189-194)."""
    p = {k: v[:, :3] for k, v in f.items() if k.endswith("_pose")}
    diff = {"tcp_to_goal_pos": lambda: f["goal_pos"] - p["tcp_pose"],
            "tcp_to_obj_pos": lambda: p["obj_pose"] - p["tcp_pose"],
            "obj_to_goal_pos": lambda: f["goal_pos"] - p["obj_pose"],
            "cubeA_to_cubeB_pos": lambda: p["cubeB_pose"] - p["cubeA_pose"],
            "cubeB_to_cubeC_pos": lambda: p["cubeC_pose"] - p["cubeB_pose"],
            "cubeA_to_cubeC_pos": lambda: p["cubeC_pose"] - p["cubeA_pose"]}
    for c in "ABC":
        diff[f"tcp_to_cube{c}_pos"] = (lambda c=c: p[f"cube{c}_pose"] - p["tcp_pose"])
    cols = [f["qpos"], f["qvel"]]
    for name in LAYOUTS[env_id][2]:
        v = f[name] if name in f else diff[name]()
        cols.append(v.reshape(len(v), -1).float())       # a flag is one column (common.py flatten_state_dict)
    x = torch.cat(cols, dim=1)
    assert x.shape == (len(x), DIMS[env_id]), (env_id, x.shape)
    return x


def fields(n, **poses):
    f = {"qpos": torch.randn(n, 9), "qvel": torch.randn(n, 9), "tcp_pose": rand_pose(n),
         "is_grasped": torch.rand(n) < 0.5}
    f.update(poses)
    return f


def check_pick(out, f, flag, goal, obj_pose, what):
    """out against PickCube's slots (pick_cube.py:132-145): qpos 0:9, qvel 9:18, is_grasped 18, tcp_pose 19:26,
    goal 26:29, obj_pose 29:36, tcp_to_obj 36:39 (obj - tcp), obj_to_goal 39:42 (goal - obj)."""
    assert out.shape == (len(goal), 42), f"{what}: shape {tuple(out.shape)}"
    close(out[:, 0:9], f["qpos"], f"{what} qpos")
    close(out[:, 9:18], f["qvel"], f"{what} qvel")
    close(out[:, 18], flag.float(), f"{what} is_grasped")
    close(out[:, 19:26], f["tcp_pose"], f"{what} tcp_pose")
    close(out[:, 26:29], goal, f"{what} goal")
    close(out[:, 29:36], obj_pose, f"{what} obj_pose")
    close(out[:, 36:39], obj_pose[:, :3] - f["tcp_pose"][:, :3], f"{what} tcp_to_obj")
    close(out[:, 39:42], goal - obj_pose[:, :3], f"{what} obj_to_goal")


# --------------------------------------------------------------------------------------------------------------------
def expr_width(e):
    """A field's width from its expression: x.raw_pose 7, x.p 3, info[...] 1, a difference of two equal widths."""
    if isinstance(e, ast.Attribute) and e.attr == "raw_pose":
        return 7
    if isinstance(e, ast.Attribute) and e.attr == "p":
        return 3
    if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) and e.value.id == "info":
        return 1
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Sub):
        a, b = expr_width(e.left), expr_width(e.right)
        assert a == b, f"widths {a} and {b} in {ast.unparse(e)}"
        return a
    raise AssertionError(f"no width rule for {ast.unparse(e)}")


def extras_in_source(path, cls):
    """(name, width) of each keyword of the dict(...) and obs.update(...) calls in cls._get_obs_extra, in source
    order, the width from the keyword's expression."""
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "_get_obs_extra":
                    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and (
                        (isinstance(c.func, ast.Name) and c.func.id == "dict")
                        or (isinstance(c.func, ast.Attribute) and c.func.attr == "update"))]
                    calls.sort(key=lambda c: (c.lineno, c.col_offset))
                    return [(k.arg, expr_width(k.value)) for c in calls for k in c.keywords]
    raise KeyError(f"{cls}._get_obs_extra not in {path}")


def test_layouts_match_source():
    if MS is None:
        print("    (skipped: no ManiSkill source; pass --ms_src)")
        return
    for env_id, (file, cls, names) in LAYOUTS.items():
        found = extras_in_source(MS / "envs/tasks/tabletop" / file, cls)
        assert [k for k, _ in found] == names, f"{env_id}: source {[k for k, _ in found]}, transcribed {names}"
        assert [w for _, w in found] == [width(k) for k in names], \
            f"{env_id}: widths in the source {found}, by name {[width(k) for k in names]}"
        assert 18 + sum(w for _, w in found) == DIMS[env_id], f"{env_id}: {18 + sum(w for _, w in found)} dims"
    sb = (MS / "envs/tasks/tabletop/stack_cube.py").read_text()
    assert '"is_cubeA_grasped": is_cubeA_grasped' in sb, "StackCube's evaluate() no longer returns is_cubeA_grasped"


def test_layout_dims():
    for env_id, (_, _, names) in LAYOUTS.items():
        assert 18 + sum(width(k) for k in names) == DIMS[env_id], env_id


def test_pick_layout_slots():
    """The slots check_pick assumes are PickCube's own layout: identity on a PickCube observation passes it."""
    n = N
    f = fields(n, goal_pos=torch.rand(n, 3), obj_pose=rand_pose(n))
    x = build("PickCube-v1", f)
    check_pick(A.identity(x), f, f["is_grasped"], f["goal_pos"], f["obj_pose"], "PickCube identity")


def test_stack_to_pick():
    n = N
    f = fields(n, cubeA_pose=rand_pose(n), cubeB_pose=rand_pose(n))
    g = torch.rand(n) < 0.5
    out = A.stack_to_pick(build("StackCube-v1", f), g)
    goal = f["cubeB_pose"][:, :3] + torch.tensor([0.0, 0.0, 0.04])         # cubeA resting on cubeB
    check_pick(out, f, g, goal, f["cubeA_pose"], "stack_to_pick")


def test_push_to_pick():
    n = N
    for env_id in ("PushCube-v1", "PullCube-v1"):
        target = torch.rand(n, 3) * 0.4 - 0.2
        # every z random (the disc sits at 1e-3 and a resting cube at 0.02, push_cube.py:157,171): a z read from the
        # wrong column cannot hide behind the resting values
        f = fields(n, goal_pos=target, obj_pose=rand_pose(n))
        g = torch.rand(n) < 0.5
        goal = torch.cat([target[:, :2], torch.full((n, 1), 0.02)], dim=1)  # a resting cube's centre
        check_pick(A.push_to_pick(build(env_id, f), g), f, g, goal, f["obj_pose"], f"push_to_pick {env_id}")


def test_sphere_to_pick():
    n = N
    bin_pos = torch.rand(n, 3) * 0.2 - 0.1
    # bin z random (it is 0.0025 in the task, place_sphere.py:179): the goal must follow the bin's own z
    f = fields(n, bin_pos=bin_pos, obj_pose=rand_pose(n))
    x = build("PlaceSphere-v1", f)
    goal = bin_pos + torch.tensor([0.0, 0.0, 0.02 + 0.0025])                # radius + floor half thickness
    obj = torch.cat([f["obj_pose"][:, :3], torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n)], dim=1)
    check_pick(A.sphere_to_pick(x), f, f["is_grasped"], goal, obj, "sphere_to_pick (observed flag)")
    g = ~f["is_grasped"]
    check_pick(A.sphere_to_pick(x, g), f, g, goal, obj, "sphere_to_pick (given flag)")
    # a sphere resting in the bin as evaluate() wants it (place_sphere.py:188-191): at its goal
    f["obj_pose"] = torch.cat([goal, unit(torch.randn(n, 4))], dim=1)
    close(A.sphere_to_pick(build("PlaceSphere-v1", f))[:, 39:42], torch.zeros(n, 3), "sphere in the bin", 1e-7)


def test_ycb_to_pick():
    n = N
    f = fields(n, goal_pos=torch.rand(n, 3), obj_pose=rand_pose(n))
    x = build("PickSingleYCB-v1", f)
    check_pick(A.ycb_to_pick(x), f, f["is_grasped"], f["goal_pos"], f["obj_pose"], "ycb_to_pick")
    idx = list(range(18)) + [28] + list(range(18, 28)) + list(range(32, 45))     # a permutation, tcp_to_goal dropped
    assert torch.equal(A.ycb_to_pick(x), x[:, idx]), "ycb_to_pick is not the column permutation"


def test_peg_to_pick():
    n = N
    f = fields(n, obj_pose=rand_pose(n))
    g = torch.rand(n) < 0.5
    out = A.peg_to_pick(build("LiftPegUpright-v1", f), g)
    p, q = f["obj_pose"][:, :3], f["obj_pose"][:, 3:]
    u = rot(q)[:, :, 0]                                    # the peg's long axis (half sizes 0.12, 0.025, 0.025)
    grip = p + 0.09 * u
    foot = p - 0.12 * u
    goal = torch.cat([foot[:, :2], torch.full((n, 1), 0.21)], dim=1)      # grip point of the peg upright on its foot
    check_pick(out, f, g, goal, torch.cat([grip, out[:, 32:36]], dim=1), "peg_to_pick")
    close(rot(out[:, 32:36])[:, :, 0], u, "peg_to_pick keeps the long axis", 1e-5)
    # lying at spawn (lift_peg_upright.py:81-84): identity orientation at every resting roll, goal 0.21 above the
    # -x end
    for k in range(4):
        q_k = A.quat_mul(axis_quat("x", math.pi / 2, n), axis_quat("x", k * math.pi / 2, n))
        spawn = torch.cat([torch.rand(n, 2) * 0.2 - 0.1, torch.full((n, 1), 0.025), q_k], 1)
        f["obj_pose"] = spawn
        out = A.peg_to_pick(build("LiftPegUpright-v1", f), g)
        close(out[:, 32:36], torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n), f"spawn, roll {90 * k}: reads as identity", 1e-6)
    close(out[:, 29:32], spawn[:, :3] + torch.tensor([0.09, 0.0, 0.0]), "spawn grip point", 1e-6)
    close(out[:, 26:29], torch.cat([spawn[:, :2] - torch.tensor([0.12, 0.0]), torch.full((n, 1), 0.21)], 1),
          "spawn goal", 1e-6)
    # upright on its -x end, centre at 0.12 (lift_peg_upright.py:96): the grip point is at its goal
    up = torch.cat([torch.rand(n, 2) * 0.2 - 0.1, torch.full((n, 1), 0.12),
                    A.quat_mul(axis_quat("y", -math.pi / 2, n), axis_quat("x", math.pi / 2, n))], dim=1)
    f["obj_pose"] = up
    close(A.peg_to_pick(build("LiftPegUpright-v1", f), g)[:, 39:42], torch.zeros(n, 3), "upright peg at goal", 1e-6)
    # continuous through upright, inside the success window (|tilt - 90 deg| < 0.08 rad, lift_peg_upright.py:89-96):
    # tilts of 89.9 / 90 / 90.1 deg about world y from the spawn pose, each wobbled by 0.5 deg about world x or y, move
    # the orientation slot by at most the rotation angle between them
    qs, spawn_q = [], axis_quat("x", math.pi / 2, 1)
    for tilt in (89.9, 90.0, 90.1):
        base = A.quat_mul(axis_quat("y", -math.radians(tilt), 1), spawn_q)
        qs.append((tilt, "none", base))
        for ax in "xy":
            for sgn in (1, -1):
                qs.append((tilt, f"{'+' if sgn > 0 else '-'}{ax}",
                           A.quat_mul(axis_quat(ax, sgn * math.radians(0.5), 1), base)))
    for i in range(len(qs)):
        for j in range(i):
            qi, qj = qs[i][2], qs[j][2]
            angle = 2 * math.acos(min(1.0, abs(float((qi * qj).sum()))))
            d = float(torch.linalg.norm(A.peg_swing(qi) - A.peg_swing(qj), dim=1)[0])
            assert d <= angle + 1e-6, f"peg orientation jumps near upright: {qs[i][:2]} vs {qs[j][:2]}: {d:.3f} > {angle:.4f}"


def test_beside_to_pick():
    n = N
    f = fields(n, cubeA_pose=rand_pose(n), cubeB_pose=rand_pose(n), cubeC_pose=rand_pose(n))     # z random too
    g = torch.rand(n) < 0.5
    out = A.beside_to_pick(build("StackPyramid-v1", f), g)
    a, b = f["cubeA_pose"][:, :3], f["cubeB_pose"][:, :3]
    goal = torch.cat([b[:, :2] + 0.05 * unit(a[:, :2] - b[:, :2]), torch.full((n, 1), 0.02)], dim=1)
    check_pick(out, f, g, goal, f["cubeA_pose"], "beside_to_pick")
    close(torch.linalg.norm(goal[:, :2] - b[:, :2], dim=1), torch.full((n,), 0.05), "goal 5 cm from cubeB", 1e-6)
    # cubeA already 5 cm from cubeB on the table: at its goal
    d = unit(torch.randn(n, 2))
    b = torch.cat([b[:, :2], torch.full((n, 1), 0.02)], dim=1)
    f["cubeB_pose"] = torch.cat([b, f["cubeB_pose"][:, 3:]], dim=1)
    f["cubeA_pose"] = torch.cat([b[:, :2] + 0.05 * d, torch.full((n, 1), 0.02), unit(torch.randn(n, 4))], dim=1)
    close(A.beside_to_pick(build("StackPyramid-v1", f), g)[:, 39:42], torch.zeros(n, 3), "cubeA beside", 1e-6)


def test_onpair_to_pick():
    n = N
    f = fields(n, cubeA_pose=rand_pose(n), cubeB_pose=rand_pose(n), cubeC_pose=rand_pose(n))     # z random too
    g = torch.rand(n) < 0.5
    out = A.onpair_to_pick(build("StackPyramid-v1", f), g)
    goal = (f["cubeA_pose"][:, :3] + f["cubeB_pose"][:, :3]) / 2 + torch.tensor([0.0, 0.0, 0.04])
    check_pick(out, f, g, goal, f["cubeC_pose"], "onpair_to_pick")


def test_quaternions():
    n = N
    a, b = unit(torch.randn(n, 4)), unit(torch.randn(n, 4))
    close(rot(A.quat_mul(a, b)), rot(a) @ rot(b), "quat_mul composes rotations", 1e-5)
    close(A.quat_x_axis(a), rot(a)[:, :, 0], "quat_x_axis", 1e-6)
    u = rot(a)[:, :, 0]
    ok = u[:, 0] > -0.99                                    # away from the end-over-end singularity
    a, u = a[ok], u[ok]
    s = A.peg_swing(a)
    close(torch.linalg.norm(s, dim=1), torch.ones(len(s)), "peg_swing unit norm", 1e-5)
    assert (s[:, 0] >= 0).all(), "peg_swing: w < 0"
    close(rot(s)[:, :, 0], u, "peg_swing keeps the long axis", 1e-5)
    close(s[:, 1], torch.zeros(len(s)), "peg_swing: no roll about the world x axis", 1e-7)
    for ang in (0.3, math.pi / 2, 2.0):                       # roll about the peg's own axis does not change it
        close(A.peg_swing(A.quat_mul(a, axis_quat("x", ang, len(a)))), s, f"peg_swing roll invariant ({ang:.2f})", 1e-5)
    psi = torch.rand(n) * 6.0 - 3.0                           # yaw on the table: yaw-only, w > 0
    qz = torch.stack([torch.cos(psi / 2), torch.zeros(n), torch.zeros(n), torch.sin(psi / 2)], dim=1)
    close(A.peg_swing(A.quat_mul(qz, axis_quat("x", math.pi / 2, n))), qz, "peg_swing: a lying peg's yaw", 1e-5)


# The object each task's success test is about, written from the task files independently of TO_PICK: the env
# attribute whose grasp flag a map that takes the flag from the env must read (None: the observation has the flag).
TASK_OBJECT = {"StackCube-v1": "cubeA", "StackCubeNarrow-v1": "cubeA",        # stack_cube.py: cubeA on cubeB
               "PushCube-v1": "obj", "PullCube-v1": "obj", "PushCubeDir-v1": "obj", "PushRetreat-v1": "obj",
               "PushCubeNoGrasp-v1": "obj", "PullCubeNoGrasp-v1": "obj",      # push_cube.py / pull_cube.py self.obj
               "LiftPegUpright-v1": "peg", "LiftPegUprightG-v1": "peg",       # lift_peg_upright.py self.peg
               "PyramidBeside-v1": "cubeA",                                   # A next to B (stack_pyramid.py:170-172)
               "PyramidOnPair-v1": "cubeC",                                   # C on A and B (stack_pyramid.py:173-174)
               "PlaceSphere-v1": None, "PickSingleYCB-v1": None, "PickYCBSubset-v1": None}    # observed flag


def test_registry():
    assert A.ADAPTERS["push"] is A.identity and A.ADAPTERS["stack_to_pick"] is A.stack_to_pick
    assert set(A.REL) == {"push", "stack"}
    assert A.STACK_GOAL_DZ == 0.04
    for name, t in A.TO_PICK.items():
        assert A.ADAPTERS[name] is t.fn, name
        assert (t.obj is None) == (t.final_key is None), f"{name}: obj and final_key go together"
    for env_id, name in A.ENV_ADAPTER.items():
        assert name is None or name in A.TO_PICK, env_id
        if name is not None:
            assert env_id in TASK_OBJECT, f"{env_id}: add the object its success test is about to TASK_OBJECT"
            assert A.TO_PICK[name].obj == TASK_OBJECT[env_id], \
                f"{env_id} reads {name}, whose flag is about {A.TO_PICK[name].obj}; the task is about {TASK_OBJECT[env_id]}"
    for env_id, (module, parent) in VARIANTS.items():      # StackPyramidNarrow needs two maps, one per sub-skill
        assert env_id in A.ENV_ADAPTER or env_id == "StackPyramidNarrow-v1", f"{env_id} not in ENV_ADAPTER"
        if parent in A.ENV_ADAPTER:
            assert A.ENV_ADAPTER[env_id] == A.ENV_ADAPTER[parent], env_id


def test_env_modules():
    """The wrwl variants keep their parent's observation and report the flags TO_PICK reads at final observations."""
    for env_id, (module, parent) in VARIANTS.items():
        src = (WRWL / "envs" / module).read_text()
        assert f'"{env_id}"' in src, f"{module} does not register {env_id}"
        assert "_get_obs_extra" not in src, f"{module} changes the observation"
        cls = LAYOUTS[parent][1]
        assert re.search(rf"\b{cls}\b", src), f"{module}: not derived from {cls}"
        name = A.ENV_ADAPTER.get(env_id)
        key = A.TO_PICK[name].final_key if name else None
        if key:
            obj = A.TO_PICK[name].obj
            m = re.findall(rf'info\["{key}"\]\s*=\s*self\.agent\.is_grasping\(self\.(\w+)\)', src)
            assert m == [obj], f"{module}: info[{key!r}] is is_grasping of {m}, the map's flag is about {obj!r}"
            if MS is not None:     # the parent task builds that actor (lift_peg_upright.py:65 self.peg = ...)
                psrc = (MS / "envs/tasks/tabletop" / LAYOUTS[parent][0]).read_text()
                assert re.search(rf"self\.{obj}\s*=", psrc), f"{LAYOUTS[parent][0]} builds no self.{obj}"


def test_ycb_rule():
    """wrwl/envs/ycb_subset.py's subset rule on synthetic boxes; with the YCB assets, the list it gives here."""
    try:
        import wrwl.envs.ycb_subset as Y
    except ImportError as e:
        print(f"    (skipped: {e})")
        return
    box = lambda x, y, z, s=1.0: {"bbox": {"min": [-x / 2, -y / 2, -z / 2], "max": [x / 2, y / 2, z / 2]},  # noqa: E731
                                  "scales": [s]}
    info = {"can": box(0.066, 0.066, 0.101), "edge": box(0.12, 0.07, 0.12), "brick": box(0.05, 0.075, 0.05),
            "scaled_in": box(0.01, 0.01, 0.015, 5.0),                     # 5 x 5 x 7.5 cm at scale 5 (1 cm unscaled)
            "scaled_out": box(0.05, 0.05, 0.1, 1.5),                      # 7.5 x 7.5 x 15 cm at scale 1.5
            "tuna": box(0.085, 0.085, 0.033),                             # both horizontal extents > 7 cm
            "marker": box(0.121, 0.018, 0.018),                           # thinner than 2 cm
            "banana": box(0.198, 0.036, 0.04),                            # longer than 12 cm
            "tall": box(0.06, 0.06, 0.14), "toobig": box(0.102, 0.102, 0.139), "tiny": box(0.016, 0.016, 0.016)}
    got = Y.rule_models(list(info), info)
    assert got == ["can", "edge", "brick", "scaled_in"], got
    j = Path(str(Y.ASSET_DIR)) / "assets/mani_skill2_ycb/info_pick_v0.json"
    if not j.exists():
        print(f"    (no YCB assets at {j}: python -m mani_skill.utils.download_asset ycb -y)")
        return
    real = Y.load_json(j)
    ids = [k for k in real if k not in ("022_windex_bottle", "028_skillet_lid", "029_plate", "059_chain")]  # :71-77
    keep = Y.rule_models(ids, real)
    print(f"    the rule keeps {len(keep)} of {len(ids)} models: " +
          ", ".join(f"{m} {'x'.join(f'{100 * v:.1f}' for v in Y.model_extents(real, m))} cm" for m in keep))
    assert len(keep) >= Y.RULE_MIN_MODELS, keep


def test_suite_tasks_table():
    """scripts/suite_tasks.sh's env ids and --obs_adapter names agree with ENV_ADAPTER."""
    sh = Path(__file__).resolve().parent / "suite_tasks.sh"
    if not sh.exists():
        print("    (skipped: no suite_tasks.sh next to this script)")
        return
    rows = re.findall(r"^\s*(\w+)\) echo \"--env_id=(\S+) .*?(?:--obs_adapter=(\w+))?\" ;;", sh.read_text(), re.M)
    assert len(rows) >= 9, rows
    for key, env_id, name in rows:
        assert "_" not in key, f"task key {key} has an underscore"
        assert env_id in A.ENV_ADAPTER, f"{key}: {env_id} has no entry in ENV_ADAPTER"
        assert (A.ENV_ADAPTER[env_id] or "") == name, f"{key}: suite_tasks.sh {name!r}, ENV_ADAPTER {A.ENV_ADAPTER[env_id]!r}"


def main():
    global MS
    p = argparse.ArgumentParser()
    p.add_argument("--ms_src", default="", help="the mani_skill package directory (default: the installed one)")
    a = p.parse_args()
    if a.ms_src:
        MS = Path(a.ms_src)
    else:
        spec = importlib.util.find_spec("mani_skill")
        MS = Path(spec.submodule_search_locations[0]) if spec and spec.submodule_search_locations else None
    print(f"ManiSkill source: {MS}")
    torch.manual_seed(0)
    fails = 0
    for name, fn in [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {name}: {e}")
    print("ALL OK" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
