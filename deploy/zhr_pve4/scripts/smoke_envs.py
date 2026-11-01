"""GPU smoke test of the task suite (physx_cuda, --envs per task, default 4). For every task key it builds the env as
scripts/suite_tasks.sh does (and checks that table agrees with this one), maps the observation with the env's adapter
(wrwl.adapters.ENV_ADAPTER) and checks, at the reset and at steps 1, 10, 25 and 50 of the expert's rollouts:
  - the env's flat dim, and the mapped (n, 42), finite;
  - every Pick slot against the simulator: qpos, qvel, the grasp flag (agent.is_grasping of the task's object),
    tcp_pose, the goal (recomputed here from the actors), the object's position and orientation (LiftPeg: the swing
    from e_x to its long axis, from ManiSkill's rotation matrix), and tcp_to_obj = obj - tcp, obj_to_goal = goal - obj;
  - for maps whose flag comes from the env: the map's object is this table's, info[final_key] equals is_grasping of
    that object after every step, and after an auto-reset in ManiSkillVectorEnv final_info has it and the final
    observation maps, as wrwl/ppo.py reads them; for the NoGrasp tasks, info["fail"] equals "grasped at some step
    since the reset" after every step.
It then rolls out the Pick expert for one episode per env, deterministic (det) and with its std and the gripper std
floored at 1.0 (floor, iteration 0 of the --min_grip_std=1.0 runs), and prints success_once, grasped once and, for
the NoGrasp tasks, fail rates. At 4 envs the rates only show the task runs; --envs=256 gives zero-shot numbers
(diagnostic only: gate Z is scripts/nsa_eval.py's number, scripts/suite_gate.py).
PickYCBSubset also prints the models wrwl/envs/ycb_subset.py's rule keeps, with their extents from the asset json and
from the collision meshes, and per model the det success. The rule fixes the list; these numbers never edit it.
--time_eval also times one evaluation as wrwl/ppo.py runs it (--eval_envs envs built with reconfiguration_freq=1,
a reset, 50 steps and the auto-reset at the end), per task, for the GPU budget (YCB rebuilds its mesh scene twice).
Run it like the queue's jobs, so it waits for and yields to other users' GPU jobs:
  bash scripts/with_gpu_mem.sh 6000 bash scripts/gpu_polite.sh python scripts/smoke_envs.py \
      --pick=runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt [--tasks=sphere,ycb] [--envs=4] [--time_eval] [--out=x.json]
"""
import argparse
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch
from mani_skill.utils import gym_utils
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import wrwl.envs.lift_peg  # noqa: F401
import wrwl.envs.place_cube  # noqa: F401
import wrwl.envs.push_nograsp  # noqa: F401
import wrwl.envs.pyramid_steps  # noqa: F401
import wrwl.envs.stack_narrow  # noqa: F401
import wrwl.envs.ycb_subset as ycb_subset
from wrwl.adapters import ENV_ADAPTER, TO_PICK
from wrwl.nsa import build_actor

WC = {"robot_uids": "panda_wristcam", "reward_mode": "sparse"}
# task key: (env id, env kwargs as in scripts/suite_tasks.sh, flat state dim, the actor the grasp flag is about)
TASKS = {"place": ("PlaceCube-v1", WC, 42, "cube"),
         "stack": ("StackCubeNarrow-v1", {"reward_mode": "sparse"}, 48, "cubeA"),
         "sphere": ("PlaceSphere-v1", WC, 39, "obj"),
         "ycb": ("PickYCBSubset-v1", WC, 45, "obj"),
         "pushng": ("PushCubeNoGrasp-v1", WC, 35, "obj"),
         "pullng": ("PullCubeNoGrasp-v1", WC, 35, "obj"),
         "beside": ("PyramidBeside-v1", WC, 64, "cubeA"),
         "onpair": ("PyramidOnPair-v1", WC, 64, "cubeC"),
         "liftpeg": ("LiftPegUprightG-v1", WC, 32, "peg")}
TOL = 1e-4
p = argparse.ArgumentParser()
p.add_argument("--pick", default="runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt")
p.add_argument("--tasks", default=",".join(TASKS))
p.add_argument("--envs", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="")
p.add_argument("--time_eval", action="store_true", help="time one ppo.py-style evaluation per task")
p.add_argument("--eval_envs", type=int, default=256)
a = p.parse_args()
torch.set_grad_enabled(False)
dev = "cuda"
sd = torch.load(a.pick, map_location="cpu")
actor = build_actor(sd, 42, 8).to(dev).eval()
std = sd["actor_logstd"].reshape(1, -1).exp().to(dev)
std[:, -1] = std[:, -1].clamp(min=1.0)      # wrwl/ppo.py Agent.dist with --min_grip_std=1.0


def queue_flags(key):
    """(env id, env kwargs, adapter) of `task_env key` in scripts/suite_tasks.sh, or None without the file."""
    sh = Path(__file__).resolve().parent / "suite_tasks.sh"
    if not sh.exists():
        return None
    out = subprocess.run(["bash", "-c", f'source "{sh}"; task_env {key}'], capture_output=True, text=True).stdout
    env_id = re.search(r"--env_id=(\S+)", out)
    kw = re.search(r"--env_kwargs='([^']*)'", out)
    ad = re.search(r"--obs_adapter=(\w+)", out)
    return (env_id and env_id.group(1), kw and json.loads(kw.group(1)), ad and ad.group(1))


def expected(key, b):
    """(goal, object position, object quaternion) from the actors, as the task's map should report them (LiftPeg:
    the swing taking e_x to the peg's long axis, computed here from ManiSkill's rotation matrix)."""
    up = lambda z: torch.tensor([0.0, 0.0, z], device=b.device)    # noqa: E731
    if key == "place":
        return b.goal_site.pose.p, b.cube.pose.p, b.cube.pose.q
    if key == "stack":
        return b.cubeB.pose.p + up(0.04), b.cubeA.pose.p, b.cubeA.pose.q
    if key == "sphere":         # the sphere's centre resting in the bin (place_sphere.py:188-191); identity orientation
        q = torch.zeros_like(b.obj.pose.q)
        q[:, 0] = 1.0
        return b.bin.pose.p + up(b.radius + b.block_half_size[0]), b.obj.pose.p, q
    if key == "ycb":
        return b.goal_site.pose.p, b.obj.pose.p, b.obj.pose.q
    if key in ("pushng", "pullng"):     # the target's xy at a resting cube's centre height
        g = b.goal_region.pose.p.clone()
        g[:, 2] = b.cube_half_size
        return g, b.obj.pose.p, b.obj.pose.q
    if key == "beside":                 # 5 cm from cubeB towards cubeA, on the table
        pa, pb = b.cubeA.pose.p, b.cubeB.pose.p
        d = (pa - pb)[:, :2]
        g = pb.clone()
        g[:, :2] = pb[:, :2] + 0.05 * d / torch.linalg.norm(d, dim=1, keepdim=True)
        g[:, 2] = 0.02
        return g, pa, b.cubeA.pose.q
    if key == "onpair":
        return (b.cubeA.pose.p + b.cubeB.pose.p) / 2 + up(0.04), b.cubeC.pose.p, b.cubeC.pose.q
    if key == "liftpeg":                # grip point 0.09 towards +x; goal 0.21 above the -x end
        u = quaternion_to_matrix(b.peg.pose.q)[:, :, 0]
        g = b.peg.pose.p - b.peg_half_length * u
        g[:, 2] = b.peg_half_length + 0.09
        sw = torch.stack([1 + u[:, 0], torch.zeros_like(u[:, 0]), -u[:, 2], u[:, 1]], dim=1)
        return g, b.peg.pose.p + 0.09 * u, sw / torch.linalg.norm(sw, dim=1, keepdim=True)
    raise KeyError(key)


def slot_errors(key, x, b, obj):
    goal, obj_p, obj_q = expected(key, b)
    err = {"qpos": x[:, 0:9] - b.agent.robot.get_qpos(), "qvel": x[:, 9:18] - b.agent.robot.get_qvel(),
           "flag": x[:, 18] - b.agent.is_grasping(getattr(b, obj)).float(), "tcp": x[:, 19:26] - b.agent.tcp.pose.raw_pose,
           "goal": x[:, 26:29] - goal, "obj_p": x[:, 29:32] - obj_p,
           "tcp_to_obj": x[:, 36:39] - (x[:, 29:32] - x[:, 19:22]),
           "obj_to_goal": x[:, 39:42] - (x[:, 26:29] - x[:, 29:32])}
    err["obj_q"] = x[:, 32:36] - obj_q
    if key == "liftpeg":            # the slot's long axis is the peg's
        err["obj_axis"] = quaternion_to_matrix(x[:, 32:36])[:, :, 0] - quaternion_to_matrix(b.peg.pose.q)[:, :, 0]
    return {k: v.abs().max().item() for k, v in err.items()}


def run(key):
    env_id, kw, dim, obj = TASKS[key]
    row = {"env": env_id, "adapter": ENV_ADAPTER[env_id]}
    q = queue_flags(key)
    if q is not None:
        assert q == (env_id, kw, ENV_ADAPTER[env_id]), f"suite_tasks.sh {q} != {(env_id, kw, ENV_ADAPTER[env_id])}"
    ad = TO_PICK[ENV_ADAPTER[env_id]] if ENV_ADAPTER[env_id] else None
    assert ad is None or ad.obj is None or ad.obj == obj, f"TO_PICK says the flag is about {ad.obj}, this table {obj}"
    env = gym.make(env_id, num_envs=a.envs, obs_mode="state", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cuda", render_mode=None, **kw)
    b = env.unwrapped
    row["max_episode_steps"] = gym_utils.find_max_episode_steps_value(env)

    def view(o):
        if ad is None:
            return o
        return ad.fn(o, b.agent.is_grasping(getattr(b, ad.obj)) if ad.obj else None)

    worst = defaultdict(float)

    def check(o):
        assert o.shape == (a.envs, dim), f"flat obs {tuple(o.shape)}, expected ({a.envs}, {dim})"
        x = view(o)
        assert x.shape == (a.envs, 42) and torch.isfinite(x).all(), f"mapped obs {tuple(x.shape)} or not finite"
        for k, v in slot_errors(key, x, b, obj).items():
            worst[k] = max(worst[k], v)

    obs, info = env.reset(seed=a.seed)
    check(obs)
    if ad is not None and ad.obj is not None:
        assert ad.final_key in info, f"info has no {ad.final_key}"
    torch.manual_seed(a.seed)
    for pol in ("det", "floor"):
        obs, info = env.reset(seed=a.seed)
        models = [o.name.rsplit("-", 1)[0] for o in b._objs] if key == "ycb" else None
        succ = torch.zeros(a.envs, dtype=torch.bool, device=dev)
        held, fail = succ.clone(), succ.clone()
        ever = succ.clone()     # grasped at some step since the reset (the NoGrasp latch)
        for t in range(1, row["max_episode_steps"] + 1):
            mean = actor(view(obs))
            act = mean if pol == "det" else mean + std * torch.randn_like(mean)
            obs, _, _, _, info = env.step(act.clamp(-1, 1))
            if t in (1, 10, 25, 50):
                check(obs)
            g_now = b.agent.is_grasping(getattr(b, obj))
            if ad is not None and ad.obj is not None:    # what ppo.py reads at final observations, at every step
                assert torch.equal(info[ad.final_key], g_now), f"info[{ad.final_key!r}] is not is_grasping({obj})"
            ever |= g_now
            if "fail" in info:
                assert torch.equal(info["fail"], ever), "fail is not 'grasped at some step since the reset'"
            succ |= info["success"]
            held |= b.agent.is_grasping(getattr(b, obj))
            if "fail" in info:
                fail |= info["fail"]
        row[pol] = {"success": succ.float().mean().item(), "grasped": held.float().mean().item(),
                    "fail": fail.float().mean().item() if "fail" in info else None}
        if models is not None and pol == "det":
            per = defaultdict(list)
            for m, s in zip(models, succ.tolist()):
                per[m].append(s)
            row["ycb_det_success"] = {m: [sum(v) / len(v), len(v)] for m, v in sorted(per.items())}
    if key == "ycb":
        info_json = ycb_subset.load_json(ycb_subset.ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")
        row["ycb_rule_models"] = list(b.all_model_ids)
        assert row["ycb_rule_models"] == ycb_subset.rule_models(b.ycb_ids.tolist(), info_json), "not the rule's list"
        row["ycb_json_cm"] = {m: [round(float(e) * 100, 1) for e in ycb_subset.model_extents(info_json, m)]
                              for m in row["ycb_rule_models"]}
        row["ycb_mesh_cm"] = {o.name.rsplit("-", 1)[0]: [round(float(e) * 100, 1) for e in
                                                         o.get_first_collision_mesh(to_world_frame=False).bounding_box.extents]
                              for o in b._objs}
        row["ycb_all_models"] = sorted(b.ycb_ids.tolist())
    row["slot_max_error"] = dict(worst)
    bad = {k: v for k, v in worst.items() if v > TOL}
    assert not bad, f"slots off by more than {TOL}: {bad}"
    if ad is not None and ad.obj is not None:   # one auto-reset through ManiSkillVectorEnv, as in training
        venv = ManiSkillVectorEnv(env, a.envs, ignore_terminations=False, record_metrics=True)
        obs, _ = venv.reset(seed=a.seed)
        for t in range(row["max_episode_steps"] + 5):
            g = b.agent.is_grasping(getattr(b, ad.obj))
            obs, _, _, _, infos = venv.step(actor(ad.fn(obs, g)).clamp(-1, 1))
            if "final_info" in infos:
                assert ad.final_key in infos["final_info"], f"final_info has no {ad.final_key}"
                x = ad.fn(infos["final_observation"], infos["final_info"][ad.final_key])
                assert x.shape == (a.envs, 42) and torch.isfinite(x).all(), "final observation does not map"
                row["final_info"] = f"{ad.final_key} at step {t + 1}"
                break
        else:
            raise AssertionError("no episode ended")
    env.close()
    if a.time_eval:     # one evaluation as wrwl/ppo.py runs it (ppo.py:687-691, 942-953)
        eenv = ManiSkillVectorEnv(gym.make(env_id, num_envs=a.eval_envs, obs_mode="state", control_mode="pd_joint_delta_pos",
                                           sim_backend="physx_cuda", render_mode=None, reconfiguration_freq=1, **kw),
                                  a.eval_envs, ignore_terminations=True, record_metrics=True)
        eb = eenv._env.unwrapped
        eenv.reset(seed=a.seed)             # the first build, which ppo.py does once before training
        times = []
        for _ in range(2):
            torch.cuda.synchronize()
            t0 = time.time()
            o, _ = eenv.reset()
            for _ in range(row["max_episode_steps"]):
                g = eb.agent.is_grasping(getattr(eb, ad.obj)) if ad is not None and ad.obj else None
                o, _, _, _, _ = eenv.step(actor(ad.fn(o, g) if ad is not None else o).clamp(-1, 1))
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        eenv.close()
        row["eval_seconds"] = min(times)
    return row


res, failed = {}, []
for key in a.tasks.split(","):
    try:
        res[key] = r = run(key)
        cells = " ".join(f"{pol} {r[pol]['success']:.2f}/{r[pol]['grasped']:.2f}"
                         + (f"/fail {r[pol]['fail']:.2f}" if r[pol]["fail"] is not None else "") for pol in ("det", "floor"))
        print(f"ok    {key:8s} {r['env']:20s} {str(r['adapter']):15s} steps {r['max_episode_steps']}  "
              f"slots <= {max(r['slot_max_error'].values()):.1e}  {cells}  (success/grasped)  {r.get('final_info', '')}",
              flush=True)
        if "eval_seconds" in r:
            print(f"        one evaluation ({a.eval_envs} envs, reconfiguration_freq=1): {r['eval_seconds']:.1f} s")
        if key == "ycb":
            print(f"        the rule keeps {len(r['ycb_rule_models'])} of {len(r['ycb_all_models'])} models "
                  "(det success per model is diagnostic only; the list is not edited):")
            for m in r["ycb_rule_models"]:
                s, k = r["ycb_det_success"].get(m, [float("nan"), 0])
                print(f"        {m:28s} json {r['ycb_json_cm'][m]} cm  mesh {r['ycb_mesh_cm'].get(m, 'not drawn')} cm  "
                      f"det {s:.2f} of {k}")
    except Exception as e:      # report and go on with the next task
        failed.append(key)
        res[key] = {"error": repr(e)}
        print(f"FAIL  {key:8s} {e!r}", flush=True)
print("ALL OK" if not failed else f"FAILED: {' '.join(failed)}")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
raise SystemExit(1 if failed else 0)
