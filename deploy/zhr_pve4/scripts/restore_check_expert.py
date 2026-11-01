"""Restore fidelity under the competent policy (G1 condition 2), the precondition of the branch-rollout labels.

Per expert task: run S episodes of the stochastic expert with a fixed noise sequence, snapshot the full simulator state
at a few times t0 and finish the episodes. Then restore every snapshot into K replicas and replay the same expert with
the same noise from t0. A faithful restore reproduces the original outcome. Reported per task and t0:
    agree      replicas whose outcome (success in [t0, H)) equals the original's        G1 needs >= 0.95 overall
    consensus  states whose K replicas all agree with each other (separates a lossy restore from sim nondeterminism)
    time       replicas whose first success step equals the original's
    pos_err    final mover position error against the original, median and p95 (metres)
    spread     final mover position spread among the K replicas of one state (0 = the replay itself is deterministic)
    by phase   agreement separately for free (gripper > 3 cm from the mover), near/contact, and grasped states
States that already succeeded before t0 are left out. Right after a restore the contact-based grasp flags are stale,
so the first replayed action is computed from the observation cached in the original run.
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch
from mani_skill.utils.registration import REGISTERED_ENVS

from wrwl.envs.unified import SPECS
from wrwl.experts import EXPERT_NAMES, flat_state, load_expert

p = argparse.ArgumentParser()
p.add_argument("--tasks", default=",".join(EXPERT_NAMES))
p.add_argument("--states", type=int, default=128)
p.add_argument("--replicas", type=int, default=8)
p.add_argument("--fracs", default="0.2,0.4,0.6", help="snapshot times as fractions of the horizon")
p.add_argument("--out", default=str(Path.home() / "pve/data/restore_expert.json"))
a = p.parse_args()
dev = "cuda"
S, K = a.states, a.replicas


def make(task, n):
    return gym.make(task, num_envs=n, obs_mode="state_dict", control_mode="pd_joint_delta_pos",
                    sim_backend="physx_cuda", render_mode=None)


def first_success(succ_t, info, t):
    newly = info["success"].bool() & (succ_t < 0)
    succ_t[newly] = t + 1


results = {}
for task in a.tasks.split(","):
    H = REGISTERED_ENVS[task].max_episode_steps
    t0s = sorted({int(round(f * H)) for f in map(float, a.fracs.split(","))})
    mover, tool = SPECS[task].mover_actor, SPECS[task].tool_actor
    env = make(task, S)
    base = env.unwrapped
    obs, _ = env.reset(seed=7)
    expert = load_expert(task, flat_state(obs).shape[1])
    std = expert.actor_logstd.exp().detach()
    noise = torch.randn(H, S, std.shape[-1], generator=torch.Generator(device=dev).manual_seed(0), device=dev)
    snaps = {}
    succ_t = torch.full((S,), -1, device=dev)
    with torch.no_grad():
        for t in range(H):
            if t in t0s:
                sd = base.get_state_dict()
                grasp = base.agent.is_grasping(getattr(base, mover))
                if tool is not None:
                    grasp = grasp | base.agent.is_grasping(getattr(base, tool))
                near = torch.linalg.norm(base.agent.tcp.pose.p - getattr(base, mover).pose.p, dim=1) < 0.03
                phase = torch.where(grasp, 2, torch.where(near, 1, 0))
                snaps[t] = ({k: {n: v.clone() for n, v in d.items()} for k, d in sd.items()}, succ_t.clone(),
                            flat_state(obs).clone(), phase)
            obs, _, _, _, info = env.step((expert.actor_mean(flat_state(obs)) + std * noise[t]).clamp(-1, 1))
            first_success(succ_t, info, t)
    final_pos = getattr(base, mover).pose.p.clone()
    env.close()

    rep = make(task, S * K)
    rbase = rep.unwrapped
    rep.reset(seed=8)
    results[task] = {}
    for t0 in t0s:
        sd, prev, x0, phase = snaps[t0]
        batch = {k: {n: v.repeat_interleave(K, 0) for n, v in d.items()} for k, d in sd.items()}
        obs, _ = rep.reset(options=dict(reset_to_env_states=dict(env_states=batch)))
        r_succ = torch.full((S * K,), -1, device=dev)
        with torch.no_grad():
            for t in range(t0, H):
                x = x0.repeat_interleave(K, 0) if t == t0 else flat_state(obs)
                act = expert.actor_mean(x) + std * noise[t].repeat_interleave(K, 0)
                obs, _, _, _, info = rep.step(act.clamp(-1, 1))
                first_success(r_succ, info, t)
        if not (prev < 0).any():
            results[task][t0] = dict(n_states=0)
            print(f"{task:20s} t0={t0:3d} every episode already succeeded, skipped", flush=True)
            continue
        valid = (prev < 0).repeat_interleave(K)
        orig_out = (succ_t > 0).repeat_interleave(K)
        rep_out = r_succ > 0
        same = (rep_out == orig_out)[valid].float()
        same_all = (rep_out == orig_out).float().view(S, K)
        by_phase = {name: (same_all[(prev < 0) & (phase == i)].mean().item(), int(((prev < 0) & (phase == i)).sum()))
                    for i, name in enumerate(["free", "near", "grasped"])}
        cons = rep_out.view(S, K).float()
        cons = ((cons.min(1).values == cons.max(1).values)[prev < 0]).float()
        pos = getattr(rbase, mover).pose.p
        err = torch.linalg.norm(pos - final_pos.repeat_interleave(K, 0), dim=1)[valid]
        pk = pos.view(S, K, 3)[prev < 0]
        spread = torch.linalg.norm(pk - pk.mean(1, keepdim=True), dim=2).max(1).values
        results[task][t0] = dict(n_states=int((prev < 0).sum()), agree=same.mean().item(),
                                 consensus=cons.mean().item(),
                                 time=(r_succ == succ_t.repeat_interleave(K))[valid].float().mean().item(),
                                 orig_success=(succ_t > 0)[prev < 0].float().mean().item(),
                                 pos_err_median=err.median().item(), pos_err_p95=err.quantile(0.95).item(),
                                 spread_median=spread.median().item(), by_phase=by_phase)
        r = results[task][t0]
        print(f"{task:20s} t0={t0:3d} states={r['n_states']:3d} agree={r['agree']:.3f} consensus={r['consensus']:.3f} "
              f"time={r['time']:.3f} orig_success={r['orig_success']:.2f} pos_err median={r['pos_err_median']*1e3:.2f}mm "
              f"p95={r['pos_err_p95']*1e3:.2f}mm spread median={r['spread_median']*1e3:.2f}mm  by phase: "
              + " ".join(f"{k}={v[0]:.2f}(n={v[1]})" for k, v in by_phase.items() if v[1]), flush=True)
    rep.close()
    torch.cuda.empty_cache()

agree = [r["agree"] * r["n_states"] for t in results.values() for r in t.values() if r["n_states"]]
n = [r["n_states"] for t in results.values() for r in t.values() if r["n_states"]]
overall = sum(agree) / max(sum(n), 1)
print(f"overall agreement {overall:.3f}  (G1 condition 2 needs >= 0.95)")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(dict(per_task=results, overall_agree=overall), open(a.out, "w"), indent=1)
print("saved", a.out)
