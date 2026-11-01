"""The source's own 'my task is done' detector, read off its frozen hidden layer: the automatic locator of the seam.

Rolls the source expert out on its own task, labels each state with the source task's success flag, and fits a
logistic probe on the source actor's last hidden layer. At transfer time the seam is wherever this probe fires: the
source believes it is finished, yet the target episode has not paid (a target success ends the episode, so every
training state where the probe fires is a refuted 'done'). No target reward, demonstration or hand-written region is
used. --check instead rolls the source out in the target task and reports how what the probe flags overlaps the
hand-written region (pick: cube held within --radius of the goal; push: cube within --radius of the goal in xy).
  python scripts/train_done_probe.py --src=runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt --out=data/done_probe_pick_c1.pt
  python scripts/train_done_probe.py --src=... --out=data/done_probe_pick_c1.pt --check --tgt_env=StackCubeNarrow-v1 --adapter=stack_to_pick
  python scripts/train_done_probe.py --src=runs/experts/runs/PushCube_c1/final_ckpt.pt --src_env=PushCube-v1 \
      --src_kwargs='{}' --obs_dim=35 --out=data/done_probe_push_c1.pt
"""
import argparse
import json

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

import wrwl.envs.place_cube  # noqa: F401
import wrwl.envs.push_retreat  # noqa: F401
import wrwl.envs.stack_narrow  # noqa: F401
from wrwl.adapters import stack_to_pick
from wrwl.experts import flat_state, load_policy

p = argparse.ArgumentParser()
p.add_argument("--src", required=True, help="source expert checkpoint")
p.add_argument("--src_env", default="PickCube-v1")
p.add_argument("--src_kwargs", default='{"robot_uids": "panda_wristcam"}')
p.add_argument("--obs_dim", type=int, default=42)
p.add_argument("--envs", type=int, default=512)
p.add_argument("--batches", type=int, default=4)
p.add_argument("--wd", type=float, default=1e-3)
p.add_argument("--check", "--check_stack", action="store_true", help="load --out and check it in the target task")
p.add_argument("--tgt_env", default="StackCubeNarrow-v1")
p.add_argument("--tgt_kwargs", default="{}")
p.add_argument("--adapter", default="stack_to_pick", choices=["stack_to_pick", "none"])
p.add_argument("--seam_rule", default="pick", choices=["pick", "push"])
p.add_argument("--radius", type=float, default=0.025)
p.add_argument("--out", required=True)
a = p.parse_args()
dev = "cuda"
torch.manual_seed(0)
src = load_policy(a.src, a.obs_dim)
trunk = src.actor_mean[:-1]


def auroc(score, y):
    o = score.argsort()
    r = torch.empty_like(score)
    r[o] = torch.arange(1, len(score) + 1, device=score.device, dtype=score.dtype)
    npos = y.sum()
    return ((r[y.bool()].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))).item()


def region(x):
    if a.seam_rule == "push":
        return torch.linalg.norm(x[:, 28:30] - x[:, 25:27], dim=1) < a.radius
    return (x[:, 18] > 0.5) & (torch.linalg.norm(x[:, 39:42], dim=1) <= a.radius)


if not a.check:
    env = gym.make(a.src_env, num_envs=a.envs, obs_mode="state", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cuda", render_mode=None, **json.loads(a.src_kwargs))
    H, Y = [], []
    with torch.no_grad():
        for b in range(a.batches):
            obs, _ = env.reset(seed=2000 + b)
            for _ in range(50):
                mean = src.actor_mean(obs)
                H.append(trunk(obs))
                obs, _, _, _, info = env.step((mean + src.actor_logstd.exp() * torch.randn_like(mean)).clamp(-1, 1))
                Y.append(info["success"].float())    # the source's own success test, evaluated after this action
    env.close()
    H, Y = torch.cat(H), torch.cat(Y)
    n = len(Y)
    perm = torch.randperm(n, device=dev)
    tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    w = torch.zeros(H.shape[1], device=dev, requires_grad=True)
    b0 = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b0], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(H[tr] @ w + b0, Y[tr]) + a.wd * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        s = H[te] @ w + b0
        acc = ((s > 0).float() == Y[te]).float().mean().item()
        print(f"source states {n}, done rate {Y.mean():.3f}; held-out AUROC {auroc(s, Y[te]):.3f}, accuracy {acc:.3f}")
    torch.save({"w": w.detach().cpu(), "b": b0.detach().cpu(), "src": a.src}, a.out)
    print("saved", a.out)
else:
    pr = torch.load(a.out)
    w, b0 = pr["w"].to(dev), pr["b"].to(dev)
    stack = a.adapter == "stack_to_pick"
    env = gym.make(a.tgt_env, num_envs=a.envs, obs_mode="state_dict" if stack else "state", reward_mode="sparse",
                   control_mode="pd_joint_delta_pos", sim_backend="physx_cuda", render_mode=None,
                   **json.loads(a.tgt_kwargs))
    base = env.unwrapped
    F_, R_ = [], []
    with torch.no_grad():
        for b in range(a.batches):
            obs, _ = env.reset(seed=3000 + b)
            for _ in range(50):
                x = stack_to_pick(flat_state(obs), base.agent.is_grasping(base.cubeA)) if stack else obs
                F_.append((trunk(x) @ w + b0) > 0)
                R_.append(region(x))
                mean = src.actor_mean(x)
                obs, *_ = env.step((mean + src.actor_logstd.exp() * torch.randn_like(mean)).clamp(-1, 1))
    env.close()
    f, r = torch.cat(F_), torch.cat(R_)
    print(f"{a.tgt_env} states visited by the source: probe fires on {f.float().mean():.3f}, hand-written region "
          f"{r.float().mean():.3f}; P(region | fires) {(f & r).sum() / f.sum().clamp_min(1):.3f}, "
          f"P(fires | region) {(f & r).sum() / r.sum().clamp_min(1):.3f}")
