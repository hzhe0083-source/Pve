"""Gold criticality labels by branch rollouts (F1, F2), restored simulator states processed in fixed-size batches.

For each sampled state s: restore it in M*N envs, play M first action chunks drawn from the mixture proposal
mu = 1/2 pi_task + 1/2 pi_base (chunk-level mixture), replay each chunk N times with independent expert continuations,
and record success within the remaining horizon. Then, with q_m the success rate after chunk m:
    w_m        = p_base(chunk_m) / mu(chunk_m)                 (reweight to pi_base)
    tilt_m    ~= w_m * exp(q_m / tau)                          (pi*_tau  ~  pi_base * exp(Q / tau))
    KL(s)      = sum_m tilt_m q_m / tau - log( sum_m w_m e^{q_m/tau} / sum_m w_m )      -> c_bar = 2 tau^2 KL
    c*(s)      = Var_tilt(q) - E_tilt[q(1-q)] / (N-1)          (binomial-noise corrected)
    c_task(s)  = Var over the expert-drawn chunks of q, noise corrected (criticality under the expert itself)
Split-half: the N continuations are split in two halves and c* is recomputed from each, to give a noise ceiling.

Right after a restore, ManiSkill's grasp flags (Panda.is_grasping, read from the last step's contact impulses) are
stale, so the labelled observation and the expert's first action come from the values cached by collect_rollouts.
"""
import argparse
import math
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

from wrwl.experts import flat_state, load_expert

p = argparse.ArgumentParser()
p.add_argument("--rollouts", default=str(Path.home() / "pve/data/rollouts.pt"))
p.add_argument("--out", default=str(Path.home() / "pve/data/labels.pt"))
p.add_argument("--tasks", default="", help="comma-separated subset of the rollout tasks (default: all)")
p.add_argument("--states", type=int, default=256)
p.add_argument("--batch-states", type=int, default=32, help="states restored together (env count = this * M * N)")
p.add_argument("--chunks", type=int, default=16)
p.add_argument("--conts", type=int, default=8)
p.add_argument("--chunk-len", type=int, default=5)
p.add_argument("--tau", type=float, default=0.2)
p.add_argument("--base-std", type=float, default=float(math.exp(-0.5)))
a = p.parse_args()
dev = "cuda"
data = torch.load(a.rollouts, weights_only=False)
if a.tasks:
    data = {t: data[t] for t in a.tasks.split(",")}
M, N, K, SB = a.chunks, a.conts, a.chunk_len, a.batch_states
g = torch.Generator().manual_seed(0)


def log_normal(x, mean, std):
    return (-0.5 * ((x - mean) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)


def labels_from(q, logw, n):
    """KL(pi*_tau || pi_base) and the noise-corrected tilted variance c* from per-chunk success rates q (S, M)."""
    tilt = torch.softmax(logw + q / a.tau, dim=1)
    log_z = torch.logsumexp(logw + q / a.tau, dim=1) - torch.logsumexp(logw, dim=1)
    kl = (tilt * q).sum(1) / a.tau - log_z
    noise = q * (1 - q) / max(n - 1, 1)
    c_star = ((tilt * q ** 2).sum(1) - (tilt * q).sum(1) ** 2 - (tilt * noise).sum(1)).clamp_min(0)
    return kl, c_star


def spearman(x, y):
    rx, ry = x.argsort().argsort().float(), y.argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return ((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12)).item()


out = {}
for task, d in data.items():
    horizon = d["horizon"]
    n_env = d["obs"].shape[0]
    snaps, snap_t = d["states"], d["state_t"]
    # candidate (snapshot, env) pairs that have not succeeded yet and leave room for the chunk
    cands = [(si, e) for si, t in enumerate(snap_t) for e in range(n_env)
             if (d["succ_time"][e] < 0 or d["succ_time"][e] > t) and t + K < horizon]
    chosen = [cands[i] for i in torch.randperm(len(cands), generator=g)[:a.states].tolist()]
    S = len(chosen)
    env = gym.make(task, num_envs=SB * M * N, obs_mode="state_dict", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cuda", render_mode=None)
    obs, _ = env.reset(seed=0)
    expert = load_expert(task, flat_state(obs).shape[1])
    e_std = expert.actor_logstd.exp().detach()
    use_expert = (torch.rand(S, M, generator=g) < 0.5).to(dev)
    noise = torch.randn(S, M, K, 8, generator=g).to(dev)
    logp_task = torch.zeros(S, M, device=dev)
    logp_base = torch.zeros(S, M, device=dev)
    success = torch.zeros(S, M, N, dtype=torch.bool, device=dev)
    t0_all = torch.tensor([snap_t[si] for si, _ in chosen], device=dev)
    mean0_all = torch.stack([d["act_mean"][e, snap_t[si]] for si, e in chosen]).to(dev)
    for b0 in range(0, S, SB):
        idx = list(range(b0, min(S, b0 + SB)))
        idx_pad = idx + [idx[0]] * (SB - len(idx))       # the env count is fixed, so pad the last batch
        rows = []
        for i in idx_pad:
            si, e = chosen[i]
            rows.append({k: {n: v[e:e + 1] for n, v in dd.items()} for k, dd in snaps[si].items()})
        batch_state = {k: {n: torch.cat([r[k][n] for r in rows]).repeat_interleave(M * N, dim=0).to(dev)
                           for n in rows[0][k]} for k in rows[0]}
        obs, _ = env.reset(options=dict(reset_to_env_states=dict(env_states=batch_state)))
        remaining = (horizon - t0_all[idx_pad]).repeat_interleave(M * N)
        ue, nz, mean0 = use_expert[idx_pad], noise[idx_pad], mean0_all[idx_pad]
        lp_t = torch.zeros(SB, M, device=dev)
        lp_b = torch.zeros(SB, M, device=dev)
        succ = torch.zeros(SB * M * N, dtype=torch.bool, device=dev)
        with torch.no_grad():
            for t in range(horizon):
                mean = expert.actor_mean(flat_state(obs))
                if t < K:
                    # same state for all N replicas of a chunk; step 0 uses the cached mean (stale grasp flags)
                    mean_sm = mean0[:, None].expand(SB, M, -1) if t == 0 else mean.view(SB, M, N, -1)[:, :, 0]
                    a_exp = mean_sm + e_std * nz[:, :, t]
                    a_base = a.base_std * nz[:, :, t]
                    act_sm = torch.where(ue[..., None], a_exp, a_base)
                    lp_t += log_normal(act_sm, mean_sm, e_std)
                    lp_b += log_normal(act_sm, torch.zeros_like(act_sm), torch.full_like(act_sm, a.base_std))
                    act = act_sm.repeat_interleave(N, dim=1).view(SB * M * N, -1)
                else:
                    act = mean + e_std * torch.randn_like(mean)
                obs, _, _, _, info = env.step(act.clamp(-1, 1))
                succ |= info["success"].bool() & (t < remaining)
        n = len(idx)
        logp_task[idx], logp_base[idx] = lp_t[:n], lp_b[:n]
        success[idx] = succ.view(SB, M, N)[:n]
    env.close()
    torch.cuda.empty_cache()

    log_mu = torch.logaddexp(logp_task, logp_base) - math.log(2)
    logw = logp_base - log_mu
    q = success.float().mean(-1)                                           # (S, M)
    kl, c_star = labels_from(q, logw, N)
    _, c_h1 = labels_from(success[..., :N // 2].float().mean(-1), logw, N // 2)
    _, c_h2 = labels_from(success[..., N // 2:].float().mean(-1), logw, N - N // 2)
    noise_term = q * (1 - q) / max(N - 1, 1)
    qe = torch.where(use_expert, q, torch.nan)
    n_e = use_expert.sum(1).clamp_min(2)
    mean_e = torch.nanmean(qe, dim=1)
    var_e = torch.nansum((qe - mean_e[:, None]) ** 2, dim=1) / (n_e - 1)
    c_task = (var_e - torch.nanmean(torch.where(use_expert, noise_term, torch.nan), dim=1)).clamp_min(0)
    top = c_star >= c_star.quantile(0.9)
    top1, top2 = c_h1 >= c_h1.quantile(0.9), c_h2 >= c_h2.quantile(0.9)
    out[task] = dict(obs=torch.stack([d["obs"][e, snap_t[si]] for si, e in chosen]), t=t0_all.cpu(),
                     succ_time=torch.tensor([d["succ_time"][e] for _, e in chosen]),
                     chosen=torch.tensor(chosen), q=q.cpu(), kl=kl.cpu(), c_bar=(2 * a.tau ** 2 * kl).cpu(),
                     c_star=c_star.cpu(), c_star_h1=c_h1.cpu(), c_star_h2=c_h2.cpu(), c_task=c_task.cpu(),
                     p_state=q.mean(1).cpu(), tau=a.tau, M=M, N=N, K=K)
    print(f"{task:20s} states={S} mean q={q.mean():.3f} c*: median={c_star.median():.4f} p90={c_star.quantile(0.9):.4f} "
          f"KL nats: median={kl.median():.3f} p90={kl.quantile(0.9):.3f}  split-half c*: spearman="
          f"{spearman(c_h1, c_h2):.2f} top10% overlap={(top1 & top2).sum().item() / max(top1.sum().item(), 1):.2f} "
          f"(full-set top10% n={top.sum().item()})", flush=True)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(out, a.out)
print("saved", a.out)
