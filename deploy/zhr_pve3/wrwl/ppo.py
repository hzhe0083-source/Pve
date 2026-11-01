"""State-based PPO for the WRWL mechanism experiments.

Derived from ManiSkill's examples/baselines/ppo/ppo.py (v3.0.1, Apache-2.0). Added on top of it:
  * --env_kwargs     JSON kwargs for gym.make (e.g. '{"theta_deg": 90}' for PushCubeDir-v1)
  * --init_ckpt      start from another policy's weights (the "fine-tune" arm)
  * --prior_ckpt     a frozen source policy; adds lambda_t * E[g(s) * KL(pi_theta || prior)(s)] to the loss
                     (reverse KL, F8's reuse term), lambda_t decays linearly to 0 over --prior_anneal_frac of training
  * --gate           where the prior term applies: all (kickstarting), none, reach (copy only while the gripper is far
                     from the object), contact (the inverse, a knock-out control)
  * eval.csv         one row per evaluation, for steps-to-threshold measurements; optional early stop
  * --loss cov       F8's covariant surrogate  E[r * A/sigma - (r-1)^2 / (2 sqrt(2 Delta))]  with A normalised per batch
                     and divided by sqrt(decisions per episode), so the per-episode KL is about Delta
  * --control_freq / --max_episode_steps   refine or coarsen the decision granularity for the same physical task
  * train.csv        per-iteration KL between successive policies, per decision and per episode; the success rate of
                     the episodes that ended in this iteration's (sampled) rollout; the action std (arm mean, gripper)
  * --unified        78-dim unified observation (wrwl.envs.unified); the actor is a SharedPolicy, optionally
                     initialised from a distilled shared network (--shared_ckpt, --init_from_shared)
  * --init_task_like a new task starts as a copy of a designated source: its one-hot column takes that source's weights
  * --prior_self     the prior is a frozen copy of the initial network (in-network reuse: KL to what the library did)
  * --copy_std       also pull the (state-independent) log-std towards the prior's; off by default, since a gate cannot
                     confine a global std to the gated states (the copy term then moves only the mean)
  * --copy_loss      rkl = KL(pi || prior), fkl = KL(prior || pi), rkl_def = KL(pi || 1/2 prior + 1/2 pi_base) with
                     pi_base = N(0, --base_std), estimated from 8 reparameterised samples (the defensive prior of F6)
  * --obs_adapter stack_to_pick   StackCube seen through PickCube's observation layout (cube = cubeA, goal = 4 cm above
                     cubeB, grasp flag from the env), so a PickCube expert can be the init or the prior
  * --init_actor_only  load only the actor from --init_ckpt (the critic was trained on another task and reward)
  * --gate oracle    copy where the gold copy loss of the k nearest labelled states (label_copyloss.py, source
                     --oracle_source) is below --oracle_thresh; states with no labelled neighbour within the fallback
                     radius copy; train.csv logs the copy fraction and the fallback fraction
  * --bootstrap_at_done truncated   bootstrap V(final obs) only at time-outs, so with sparse reward a success ends the
                     return at 1 and V is a success probability (the official script bootstraps at every done)
  * --gate pregrasp  (unified obs) copy while nothing is held, learn once the mover or the tool is grasped
  * --oracle_shuffle  control for --gate oracle: permute the labels over the labelled states
  * --save_every N / --halt_at N   save ckpt_<step>.pt every N steps; stop at N steps without changing the
                     schedules, which still span --total_timesteps
  * --freeze_trunk   with --init_ckpt: keep the source actor's hidden layers fixed (reuse its features as they are);
                     only the last actor layer and the log-std learn
  * --explore_boost  (stack_to_pick) add this to the gripper's log-std wherever the cube is held within --explore_radius
                     of the goal, the region where the PickCube and StackCube experts disagree (scripts/conflict_probe.py);
                     the boost is part of the policy's distribution, so rollouts and PPO ratios use the same std
  * --min_grip_std S  a floor on the gripper std (a2r's --min_grip_std recipe); with --grip_floor_seam only where the
                     cube is held within --explore_radius of the goal, with --grip_floor_anti everywhere except there
  * --seam_probe P   take the boost / floor region from the source's own done-probe (scripts/train_done_probe.py) on
                     the source actor's hidden layer instead of the hand-written region: no target labels involved
  * --seam_rule pick|push   the hand-written region for PickCube-layout (held at the goal) or PushCube-layout (cube in
                     the goal disc) observations; --seam_dims gripper|arm|all   which action dims get the boost / floor
  * --seam_gate time|grasp   controls for the seam's location: the region is "step >= --seam_t0 of the episode" or
                     "the cube is held", fixed per step in the rollout and reused in the updates (default state)
  * --seam_gate latch  the region (--seam_rule or --seam_probe) from the first step it fires to the end of the episode.
                     Both the rule and the probe need the cube held, so with the state gate the floor switches off as
                     soon as the gripper opens and the hand closes again; latched, it stays on after the release
  * --critic_warmup N  value-only updates for the first N iterations while the initial actor keeps acting
  * --stop_after N   early stopping (--stop_success) is allowed only from N steps on
  * --train_units F  with --freeze_trunk: after --select_iters head-only iterations, the fraction F of each hidden
                     layer's units (rows: incoming weights + bias) learns too, the other rows stay frozen (their
                     gradients are zeroed). --train_units_by grad takes the rows with the largest summed squared policy
                     gradient over those iterations (the key parameters for the new task), random as many at random;
                     the chosen rows and the importance go to train_units.json
  * --protect_old block|gpm   learn the new skill in the shared hidden layers while keeping the old skill's actions,
                     read through its own frozen output layer, unchanged to first order (Protector below): after every
                     optimizer step the applied update of each hidden Linear loses the block P_out dW P_in (block) or
                     every input direction the old skill occupies (gpm); monitor logs the same quantities for plain
                     fine-tuning without projecting anything. --protect_states: the old skill's states;
                     --protect_energy: subspace size; --protect_every N: re-estimate the subspaces every N iterations.
                     protect.csv logs the share of each update removed and the old skill's action drift.
                     --protect_grad: project the gradient before the optimizer step as well (block, gpm).
                     Baselines on the same states, nothing projected, a penalty times --protect_coef added to the loss:
                     distill (the old skill's mean-action drift on a batch of its states, LwF-style), ewc (diagonal
                     Fisher of the old skill's Gaussian mean at the start), l2 (L2-SP: squared distance to the start)
  * --protect_env E --protect_dagger N   every N iterations the old skill (old head on the current hidden layers) is
                     rolled out deterministically on its own task E (--protect_env_kwargs) from fresh starts;
                     old_eval.csv logs its success there, and with distill its states before each env's first success,
                     labelled by the frozen original actor, become half of every distill batch (DAgger-style: the
                     states the old skill actually reaches now, not only the ones it reached before); the last row of
                     old_eval.csv is from retention_eval.py's starts (seed 0)
  * --protect_old fn  FN-PPO: every update of the hidden layers (the gradient before clipping and Adam and the realised
                     step after it, whatever --protect_grad says) loses its exact first-order effect on the old skill's
                     sigma-scaled mean actions (old head) on --fn_anchors anchor states: J dW = 0 for the Jacobian over
                     all hidden weights jointly, not a factored subspace. Anchors and (J J^T + delta I)^-1 are redrawn
                     and refactored every iteration at the current weights: --fn_pool_frac of them from the states the
                     old skill reaches through the current hidden layers (needs --protect_env; rolled out every
                     --protect_dagger iterations and once before training), the rest from --protect_states, each part
                     >= 50% held and >= 20% within 2.5 cm of the goal (PickCube layout). --fn_warmup N: the hidden
                     layers stay fixed for the first N iterations while head, log-std and critic train. fn.csv logs the
                     share of each gradient (rho_g) and step (rho_s, also protect.csv's removed_l*) removed per layer,
                     the anchors, the Gram (skipped: it could not be factored, so the hidden layers stayed fixed that
                     iteration) and the old skill's drift on its closed-loop states
  * --protect_strat  with distill (KL-CL): every penalty is on a fresh draw of fn's anchors (Protector.draw:
                     --fn_anchors states, --fn_pool_frac of them from the closed-loop pool, each part stratified, own
                     generator) instead of 4096 uniform states, and the pool starts with a rollout before training as
                     fn's does: the soft penalty on the states fn protects hard
  * --head_lr        the actor's output layer and log-std get their own Adam learning rate (default: one group)
  * --nullspace_adapter H   keep the old skill exactly: the old hidden layers stay frozen and a new adapter of width H
                     writes new features into the last hidden layer only in directions the old output layer cannot
                     read (wrwl/nsa.py); both skills read the same last hidden layer. --nsa_base CKPT continues a chain:
                     the levels and the head of that adapter run stay frozen and one more level protects every head
                     so far; the new head starts as that run's head (--nsa_head_from base) or as --init_ckpt's head
                     and log-std (--nsa_head_from init). --nsa_project False: the prefix-sum ablation (no projection,
                     the adapter's last layer starts at zero; the old skill is exact only through its own level, which
                     old_eval.csv then reads)
  * --nsa_proj random   control for the projection's subspace: the newest adapter's write is projected away from as
                     many random orthonormal rows (own generator, seed 10007 + --seed) as the heads have instead of the
                     heads' rows, so the old head reads the write; the random draw starts in the null space of both, so
                     the copied head still starts as the old policy (wrwl/nsa.py)
  * --nsa_mask visible|zero   controls for rule F (scripts/why_test.py): the newest adapter's write is multiplied by
                     B^T B (only what the old heads read) or by 0 (no write) instead of the heads' projection; the
                     random draw starts in the null space, so the copied head still starts as the old policy
  * --nsa_restore L  with --nsa_project False (the prefix sum, one level): adds L * mean ||W_old g(x)||^2 over each
                     minibatch's states (on-policy, no old data) to the loss, W_old the old head and g the adapter's
                     write: a restoring force on what the old head reads of the write; train.csv logs it (restore_pen)
  * --adapter_lr     the newest adapter's own Adam learning rate (default: one group); the head, log-std and critic
                     keep --learning_rate (the head and log-std --head_lr if given)
  * stage columns (PickCube layout, wrwl/phase_log.py): train.csv counts, per iteration, the training episodes that
                     ended (st_done) and how many were grasped, held at the goal, held and succeeded, succeeded, held
                     but never released, lost after a grasp, never grasped; the gripper mean in goal / carry / free
                     states and the action means' squared deviation from the start policy there. eval.csv the same
                     stage shares for the deterministic evaluation (ev_*)
  * train.csv also logs the share of rollout states in the seam and the std actually applied (arm mean, gripper);
    args.json records the md5 of this file
"""
import copy
import hashlib
import math
import csv
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.distributions.normal import Normal

import wrwl.envs.place_cube  # noqa: F401  registers PlaceCube-v1
import wrwl.envs.push_dir  # noqa: F401  registers PushCubeDir-v1
import wrwl.envs.push_retreat  # noqa: F401  registers PushRetreat-v1
import wrwl.envs.stack_narrow  # noqa: F401  registers StackCubeNarrow-v1
from wrwl.adapters import stack_to_pick
from wrwl.nsa import add_level, build_actor
from wrwl.phase_log import COUNTS, EVAL_COLS, INFO_KEYS, PHASE_COLS, EpisodeStages, phase_stats
from wrwl.envs.unified import GRASP_DIMS, OBS_DIM, TASKS, unify
from wrwl.shared import SharedPolicy

# Indices of the TCP and the manipulated object's position in each task's flat state observation
# (qpos 9, qvel 9, tcp_pose 7, then the task's extras in insertion order).
OBJ_POS_IDX = {"PushCube-v1": 28, "PushCubeDir-v1": 28, "PullCube-v1": 28, "PickCube-v1": 29}
TCP_POS_IDX = {"PushCube-v1": 18, "PushCubeDir-v1": 18, "PullCube-v1": 18, "PickCube-v1": 19}


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    save_model: bool = True
    env_id: str = "PushCubeDir-v1"
    env_kwargs: str = "{}"
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    num_envs: int = 2048
    num_eval_envs: int = 64
    partial_reset: bool = True
    eval_partial_reset: bool = False
    num_steps: int = 20
    num_eval_steps: int = 50
    control_mode: Optional[str] = "pd_joint_delta_pos"
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    reward_scale: float = 1.0
    bootstrap_at_done: str = "all"
    eval_freq: int = 5
    init_ckpt: Optional[str] = None
    prior_ckpt: Optional[str] = None
    prior_coef: float = 0.0
    prior_anneal_frac: float = 0.5
    gate: str = "all"
    gate_dist: float = 0.05
    copy_loss: str = "rkl"
    copy_std: bool = False
    base_std: float = 0.6065306597126334
    obs_adapter: Optional[str] = None
    init_actor_only: bool = False
    oracle_labels: Optional[str] = None
    oracle_source: Optional[str] = None
    oracle_k: int = 5
    oracle_thresh: float = 0.1
    oracle_shuffle: bool = False
    save_every: int = 0
    halt_at: int = 0
    freeze_trunk: bool = False
    explore_boost: float = 0.0
    explore_radius: float = 0.025
    min_grip_std: float = 0.0
    grip_floor_seam: bool = False
    grip_floor_anti: bool = False
    seam_probe: Optional[str] = None
    seam_rule: str = "pick"
    seam_dims: str = "gripper"
    seam_gate: str = "state"
    seam_t0: int = 16
    critic_warmup: int = 0
    stop_after: int = 0
    train_units: float = 0.0
    train_units_by: str = "grad"
    select_iters: int = 5
    protect_old: str = "none"
    protect_states: Optional[str] = None
    protect_energy: float = 0.95
    protect_every: int = 0
    protect_coef: float = 1.0
    protect_grad: bool = False
    protect_env: Optional[str] = None
    protect_env_kwargs: str = '{"robot_uids": "panda_wristcam"}'
    protect_dagger: int = 0
    fn_anchors: int = 512
    fn_pool_frac: float = 0.5
    fn_warmup: int = 0
    protect_strat: bool = False
    head_lr: Optional[float] = None
    nullspace_adapter: int = 0
    nsa_base: Optional[str] = None
    nsa_head_from: str = "base"
    nsa_project: tyro.conf.FlagConversionOff[bool] = True
    nsa_init: Optional[str] = None
    nsa_freeze_adapter: bool = False
    nsa_proj: str = "heads"
    nsa_restore: float = 0.0
    adapter_lr: Optional[float] = None
    nsa_mask: str = "null"
    stop_success: Optional[float] = None
    stop_patience: int = 2
    loss: str = "clip"
    delta: float = 0.5
    control_freq: Optional[int] = None
    max_episode_steps: Optional[int] = None
    unified: bool = False
    init_task_like: Optional[str] = None
    prior_self: bool = False
    shared_ckpt: Optional[str] = None
    init_from_shared: bool = False
    init_logstd: float = -0.5
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    # set on the learner only (--explore_boost, --min_grip_std, --seam_probe); priors keep these defaults
    explore_boost, explore_radius, min_grip_std, grip_floor_seam, seam_fn = 0.0, 0.025, 0.0, False, None
    grip_floor_anti, seam_rule, dim_mask = False, "pick", None     # dim_mask None: the gripper only
    mask_override = None    # --seam_gate time / grasp: the region for this batch, fixed in the rollout and reused

    def at_goal(self, x):
        """The seam: a mask set from outside (--seam_gate time / grasp), where the source's done-probe fires
        (--seam_probe), else the hand-written region. pick: in PickCube's layout the cube is held (flag at 18) within
        explore_radius of the goal (cube -> goal at 39:42); push: in PushCube's layout the cube (28:31) is within
        explore_radius of the goal (25:28) in xy."""
        if self.mask_override is not None:
            return self.mask_override
        if self.seam_fn is not None:
            return self.seam_fn(x)
        if self.seam_rule == "push":
            return torch.linalg.norm(x[:, 28:30] - x[:, 25:27], dim=1) < self.explore_radius
        return (x[:, 18] > 0.5) & (torch.linalg.norm(x[:, 39:42], dim=1) <= self.explore_radius)

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_value(self, x):
        return self.critic(x)

    def dist(self, x):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        if self.dim_mask is not None:       # --seam_dims arm / all: the same boost or floor on those action dims
            if self.explore_boost:
                logstd = logstd + self.explore_boost * (self.dim_mask & self.at_goal(x).unsqueeze(1)).float()
            if self.min_grip_std:
                sel = self.dim_mask.expand_as(logstd)
                if self.grip_floor_seam:
                    m = self.at_goal(x)
                    sel = sel & (~m if self.grip_floor_anti else m).unsqueeze(1)
                logstd = torch.where(sel, logstd.clamp(min=math.log(self.min_grip_std)), logstd)
            return mean, logstd
        if self.explore_boost:
            boost = torch.zeros_like(logstd)
            boost[:, -1] = self.explore_boost * self.at_goal(x).float()
            logstd = logstd + boost
        if self.min_grip_std:       # a floor on the gripper std, everywhere, only at the seam, or everywhere but there
            g = logstd[:, -1].clamp(min=math.log(self.min_grip_std))
            if self.grip_floor_seam:
                m = self.at_goal(x)
                g = torch.where(~m if self.grip_floor_anti else m, g, logstd[:, -1])
            logstd = torch.cat([logstd[:, :-1], g.unsqueeze(1)], 1)
        return mean, logstd

    def get_action(self, x, deterministic=False):
        mean, logstd = self.dist(x)
        return mean if deterministic else Normal(mean, logstd.exp()).sample()

    def get_action_and_value(self, x, action=None):
        mean, logstd = self.dist(x)
        probs = Normal(mean, logstd.exp())
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class SharedAgent(nn.Module):
    """Actor = SharedPolicy over the unified observation (trunk = the latent space); separate critic."""

    def __init__(self, act_dim, init_logstd=-0.5):
        super().__init__()
        self.actor = SharedPolicy(OBS_DIM, act_dim)
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                layer_init(m)
        layer_init(self.actor.head[-1], std=0.01 * np.sqrt(2))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(OBS_DIM, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * init_logstd)

    def _norm(self, x):
        return (x - self.actor.obs_mean) / self.actor.obs_std

    def get_value(self, x):
        return self.critic(self._norm(x))

    def dist(self, x):
        mean = self.actor(x)
        return mean, self.actor_logstd.expand_as(mean)

    def get_action(self, x, deterministic=False):
        mean, logstd = self.dist(x)
        return mean if deterministic else Normal(mean, logstd.exp()).sample()

    def get_action_and_value(self, x, action=None):
        mean, logstd = self.dist(x)
        probs = Normal(mean, logstd.exp())
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x)


class Protector:
    """--protect_old: the new skill trains the shared hidden layers while the old skill keeps its own frozen output
    layer. For each hidden Linear, with augmented input x~ = [x; 1] on the old skill's states:
      C = E[x~ x~^T]   input directions the old skill occupies
      Q = E[J^T J]     J = d mu_old / d z through the later layers and the old head: output directions its actions read
    P_in, P_out project on the top eigenvectors of C, Q holding `energy` of their trace. After each optimizer step the
    applied update dW~ becomes dW~ - P_out dW~ P_in (block) or dW~ (I - P_in) (gpm). The update is projected, not only
    the gradient: Adam rescales each coordinate, so a projected gradient would not give a projected step. With
    --protect_grad the gradient is projected too (project_grad), so Adam's moments never hold the blocked part. This
    keeps the old skill's mean actions unchanged only to first order and inside the kept subspaces (Kronecker-factored,
    layer by layer); protect.csv logs the true drift so the residual is monitored.
    The baselines project nothing (project() measures as monitor) and add penalty() to the loss; distill with strat
    (--protect_strat) takes it on fn's anchors (draw()).
    fn (FN-PPO) is exact to first order instead: J = d(mu_old / sigma) / d theta on M anchor states (rows (n, a), theta
    all hidden weights), and every update D (the gradient before clipping and Adam, and the realised step after it)
    loses J^T alpha, alpha = (J J^T + delta I)^-1 J D, so J D = 0 up to delta jointly over the layers. Per layer
    d mu_a(x_n) / dW~ = G[n, a] Z[n]^T (G = d(mu_old_a / sigma_a) / d(the layer's output), Z = [input; 1]), so
    J J^T = sum_l (G G^T) * (Z Z^T repeated over a, b) and the removed part is (sum_a alpha[n, a] G[n, a])^T Z: no
    per-parameter Jacobian is formed. refresh() redraws the anchors (pool_frac of them from the states the old skill
    reaches now through the current hidden layers, add_states, the rest from the stored ones, each part
    phase-stratified) and refactors the Gram once per iteration at the current weights; within the iteration the
    projection is linearised there."""
    fn_delta = 1e-4     # fn: delta = fn_delta * tr(J J^T) / (M A)

    def __init__(self, actor, old_head, states, energy, mode, sigma, n_anchors=512, pool_frac=0.5, seed=0,
                 strat=False):
        self.actor, self.old_head, self.states, self.energy, self.mode = actor, old_head, states, energy, mode
        self.lin = [m for m in actor[:-1] if isinstance(m, nn.Linear)]
        self.sigma = sigma
        with torch.no_grad():
            self.ref = self.old_mu(states[:2000])
            self.ref_all = self.old_mu(states) if mode == "distill" else None
        self.teacher = copy.deepcopy(actor).requires_grad_(False)     # the original old skill, for new states
        self.dx = self.dy = None                                       # --protect_dagger: states it reaches now
        self.w0 = self.snapshot()
        self.Z = self.G = self.Kinv = None                             # fn: this iteration's rows and inverse Gram
        self.strat_pen = strat                                         # --protect_strat: distill on fn's anchors
        if mode == "fn" or strat:   # the anchors have their own generator, so the training RNG is untouched
            self.n_anchors, self.pool_frac, self.gen = n_anchors, pool_frac, torch.Generator().manual_seed(seed)
        if mode != "fn":            # fn has no subspaces
            self.estimate()
        self.F0 = [f.clone() for f in self.F] if mode == "ewc" else None   # the Fisher at the old solution

    def old_mu(self, x):
        return self.old_head(self.actor[:-1](x))

    def estimate(self):
        from torch.func import jacrev, vmap
        mods = list(self.actor[:-1]) + [self.old_head]
        self.P, self.F = [], []
        h = self.states
        for i, m in enumerate(mods[:-1]):
            if isinstance(m, nn.Linear):
                with torch.no_grad():
                    xa = torch.cat([h, torch.ones(len(h), 1, device=h.device)], 1)
                    C = xa.T @ xa / len(xa)
                    z = m(h)
                later = mods[i + 1:]

                def f(zz):
                    for mm in later:
                        zz = mm(zz)
                    return zz
                J = torch.cat([vmap(jacrev(f))(z[k:k + 2000]).detach() for k in range(0, len(z), 2000)])
                J = J / self.sigma[None, :, None]
                Q = torch.einsum("nao,nap->op", J, J) / len(J)
                self.P.append((self.top(C), self.top(Q)))
                # exact diagonal Fisher of the mean: d mu_a / dW~[o,i] = J[a,o] x~[i], per state, then averaged
                self.F.append(J.pow(2).sum(1).T @ xa.pow(2) / len(xa))
                h = z
            else:
                with torch.no_grad():
                    h = m(h)

    def top(self, M):
        ev, V = torch.linalg.eigh(M)
        ev, V = ev.flip(0).clamp_min(0), V.flip(1)
        k = int((ev.cumsum(0) / ev.sum() < self.energy).sum()) + 1
        return V[:, :k] @ V[:, :k].T

    def snapshot(self):
        return [torch.cat([l.weight.detach(), l.bias.detach()[:, None]], 1).clone() for l in self.lin]

    def project_grad(self):
        """--protect_grad: remove the same part from the gradient before the optimizer step. Projecting only the step
        leaves the blocked gradient (most of the new skill's gradient lies in the old skill's subspace) in Adam's
        moments, so every step Adam turns it into a full-size step and the projection leaves its remainder in the
        directions that are not blocked. fn returns the share of each layer's gradient removed (a device tensor; None:
        nothing done)."""
        if self.mode == "fn":
            if self.Kinv is None or any(l.weight.grad is None for l in self.lin):
                return None
            with torch.no_grad():
                D = [torch.cat([l.weight.grad, l.bias.grad[:, None]], 1) for l in self.lin]
                cut = self.fn_cut(D)
                for l, c in zip(self.lin, cut):
                    l.weight.grad.sub_(c[:, :-1])
                    l.bias.grad.sub_(c[:, -1])
                return self.share(cut, D)
        with torch.no_grad():
            for l, (Pin, Pout) in zip(self.lin, self.P):
                if l.weight.grad is None:
                    continue
                G = torch.cat([l.weight.grad, l.bias.grad[:, None]], 1)
                G = G - (G @ Pin if self.mode == "gpm" else Pout @ G @ Pin)
                l.weight.grad.copy_(G[:, :-1])
                l.bias.grad.copy_(G[:, -1])

    def project(self, before):
        """Replace the step just taken by its allowed part; returns the share of the step removed per layer (fn: a
        device tensor)."""
        if self.mode == "fn":   # None: the Gram could not be factored this iteration, nothing projected
            if self.Kinv is None:
                return None
            with torch.no_grad():
                D = [torch.cat([l.weight, l.bias[:, None]], 1) - w0 for l, w0 in zip(self.lin, before)]
                cut = self.fn_cut(D)
                for l, c in zip(self.lin, cut):
                    l.weight.sub_(c[:, :-1])
                    l.bias.sub_(c[:, -1])
                return self.share(cut, D)
        removed = []
        with torch.no_grad():
            for l, w0, (Pin, Pout) in zip(self.lin, before, self.P):
                dW = torch.cat([l.weight, l.bias[:, None]], 1) - w0
                cut = dW @ Pin if self.mode == "gpm" else Pout @ dW @ Pin
                removed.append((cut.pow(2).sum() / dW.pow(2).sum().clamp_min(1e-30)).item())
                if self.mode not in ("block", "gpm"):   # measure what block would remove, change nothing
                    continue
                dW = dW - cut
                l.weight.copy_(w0[:, :-1] + dW[:, :-1])
                l.bias.copy_(w0[:, -1] + dW[:, -1])
        return removed

    def drift(self):
        """The old skill's mean-action drift on held states, in its own std units (squared, summed over dims)."""
        with torch.no_grad():
            return (((self.old_mu(self.states[:2000]) - self.ref) / self.sigma) ** 2).sum(1).mean().item()

    def add_states(self, x, cap=20000):
        """--protect_dagger: keep the latest `cap` states the old skill reached, labelled by the original actor.
        Non-finite states (a diverged simulation: old_rollout never resets it, so it adds one per remaining step) are
        dropped: one among fn's anchors makes the Gram NaN (the hidden layers then stay fixed that iteration), one in a
        distill batch makes every parameter NaN."""
        ok = torch.isfinite(x).all(1)
        if not ok.all():
            print(f"add_states: {int((~ok).sum())} of {len(x)} states not finite, dropped", flush=True)
            x = x[ok]
            if not len(x):      # keep the pool as it was: an empty one breaks penalty()'s randint
                return
        with torch.no_grad():
            y = self.teacher(x)
        self.dx = x if self.dx is None else torch.cat([self.dx, x])[-cap:]
        self.dy = y if self.dy is None else torch.cat([self.dy, y])[-cap:]

    def penalty(self, n=4096):
        """The baselines' loss term (differentiable in the hidden layers). distill with --protect_strat: on a fresh
        draw of fn's anchors instead of n uniform states (one stratified draw of n would deplete the strata)."""
        if self.mode == "distill" and self.strat_pen:
            mp, mb = self.draw()
            x, y = self.states[mb], self.ref_all[mb]
            if mp is not None:
                x, y = torch.cat([self.dx[mp], x]), torch.cat([self.dy[mp], y])
            return (((self.old_mu(x) - y) / self.sigma) ** 2).sum(1).mean()
        if self.mode == "distill":
            i = torch.randint(len(self.states), (n,), device=self.states.device)
            x, y = self.states[i], self.ref_all[i]
            if self.dx is not None:                 # half from the fixed states, half from the ones it reaches now
                j = torch.randint(len(self.dx), (n // 2,), device=self.dx.device)
                x, y = torch.cat([x[:n - n // 2], self.dx[j]]), torch.cat([y[:n - n // 2], self.dy[j]])
            return (((self.old_mu(x) - y) / self.sigma) ** 2).sum(1).mean()
        dW = [torch.cat([l.weight, l.bias[:, None]], 1) - w0 for l, w0 in zip(self.lin, self.w0)]
        if self.mode == "ewc":
            return sum((f * d.pow(2)).sum() for f, d in zip(self.F0, dW))
        return sum(d.pow(2).sum() for d in dW)

    def strat(self, x, n):
        """fn: the mask of n distinct rows of x (PickCube layout), at least 20% with the cube within 2.5 cm of the goal
        and at least 50% held, as far as x has such states, the rest uniform."""
        held, near = x[:, 18] > 0.5, x[:, 39:42].norm(dim=1) <= 0.025
        take = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        for pct, ok in ((20, near), (50, held), (100, torch.ones_like(held))):     # top each quota up, in this order
            k = min(-(-n * pct // 100) - int((take & ok).sum()), n - int(take.sum()))
            if k > 0:   # a fresh order per stratum: one shared order would bias the later draws against earlier strata
                p = torch.randperm(len(x), generator=self.gen).to(x.device)
                take[p[(ok & ~take)[p]][:k]] = True
        return take

    def draw(self):
        """fn's anchors (and --protect_strat's penalty states): pool_frac of n_anchors from the closed-loop pool, the
        rest from the stored states, each part stratified. Returns the two row masks (pool: None if it gives none)."""
        n_pool = min(round(self.n_anchors * self.pool_frac), len(self.dx)) if self.dx is not None else 0
        mp = self.strat(self.dx, n_pool) if n_pool else None
        return mp, self.strat(self.states, self.n_anchors - n_pool)

    def rows(self, x):
        """fn: at the current weights, per hidden Linear, Z = [input; 1] (M x in+1) and G = d(mu_old / sigma) / d(its
        output) (M x A x out) through the later layers and the old head."""
        from torch.func import jacrev, vmap
        mods = list(self.actor[:-1]) + [self.old_head]
        Z, G, h = [], [], x
        with torch.no_grad():
            for i, m in enumerate(mods[:-1]):
                z = m(h)
                if isinstance(m, nn.Linear):
                    later = mods[i + 1:]

                    def f(zz):
                        for mm in later:
                            zz = mm(zz)
                        return zz
                    Z.append(torch.cat([h, torch.ones_like(h[:, :1])], 1))
                    G.append(vmap(jacrev(f))(z) / self.sigma[None, :, None])
                h = z
        return Z, G

    @staticmethod
    def gram(Z, G):
        """fn: J J^T in float64, rows (n, a): the sum over layers of (G G^T) * (Z Z^T repeated over a, b)."""
        M, A = G[0].shape[:2]
        K = torch.zeros(M, A, M, A, dtype=torch.float64, device=G[0].device)
        for z, g in zip(Z, G):
            g, z = g.double().reshape(M * A, -1), z.double()
            K += (g @ g.T).view(M, A, M, A) * (z @ z.T)[:, None, :, None]
        return K.view(M * A, M * A)

    def refresh(self):
        """fn, once per iteration before the updates: draw the anchors (pool_frac of them from the closed-loop pool,
        the rest from the stored states, each part stratified), take their rows at the current weights and invert
        J J^T + delta I (float64 Cholesky; if it fails, delta x10 up to 3 times, then the hidden layers stay fixed this
        iteration). Returns fn.csv's anchors_pool, anchors_bank, frac_held, frac_near, k_trace, delta_used, skipped."""
        mp, mb = self.draw()
        xp, xb = self.dx[mp] if mp is not None else self.states[:0], self.states[mb]
        x = torch.cat([xp, xb])
        self.Z, self.G = self.rows(x)
        K = self.gram(self.Z, self.G)
        tr = K.trace().item()
        delta, self.Kinv = self.fn_delta * tr / len(K), None
        K.diagonal().add_(delta)
        for k in range(4):
            if k:       # not positive definite in float64: delta x10
                K.diagonal().add_(9 * delta)
                delta *= 10
            L, info = torch.linalg.cholesky_ex(K)
            if info.item() == 0 and L.diagonal().isfinite().all():
                self.Kinv = torch.cholesky_inverse(L)   # explicit, so each projection is one matrix-vector product
                break
        return [len(xp), len(xb), (x[:, 18] > 0.5).float().mean().item(),
                (x[:, 39:42].norm(dim=1) <= 0.025).float().mean().item(), tr, delta, int(self.Kinv is None)]

    def fn_cut(self, D):
        """fn: the part of the per-layer updates D (out x in+1) that moves the old skill's actions on the anchors to
        first order, J^T alpha with alpha = (J J^T + delta I)^-1 J D; per layer (sum_a alpha[n, a] G[n, a])^T Z."""
        v = sum(torch.einsum("nao,no->na", g, z @ d.T) for g, z, d in zip(self.G, self.Z, D))
        alpha = (self.Kinv @ v.reshape(-1).double()).to(v.dtype).view_as(v)
        return [torch.einsum("na,nao->on", alpha, g) @ z for g, z in zip(self.G, self.Z)]

    @staticmethod
    def share(cut, D):
        """The share of each layer's update that cut holds (squared norms), left on the device: main() averages the
        iteration's shares and reads them once, so the projections add no host sync to the Adam steps."""
        return torch.stack([c.pow(2).sum() / d.pow(2).sum().clamp_min(1e-30) for c, d in zip(cut, D)])

    def drift_pool(self, n=2000):
        """drift() on n states spread over the closed-loop pool, against the original actor's labels."""
        if self.dx is None:
            return float("nan")
        i = torch.linspace(0, len(self.dx) - 1, min(n, len(self.dx)), device=self.dx.device).long()
        with torch.no_grad():
            return (((self.old_mu(self.dx[i]) - self.dy[i]) / self.sigma) ** 2).sum(1).mean().item()


def gaussian_kl(mean_q, logstd_q, mean_p, logstd_p):
    """KL(q || p) for diagonal Gaussians, summed over action dims."""
    var_q, var_p = (2 * logstd_q).exp(), (2 * logstd_p).exp()
    return (logstd_p - logstd_q + (var_q + (mean_q - mean_p) ** 2) / (2 * var_p) - 0.5).sum(-1)


def gate_mask(obs, env_id, gate, dist, unified=False):
    """Per-state weight g(s) of the prior term."""
    if gate == "all":
        return torch.ones(obs.shape[0], device=obs.device)
    if gate == "none":
        return torch.zeros(obs.shape[0], device=obs.device)
    if gate == "pregrasp":
        assert unified, "pregrasp reads the grasp flags of the unified observation"
        return (obs[:, GRASP_DIMS].max(1).values < 0.5).float()
    j, k = (25, 18) if unified else (OBJ_POS_IDX[env_id], TCP_POS_IDX[env_id])   # mover position, tcp position
    far = torch.linalg.norm(obs[:, k:k + 3] - obs[:, j:j + 3], dim=1) > dist
    if gate == "reach":
        return far.float()
    if gate == "contact":
        return (~far).float()
    raise ValueError(gate)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = args.exp_name or f"{args.env_id}__{args.seed}__{int(time.time())}"
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    with open(f"runs/{run_name}/args.json", "w") as f:
        code = hashlib.md5()        # every module of the wrwl package, so env and adapter changes show up too
        for src in sorted(Path(__file__).parent.rglob("*.py")):
            code.update(src.read_bytes())
        json.dump({**vars(args), "code_md5": code.hexdigest()}, f, indent=1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda")

    env_kwargs = dict(obs_mode="state_dict" if args.unified else "state", render_mode=None, sim_backend="physx_cuda",
                      control_mode=args.control_mode, **json.loads(args.env_kwargs))
    if args.control_freq is not None:
        env_kwargs["sim_config"] = dict(control_freq=args.control_freq)
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=1, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs, eval_envs = FlattenActionSpaceWrapper(envs), FlattenActionSpaceWrapper(eval_envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset,
                                   record_metrics=True)
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    train_base, eval_base = envs._env.unwrapped, eval_envs._env.unwrapped

    def to_obs(o, base, grasped=None):
        if args.unified:
            return unify(args.env_id, o, base)
        if args.obs_adapter == "stack_to_pick":
            return stack_to_pick(o, base.agent.is_grasping(base.cubeA) if grasped is None else grasped)
        return o

    obs_dim = OBS_DIM if args.unified else int(np.prod(envs.single_observation_space.shape))
    if args.obs_adapter == "stack_to_pick":
        obs_dim = 42
    act_dim = int(np.prod(envs.single_action_space.shape))

    if args.unified:
        agent = SharedAgent(act_dim, args.init_logstd).to(device)
        shared_state = torch.load(args.shared_ckpt, map_location=device)
        if args.init_from_shared:
            agent.actor.load_state_dict(shared_state)
        else:
            agent.actor.obs_mean.copy_(shared_state["obs_mean"])
            agent.actor.obs_std.copy_(shared_state["obs_std"])
    else:
        agent = Agent(obs_dim, act_dim).to(device)
    if args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location=device)
        if args.init_actor_only:
            sd = {k: v for k, v in sd.items() if k.startswith("actor")}
        res = agent.load_state_dict(sd, strict=not args.init_actor_only)
        assert sd and not res.unexpected_keys and all(k.startswith("critic") for k in res.missing_keys), res
    if args.freeze_trunk:
        assert args.init_ckpt and not args.unified, "--freeze_trunk reuses the hidden layers of --init_ckpt"
        for prm in agent.actor_mean[:-1].parameters():
            prm.requires_grad_(False)
    assert not args.nsa_freeze_adapter or (args.nullspace_adapter and args.nsa_project and args.nsa_init != "zero"), \
        "--nsa_freeze_adapter: a projected, randomly initialised --nullspace_adapter (frozen at zero it is head-only)"
    assert args.nsa_proj in ("heads", "random") and (args.nsa_proj == "heads" or (
        args.nullspace_adapter and args.nsa_project and not args.nsa_base)), "--nsa_proj random: one projected level"
    assert args.nsa_mask in ("null", "visible", "zero") and (args.nsa_mask == "null" or (
        args.nullspace_adapter and args.nsa_project and args.nsa_proj == "heads" and not args.nsa_base
        and args.nsa_init != "zero")), "--nsa_mask visible|zero: one projected level with a random draw"
    assert not args.nsa_restore or (args.nullspace_adapter and not args.nsa_project and not args.nsa_base), \
        "--nsa_restore: a restoring force on the prefix-sum adapter's write (--nsa_project False, one level)"
    assert args.adapter_lr is None or (args.nullspace_adapter and not args.nsa_freeze_adapter), \
        "--adapter_lr: the learning rate of a trained --nullspace_adapter"
    if args.nullspace_adapter:
        assert args.init_ckpt and not args.unified and not args.freeze_trunk and not args.train_units \
            and args.protect_old == "none" and not args.seam_probe and args.nsa_head_from in ("base", "init") \
            and args.nsa_init in (None, "rand", "zero")
        if args.nsa_base:       # continue a chain: that run's levels and head, then one more level on top
            base_sd = torch.load(args.nsa_base, map_location="cpu")
            agent.actor_mean = build_actor(base_sd, obs_dim, act_dim).to(device)
            if args.nsa_head_from == "base":
                agent.actor_logstd.data.copy_(base_sd["actor_logstd"])
        agent.actor_mean = add_level(agent.actor_mean, obs_dim, args.nullspace_adapter, args.nsa_project,
                                     args.nsa_init, args.nsa_proj, 10_007 + args.seed, args.nsa_mask).to(device)
        if args.nsa_base and args.nsa_head_from == "init":     # the head (and log-std, kept) from --init_ckpt
            init_head = build_actor(torch.load(args.init_ckpt, map_location="cpu"), obs_dim, act_dim)[-1]
            agent.actor_mean[-1].load_state_dict(init_head.state_dict())
        if args.nsa_freeze_adapter:     # control: the head reads the newest adapter's fixed random null-space features
            agent.actor_mean[0].adapter.requires_grad_(False)
    restore_w = agent.actor_mean[-1].weight.detach().clone() if args.nsa_restore else None  # the old head's rows
    unit_masks = None   # --train_units: per hidden Linear, the rows that learn (none during the selection iterations)
    if args.train_units:
        assert args.freeze_trunk and 0 < args.train_units <= 1 and args.train_units_by in ("grad", "random")
        assert args.select_iters >= 1
        hidden = [m for m in agent.actor_mean[:-1] if isinstance(m, nn.Linear)]
        for prm in agent.actor_mean[:-1].parameters():
            prm.requires_grad_(True)
        unit_masks = [torch.zeros(l.out_features, dtype=torch.bool, device=device) for l in hidden]
        unit_imp = [torch.zeros(l.out_features, device=device) for l in hidden]
    protector, rem_acc, grad_acc = None, [], []
    fn = args.protect_old == "fn"
    assert not args.fn_warmup or fn, "--fn_warmup belongs to --protect_old fn"
    assert not args.protect_strat or args.protect_old == "distill", "--protect_strat: distill on fn's anchors"
    if args.protect_old != "none":
        assert args.protect_old in ("block", "gpm", "monitor", "distill", "ewc", "l2", "fn") and args.init_ckpt \
            and args.protect_states and (not args.protect_grad or args.protect_old in ("block", "gpm", "fn")) \
            and not args.freeze_trunk
        ps = torch.load(args.protect_states, map_location="cpu")
        xs = ps["x"][ps["keep"]].float()
        xs = xs[torch.randperm(len(xs), generator=torch.Generator().manual_seed(0))[:20000]].to(device)
        old_head = copy.deepcopy(agent.actor_mean[-1]).requires_grad_(False)    # the old skill keeps its own head
        if fn:      # closed-loop anchors, a Gram per iteration (no fixed subspaces to re-estimate), PickCube's strata
            assert args.protect_env and args.protect_dagger >= 1 and not args.protect_every and args.fn_anchors > 0 \
                and 0 <= args.fn_pool_frac <= 1 and xs.shape[1] == 42
        assert not args.protect_strat or (args.fn_anchors > 0 and 0 <= args.fn_pool_frac <= 1 and xs.shape[1] == 42)
        protector = Protector(agent.actor_mean, old_head, xs, args.protect_energy, args.protect_old,
                              agent.actor_logstd.detach().exp().flatten(), args.fn_anchors, args.fn_pool_frac,
                              args.seed, args.protect_strat)
        prot_f = open(f"runs/{run_name}/protect.csv", "w", newline="")
        prot_w = csv.writer(prot_f)
        prot_w.writerow(["iteration", "global_step", "old_drift", "removed_l1", "removed_l2", "removed_l3"])
        prot_w.writerow([0, 0, protector.drift(), 0.0, 0.0, 0.0])
        if fn:
            fn_f = open(f"runs/{run_name}/fn.csv", "w", newline="")
            fn_w = csv.writer(fn_f)
            fn_w.writerow(["iteration", "global_step", "rho_g_l1", "rho_g_l2", "rho_g_l3", "rho_s_l1", "rho_s_l2",
                           "rho_s_l3", "anchors_pool", "anchors_bank", "frac_held", "frac_near", "k_trace",
                           "delta_used", "skipped", "drift_pool"])
    old_env = None
    if args.protect_env:
        # the head at this point is the old skill's own (in a chain only with --nsa_head_from init: Pick's)
        assert args.init_ckpt and args.protect_dagger >= 1 and not (args.nsa_base and args.nsa_head_from == "base")
        old_env = gym.make(args.protect_env, num_envs=256, obs_mode="state", render_mode=None, sim_backend="physx_cuda",
                           control_mode=args.control_mode, **json.loads(args.protect_env_kwargs))
        old_steps = gym_utils.find_max_episode_steps_value(old_env)
        old_head0 = copy.deepcopy(agent.actor_mean[-1]).requires_grad_(False)     # the old skill's own head
        old_f = open(f"runs/{run_name}/old_eval.csv", "w", newline="")
        old_w = csv.writer(old_f)
        old_w.writerow(["iteration", "global_step", "old_success", "old_states"])

        def old_feat(x):
            """What the old head reads: the last hidden layer, or with --nsa_project False its own level (below)."""
            return agent.actor_mean[0].trunk(x) if args.nullspace_adapter and not args.nsa_project \
                else agent.actor_mean[:-1](x)

        def old_rollout(seed):
            """The old skill (old head on the current hidden layers), deterministic, on its own task: its states
            before each env's first success, and its success rate."""
            o, _ = old_env.reset(seed=seed)
            done = torch.zeros(len(o), dtype=torch.bool, device=o.device)
            xs = []
            with torch.no_grad():
                for _ in range(old_steps):
                    xs.append(o[~done])
                    o, _, _, _, inf = old_env.step(old_head0(old_feat(o)).clamp(-1, 1))
                    done |= inf["success"]
            return torch.cat(xs), done.float().mean().item()
    if fn or (args.protect_strat and args.protect_env):     # the closed-loop pool is never empty: the old skill's
        ox, osucc = old_rollout(args.seed * 100003)            # own states through the initial hidden layers
        protector.add_states(ox)
        old_w.writerow([0, 0, osucc, len(ox)])
        old_f.flush()
    assert args.seam_rule in ("pick", "push") and args.seam_dims in ("gripper", "arm", "all")
    agent.seam_rule = args.seam_rule
    if args.seam_dims != "gripper":
        agent.dim_mask = torch.ones(act_dim, dtype=torch.bool, device=device)
        if args.seam_dims == "arm":
            agent.dim_mask[-1] = False
    if args.explore_boost:
        assert obs_dim == (35 if args.seam_rule == "push" else 42) or args.seam_probe, "seam rule vs observation layout"
        agent.explore_boost, agent.explore_radius = args.explore_boost, args.explore_radius
    if args.min_grip_std:
        assert not args.grip_floor_seam or args.seam_probe or obs_dim == (35 if args.seam_rule == "push" else 42), \
            "seam rule vs observation layout"
        agent.min_grip_std, agent.grip_floor_seam = args.min_grip_std, args.grip_floor_seam
        agent.grip_floor_anti, agent.explore_radius = args.grip_floor_anti, args.explore_radius
        assert args.grip_floor_seam or not args.grip_floor_anti, "--grip_floor_anti inverts --grip_floor_seam"
    if args.seam_probe:     # the seam is where the source's own done-probe fires (scripts/train_done_probe.py)
        assert args.explore_boost or args.grip_floor_seam, "--seam_probe only moves the boost or floor region"
        src_ckpt = args.init_ckpt or args.prior_ckpt
        assert src_ckpt, "--seam_probe reads the hidden layer of the source (--init_ckpt or --prior_ckpt)"
        probe = torch.load(args.seam_probe)
        src_net = Agent(obs_dim, act_dim).to(device)
        src_net.load_state_dict(torch.load(src_ckpt, map_location=device))
        src_trunk = src_net.actor_mean[:-1].requires_grad_(False)
        pw, pb = probe["w"].to(device), probe["b"].to(device)

        def seam_fn(x):     # a closure, so the frozen source copy is neither a submodule nor saved nor trained
            with torch.no_grad():
                return (src_trunk(x) @ pw + pb) > 0

        agent.seam_fn = seam_fn
    oracle = None
    if args.gate == "oracle":
        L = torch.load(args.oracle_labels, weights_only=False)
        cols = {w: j for j, (src, w) in enumerate(L["branches"]) if src == args.oracle_source}
        v_star = L["Q"][:, 0]
        delta = v_star - L["Q"][:, cols[1]]
        if 3 in cols:   # the 3-chunk window only where pi* still has >= 15 steps afterwards, else it measures the deadline
            ok3 = L["t"] + 3 * L["chunk_len"] + 15 <= L["horizon"]
            delta = torch.where(ok3, torch.maximum(delta, v_star - L["Q"][:, cols[3]]), delta)
        keep = v_star > 0.1          # where the expert itself fails there is no copy loss to measure
        lx = L["x"][keep].to(device)
        if args.obs_adapter == "stack_to_pick":
            lx = stack_to_pick(lx, L["grasp"][keep].to(device))
        scale = lx.std(0).clamp_min(1e-2)
        lxn = lx / scale
        safe = (delta[keep] <= args.oracle_thresh).float().to(device)
        if args.oracle_shuffle:   # control: the same neighbourhoods and copy fraction, labels on random states
            safe = safe[torch.randperm(len(safe), generator=torch.Generator().manual_seed(args.seed)).to(device)]
        dd, ii = torch.cdist(lxn, lxn).topk(args.oracle_k + 1, largest=False)
        loo = (((safe[ii[:, 1:]].mean(1) >= 0.5).float()) == safe).float().mean()
        oracle = dict(x=lxn, scale=scale, safe=safe, radius=1.5 * dd[:, 1].quantile(0.95))
        print(f"oracle gate: {int(keep.sum())} of {len(keep)} labelled states kept (V* > 0.1), safe fraction "
              f"{safe.mean():.2f}, leave-one-out kNN agreement {loo:.2f}, fallback radius {oracle['radius']:.2f}",
              flush=True)
    if args.init_task_like:
        task0 = OBS_DIM - len(TASKS)
        with torch.no_grad():
            w = agent.actor.trunk[0].weight
            w[:, task0 + TASKS.index(args.env_id)] = w[:, task0 + TASKS.index(args.init_task_like)]
    prior = None
    if args.prior_self:
        prior = copy.deepcopy(agent)
    elif args.prior_ckpt:
        assert not args.unified, "a flat-state prior checkpoint cannot read the unified observation"
        prior = Agent(obs_dim, act_dim).to(device)
        prior.load_state_dict(torch.load(args.prior_ckpt, map_location=device))
    if prior is not None:
        prior.eval()
        for prm in prior.parameters():
            prm.requires_grad_(False)
    if args.head_lr is None and args.adapter_lr is None:
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    else:       # the actor's output layer and log-std, and / or the newest adapter, in their own groups
        assert not args.unified
        own = [*agent.actor_mean[-1].parameters(), agent.actor_logstd] if args.head_lr is not None else []
        ada = list(agent.actor_mean[0].adapter.parameters()) if args.adapter_lr is not None else []
        groups = [{"params": [p for p in agent.parameters() if all(p is not q for q in own + ada)]}]
        groups += [{"params": own, "lr": args.head_lr}] if own else []
        groups += [{"params": ada, "lr": args.adapter_lr}] if ada else []
        optimizer = optim.Adam(groups, lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = to_obs(next_obs, train_base)
    eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    act_low = torch.from_numpy(envs.single_action_space.low).to(device)
    act_high = torch.from_numpy(envs.single_action_space.high).to(device)
    csv_f = open(f"runs/{run_name}/eval.csv", "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["iteration", "global_step", "success_once", "success_at_end", "return", "prior_coef", "wall_s",
                    *EVAL_COLS])
    kl_f = open(f"runs/{run_name}/train.csv", "w", newline="")
    kl_w = csv.writer(kl_f)
    kl_w.writerow(["iteration", "global_step", "kl_per_decision", "kl_per_episode", "decisions_per_episode",
                   "copy_fraction", "fallback_fraction", "train_success", "train_episodes", "std_arm", "std_gripper",
                   "seam_frac", "applied_std_arm", "applied_std_gripper", *(f"st_{c}" for c in COUNTS), *PHASE_COLS,
                   "restore_pen"])
    # stage logging (wrwl/phase_log.py): the env's own grasp / placed flags, PickCube layout only
    st_keys = INFO_KEYS.get(args.env_id) if not args.unified and args.obs_adapter is None else None
    tr_st = EpisodeStages(args.num_envs, device, st_keys) if st_keys else None
    ev_st = EpisodeStages(args.num_eval_envs, device, st_keys) if st_keys else None
    phase_on = not args.unified and obs_dim == 42
    actor0 = copy.deepcopy(agent.actor_mean).requires_grad_(False) if phase_on else None   # the start policy
    nan_cols = lambda n: [float("nan")] * n     # noqa: E731
    gate_stats = (float("nan"), float("nan"))
    hits = 0
    last_save = 0
    assert args.seam_gate in ("state", "time", "grasp", "latch")
    seam_used = bool(args.explore_boost or (args.min_grip_std and args.grip_floor_seam))
    if args.seam_gate != "state":   # the region does not follow from the observation: fix it per step, reuse it in updates
        assert seam_used, "--seam_gate sets the boost or floor region"
        assert args.seam_gate == "latch" or not args.seam_probe, "--seam_gate time / grasp replace the region"
        assert args.seam_gate != "grasp" or obs_dim == 42, "the grasp gate reads PickCube's grasp flag"
        gates = torch.zeros((args.num_steps, args.num_envs), dtype=torch.bool, device=device)
        t_env = torch.zeros(args.num_envs, dtype=torch.long, device=device)    # steps since this env's reset
        latch = torch.zeros(args.num_envs, dtype=torch.bool, device=device)    # the region fired earlier this episode

    for iteration in range(1, args.num_iterations + 1):
        frac_done = (iteration - 1) / args.num_iterations
        lam = args.prior_coef * max(0.0, 1.0 - frac_done / args.prior_anneal_frac) if prior is not None else 0.0
        if args.save_every and global_step // args.save_every > last_save:
            last_save = global_step // args.save_every
            torch.save(agent.state_dict(), f"runs/{run_name}/ckpt_{global_step}.pt")
        if args.halt_at and global_step >= args.halt_at:   # stop early, the schedule still spans total_timesteps
            print("halt at", global_step, flush=True)
            break
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1 or args.eval_freq == 1:
            metrics = defaultdict(list)
            eval_obs, _ = eval_envs.reset()
            eval_obs = to_obs(eval_obs, eval_base)
            if ev_st is not None:
                ev_st.reset()
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, _, _, _, ei = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    eval_obs = to_obs(eval_obs, eval_base)
                    if ev_st is not None:
                        ev_st.update(ei)
                    if "final_info" in ei:
                        for k, v in ei["final_info"]["episode"].items():
                            metrics[k].append(v[ei["_final_info"]])
            row = {k: torch.cat(v).float().mean().item() for k, v in metrics.items() if len(v)}
            csv_w.writerow([iteration, global_step, row.get("success_once", float("nan")),
                            row.get("success_at_end", float("nan")), row.get("return", float("nan")), lam,
                            round(time.time() - start_time, 1),
                            *(ev_st.fractions() if ev_st is not None else nan_cols(len(EVAL_COLS)))])
            csv_f.flush()
            print(f"iter {iteration} step {global_step} eval success_once={row.get('success_once', float('nan')):.3f}"
                  f" lam={lam:.3f}", flush=True)
            if args.stop_success is not None and global_step >= args.stop_after \
                    and row.get("success_once", 0.0) >= args.stop_success:
                hits += 1
                if hits >= args.stop_patience:
                    print("early stop: reached", args.stop_success, flush=True)
                    break
            else:
                hits = 0

        n_done, n_succ = 0, 0.0      # episodes finished in this iteration's rollout, and how many succeeded
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            if args.seam_gate != "state":   # a just-reset env's grasp flag can be stale, so not on its first step
                if args.seam_gate == "latch":
                    latch = (latch | agent.at_goal(next_obs)) & (next_done == 0)
                    gates[step] = latch
                else:
                    gates[step] = t_env >= args.seam_t0 if args.seam_gate == "time" else \
                        (next_obs[:, 18] > 0.5) & (next_done == 0)
                agent.mask_override = gates[step]
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            agent.mask_override = None
            actions[step] = action
            logprobs[step] = logprob
            next_obs, reward, terminations, truncations, infos = envs.step(torch.clamp(action, act_low, act_high))
            if tr_st is not None:
                tr_st.update(infos)
            g_next = None
            if args.obs_adapter == "stack_to_pick":     # contact flags are stale on envs that were just auto-reset
                g_next = train_base.agent.is_grasping(train_base.cubeA)
                if "_final_info" in infos:
                    g_next = g_next & ~infos["_final_info"]
            next_obs = to_obs(next_obs, train_base, g_next)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            if args.seam_gate != "state":
                t_env = (t_env + 1) * (1 - next_done.long())
            rewards[step] = reward.view(-1) * args.reward_scale
            if "final_info" in infos:
                n_done += int(infos["_final_info"].sum())
                n_succ += float(infos["final_info"]["episode"]["success_once"][infos["_final_info"]].float().sum())
                done_mask = infos["_final_info"]
                if args.bootstrap_at_done == "truncated":
                    done_mask = done_mask & truncations.bool() & ~terminations.bool()
                g_fin = infos["final_info"]["is_cubeA_grasped"] if args.obs_adapter == "stack_to_pick" else None
                final_obs = to_obs(infos["final_observation"], train_base, g_fin)
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = \
                        agent.get_value(final_obs[done_mask]).view(-1)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done, nextvalues = 1.0 - next_done, next_value
                else:
                    next_not_done, nextvalues = 1.0 - dones[t + 1], values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + args.gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape(-1, obs_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1, act_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_gates = gates.reshape(-1) if args.seam_gate != "state" else None
        if prior is not None and lam > 0:
            with torch.no_grad():
                b_prior_mean, b_prior_logstd = prior.dist(b_obs)
                if oracle is not None:
                    gs, fb = [], []
                    for c0 in range(0, len(b_obs), 8192):
                        d, i = torch.cdist(b_obs[c0:c0 + 8192] / oracle["scale"], oracle["x"]).topk(
                            args.oracle_k, largest=False)
                        far = d[:, 0] > oracle["radius"]
                        gs.append(torch.where(far, 1.0, (oracle["safe"][i].mean(1) >= 0.5).float()))
                        fb.append(far.float())
                    b_gate = torch.cat(gs)
                    gate_stats = (b_gate.mean().item(), torch.cat(fb).mean().item())
                else:
                    b_gate = gate_mask(b_obs, args.env_id, args.gate, args.gate_dist, args.unified)
                    gate_stats = (b_gate.mean().item(), 0.0)
        with torch.no_grad():       # the std actually applied in this rollout (the policy before the update)
            app, seam_n = [], 0.0
            for start in range(0, args.batch_size, 65536):
                sl = slice(start, start + 65536)
                agent.mask_override = b_gates[sl] if b_gates is not None else None
                app.append(agent.dist(b_obs[sl])[1].exp())
                if seam_used:
                    seam_n += agent.at_goal(b_obs[sl]).float().sum().item()
                agent.mask_override = None
            app = torch.cat(app)
        ph_row = nan_cols(len(PHASE_COLS))
        if phase_on:    # the rollout policy's action means by stage, and their deviation from the start policy
            with torch.no_grad():
                ch = range(0, args.batch_size, 65536)
                mu = torch.cat([agent.actor_mean(b_obs[c:c + 65536]) for c in ch])
                mu0 = torch.cat([actor0(b_obs[c:c + 65536]) for c in ch])
                ph_row = phase_stats(b_obs, dones.reshape(-1) == 0, mu, mu0)
        pen_acc = []    # --nsa_restore: the penalty on each minibatch
        if fn:      # fresh anchors and their Gram at the current weights, for all of this iteration's updates
            fn_row = protector.refresh()
            if fn_row[-1]:
                print(f"fn: Gram not positive definite up to delta {fn_row[-2]:.2e}, the hidden layers stay fixed in "
                      f"iteration {iteration}", flush=True)
        skip = fn and fn_row[-1]    # no inverse Gram: never move the hidden layers unprojected
        fn_warm = fn and args.critic_warmup < iteration and (iteration <= args.fn_warmup or skip)  # trunk stays fixed
        fn_move = fn and iteration > max(args.critic_warmup, args.fn_warmup) and not skip  # both projections apply

        agent.train()
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start:start + args.minibatch_size]
                if b_gates is not None:
                    agent.mask_override = b_gates[mb]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                if args.loss == "clip" and args.target_kl is not None and approx_kl > args.target_kl:
                    break
                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                if args.loss == "cov":
                    # F8: per-decision optimum r - 1 = sqrt(2 Delta) * A / sqrt(H), so the per-episode KL is ~Delta
                    a_cov = mb_adv / np.sqrt(max_episode_steps)
                    pg_loss = -(ratio * a_cov - (ratio - 1) ** 2 / (2 * np.sqrt(2 * args.delta))).mean()
                else:
                    pg_loss = torch.max(-mb_adv * ratio,
                                        -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb]) ** 2).mean()
                loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss
                if iteration <= args.critic_warmup:     # value-only updates while the initial actor keeps acting
                    loss = args.vf_coef * v_loss
                elif prior is not None and lam > 0:
                    mean, logstd = agent.dist(b_obs[mb])
                    if not args.copy_std:
                        logstd = logstd.detach()
                    if args.copy_loss == "fkl":     # KL(prior || pi): mass-covering, kickstarting's cross-entropy
                        kl = gaussian_kl(b_prior_mean[mb], b_prior_logstd[mb], mean, logstd)
                    elif args.copy_loss == "rkl_def":
                        std = logstd.exp()
                        a_s = mean + std * torch.randn(8, *mean.shape, device=device)
                        lq = Normal(mean, std).log_prob(a_s).sum(-1)
                        lp = Normal(b_prior_mean[mb], b_prior_logstd[mb].exp()).log_prob(a_s).sum(-1)
                        lb = Normal(torch.zeros_like(mean), torch.full_like(mean, args.base_std)).log_prob(a_s).sum(-1)
                        kl = (lq - torch.logaddexp(lp, lb) + math.log(2)).mean(0)
                    else:                             # KL(pi || prior): mode-seeking
                        kl = gaussian_kl(mean, logstd, b_prior_mean[mb], b_prior_logstd[mb])
                    loss = loss + lam * (b_gate[mb] * kl).mean()
                if protector is not None and protector.mode in ("distill", "ewc", "l2") \
                        and iteration > args.critic_warmup:
                    loss = loss + args.protect_coef * protector.penalty()
                if restore_w is not None and iteration > args.critic_warmup:   # W_old g(x) on this minibatch's states
                    lvl, xb = agent.actor_mean[0], b_obs[mb]
                    with torch.no_grad():
                        h2 = lvl.trunk[:4](xb)
                    pen = (lvl.adapter(torch.cat([xb, h2], 1)) @ restore_w.T).pow(2).sum(1).mean()
                    loss = loss + args.nsa_restore * pen
                    pen_acc.append(pen.detach())
                agent.mask_override = None
                optimizer.zero_grad(set_to_none=True)   # actor grads stay None during --critic_warmup
                loss.backward()
                if unit_masks is not None:
                    for l, m, imp in zip(hidden, unit_masks, unit_imp):
                        if l.weight.grad is None:
                            continue
                        if iteration <= args.select_iters:
                            imp += l.weight.grad.pow(2).sum(1) + l.bias.grad.pow(2)
                        l.weight.grad[~m] = 0     # zero grads: Adam's moments stay 0, so frozen rows never move
                        l.bias.grad[~m] = 0
                if fn_warm:     # no trunk gradient, so Adam leaves the trunk (and its moments) untouched
                    for l in protector.lin:
                        l.weight.grad = l.bias.grad = None
                if args.protect_grad or fn_move:    # fn always projects the gradient too
                    rho = protector.project_grad()
                    if rho is not None:
                        grad_acc.append(rho)
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                before = protector.snapshot() if protector is not None else None
                optimizer.step()
                if protector is not None and (fn_move or not fn):
                    rho = protector.project(before)
                    if rho is not None:
                        rem_acc.append(rho)
            if args.loss == "clip" and args.target_kl is not None and approx_kl > args.target_kl:
                break
        agent.mask_override = None      # the target_kl break leaves a minibatch's mask behind
        if protector is not None:
            r = (torch.stack(rem_acc) if fn else torch.tensor(rem_acc)).mean(0).tolist() if rem_acc \
                else [float("nan")] * 3
            prot_w.writerow([iteration, global_step, protector.drift(), *r])
            prot_f.flush()
            rem_acc = []
            if fn:
                rg = torch.stack(grad_acc).mean(0).tolist() if grad_acc else [float("nan")] * 3
                fn_w.writerow([iteration, global_step, *rg, *r, *fn_row, protector.drift_pool()])
                fn_f.flush()
                grad_acc = []
            if args.protect_every and iteration % args.protect_every == 0:
                protector.estimate()
        if old_env is not None and iteration % args.protect_dagger == 0:
            ox, osucc = old_rollout(args.seed * 100003 + iteration)    # never the retention_eval starts (seed 0)
            if args.protect_old in ("distill", "fn"):
                protector.add_states(ox)
            old_w.writerow([iteration, global_step, osucc, len(ox)])
            old_f.flush()
        if unit_masks is not None and iteration == args.select_iters:
            gen = torch.Generator().manual_seed(args.seed + 7)
            for l, m, imp in zip(hidden, unit_masks, unit_imp):
                k = max(1, round(args.train_units * l.out_features))
                idx = imp.topk(k).indices if args.train_units_by == "grad" else \
                    torch.randperm(l.out_features, generator=gen)[:k].to(device)
                m[idx] = True
            json.dump({"by": args.train_units_by, "frac": args.train_units, "select_iters": args.select_iters,
                       "rows": [m.nonzero().flatten().tolist() for m in unit_masks],
                       "importance": [imp.tolist() for imp in unit_imp]}, open(f"runs/{run_name}/train_units.json", "w"))

        with torch.no_grad():
            kls = []
            for start in range(0, args.batch_size, 65536):
                sl = slice(start, start + 65536)
                agent.mask_override = b_gates[sl] if b_gates is not None else None
                _, nlp, _, _ = agent.get_action_and_value(b_obs[sl], b_actions[sl])
                lr_ = nlp - b_logprobs[sl]
                kls.append(((lr_.exp() - 1) - lr_))
                agent.mask_override = None
            kl_dec = torch.cat(kls).mean().item()
            std = agent.actor_logstd.exp().flatten()      # the learned std, without --explore_boost
        kl_w.writerow([iteration, global_step, kl_dec, kl_dec * max_episode_steps, max_episode_steps,
                       *(gate_stats if prior is not None and lam > 0 else (float("nan"), float("nan"))),
                       n_succ / max(1, n_done), n_done, std[:-1].mean().item(), std[-1].item(),
                       seam_n / args.batch_size if seam_used else float("nan"),
                       app[:, :-1].mean().item(), app[:, -1].mean().item(),
                       *(tr_st.pop() if tr_st is not None else nan_cols(len(COUNTS))), *ph_row,
                       torch.stack(pen_acc).mean().item() if pen_acc else float("nan")])
        if iteration % 10 == 0:
            kl_f.flush()

    kl_f.close()
    if old_env is not None:     # from retention_eval.py's starts (seed 0), so the final number is comparable
        ox, osucc = old_rollout(0)
        old_w.writerow([iteration, global_step, osucc, len(ox)])
        old_f.close()
        print(f"old skill on {args.protect_env} at the end: {osucc:.3f}", flush=True)
        if args.nullspace_adapter:
            with torch.no_grad():
                gap = (old_head0(agent.actor_mean[:-1](ox)) - old_head0(agent.actor_mean[0].trunk(ox))).abs().max()
            print(f"nullspace adapter: max |old head on h~ - old head on h| = {gap.item():.2e}", flush=True)
    if args.save_model:
        torch.save(agent.state_dict(), f"runs/{run_name}/final_ckpt.pt")
    csv_f.close()
    print(f"done {run_name} steps={global_step} wall={time.time() - start_time:.0f}s "
          f"SPS={int(global_step / (time.time() - start_time))}", flush=True)
    envs.close()
    eval_envs.close()
