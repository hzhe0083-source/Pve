"""Null-space skill adapters: add a skill to one policy without changing what any earlier skill does.

A policy is a shared trunk (the PickCube expert's three tanh layers) plus one linear output layer per skill. To add skill
k, the trunk and everything added before stay frozen, and a new adapter g_k (reading the input and the trunk's second
hidden layer) writes into the last hidden layer only in directions no earlier output layer can read:
    h_k = h_{k-1} + (I - P_k) g_k(x, h2),   P_k = projector onto the rows of every earlier head (8 per skill).
Every head reads the same h, so an earlier head's output is unchanged at every state, including states it reaches
only after the change (to float precision; no stored states, no approximation), and a later skill can read what the
earlier adapters wrote. The adapter starts random, not zero: the new head starts as a copy of an old head, so at the
start it cannot see these directions either (training starts from that old policy all the same), and the head then
reads random features of the new task's states. With the projection, the adapter gets no gradient until the copied
head first moves off the protected rows, whatever its init.
"""
import re

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NullSpaceTrunk(nn.Module):
    """One level: `trunk` (the base nn.Sequential of 3 tanh Linear layers, or the previous level) frozen, plus an
    adapter whose output is projected away from `protected` (rows of every earlier head, k x 256).
    project=False is the prefix-sum ablation: no projection (null = I) and the adapter's last layer starts at zero, so
    the copied head still starts as the old policy; each earlier skill then stays exact only when it reads its own
    level (h0 + g1 + ... + gj), not the last one. init="rand" with project=False: the same random draw as the projected
    level, projected once at construction (so it still starts as the old policy) and trained unprojected; init="zero"
    with project=True: a zero last layer.
    proj="random" (control nsa<H>rp, needs project=True): the write is projected away from as many random orthonormal
    rows (Haar, from its own generator seeded proj_seed, so the global stream and the adapter's draw are nsa<H>'s) as
    the heads have, I - Br^T Br, instead of the heads' rows; the old heads then read the write. The random draw starts
    in the null space of the heads and of Br together, so the copied head still starts as the old policy.
    mask="visible" / "zero" (controls for rule F, need project=True and proj="heads"): the write is multiplied by
    B^T B (only the directions the earlier heads read) or by 0 (no write) instead of I - B^T B; a random draw starts
    projected once into the null space, as for npr, so the copied head still starts as the old policy."""

    def __init__(self, trunk, protected, obs_dim, width, project=True, init=None, proj="heads", proj_seed=0,
                 mask="null"):
        super().__init__()
        assert proj in ("heads", "random") and (proj == "heads" or project), "proj='random' replaces the projection"
        assert mask in ("null", "visible", "zero") and (mask == "null" or (project and proj == "heads")), \
            "mask='visible' / 'zero' replaces the heads' projection"
        self.trunk = trunk.requires_grad_(False)
        d = protected.shape[1]
        rand = project if init is None else init == "rand"     # None: random with the projection, zero without
        self.adapter = nn.Sequential(layer_init(nn.Linear(obs_dim + d, width)), nn.Tanh(),
                                     layer_init(nn.Linear(width, d), std=1.0 if rand else 0.0))
        # an orthonormal basis of the protected rows in float64 (a float32 pinv of several heads' rows leaks ~1e-4)
        _, s, vh = torch.linalg.svd(protected.double(), full_matrices=False)
        b = vh[s > s[0] * 1e-9] if len(s) and s[0] > 0 else vh[:0]
        eye = torch.eye(d, dtype=torch.float64, device=protected.device)
        keep = eye - b.T @ b
        start = keep if rand and not project else None     # nsa<H>npr: nsa<H>'s draw written once into the null space
        if proj == "random":
            q, _ = torch.linalg.qr(torch.randn(d, len(b), generator=torch.Generator().manual_seed(proj_seed),
                                               dtype=torch.float64))
            br = q.T.to(protected.device)                   # len(b) x d, orthonormal rows spanning a Haar subspace
            _, s2, vh2 = torch.linalg.svd(torch.cat([b, br]), full_matrices=False)
            bj = vh2[s2 > s2[0] * 1e-9]
            start, keep = (eye - bj.T @ bj if rand else None), eye - br.T @ br
        if mask != "null":          # the draw starts in the null space, as npr's; the write is then masked
            start, keep = (keep if rand else None), (eye - keep if mask == "visible" else torch.zeros_like(eye))
        if start is not None:       # so the copied head still starts as the old policy; the draw then trains under
            w = self.adapter[2].weight      # null (no projection for npr, I - Br^T Br for rp)
            with torch.no_grad():
                w.copy_((start.to(w.device) @ w.double()).float())
        self.register_buffer("null", (keep if project else eye).float())

    def features(self, x):
        """The base trunk's second hidden layer and the last hidden layer with every adapter up to this level."""
        if isinstance(self.trunk, NullSpaceTrunk):
            h2, h = self.trunk.features(x)
        else:
            h2 = self.trunk[:4](x)
            h = self.trunk[4:](h2)
        return h2, h + self.adapter(torch.cat([x, h2], 1)) @ self.null

    def forward(self, x):
        return self.features(x)[1]

    def protected_rows(self):
        """An orthonormal basis (rows) of the directions this level keeps clear."""
        ev, V = torch.linalg.eigh(torch.eye(len(self.null), device=self.null.device) - self.null)
        return V[:, ev > 0.5].T

    def level(self, k):
        """The module of level k (1 = the first adapter); levels below k ignore everything added after them."""
        m = self
        while depth(m) > k:
            m = m.trunk
        return m


def depth(m):
    return 1 + depth(m.trunk) if isinstance(m, NullSpaceTrunk) else 0


def add_level(actor_mean, obs_dim, width, project=True, init=None, proj="heads", proj_seed=0, mask="null"):
    """actor_mean = nn.Sequential(trunk or level, head) -> the same with one more level. The level protects every
    head so far: the rows the previous level protects plus the current head (whose copy keeps training as the new
    skill's head). project=False: the prefix-sum ablation (NullSpaceTrunk). proj="random": the same-rank random
    projection control (NullSpaceTrunk)."""
    inner, head = actor_mean[0] if len(actor_mean) == 2 else actor_mean[:-1], actor_mean[-1]
    w = head.weight.detach()
    rows = torch.cat([inner.protected_rows(), w]) if isinstance(inner, NullSpaceTrunk) else w
    return nn.Sequential(NullSpaceTrunk(inner, rows, obs_dim, width, project, init, proj, proj_seed, mask), head)


def build_actor(sd, obs_dim, act_dim, prefix="actor_mean."):
    """Rebuild actor_mean (any number of levels) from a checkpoint's state dict and load it."""
    keys = [k for k in sd if k.startswith(prefix)]
    n = max((k[len(prefix):].count("trunk.") for k in keys), default=0)
    widths = {}
    for k in keys:
        m = re.fullmatch(re.escape(prefix) + r"0\.((?:trunk\.)*)adapter\.0\.weight", k)
        if m:
            widths[n - m.group(1).count("trunk.")] = sd[k].shape[0]
    base = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 256),
                         nn.Tanh())
    m = base
    for lv in range(1, n + 1):
        m = NullSpaceTrunk(m, torch.zeros(8 * lv, 256), obs_dim, widths[lv])
    actor = nn.Sequential(m, nn.Linear(256, act_dim)) if n else nn.Sequential(*base, nn.Linear(256, act_dim))
    actor.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)})
    return actor
