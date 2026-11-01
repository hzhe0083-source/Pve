"""Shared multi-task policy over the unified observation: a trunk (the latent space) plus an action head."""
import torch
from torch import nn

from wrwl.envs.unified import CATEGORICAL_DIMS, OBS_DIM


class SharedPolicy(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=8, width=256, latent_dim=256):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.trunk = nn.Sequential(nn.Linear(obs_dim, width), nn.Tanh(), nn.Linear(width, latent_dim), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(latent_dim, width), nn.Tanh(), nn.Linear(width, act_dim))

    def set_normalizer(self, obs):
        self.obs_mean.copy_(obs.mean(0))
        self.obs_std.copy_(obs.std(0).clamp_min(1e-2))
        self.obs_mean[CATEGORICAL_DIMS] = 0.0
        self.obs_std[CATEGORICAL_DIMS] = 1.0

    def latent(self, obs):
        return self.trunk((obs - self.obs_mean) / self.obs_std)

    def forward(self, obs):
        return self.head(self.latent(obs))
