"""
PPO actor-critic network with parameter sharing across agents.

All agents are homogeneous trains, so a single network is shared by every agent
in the environment. Each agent independently receives its own flattened tree
observation and produces its own action; the policy just sees a batch of agents.

The network is intentionally small (two-layer 256-unit MLP, Tanh activations) to
keep CPU training tractable on a laptop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


N_ACTIONS: int = 5  # DO_NOTHING, MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT, STOP_MOVING


@dataclass
class NetConfig:
    obs_dim: int
    hidden: int = 256
    n_actions: int = N_ACTIONS


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Orthogonal init with zero bias -- standard PPO practice."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    return module


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic with separate policy and value heads."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(cfg.obs_dim, cfg.hidden), gain=2.0 ** 0.5),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(cfg.hidden, cfg.hidden), gain=2.0 ** 0.5),
            nn.Tanh(),
        )
        # Policy head: small gain so initial actions are near-uniform.
        self.policy_head = _orthogonal_init(nn.Linear(cfg.hidden, cfg.n_actions), gain=0.01)
        # Value head: unit gain.
        self.value_head = _orthogonal_init(nn.Linear(cfg.hidden, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions plus return log-probs and state-values (no grad)."""
        logits, value = self.forward(obs)
        if deterministic:
            action = logits.argmax(dim=-1)
            log_prob = F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate log-probs / entropy / values for the PPO update."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value


def save_checkpoint(model: ActorCritic, path: str, extra: Optional[dict] = None) -> None:
    """Persist model + config so :func:`load_checkpoint` can rebuild a working network."""
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "obs_dim": model.cfg.obs_dim,
            "hidden": model.cfg.hidden,
            "n_actions": model.cfg.n_actions,
        },
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> ActorCritic:
    """Rebuild an :class:`ActorCritic` from a file saved by :func:`save_checkpoint`."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = NetConfig(**payload["config"])
    model = ActorCritic(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
