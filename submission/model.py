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

    @staticmethod
    def _apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Set logits for invalid actions to -inf so they're never sampled.

        ``mask`` is a bool tensor of the same shape as ``logits`` (True = valid).
        We use ``-1e9`` rather than ``-inf`` to avoid NaN cascades if every
        action happens to be masked out -- the categorical will still degenerate
        but won't break grad computation.
        """
        return logits.masked_fill(~mask, -1e9)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None,
            deterministic: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions plus return log-probs and state-values (no grad)."""
        logits, value = self.forward(obs)
        if action_mask is not None:
            logits = self._apply_mask(logits, action_mask)
        if deterministic:
            action = logits.argmax(dim=-1)
            log_prob = F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor,
                 action_mask: Optional[torch.Tensor] = None,
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate log-probs / entropy / values for the PPO update."""
        logits, value = self.forward(obs)
        if action_mask is not None:
            logits = self._apply_mask(logits, action_mask)
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


# ---------------------------------------------------------------------------
# MAPPO: centralized critic.
# ---------------------------------------------------------------------------
# The critic is *only* used during training. At inference we deploy the
# decentralized policy alone (ActorCritic.policy_head + trunk), which is what
# the Docker image ships.
#
# Pooled global state layout (see training/global_state.py):
#   [mean over agents of per-agent obs] (obs_dim,)
#   [max over agents of per-agent obs]  (obs_dim,)
#   [env-level scalars]                 (N_ENV_SCALARS,)
# => total dim: 2 * obs_dim + N_ENV_SCALARS
#
# A separate value network rather than a head on the shared trunk: the trunk
# is sized for per-agent obs (~283 dims); the global state is much bigger
# (~575 dims). Sharing trunks would force compromises.

N_ENV_SCALARS: int = 6  # see global_state.compute_global_state


def get_global_state_dim(obs_dim: int) -> int:
    """Dim of the pooled global state (option a)."""
    return 2 * obs_dim + N_ENV_SCALARS


@dataclass
class CriticConfig:
    global_state_dim: int
    hidden: int = 256


class CentralizedCritic(nn.Module):
    """Standalone value network operating on a global state.

    Two-layer Tanh MLP, deliberately matching the policy's architecture style
    so the gradient scales stay similar. Outputs a single scalar value used as
    the value estimate for every agent's per-step advantage computation -- the
    "centralized value" of MAPPO.
    """

    def __init__(self, cfg: CriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(cfg.global_state_dim, cfg.hidden), gain=2.0 ** 0.5),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(cfg.hidden, cfg.hidden), gain=2.0 ** 0.5),
            nn.Tanh(),
        )
        self.value_head = _orthogonal_init(nn.Linear(cfg.hidden, 1), gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return scalar value per global-state row. Shape: (batch,)."""
        return self.value_head(self.trunk(global_state)).squeeze(-1)


def save_mappo_checkpoint(
    policy: ActorCritic,
    critic: Optional[CentralizedCritic],
    path: str,
    extra: Optional[dict] = None,
) -> None:
    """Persist both the policy and the centralized critic (if present).

    Compatible with :func:`load_checkpoint` for backward compatibility:
    callers that don't care about the critic will just load the policy half.
    """
    payload = {
        "state_dict": policy.state_dict(),
        "config": {
            "obs_dim": policy.cfg.obs_dim,
            "hidden": policy.cfg.hidden,
            "n_actions": policy.cfg.n_actions,
        },
    }
    if critic is not None:
        payload["critic_state_dict"] = critic.state_dict()
        payload["critic_config"] = {
            "global_state_dim": critic.cfg.global_state_dim,
            "hidden": critic.cfg.hidden,
        }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_mappo_checkpoint(
    path: str, map_location: str = "cpu"
) -> Tuple[ActorCritic, Optional[CentralizedCritic]]:
    """Load both policy + critic. Critic is None if absent from the file.

    Falls back to behaving like :func:`load_checkpoint` if the file was saved
    without a critic (e.g. by an older training run).
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = NetConfig(**payload["config"])
    policy = ActorCritic(cfg)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()

    critic = None
    if "critic_state_dict" in payload:
        c_cfg = CriticConfig(**payload["critic_config"])
        critic = CentralizedCritic(c_cfg)
        critic.load_state_dict(payload["critic_state_dict"])
        critic.eval()
    return policy, critic
