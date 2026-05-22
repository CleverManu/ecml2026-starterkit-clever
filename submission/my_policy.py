"""
Inference-side policy for the ECML 2026 Flatland Competition.

Implements the :class:`flatland.core.policy.Policy` interface so it can be loaded
by ``flatland-trajectory-generate-from-policy`` via the ``POLICY`` env var:

::

    ENV POLICY=my_orga.my_policy.MyPolicy

At construction time the policy loads the checkpoint that was shipped inside the
Docker image (``my_orga/checkpoint.pt``). If no checkpoint is found, the policy
falls back to a freshly initialized random network rather than crashing, so the
container is still runnable for debugging.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from flatland.core.policy import Policy

from submission.model import ActorCritic, NetConfig, load_checkpoint
from submission.my_observation_builder import get_obs_dim

CHECKPOINT_FILENAME = "checkpoint.pt"


def _default_checkpoint_path() -> str:
    """Path to checkpoint shipped alongside this module."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_FILENAME)


class MyPolicy(Policy):
    """PPO policy that produces actions from flattened tree observations."""

    def __init__(self, checkpoint_path: Optional[str] = None, deterministic: bool = True) -> None:
        super().__init__()
        self.deterministic = deterministic
        self.device = torch.device("cpu")  # Evaluation containers are CPU-only.

        path = checkpoint_path or _default_checkpoint_path()
        if os.path.isfile(path):
            self.model = load_checkpoint(path, map_location="cpu")
        else:
            # Fallback: random-init network. Lets the container start even if no
            # checkpoint was bundled (useful as a regression test).
            print(f"[MyPolicy] WARNING: checkpoint not found at {path}; using random weights.")
            self.model = ActorCritic(NetConfig(obs_dim=get_obs_dim()))
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------
    def act(self, observation, **kwargs) -> int:
        """Single-agent inference (delegated to by the default ``act_many``)."""
        if observation is None:
            return 0  # DO_NOTHING for agents with no observation yet.
        obs = np.asarray(observation, dtype=np.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0)  # batch of 1
            action, _, _ = self.model.act(obs_t, deterministic=self.deterministic)
        return int(action.item())

    def act_many(self, handles: List[int], observations: List, **kwargs) -> Dict[int, int]:
        """
        Batched inference: one forward pass for all agents at this timestep.

        Saves a lot of Python overhead vs the default per-agent loop.
        """
        valid_idx, batch = [], []
        for i, obs in enumerate(observations):
            if obs is None:
                continue
            valid_idx.append(i)
            batch.append(np.asarray(obs, dtype=np.float32))

        actions: Dict[int, int] = {h: 0 for h in handles}  # DO_NOTHING default
        if not batch:
            return actions

        obs_t = torch.from_numpy(np.stack(batch))
        with torch.no_grad():
            chosen, _, _ = self.model.act(obs_t, deterministic=self.deterministic)
        chosen = chosen.cpu().numpy()

        for j, i in enumerate(valid_idx):
            actions[handles[i]] = int(chosen[j])
        return actions
