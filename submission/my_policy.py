"""
Inference-side policy for the ECML 2026 Flatland Competition.

Implements the :class:`flatland.core.policy.Policy` interface so it can be loaded
by ``flatland-trajectory-generate-from-policy`` via the ``POLICY`` env var:

::

    ENV POLICY=submission.my_policy.MyPolicy

At construction time the policy loads the checkpoint that was shipped inside the
Docker image (``submission/checkpoint.pt``). If no checkpoint is found, the policy
falls back to a freshly initialized random network rather than crashing, so the
container is still runnable for debugging.

Action masking
--------------
If the model was trained with the V3 observation builder
(:class:`submission.my_observation_builder.MyObservationBuilderV3`), every obs
already has a 5-bool mask appended at the end. We extract it here and pass it
to the model. The V1/V2 obs builders produce obs without an embedded mask, so
no masking is applied in those cases.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from flatland.core.policy import Policy

from submission.model import ActorCritic, NetConfig, load_checkpoint
from submission.my_observation_builder import extract_mask_from_obs, get_obs_dim

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
        mask = extract_mask_from_obs(obs)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            mask_t = torch.from_numpy(mask).unsqueeze(0) if mask is not None else None
            action, _, _ = self.model.act(obs_t, action_mask=mask_t,
                                          deterministic=self.deterministic)
        return int(action.item())

    def act_many(self, handles: List[int], observations: List, **kwargs) -> Dict[int, int]:
        """Batched inference: one forward pass for all agents at this timestep.

        Extracts the action mask from V3 observations automatically.
        """
        valid_idx, batch_obs = [], []
        for i, obs in enumerate(observations):
            if obs is None:
                continue
            valid_idx.append(i)
            batch_obs.append(np.asarray(obs, dtype=np.float32))

        actions: Dict[int, int] = {h: 0 for h in handles}  # DO_NOTHING default
        if not batch_obs:
            return actions

        obs_arr = np.stack(batch_obs)
        obs_t = torch.from_numpy(obs_arr)
        mask_t = None
        mask_arr = extract_mask_from_obs(obs_arr)
        if mask_arr is not None:
            mask_t = torch.from_numpy(mask_arr)
        with torch.no_grad():
            chosen, _, _ = self.model.act(
                obs_t, action_mask=mask_t, deterministic=self.deterministic,
            )
        chosen = chosen.cpu().numpy()

        for j, i in enumerate(valid_idx):
            actions[handles[i]] = int(chosen[j])
        return actions
