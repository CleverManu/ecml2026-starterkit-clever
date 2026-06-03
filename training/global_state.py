"""
Global state computation for MAPPO's centralized critic.

The centralized critic is *only* used at training time; it sees information
that no single agent has access to, which lets it produce cleaner value
targets and therefore cleaner advantage estimates. At inference (Docker
submission), only the decentralized policy runs, so global state is never
computed there.

Pooled summary (option a from our design discussion)
-----------------------------------------------------
For each environment step:

  [mean over all agents of per-agent obs] (obs_dim features)
  [max  over all agents of per-agent obs] (obs_dim features)
  [env-level scalars]                     (N_ENV_SCALARS features)

The mean+max pooling is permutation-invariant (which the global state should
be -- "agent 3's obs" carries no inherent meaning when agents are
interchangeable parameter-sharing trains) and a fixed dimension regardless of
agent count (critical: the competition spans 8-532 agents per env, so any
per-agent concatenation explodes).

We use simple Python loops + numpy rather than tensor ops because this gets
called once per env step in the rollout collector, and the cost is dominated
by the obs builder anyway. Future optimization: precompute the obs stack
once and reuse it for both the policy and the global state.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# Number of env-level scalars in the global state. If you change this, update
# ``submission.model.N_ENV_SCALARS`` to match (it controls the critic input dim).
N_ENV_SCALARS: int = 6


def compute_global_state(env, obs_dict: Dict[int, np.ndarray]) -> np.ndarray:
    """Build a fixed-size global summary from the env + current obs dict.

    Layout:
        positions [0 : obs_dim)        mean over agents of per-agent obs
        positions [obs_dim : 2*obs_dim) max over agents of per-agent obs
        positions [2*obs_dim : ]        env-level scalars (see below)

    Env-level scalars (in order):
        - elapsed / max_episode_steps    (in [0, 1])
        - n_arrived / n_total            (in [0, 1])
        - n_on_map / n_total             (in [0, 1])
        - n_off_map / n_total            (in [0, 1])
        - n_malfunctioning / n_total     (in [0, 1])
        - sum of malfunction-down-counters, normalized by n_total*20
          (rough indicator of accumulated malfunction time)

    Returns a 1-D ``np.float32`` of length ``2 * obs_dim + N_ENV_SCALARS``.
    If no agents have observations yet (right after env construction, before
    any reset), returns zeros.
    """
    # Stack per-agent obs (those that exist). Off-map agents may have None.
    valid_obs = [np.asarray(o, dtype=np.float32)
                 for o in obs_dict.values() if o is not None]
    if not valid_obs:
        # No agents observable yet (env just constructed, no reset called).
        # Return zeros; the trainer should be aware not to hit this case
        # because all reset paths go through env.reset() which populates obs.
        return np.zeros(N_ENV_SCALARS, dtype=np.float32)  # caller handles dim

    stacked = np.stack(valid_obs)  # (n_valid_agents, obs_dim)
    obs_mean = stacked.mean(axis=0).astype(np.float32)
    obs_max = stacked.max(axis=0).astype(np.float32)

    # Env-level scalars.
    n_total = len(env.agents)
    n_arrived = 0
    n_on_map = 0
    n_off_map = 0
    n_malfunctioning = 0
    sum_malfunction_remaining = 0
    for a in env.agents:
        state = int(a.state)
        if state == 6:
            n_arrived += 1
        if a.position is not None:
            n_on_map += 1
        else:
            n_off_map += 1
        if (getattr(a, "malfunction_handler", None) is not None
                and a.malfunction_handler.malfunction_down_counter > 0):
            n_malfunctioning += 1
            sum_malfunction_remaining += a.malfunction_handler.malfunction_down_counter

    elapsed = float(getattr(env, "_elapsed_steps", 0))
    max_steps = float(getattr(env, "_max_episode_steps", 1) or 1)

    scalars = np.array([
        np.clip(elapsed / max_steps, 0.0, 1.0),
        n_arrived / max(n_total, 1),
        n_on_map / max(n_total, 1),
        n_off_map / max(n_total, 1),
        n_malfunctioning / max(n_total, 1),
        np.clip(sum_malfunction_remaining / (max(n_total, 1) * 20.0), 0.0, 1.0),
    ], dtype=np.float32)

    return np.concatenate([obs_mean, obs_max, scalars]).astype(np.float32)
