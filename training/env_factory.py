"""
Training environment factory.

Loads a competition scenario from disk, attaches our observation builder and the
official ECML 2026 reward shaping, and patches it with the random sampler from
PR #7 so that every ``env.reset()`` produces fresh lines and timetables on the
same fixed grid topology.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rewards import ECML2026Rewards

from submission.my_observation_builder import (
    MyObservationBuilder, MyObservationBuilderV2, MyObservationBuilderV3,
)
from training.sampling import sampling_env_generator


SAMPLING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampling")
DEFAULT_SCENARIO = os.path.join(SAMPLING_DIR, "level_0_scenario_1.pkl")


def make_training_env(
    scenario_path: str = DEFAULT_SCENARIO,
    line_length: int = 2,
    scene: Optional[str] = None,
    obs_version: str = "v1",
    n_agents_range: Optional[Tuple[int, int]] = None,
) -> RailEnv:
    """
    Build a training environment.

    Parameters
    ----------
    scenario_path
        Path to a Flatland scenario pickle. Defaults to ``level_0_scenario_1.pkl``
        from PR #7. The pickle defines the rail grid topology; lines and
        timetables are re-sampled at every reset.
    line_length
        Maximum waypoints per train. ``2`` means simple A->B, ``3`` means one
        intermediate stop. Use ``2`` for the baseline.
    scene
        Optional restriction to a subset of stations (``"scene_1"`` ... ``"scene_5"``).
        ``None`` is equivalent to ``"scene_5"`` (all stations).
    obs_version
        ``"v1"`` (default) -> ``MyObservationBuilder`` (252 features).
        ``"v2"`` -> ``MyObservationBuilderV2`` with 8 extra global features (260).
    n_agents_range
        Optional ``(low, high)`` inclusive range. If set, the env's agent count
        is re-randomized on every reset to a uniform draw from that range.
        Useful when competition eval covers a wide span of agent counts.
    """
    if obs_version == "v3":
        obs_builder = MyObservationBuilderV3()
    elif obs_version == "v2":
        obs_builder = MyObservationBuilderV2()
    else:
        obs_builder = MyObservationBuilder()
    env = RailEnvPersister.load_new(
        scenario_path,
        obs_builder=obs_builder,
        rewards=ECML2026Rewards(),
    )[0]
    env = sampling_env_generator(env, line_length=line_length, scene=scene)
    if n_agents_range is not None:
        _enable_varied_n_agents(env, n_agents_range)
    return env


def _enable_varied_n_agents(env: RailEnv, n_agents_range: Tuple[int, int]) -> None:
    """Patch ``env.reset`` so each call randomizes ``number_of_agents``.

    Flatland's line generator is called on every reset and respects
    ``env.number_of_agents``, so this is all it takes to vary the count.
    """
    lo, hi = n_agents_range
    if lo < 1 or hi < lo:
        raise ValueError(f"Invalid n_agents_range={n_agents_range}; need 1 <= lo <= hi")

    original_reset = env.reset

    def reset_with_random_n(*args, **kwargs):
        env.number_of_agents = int(np.random.randint(lo, hi + 1))
        return original_reset(*args, **kwargs)

    env.reset = reset_with_random_n
