"""
Training environment factory.

Loads a competition scenario from disk, attaches our observation builder and the
official ECML 2026 reward shaping, and patches it with the random sampler from
PR #7 so that every ``env.reset()`` produces fresh lines and timetables on the
same fixed grid topology.
"""
from __future__ import annotations

import os
from typing import Optional

from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rewards import ECML2026Rewards

from submission.my_observation_builder import MyObservationBuilder
from training.sampling import sampling_env_generator


SAMPLING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampling")
DEFAULT_SCENARIO = os.path.join(SAMPLING_DIR, "level_0_scenario_1.pkl")


def make_training_env(
    scenario_path: str = DEFAULT_SCENARIO,
    line_length: int = 2,
    scene: Optional[str] = None,
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
    """
    obs_builder = MyObservationBuilder()
    env = RailEnvPersister.load_new(
        scenario_path,
        obs_builder=obs_builder,
        rewards=ECML2026Rewards(),
    )[0]
    env = sampling_env_generator(env, line_length=line_length, scene=scene)
    return env
