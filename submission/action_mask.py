"""
Action validity masking for Flatland.

Flatland's 5 discrete actions ``(DO_NOTHING, MOVE_LEFT, MOVE_FORWARD,
MOVE_RIGHT, STOP_MOVING)`` are not all meaningful in every cell:

* On straight track, the three "move" actions all do the same thing.
* At a switch, only some turns correspond to actual rail transitions.
* Off-map (agent waiting to depart), only DO_NOTHING and MOVE_FORWARD have an
  effect; the others are equivalent to one of those.

Without masking, the policy must learn these rules from the reward signal,
which wastes capacity. With masking, the policy categorical distribution
ignores impossible actions entirely.

Flatland doesn't expose a ``get_valid_actions(handle)`` helper, so we build the
mask from ``env.rail.get_transitions(row, col, direction)``, which returns a
4-tuple of booleans ``(north, east, south, west)`` indicating which absolute
directions can be exited from the cell.

The returned mask is a length-5 ``np.bool_`` array; ``True`` = valid.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

# Flatland action indices (mirrors ``RailEnvActions``).
_DO_NOTHING = 0
_MOVE_LEFT = 1
_MOVE_FORWARD = 2
_MOVE_RIGHT = 3
_STOP_MOVING = 4

# Relative direction offset for each move action.
# Flatland directions: 0=N, 1=E, 2=S, 3=W. Turning right adds +1 modulo 4.
_ACTION_OFFSETS = {
    _MOVE_LEFT: -1,
    _MOVE_FORWARD: 0,
    _MOVE_RIGHT: +1,
}


def get_action_mask(env, handle: int) -> np.ndarray:
    """
    Compute the (5,) valid-action mask for one agent in its current state.

    Always allows ``DO_NOTHING`` and ``STOP_MOVING`` (the env handles them in
    every state). For the three move actions, looks up the rail transitions at
    the agent's current cell+direction and disables actions whose target
    direction doesn't correspond to an actual track.

    Agents that aren't on the map yet (``WAITING`` / ``READY_TO_DEPART`` /
    ``MALFUNCTION_OFF_MAP``) have ``MOVE_FORWARD`` enabled (it triggers
    departure) and the two other turns disabled (no effect off-map).
    """
    mask = np.zeros(5, dtype=bool)
    # DO_NOTHING and STOP_MOVING are always legal as "no-op" choices.
    mask[_DO_NOTHING] = True
    mask[_STOP_MOVING] = True

    agent = env.agents[handle]

    # Off-map: only MOVE_FORWARD has departure semantics.
    if agent.position is None:
        mask[_MOVE_FORWARD] = True
        return mask

    # DONE agents: action is ignored anyway; keep everything True so we don't
    # accidentally produce an all-False mask that would crash the categorical.
    if int(agent.state) == 6:  # TrainState.DONE
        mask[:] = True
        return mask

    row, col = agent.position
    direction = int(agent.direction)
    transitions = env.rail.get_transitions(((row, col), direction))  # (N, E, S, W) bools

    for action, offset in _ACTION_OFFSETS.items():
        target_dir = (direction + offset) % 4
        if transitions[target_dir]:
            mask[action] = True

    # Failsafe: if somehow no move action is valid (shouldn't happen on a
    # well-formed rail), keep MOVE_FORWARD legal so the env can handle it
    # rather than have the policy distribution explode.
    if not mask[_MOVE_LEFT] and not mask[_MOVE_FORWARD] and not mask[_MOVE_RIGHT]:
        mask[_MOVE_FORWARD] = True

    return mask


def get_action_masks(env, handles: List[int]) -> Dict[int, np.ndarray]:
    """Convenience: compute masks for multiple agents at once."""
    return {h: get_action_mask(env, h) for h in handles}
