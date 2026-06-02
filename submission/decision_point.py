"""
Decision-point logic for frame-skipping during training and inference.

The 2020 NeurIPS Flatland winning team noted: "Since the majority of cells in
Flatland are non-decision cells, in these cells we hard code a MOVE_FORWARD
action or a STOP action, which is determined solely by the occupancy of the
single reachable cell." This module implements that recipe.

For each agent at each step, we classify the state as either:

* **decision point**: the policy network is consulted; its action is used
* **non-decision**: a hard-coded default action is used; no training sample
  is recorded (frame skipping)

The collapse of episode length from "every step" to "only decisions" is
significant -- empirically 5-10x for typical Flatland envs. This makes credit
assignment over decisions much sharper and lets the policy specialize on the
states that actually matter.

What counts as a "decision point"
---------------------------------
Option (c) from our design discussion -- switches + nearby-agent events.

Concretely:

1. **Switch**: the agent's current cell has more than one valid exit direction
   given its facing direction (left/forward/right diverge). This is the only
   place ``MOVE_LEFT`` and ``MOVE_RIGHT`` semantically differ from each other
   and from ``MOVE_FORWARD``.

2. **Nearby agent**: another agent on the map within ``DECISION_RADIUS`` cells.
   This captures the moments when stopping vs proceeding matters for collision
   avoidance, which is the bulk of Flatland's hard cases.

3. **Off-map**: agents not yet on the map (``WAITING`` / ``READY_TO_DEPART``)
   are decision points: choosing when to depart is itself a decision the policy
   should learn.

Notes:

* Malfunctions are captured *partially* via (2): a malfunctioning agent within
  ``DECISION_RADIUS`` triggers a decision point. The exact malfunction-start
  tick is not separately flagged; if this turns out to matter, escalate to
  option (d) (explicit malfunction events) by extending ``is_decision_point``.

Default action for non-decision cells
-------------------------------------
Option (b) from our design discussion: ``MOVE_FORWARD`` if the next cell is
unoccupied, ``STOP_MOVING`` otherwise. No state to maintain, no replanning,
no failure modes.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from flatland.envs.fast_methods import fast_count_nonzero


DECISION_RADIUS: int = 3  # cells (Manhattan); matches the K we chose


def is_decision_point(env, handle: int) -> bool:
    """Return True iff this agent's current state is a decision point.

    See module docstring for the exact criteria.
    """
    agent = env.agents[handle]

    # Off-map agents (waiting / ready-to-depart / malfunction-off-map) are
    # always decision points: the policy chooses when to enter the map.
    if agent.position is None:
        return True

    # DONE: not really a decision; agent is gone. Return False so it gets
    # the default action (which the env ignores anyway).
    if int(agent.state) == 6:
        return False

    row, col = agent.position
    direction = int(agent.direction)

    # 1) Switch detection: more than one valid exit from this cell+direction.
    transitions = env.rail.get_transitions(((row, col), direction))
    if fast_count_nonzero(transitions) > 1:
        return True

    # 2) Nearby-agent detection: any other on-map, non-done agent within
    # DECISION_RADIUS cells (Manhattan).
    for other in env.agents:
        if other.handle == handle:
            continue
        if other.position is None or int(other.state) == 6:
            continue
        dr = abs(other.position[0] - row)
        dc = abs(other.position[1] - col)
        if dr + dc <= DECISION_RADIUS:
            return True

    return False


def default_action(env, handle: int) -> int:
    """Hard-coded action for non-decision cells.

    ``MOVE_FORWARD`` (action 2) if the cell directly ahead is unoccupied and
    the transition exists, ``STOP_MOVING`` (action 4) otherwise.

    For off-map agents, returns ``DO_NOTHING`` (action 0) -- but note that
    off-map states are *decision points* by definition, so this branch is
    rarely hit. It exists for completeness.
    """
    agent = env.agents[handle]
    if agent.position is None:
        return 0  # DO_NOTHING; off-map default (rarely reached)

    row, col = agent.position
    direction = int(agent.direction)

    # Find the next cell along agent's heading. Direction enum:
    # 0=N (-1, 0), 1=E (0, +1), 2=S (+1, 0), 3=W (0, -1).
    dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
    next_pos = (row + dr, col + dc)

    # Bounds check (shouldn't happen if rail topology is correct, but cheap to verify).
    if not (0 <= next_pos[0] < env.height and 0 <= next_pos[1] < env.width):
        return 4  # STOP_MOVING

    # Check the forward transition is valid in the rail.
    transitions = env.rail.get_transitions(((row, col), direction))
    if not transitions[direction]:
        # No valid forward transition; safest default is stop.
        return 4

    # Check occupancy: is any other on-map agent currently in the next cell?
    for other in env.agents:
        if other.handle == handle:
            continue
        if other.position == next_pos and int(other.state) != 6:
            return 4  # STOP_MOVING: avoid head-on or trailing collision

    return 2  # MOVE_FORWARD
