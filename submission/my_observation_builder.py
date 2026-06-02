"""
Custom observation builder for the ECML 2026 Flatland Competition.

Wraps :class:`flatland.envs.observations.TreeObsForRailEnv` and returns a fixed-size,
flat, normalized ``numpy`` vector per agent. Doing this in the observation builder
itself (rather than inside the policy) keeps the observation contract identical at
training and evaluation time -- the policy just consumes ``np.float32`` vectors.

Design notes
------------
* ``max_depth = 2`` gives ``1 + 4 + 16 = 21`` nodes per tree and ``21 * 12 = 252``
  features per agent. Depth 2 is a good CPU-friendly compromise: enough lookahead
  to anticipate the next switch and its outcomes, small enough that rollouts are
  fast.
* The 12 features per node are split into three groups (data / distance / agent),
  each normalized differently:
    - "data" features (distances to events) -> clipped to ``[-1, 1]`` after dividing
      by a fixed observation radius.
    - "distance" features (min-distance-to-target) -> min/max normalized.
    - "agent" features (counts, speeds) -> clipped to ``[-1, 1]``.
  This matches the normalization in :mod:`flatland.ml.observations` and the original
  2020 starter-kit conventions.
* Missing branches (``-np.inf`` in the tree) are replaced with ``-np.inf`` placeholders
  *before* the clip step, so the network learns a "no information" signal in those
  positions instead of zeros that could be confused with valid observations.

The class is parameter-less so it can be instantiated by Flatland's policy runner
via the ``OBS_BUILDER`` environment variable (see ``Dockerfile``).
"""
from typing import Optional

import numpy as np

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.observations import Node, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

# Network input dim and tree depth are referenced by the policy network, so we
# expose them as module-level constants.
TREE_DEPTH: int = 2
PREDICTION_DEPTH: int = 30
OBSERVATION_RADIUS: int = 10

_NUM_FEATURES: int = 12
_NUM_BRANCHES: int = 4
_NUM_DATA: int = 6
_NUM_DISTANCE: int = 1
_NUM_AGENT_DATA: int = 5


def _group_len(tree_depth: int, num_features: int) -> int:
    """Total number of features for one group across the full tree."""
    k = num_features
    for _ in range(tree_depth):
        k = k * _NUM_BRANCHES + num_features
    return k


def get_obs_dim(tree_depth: int = TREE_DEPTH) -> int:
    """Length of the flat observation vector for the configured tree depth."""
    return _group_len(tree_depth, _NUM_FEATURES)


def _split_node(node: Node):
    """Split one Node into its three feature groups."""
    data = np.array(
        [
            node.dist_own_target_encountered,
            node.dist_other_target_encountered,
            node.dist_other_agent_encountered,
            node.dist_potential_conflict,
            node.dist_unusable_switch,
            node.dist_to_next_branch,
        ],
        dtype=np.float64,
    )
    distance = np.array([node.dist_min_to_target], dtype=np.float64)
    agent_data = np.array(
        [
            node.num_agents_same_direction,
            node.num_agents_opposite_direction,
            node.num_agents_malfunctioning,
            node.speed_min_fractional,
            node.num_agents_ready_to_depart,
        ],
        dtype=np.float64,
    )
    return data, distance, agent_data


def _split_subtree(node, current_depth: int, max_depth: int):
    """Recursive pre-order traversal yielding three concatenated feature arrays."""
    if node == -np.inf:
        # Pad missing branch with -inf sentinels for every remaining node.
        remaining = max_depth - current_depth
        num_remaining_nodes = (_NUM_BRANCHES ** (remaining + 1) - 1) // (_NUM_BRANCHES - 1)
        return (
            np.full(num_remaining_nodes * _NUM_DATA, -np.inf),
            np.full(num_remaining_nodes * _NUM_DISTANCE, -np.inf),
            np.full(num_remaining_nodes * _NUM_AGENT_DATA, -np.inf),
        )

    data, distance, agent_data = _split_node(node)
    if not node.childs:
        return data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sd, sdi, sa = _split_subtree(node.childs[direction], current_depth + 1, max_depth)
        data = np.concatenate((data, sd))
        distance = np.concatenate((distance, sdi))
        agent_data = np.concatenate((agent_data, sa))
    return data, distance, agent_data


def _split_tree(tree: Node, max_depth: int):
    """Top-level split; pre-order traversal with root first."""
    data, distance, agent_data = _split_node(tree)
    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sd, sdi, sa = _split_subtree(tree.childs[direction], 1, max_depth)
        data = np.concatenate((data, sd))
        distance = np.concatenate((distance, sdi))
        agent_data = np.concatenate((agent_data, sa))
    return data, distance, agent_data


def _max_lt(seq: np.ndarray, val: float) -> float:
    """Largest value in ``seq`` strictly less than ``val`` and >= 0; 0 if none."""
    mask = (seq < val) & (seq >= 0)
    return float(seq[mask].max()) if mask.any() else 0.0


def _min_gt(seq: np.ndarray, val: float) -> float:
    """Smallest value in ``seq`` >= ``val``; ``np.inf`` if none."""
    mask = seq >= val
    return float(seq[mask].min()) if mask.any() else np.inf


def _norm_clip(obs: np.ndarray, clip_min: float = -1.0, clip_max: float = 1.0,
               fixed_radius: float = 0.0, normalize_to_range: bool = False) -> np.ndarray:
    """Min/max-style normalization with clipping. Matches the 2020 starter kit."""
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1.0, _max_lt(obs, 1000.0)) + 1.0
    min_obs = 0.0
    if normalize_to_range:
        min_obs = _min_gt(obs, 0.0)
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(obs / max_obs, clip_min, clip_max)
    norm = abs(max_obs - min_obs)
    return np.clip((obs - min_obs) / norm, clip_min, clip_max)


def flatten_normalized(tree: Node, max_depth: int = TREE_DEPTH,
                       observation_radius: int = OBSERVATION_RADIUS) -> np.ndarray:
    """Convert a Tree obs Node into a flat normalized vector of length ``get_obs_dim``."""
    data, distance, agent_data = _split_tree(tree, max_depth)
    data = _norm_clip(data, fixed_radius=observation_radius)
    distance = _norm_clip(distance, normalize_to_range=True)
    agent_data = np.clip(agent_data, -1.0, 1.0)
    return np.concatenate((data, distance, agent_data)).astype(np.float32)


class MyObservationBuilder(TreeObsForRailEnv):
    """
    Tree observation, flattened and normalized into a fixed-size ``np.float32`` vector.

    Parameter-less so it can be loaded by Flatland's policy runner via the
    ``OBS_BUILDER`` env var.
    """

    def __init__(self) -> None:
        super().__init__(
            max_depth=TREE_DEPTH,
            predictor=ShortestPathPredictorForRailEnv(max_depth=PREDICTION_DEPTH),
        )

    def get(self, handle: Optional[AgentHandle] = 0) -> np.ndarray:
        tree = super().get(handle)
        if tree is None:
            return np.zeros(get_obs_dim(self.max_depth), dtype=np.float32)
        return flatten_normalized(tree, self.max_depth)


# ---------------------------------------------------------------------------
# V2: Tree observation + 8 global/per-agent features.
# ---------------------------------------------------------------------------
# Tree obs is purely local (current cell + a few branches). The policy benefits
# from a handful of global state features so it knows things like "I'm running
# late" or "most other agents have already arrived". These are cheap to compute
# from `self.env` after every step.

N_GLOBAL_FEATURES: int = 8


def get_obs_dim_v2(tree_depth: int = TREE_DEPTH) -> int:
    """Observation dim when global features are appended."""
    return get_obs_dim(tree_depth) + N_GLOBAL_FEATURES


class MyObservationBuilderV2(MyObservationBuilder):
    """
    Tree observation + 8 normalized global/per-agent state features.

    Features appended at the end of the flat vector:
        0. fraction of total episode steps elapsed
        1. fraction of agents that have arrived
        2. fraction of agents currently moving or stopped on map
        3. fraction of agents currently malfunctioning
        4. fraction of agents still waiting to depart
        5. (earliest_departure - elapsed) / max_steps  -- "how soon should I leave"
        6. (latest_arrival   - elapsed) / max_steps  -- "how much time do I have"
        7. malfunction time remaining for this agent / 50.0 (capped at 1.0)

    All values are clipped to ``[-1, 1]``. Use this builder by setting the
    Docker env var ``OBS_BUILDER=my_orga.my_observation_builder.MyObservationBuilderV2``.
    """

    def get(self, handle: Optional[AgentHandle] = 0) -> np.ndarray:
        tree_part = super().get(handle)
        extras = self._global_features(handle)
        return np.concatenate([tree_part, extras]).astype(np.float32)

    def _global_features(self, handle: int) -> np.ndarray:
        env = self.env
        agents = env.agents
        n_agents = max(len(agents), 1)
        elapsed = float(env._elapsed_steps)
        max_steps = max(float(env._max_episode_steps), 1.0)

        # Aggregate state counts.
        n_arrived = sum(1 for a in agents if a.state == 6)         # DONE
        n_on_map = sum(1 for a in agents if a.state in (3, 4))     # MOVING, STOPPED
        n_malfunc = sum(1 for a in agents if a.state == 5)         # MALFUNCTION
        n_waiting = sum(1 for a in agents if a.state in (0, 1, 2)) # WAITING, READY_TO_DEPART, MALF_OFF_MAP

        # Per-agent timing.
        agent = agents[handle]
        ed = agent.earliest_departure if agent.earliest_departure is not None else 0
        la = agent.latest_arrival if agent.latest_arrival is not None else int(max_steps)
        time_to_dep = (ed - elapsed) / max_steps
        time_to_arr = (la - elapsed) / max_steps

        mf_remaining = 0.0
        try:
            mf_remaining = float(agent.malfunction_handler.malfunction_down_counter) / 50.0
        except AttributeError:
            pass

        feats = np.array([
            elapsed / max_steps,
            n_arrived / n_agents,
            n_on_map / n_agents,
            n_malfunc / n_agents,
            n_waiting / n_agents,
            time_to_dep,
            time_to_arr,
            mf_remaining,
        ], dtype=np.float32)
        return np.clip(feats, -1.0, 1.0)


# ---------------------------------------------------------------------------
# V3: V2 + 5 action-mask features at the end.
# ---------------------------------------------------------------------------
# The point of having the mask *in* the observation (rather than passing it
# alongside) is that Flatland's :class:`PolicyRunner` doesn't give the policy
# access to the env at inference time -- but it does invoke the observation
# builder, which has ``self.env``. So embedding the mask in the obs vector
# is the only way masking can work consistently at both train and eval.

N_MASK_FEATURES: int = 5


def get_obs_dim_v3(tree_depth: int = TREE_DEPTH) -> int:
    return get_obs_dim_v2(tree_depth) + N_MASK_FEATURES


class MyObservationBuilderV3(MyObservationBuilderV2):
    """
    Tree obs (V1) + 8 global features (V2) + 5 action-mask features.

    Total obs dim: ``252 + 8 + 5 = 265``.

    The mask is computed on the same agent state the policy will see and is
    appended as the **final 5 features**, in the order
    ``(DO_NOTHING, MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT, STOP_MOVING)``.
    Values are ``1.0`` for valid, ``0.0`` for invalid.

    The trainer extracts these features back into a bool mask and applies
    them in the categorical's logits, both during action selection and during
    the PPO update. Inference (:class:`my_orga.my_policy.MyPolicy`) does the
    same extraction, so masking is consistent across train and eval without
    needing env access at inference time.
    """

    def get(self, handle: Optional[AgentHandle] = 0) -> np.ndarray:
        # Late import to avoid a cycle (action_mask -> obs_builder is fine,
        # but obs_builder -> action_mask only when v3 is used).
        from my_orga.action_mask import get_action_mask

        v2_part = super().get(handle)
        mask = get_action_mask(self.env, handle).astype(np.float32)
        return np.concatenate([v2_part, mask]).astype(np.float32)


def extract_mask_from_obs(obs: np.ndarray) -> Optional[np.ndarray]:
    """Pull a (..., 5) bool mask out of a V3 or V4 obs batch, or return None.

    Both V3 and V4 keep the 5 mask features at positions ``[-5:]`` so this
    helper works on either. Returns ``None`` for V1/V2 obs sizes.
    """
    if obs.shape[-1] not in (get_obs_dim_v3(), get_obs_dim_v4()):
        return None
    return (obs[..., -N_MASK_FEATURES:] > 0.5)


# ---------------------------------------------------------------------------
# V4: V3 + nearest-K other-agent features.
# ---------------------------------------------------------------------------
# V4 inserts other-agent features *between* the V2 global features and the V3
# mask features. Layout:
#   [tree obs (252)] [v2 globals (8)] [nearest-K agent features (K*6)] [mask (5)]
#
# Keeping mask at the end means `extract_mask_from_obs` works unchanged on
# both V3 and V4 obs vectors. Adding features in the middle means a V4
# checkpoint can't be loaded as V3 (different obs_dim), which is enforced by
# the trainer's dim check.

# Number of nearest agents to encode. Competition uses 8-532 agents per env;
# K=3 captures the closest coordination partners without exploding obs size.
N_NEAREST_AGENTS: int = 3

# Per-agent feature layout. Keeping each feature in [-1, 1] for clean
# normalization with the rest of the obs:
#   0: dx (delta row), normalized by obs radius
#   1: dy (delta col), normalized by obs radius
#   2: on_my_path -- 1.0 if this agent is on my planned shortest path within
#                    the next OBS_RADIUS cells, else 0.0
#   3: relative direction (-1: opposite, 0: perpendicular, +1: same)
#   4: euclidean distance, normalized by obs radius (clipped to 1.0)
#   5: malfunction flag (1.0 if other agent is malfunctioning)
N_AGENT_FEATURES: int = 6
OBS_RADIUS: float = 30.0  # cells; matches typical tree-obs reach at depth 2


def get_obs_dim_v4(tree_depth: int = TREE_DEPTH) -> int:
    """V4 dim: V2 (260) + nearest-K agents (K*6) + mask (5)."""
    return (get_obs_dim_v2(tree_depth)
            + N_NEAREST_AGENTS * N_AGENT_FEATURES
            + N_MASK_FEATURES)


class MyObservationBuilderV4(MyObservationBuilderV2):
    """
    V2 base (tree + globals) + nearest-K other-agent features + mask.

    Total obs dim with defaults: ``260 + 3*6 + 5 = 283``.

    The K=3 nearest agents (by Euclidean distance) get a 6-feature block each:
    relative position, on-my-path flag, relative direction, normalized distance,
    and malfunction flag. Agents not on the map yet (no position) and DONE
    agents are excluded from the candidate set; if fewer than K other agents
    are eligible, slots are zero-padded.

    The mask features are appended *last* (positions ``-5:``) so the existing
    :func:`extract_mask_from_obs` and the V3/V4 model branches in
    :mod:`my_policy` and :mod:`rollout` work uniformly across V3 and V4.

    Other-agent features use Euclidean ordering for speed. A future
    improvement would be path-aware ordering ("nearest agent along my
    shortest path"), which is more semantically right but expensive to
    compute every step.
    """

    def get(self, handle: Optional[AgentHandle] = 0) -> np.ndarray:
        from my_orga.action_mask import get_action_mask

        v2_part = super().get(handle)  # (260,)
        agent_part = self._nearest_agent_features(handle)  # (K*6,)
        mask = get_action_mask(self.env, handle).astype(np.float32)  # (5,)
        return np.concatenate([v2_part, agent_part, mask]).astype(np.float32)

    # ------------------------------------------------------------------
    def _nearest_agent_features(self, handle: int) -> np.ndarray:
        """Build the K*6 block of nearest-other-agent features.

        Returns a zero block if the agent has no position yet (waiting to
        depart or already done) -- the network has no spatial context for an
        off-map agent, so other-agent features wouldn't be meaningful.
        """
        out = np.zeros(N_NEAREST_AGENTS * N_AGENT_FEATURES, dtype=np.float32)
        me = self.env.agents[handle]
        if me.position is None or int(me.state) == 6:
            return out

        my_row, my_col = me.position
        my_dir = int(me.direction)

        # Collect eligible other agents (on-map, not done, not me).
        candidates = []
        for other in self.env.agents:
            if other.handle == handle:
                continue
            if other.position is None or int(other.state) == 6:
                continue
            dr = other.position[0] - my_row
            dc = other.position[1] - my_col
            dist_sq = dr * dr + dc * dc
            candidates.append((dist_sq, other, dr, dc))

        if not candidates:
            return out

        candidates.sort(key=lambda x: x[0])
        # On-my-path lookup: rely on Flatland's distance map. An "on path"
        # check via a precomputed shortest-path would be cleaner but costly;
        # the proxy we use is: is the other agent closer than the obs radius
        # AND roughly along my heading direction.
        for slot in range(min(N_NEAREST_AGENTS, len(candidates))):
            _, other, dr, dc = candidates[slot]
            dist = float(np.sqrt(dr * dr + dc * dc))
            other_dir = int(other.direction) if other.direction is not None else my_dir

            # Feature 2: on_my_path (proxy). The agent is "in front of me" if
            # the relative position is along the direction I'm facing.
            #   N=0 -> dr negative
            #   E=1 -> dc positive
            #   S=2 -> dr positive
            #   W=3 -> dc negative
            on_my_path = 0.0
            if my_dir == 0 and dr < 0:
                on_my_path = 1.0
            elif my_dir == 1 and dc > 0:
                on_my_path = 1.0
            elif my_dir == 2 and dr > 0:
                on_my_path = 1.0
            elif my_dir == 3 and dc < 0:
                on_my_path = 1.0

            # Feature 3: relative direction
            #   same (diff=0)        -> +1.0
            #   opposite (diff=2)    -> -1.0
            #   perpendicular (1,3)  ->  0.0
            ddir = (other_dir - my_dir) % 4
            if ddir == 0:
                rel_dir = 1.0
            elif ddir == 2:
                rel_dir = -1.0
            else:
                rel_dir = 0.0

            # Malfunction flag
            mf = 1.0 if getattr(other, "malfunction_handler", None) is not None \
                and other.malfunction_handler.malfunction_down_counter > 0 else 0.0

            base = slot * N_AGENT_FEATURES
            out[base + 0] = np.clip(dr / OBS_RADIUS, -1.0, 1.0)
            out[base + 1] = np.clip(dc / OBS_RADIUS, -1.0, 1.0)
            out[base + 2] = on_my_path
            out[base + 3] = rel_dir
            out[base + 4] = np.clip(dist / OBS_RADIUS, 0.0, 1.0)
            out[base + 5] = mf
        return out
