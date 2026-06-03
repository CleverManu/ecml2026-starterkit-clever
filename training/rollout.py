"""
Rollout collection for parameter-sharing multi-agent PPO.

In Flatland, multiple agents step in lockstep but each agent makes its own
decisions from its own observation. We treat every (agent, timestep) pair where
``action_required`` is True as an independent training sample. All agents share
the same policy network, so a single rollout buffer collects samples from
all agents into one tensor for the PPO update.

Bootstrap values are taken at the moment an agent's transition is dropped --
either because the agent finished (``done=True``) or the rollout window ended
mid-episode. This is essentially independent GAE per agent.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from flatland.envs.rail_env import RailEnv

from submission.action_mask import get_action_mask
from submission.model import ActorCritic, CentralizedCritic
from submission.my_observation_builder import extract_mask_from_obs
from training.global_state import compute_global_state


@dataclass
class Batch:
    """Flat tensors ready for PPO update. All same length along dim 0."""
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: Optional[torch.Tensor] = None  # (N, 5) bool; None = no masking
    # MAPPO addition: per-sample global state. None for vanilla PPO.
    # Shape: (N, global_state_dim). The same global state is replicated
    # across each agent that produced a decision at that step.
    global_states: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return self.obs.shape[0]


@dataclass
class _PerAgentTrace:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    # MAPPO: snapshot of global state at the moment this action was taken.
    global_states: List[np.ndarray] = field(default_factory=list)


def _compute_gae(rewards: List[float], values: List[float], last_value: float,
                 gamma: float, lam: float) -> (np.ndarray, np.ndarray):
    """Generalized Advantage Estimation for one agent's trajectory."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


@dataclass
class RolloutCollector:
    """Collect a fixed number of env steps and return a flat training batch."""
    env: RailEnv
    model: ActorCritic
    rollout_steps: int
    # If set, use this centralized critic for value estimates instead of the
    # policy's value head. Enables MAPPO training -- the critic sees the global
    # state and produces much cleaner value targets in dense MARL settings.
    critic: Optional["CentralizedCritic"] = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 100.0  # ECML2026Rewards has very large terminal penalties;
                                 # scaling keeps the value head's targets reasonable.
    progress_reward_coef: float = 0.0  # If > 0, give a small per-step reward for
                                       # reducing min-distance to target. Helps with the
                                       # sparse-reward problem early in training.
    arrival_bonus: float = 0.0   # If > 0, add this raw-units bonus to an agent's
                                 # reward on the step it transitions to DONE.
    use_action_mask: bool = False  # Compute and apply valid-action masks each step.
    decision_points_only: bool = False  # If True, only consult policy at decision
                                        # points (switches + nearby-agent cells);
                                        # use hard-coded default action elsewhere.
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # Stats from the most recent collection
    episodes_completed: int = 0
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    success_rates: List[float] = field(default_factory=list)

    def collect(self) -> Batch:
        traces: Dict[int, _PerAgentTrace] = defaultdict(_PerAgentTrace)
        finished_advantages, finished_returns = [], []
        finished_obs, finished_actions, finished_log_probs, finished_values = [], [], [], []
        finished_masks: List[np.ndarray] = []
        finished_global_states: List[np.ndarray] = []  # MAPPO; empty when no critic

        self.episode_returns = []
        self.episode_lengths = []
        self.success_rates = []

        obs_dict, info = self.env.reset()
        # Track previous distance-to-target per agent for progress shaping.
        prev_distance: Dict[int, float] = {}
        if self.progress_reward_coef > 0:
            prev_distance = self._distances_to_target()
        # Track which agents have arrived (so we only credit the bonus on transition).
        prev_done: Dict[int, bool] = {a.handle: (a.state == 6) for a in self.env.agents}
        ep_return = 0.0
        ep_length = 0
        steps_done = 0

        while steps_done < self.rollout_steps:
            # Build batch only for agents that need a fresh action this step.
            handles = self.env.get_agent_handles()
            need_action = [
                h for h in handles
                if obs_dict.get(h) is not None and info["action_required"][h]
            ]

            # Decision-point gate: when enabled, the policy is only consulted
            # at decision points. Non-decision agents get a hard-coded default
            # action (typically MOVE_FORWARD or STOP) and no training sample
            # is recorded for them. This collapses effective episode length
            # by ~5-10x in typical Flatland envs and dramatically sharpens
            # credit assignment over the decisions that matter.
            if self.decision_points_only:
                from submission.decision_point import is_decision_point, default_action
                need = [h for h in need_action if is_decision_point(self.env, h)]
                hardcoded = [h for h in need_action if h not in need]
            else:
                need = need_action
                hardcoded = []

            actions_dict: Dict[int, int] = {h: 0 for h in handles}
            # Fill hard-coded actions for non-decision agents.
            if hardcoded:
                from submission.decision_point import default_action
                for h in hardcoded:
                    actions_dict[h] = default_action(self.env, h)

            if need:
                batch_obs = np.stack([np.asarray(obs_dict[h], dtype=np.float32) for h in need])
                obs_t = torch.from_numpy(batch_obs).to(self.device)

                # Action masks: if obs is V3/V4 the mask is already embedded in
                # the last 5 features; otherwise compute it from env.
                mask_t = None
                batch_masks = None
                if self.use_action_mask:
                    embedded = extract_mask_from_obs(batch_obs)
                    if embedded is not None:
                        batch_masks = embedded
                    else:
                        batch_masks = np.stack([get_action_mask(self.env, h) for h in need])
                    mask_t = torch.from_numpy(batch_masks).to(self.device)

                with torch.no_grad():
                    logits, values_t = self.model(obs_t)
                    if mask_t is not None:
                        logits = self.model._apply_mask(logits, mask_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    sampled = dist.sample()
                    log_probs = dist.log_prob(sampled)

                # MAPPO: replace per-agent policy value with the centralized
                # critic's value of the *global state*. Same scalar replicated
                # to each deciding agent because they share the world state.
                step_global_state = None
                if self.critic is not None:
                    step_global_state = compute_global_state(self.env, obs_dict)
                    gs_t = torch.from_numpy(step_global_state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        cv = self.critic(gs_t)  # shape (1,)
                    central_value = float(cv.item())
                    values_t = torch.full_like(values_t, central_value)

                sampled_np = sampled.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
                values_np = values_t.cpu().numpy()

                for j, h in enumerate(need):
                    actions_dict[h] = int(sampled_np[j])
                    tr = traces[h]
                    tr.obs.append(batch_obs[j])
                    tr.actions.append(int(sampled_np[j]))
                    tr.log_probs.append(float(log_probs_np[j]))
                    tr.values.append(float(values_np[j]))
                    if batch_masks is not None:
                        tr.masks.append(batch_masks[j])
                    if step_global_state is not None:
                        tr.global_states.append(step_global_state)

            next_obs, rewards, dones, next_info = self.env.step(actions_dict)
            steps_done += 1
            ep_length += 1

            # Credit rewards to the most recent decision step of each agent.
            # Scale by ``reward_scale`` so value targets stay numerically reasonable;
            # the policy gradient is invariant to scale after advantage normalization.
            inv_scale = 1.0 / self.reward_scale

            # Optional progress shaping: reward each agent for shrinking its
            # min-distance to target since the previous step. Magnitude is tiny
            # vs terminal penalties so it doesn't dominate the true objective.
            shaped: Dict[int, float] = {}
            if self.progress_reward_coef > 0:
                current_distance = self._distances_to_target()
                for h in handles:
                    prev = prev_distance.get(h, float("inf"))
                    cur = current_distance.get(h, float("inf"))
                    if prev != float("inf") and cur != float("inf"):
                        shaped[h] = self.progress_reward_coef * (prev - cur)
                prev_distance = current_distance

            # Optional per-arrival bonus: detect agents that just transitioned
            # to DONE this step. Credited in raw units (scaled by inv_scale below).
            if self.arrival_bonus > 0:
                for a in self.env.agents:
                    h = a.handle
                    is_done = (a.state == 6)
                    if is_done and not prev_done.get(h, False):
                        shaped[h] = shaped.get(h, 0.0) + self.arrival_bonus
                    prev_done[h] = is_done

            for h in handles:
                r = float(rewards.get(h, 0.0))
                ep_return += r  # logged in raw units for human readability
                r_scaled = (r + shaped.get(h, 0.0)) * inv_scale
                if traces[h].rewards.__len__() < len(traces[h].actions):
                    traces[h].rewards.append(r_scaled)
                elif traces[h].actions:
                    traces[h].rewards[-1] += r_scaled

            done_all = dones["__all__"]
            obs_dict, info = next_obs, next_info

            if done_all:
                # Arrival = agent reached its target (TrainState.DONE = 6),
                # not just the episode-end "done" flag.
                arrived = sum(int(a.state == 6) for a in self.env.agents)
                self.success_rates.append(arrived / max(len(handles), 1))
                self.episode_returns.append(ep_return)
                self.episode_lengths.append(ep_length)
                self.episodes_completed += 1
                ep_return = 0.0
                ep_length = 0

                # Flush all per-agent traces: terminal => bootstrap with 0.
                for h, tr in traces.items():
                    if not tr.actions:
                        continue
                    self._flush(
                        tr, last_value=0.0,
                        bufs=(finished_obs, finished_actions, finished_log_probs,
                              finished_values, finished_advantages, finished_returns,
                              finished_masks, finished_global_states),
                    )
                traces = defaultdict(_PerAgentTrace)
                if steps_done < self.rollout_steps:
                    obs_dict, info = self.env.reset()
                    if self.progress_reward_coef > 0:
                        prev_distance = self._distances_to_target()
                    prev_done = {a.handle: (a.state == 6) for a in self.env.agents}

        # End of rollout window: bootstrap unfinished traces from current value estimate.
        # With MAPPO, the bootstrap value comes from the centralized critic on the
        # current global state -- one scalar shared by every unfinished agent. Without
        # the critic, we bootstrap per-agent from the policy's value head as before.
        bootstrap_central = None
        if self.critic is not None and traces:
            gs_now = compute_global_state(self.env, obs_dict)
            with torch.no_grad():
                bootstrap_central = float(self.critic(
                    torch.from_numpy(gs_now).unsqueeze(0).to(self.device)
                ).item())
        for h, tr in traces.items():
            if not tr.actions:
                continue
            if bootstrap_central is not None:
                last_value = bootstrap_central
            else:
                last_value = 0.0
                obs_h = obs_dict.get(h)
                if obs_h is not None:
                    with torch.no_grad():
                        _, v = self.model(torch.from_numpy(
                            np.asarray(obs_h, dtype=np.float32)).unsqueeze(0).to(self.device))
                        last_value = float(v.item())
            self._flush(
                tr, last_value=last_value,
                bufs=(finished_obs, finished_actions, finished_log_probs,
                      finished_values, finished_advantages, finished_returns,
                      finished_masks, finished_global_states),
            )

        obs_arr = np.concatenate(finished_obs) if finished_obs else np.zeros((0, self.model.cfg.obs_dim), dtype=np.float32)
        actions_arr = np.concatenate(finished_actions) if finished_actions else np.zeros(0, dtype=np.int64)
        log_probs_arr = np.concatenate(finished_log_probs) if finished_log_probs else np.zeros(0, dtype=np.float32)
        values_arr = np.concatenate(finished_values) if finished_values else np.zeros(0, dtype=np.float32)
        advantages_arr = np.concatenate(finished_advantages) if finished_advantages else np.zeros(0, dtype=np.float32)
        returns_arr = np.concatenate(finished_returns) if finished_returns else np.zeros(0, dtype=np.float32)
        masks_t = None
        if self.use_action_mask and finished_masks:
            masks_arr = np.concatenate(finished_masks)
            masks_t = torch.from_numpy(masks_arr.astype(np.bool_))
        gs_t = None
        if self.critic is not None and finished_global_states:
            gs_arr = np.concatenate(finished_global_states)
            gs_t = torch.from_numpy(gs_arr.astype(np.float32))

        return Batch(
            obs=torch.from_numpy(obs_arr),
            actions=torch.from_numpy(actions_arr.astype(np.int64)),
            log_probs=torch.from_numpy(log_probs_arr.astype(np.float32)),
            values=torch.from_numpy(values_arr.astype(np.float32)),
            advantages=torch.from_numpy(advantages_arr.astype(np.float32)),
            returns=torch.from_numpy(returns_arr.astype(np.float32)),
            action_masks=masks_t,
            global_states=gs_t,
        )

    def _flush(self, tr: _PerAgentTrace, last_value: float, bufs) -> None:
        """Compute GAE for one agent trace and push into the rollout buffers."""
        # Pad reward list if env never delivered one after the last decision.
        while len(tr.rewards) < len(tr.actions):
            tr.rewards.append(0.0)
        adv, ret = _compute_gae(tr.rewards, tr.values, last_value,
                                self.gamma, self.gae_lambda)
        (b_obs, b_act, b_lp, b_val, b_adv, b_ret, b_mask, b_gs) = bufs
        b_obs.append(np.stack(tr.obs).astype(np.float32))
        b_act.append(np.asarray(tr.actions, dtype=np.int64))
        b_lp.append(np.asarray(tr.log_probs, dtype=np.float32))
        b_val.append(np.asarray(tr.values, dtype=np.float32))
        b_adv.append(adv.astype(np.float32))
        b_ret.append(ret.astype(np.float32))
        if tr.masks:
            b_mask.append(np.stack(tr.masks).astype(np.bool_))
        if tr.global_states:
            b_gs.append(np.stack(tr.global_states).astype(np.float32))

    def _distances_to_target(self) -> Dict[int, float]:
        """Read each active agent's shortest-path distance to its target."""
        out: Dict[int, float] = {}
        try:
            distance_map = self.env.distance_map.get()
        except Exception:
            return out
        for agent in self.env.agents:
            if agent.state == 6:  # DONE
                out[agent.handle] = 0.0
                continue
            pos = agent.position
            if pos is None:
                # Not on map yet: use initial position so departure isn't credited.
                pos = agent.initial_position
            direction = agent.direction if agent.direction is not None else agent.initial_direction
            try:
                d = float(distance_map[agent.handle, pos[0], pos[1], direction])
                if np.isfinite(d):
                    out[agent.handle] = d
            except (IndexError, TypeError):
                pass
        return out
