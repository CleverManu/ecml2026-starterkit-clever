"""
Parallel rollout collector using ``multiprocessing``.

Each worker process owns its own ``RailEnv`` and a frozen copy of the model.
At every PPO iteration, the main process broadcasts the current model weights
to all workers, asks each to collect ``rollout_steps / n_envs`` env-steps,
and concatenates the resulting batches into one PPO update.

Because the env-stepping cost (tree obs + shortest-path predictor) dominates
single-thread speed (~150 sps), N workers give roughly Nx training speedup on a
multi-core CPU. The overhead per iteration is the cost of pickling/sending the
state_dict to each worker (a few hundred KB for the default network).

Usage from training script:
    collector = ParallelRolloutCollector(
        n_envs=4, env_kwargs=dict(scene="scene_4", line_length=3),
        model=model, ppo_kwargs=dict(...),
    )
    batch = collector.collect()
    ...
    collector.close()
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.multiprocessing import set_start_method as _set_start_method

from submission.action_mask import get_action_mask
from submission.model import ActorCritic, CentralizedCritic, CriticConfig, NetConfig
from submission.my_observation_builder import extract_mask_from_obs
from training.env_factory import make_training_env
from training.global_state import compute_global_state
from training.rollout import Batch, _PerAgentTrace, _compute_gae


# Workers communicate via simple (command, payload) tuples on a Pipe.
CMD_SET_WEIGHTS = "set_weights"
CMD_SET_CRITIC_WEIGHTS = "set_critic_weights"
CMD_COLLECT = "collect"
CMD_CLOSE = "close"


def _serialize_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Move tensors to CPU and detach, ready to pickle into a child process."""
    return {k: v.detach().cpu() for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------
def _worker_loop(
    conn,
    worker_id: int,
    env_kwargs: Dict[str, Any],
    model_cfg: Dict[str, Any],
    seed: int,
    critic_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Entry point for each parallel rollout worker.

    If ``critic_cfg`` is provided, the worker also builds a CentralizedCritic
    and listens for the new ``CMD_SET_CRITIC_WEIGHTS`` command. The critic is
    used during _collect to produce centralized value estimates per step.
    """
    try:
        torch.set_num_threads(1)  # Each worker uses one thread; we get parallelism from processes.
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = make_training_env(**env_kwargs)
        # First reset uses a deterministic seed per worker so each rollout
        # starts from a different timetable distribution.
        env.reset(random_seed=seed)

        model = ActorCritic(NetConfig(**model_cfg))
        model.eval()

        critic: Optional[CentralizedCritic] = None
        if critic_cfg is not None:
            critic = CentralizedCritic(CriticConfig(**critic_cfg))
            critic.eval()

        # State carried across collect() calls: the current obs + info, the
        # in-flight per-agent traces and episode-level accumulators. This is
        # essential -- without it, each collect() would start with reset()
        # and we'd discard partial episodes.
        obs_dict, info = env.reset(random_seed=seed)
        ep_return = 0.0
        ep_length = 0

        while True:
            cmd, payload = conn.recv()

            if cmd == CMD_CLOSE:
                break

            elif cmd == CMD_SET_WEIGHTS:
                # state_dict tensors arrive in CPU; load and acknowledge.
                model.load_state_dict(payload)
                conn.send("ok")

            elif cmd == CMD_SET_CRITIC_WEIGHTS:
                if critic is not None:
                    critic.load_state_dict(payload)
                conn.send("ok")

            elif cmd == CMD_COLLECT:
                n_steps: int = payload["n_steps"]
                gamma: float = payload["gamma"]
                gae_lambda: float = payload["gae_lambda"]
                reward_scale: float = payload["reward_scale"]
                progress_coef: float = payload.get("progress_coef", 0.0)
                arrival_bonus: float = payload.get("arrival_bonus", 0.0)
                use_mask: bool = payload.get("use_action_mask", False)
                dp_only: bool = payload.get("decision_points_only", False)

                # Run the rollout and ship back the batch + episode metrics.
                result = _collect(
                    env=env, model=model, critic=critic, n_steps=n_steps,
                    gamma=gamma, gae_lambda=gae_lambda,
                    reward_scale=reward_scale, progress_coef=progress_coef,
                    arrival_bonus=arrival_bonus, use_mask=use_mask,
                    decision_points_only=dp_only,
                    obs_dict=obs_dict, info=info,
                    ep_return=ep_return, ep_length=ep_length,
                )
                conn.send(result["batch"])
                # Update carry-over state from the result.
                obs_dict = result["obs_dict"]
                info = result["info"]
                ep_return = result["ep_return"]
                ep_length = result["ep_length"]
                # Send stats as a second message so main can drain them after batch.
                conn.send({
                    "episode_returns": result["episode_returns"],
                    "episode_lengths": result["episode_lengths"],
                    "success_rates": result["success_rates"],
                })
                # Drop the big intermediate dict and run an explicit GC pass.
                # Flatland's env caches (distance map, shortest-path predictor)
                # can hold MBs that take a while to be reclaimed otherwise.
                del result
                gc.collect()
            else:
                conn.send(("error", f"unknown command: {cmd}"))
    except Exception as e:  # pragma: no cover - debugging only
        import traceback
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def _collect(
    env, model, critic, n_steps, gamma, gae_lambda, reward_scale, progress_coef,
    arrival_bonus, use_mask, decision_points_only,
    obs_dict, info, ep_return, ep_length,
) -> Dict[str, Any]:
    """Pure rollout function used by workers. Mirrors RolloutCollector.collect.

    When ``critic`` is not None, value estimates come from the centralized
    critic on a pooled global state computed each step.
    """
    from collections import defaultdict

    # Lazy imports so workers only pay this cost when DP-mode is enabled.
    if decision_points_only:
        from submission.decision_point import is_decision_point, default_action

    traces = defaultdict(_PerAgentTrace)
    finished_advantages, finished_returns = [], []
    finished_obs, finished_actions, finished_log_probs, finished_values = [], [], [], []
    finished_masks: List[np.ndarray] = []
    finished_global_states: List[np.ndarray] = []  # MAPPO only

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    success_rates: List[float] = []

    prev_distance: Dict[int, float] = {}
    if progress_coef > 0:
        prev_distance = _distances_to_target(env)
    prev_done: Dict[int, bool] = {a.handle: (a.state == 6) for a in env.agents}

    inv_scale = 1.0 / reward_scale
    steps_done = 0

    while steps_done < n_steps:
        handles = env.get_agent_handles()
        need_action = [
            h for h in handles
            if obs_dict.get(h) is not None and info["action_required"][h]
        ]

        # Decision-point gate (matches RolloutCollector.collect).
        if decision_points_only:
            need = [h for h in need_action if is_decision_point(env, h)]
            hardcoded = [h for h in need_action if h not in need]
        else:
            need = need_action
            hardcoded = []

        actions_dict: Dict[int, int] = {h: 0 for h in handles}
        if hardcoded:
            for h in hardcoded:
                actions_dict[h] = default_action(env, h)

        if need:
            batch_obs = np.stack([np.asarray(obs_dict[h], dtype=np.float32) for h in need])
            obs_t = torch.from_numpy(batch_obs)

            mask_t = None
            batch_masks = None
            if use_mask:
                embedded = extract_mask_from_obs(batch_obs)
                if embedded is not None:
                    batch_masks = embedded
                else:
                    batch_masks = np.stack([get_action_mask(env, h) for h in need])
                mask_t = torch.from_numpy(batch_masks)

            with torch.no_grad():
                logits, values_t = model(obs_t)
                if mask_t is not None:
                    logits = model._apply_mask(logits, mask_t)
                dist = torch.distributions.Categorical(logits=logits)
                sampled = dist.sample()
                log_probs = dist.log_prob(sampled)

            # MAPPO: override values with centralized critic on global state.
            step_global_state = None
            if critic is not None:
                step_global_state = compute_global_state(env, obs_dict)
                gs_t = torch.from_numpy(step_global_state).unsqueeze(0)
                with torch.no_grad():
                    cv = critic(gs_t)
                values_t = torch.full_like(values_t, float(cv.item()))

            sampled_np = sampled.numpy()
            log_probs_np = log_probs.numpy()
            values_np = values_t.numpy()
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

        next_obs, rewards, dones, next_info = env.step(actions_dict)
        steps_done += 1
        ep_length += 1

        shaped: Dict[int, float] = {}
        if progress_coef > 0:
            current_distance = _distances_to_target(env)
            for h in handles:
                prev = prev_distance.get(h, float("inf"))
                cur = current_distance.get(h, float("inf"))
                if prev != float("inf") and cur != float("inf"):
                    shaped[h] = progress_coef * (prev - cur)
            prev_distance = current_distance

        if arrival_bonus > 0:
            for a in env.agents:
                h = a.handle
                is_done = (a.state == 6)
                if is_done and not prev_done.get(h, False):
                    shaped[h] = shaped.get(h, 0.0) + arrival_bonus
                prev_done[h] = is_done

        for h in handles:
            r = float(rewards.get(h, 0.0))
            ep_return += r
            r_scaled = (r + shaped.get(h, 0.0)) * inv_scale
            if traces[h].rewards.__len__() < len(traces[h].actions):
                traces[h].rewards.append(r_scaled)
            elif traces[h].actions:
                traces[h].rewards[-1] += r_scaled

        done_all = dones["__all__"]
        obs_dict, info = next_obs, next_info

        if done_all:
            arrived = sum(int(a.state == 6) for a in env.agents)
            success_rates.append(arrived / max(len(handles), 1))
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
            ep_return = 0.0
            ep_length = 0

            for h, tr in traces.items():
                if not tr.actions:
                    continue
                _flush_to_buffers(tr, 0.0, gamma, gae_lambda,
                                  finished_obs, finished_actions,
                                  finished_log_probs, finished_values,
                                  finished_advantages, finished_returns,
                                  finished_masks, finished_global_states)
            traces = defaultdict(_PerAgentTrace)

            if steps_done < n_steps:
                obs_dict, info = env.reset()
                if progress_coef > 0:
                    prev_distance = _distances_to_target(env)
                prev_done = {a.handle: (a.state == 6) for a in env.agents}

    # Bootstrap partial traces at end of rollout window. With MAPPO, the
    # bootstrap value comes from the centralized critic on the current global
    # state (one scalar shared across all unfinished agents).
    bootstrap_central = None
    if critic is not None and traces:
        gs_now = compute_global_state(env, obs_dict)
        with torch.no_grad():
            bootstrap_central = float(critic(
                torch.from_numpy(gs_now).unsqueeze(0)
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
                    _, v = model(torch.from_numpy(
                        np.asarray(obs_h, dtype=np.float32)).unsqueeze(0))
                    last_value = float(v.item())
        _flush_to_buffers(tr, last_value, gamma, gae_lambda,
                          finished_obs, finished_actions,
                          finished_log_probs, finished_values,
                          finished_advantages, finished_returns,
                          finished_masks, finished_global_states)

    # See the matching block in RolloutCollector.collect: reset if env ended
    # exactly on the rollout boundary so next collect() can step.
    if env.dones.get("__all__", False):
        obs_dict, info = env.reset()
        ep_return = 0.0
        ep_length = 0

    obs_dim = model.cfg.obs_dim
    obs_arr = (np.concatenate(finished_obs) if finished_obs
               else np.zeros((0, obs_dim), dtype=np.float32))
    actions_arr = (np.concatenate(finished_actions) if finished_actions
                   else np.zeros(0, dtype=np.int64))
    log_probs_arr = (np.concatenate(finished_log_probs) if finished_log_probs
                     else np.zeros(0, dtype=np.float32))
    values_arr = (np.concatenate(finished_values) if finished_values
                  else np.zeros(0, dtype=np.float32))
    advantages_arr = (np.concatenate(finished_advantages) if finished_advantages
                      else np.zeros(0, dtype=np.float32))
    returns_arr = (np.concatenate(finished_returns) if finished_returns
                   else np.zeros(0, dtype=np.float32))
    masks_arr = (np.concatenate(finished_masks)
                 if (use_mask and finished_masks) else None)
    gs_arr = (np.concatenate(finished_global_states)
              if (critic is not None and finished_global_states) else None)

    return {
        "batch": {
            "obs": obs_arr, "actions": actions_arr.astype(np.int64),
            "log_probs": log_probs_arr.astype(np.float32),
            "values": values_arr.astype(np.float32),
            "advantages": advantages_arr.astype(np.float32),
            "returns": returns_arr.astype(np.float32),
            "masks": masks_arr.astype(np.bool_) if masks_arr is not None else None,
            "global_states": gs_arr.astype(np.float32) if gs_arr is not None else None,
        },
        "obs_dict": obs_dict, "info": info,
        "ep_return": ep_return, "ep_length": ep_length,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "success_rates": success_rates,
    }


def _flush_to_buffers(tr, last_value, gamma, gae_lambda,
                     b_obs, b_act, b_lp, b_val, b_adv, b_ret, b_mask, b_gs):
    while len(tr.rewards) < len(tr.actions):
        tr.rewards.append(0.0)
    adv, ret = _compute_gae(tr.rewards, tr.values, last_value, gamma, gae_lambda)
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


def _distances_to_target(env) -> Dict[int, float]:
    out: Dict[int, float] = {}
    try:
        distance_map = env.distance_map.get()
    except Exception:
        return out
    for agent in env.agents:
        if agent.state == 6:
            out[agent.handle] = 0.0
            continue
        pos = agent.position
        if pos is None:
            pos = agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        try:
            d = float(distance_map[agent.handle, pos[0], pos[1], direction])
            if np.isfinite(d):
                out[agent.handle] = d
        except (IndexError, TypeError):
            pass
    return out


# ---------------------------------------------------------------------------
# Main-process collector
# ---------------------------------------------------------------------------
@dataclass
class ParallelRolloutCollector:
    """N parallel env workers that collect rollouts cooperatively."""
    n_envs: int
    env_kwargs: Dict[str, Any]
    model: ActorCritic
    rollout_steps: int
    # MAPPO: optional centralized critic. None for vanilla PPO.
    critic: Optional[CentralizedCritic] = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 100.0
    progress_reward_coef: float = 0.0
    arrival_bonus: float = 0.0
    use_action_mask: bool = False
    decision_points_only: bool = False
    base_seed: int = 0

    # Diagnostics filled in by each collect() call.
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    success_rates: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 'spawn' is safer across platforms but slower to start; on Linux 'fork'
        # is fine and much faster. We default to fork; user can override env var.
        ctx = os.environ.get("MP_START_METHOD", "fork")
        try:
            _set_start_method(ctx, force=True)
        except RuntimeError:
            pass  # Already set elsewhere.

        self._parents: List[Any] = []
        self._workers: List[Process] = []
        model_cfg = {
            "obs_dim": self.model.cfg.obs_dim,
            "hidden": self.model.cfg.hidden,
            "n_actions": self.model.cfg.n_actions,
        }
        critic_cfg = None
        if self.critic is not None:
            critic_cfg = {
                "global_state_dim": self.critic.cfg.global_state_dim,
                "hidden": self.critic.cfg.hidden,
            }
        for i in range(self.n_envs):
            parent, child = Pipe(duplex=True)
            p = Process(
                target=_worker_loop,
                args=(child, i, dict(self.env_kwargs), model_cfg,
                      self.base_seed + i * 1000, critic_cfg),
                daemon=True,
            )
            p.start()
            self._parents.append(parent)
            self._workers.append(p)

    def collect(self) -> Batch:
        # 1. Broadcast current weights.
        sd = _serialize_state_dict(self.model.state_dict())
        for conn in self._parents:
            conn.send((CMD_SET_WEIGHTS, sd))
        for conn in self._parents:
            ack = conn.recv()
            if isinstance(ack, tuple) and ack[0] == "error":
                raise RuntimeError(f"Worker error during set_weights: {ack[1]}")

        # 1b. Broadcast critic weights too if running MAPPO.
        if self.critic is not None:
            csd = _serialize_state_dict(self.critic.state_dict())
            for conn in self._parents:
                conn.send((CMD_SET_CRITIC_WEIGHTS, csd))
            for conn in self._parents:
                ack = conn.recv()
                if isinstance(ack, tuple) and ack[0] == "error":
                    raise RuntimeError(f"Worker error during set_critic_weights: {ack[1]}")

        # 2. Kick off parallel collection.
        per_worker = max(1, self.rollout_steps // self.n_envs)
        for conn in self._parents:
            conn.send((CMD_COLLECT, {
                "n_steps": per_worker,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "reward_scale": self.reward_scale,
                "progress_coef": self.progress_reward_coef,
                "arrival_bonus": self.arrival_bonus,
                "use_action_mask": self.use_action_mask,
                "decision_points_only": self.decision_points_only,
            }))

        # 3. Drain results.
        all_batches: List[Dict[str, np.ndarray]] = []
        self.episode_returns = []
        self.episode_lengths = []
        self.success_rates = []

        for conn in self._parents:
            payload = conn.recv()
            if isinstance(payload, tuple) and payload[0] == "error":
                raise RuntimeError(f"Worker error during collect: {payload[1]}")
            all_batches.append(payload)
            stats = conn.recv()
            if isinstance(stats, tuple) and stats[0] == "error":
                raise RuntimeError(f"Worker error during collect (stats): {stats[1]}")
            self.episode_returns.extend(stats["episode_returns"])
            self.episode_lengths.extend(stats["episode_lengths"])
            self.success_rates.extend(stats["success_rates"])

        # 4. Concatenate batches.
        def _cat(key, dtype):
            parts = [b[key] for b in all_batches
                     if b.get(key) is not None and b[key].shape[0] > 0]
            if not parts:
                if key == "obs":
                    return np.zeros((0, self.model.cfg.obs_dim), dtype=dtype)
                if key in ("masks", "global_states"):
                    return None
                return np.zeros(0, dtype=dtype)
            return np.concatenate(parts, axis=0)

        masks_arr = None
        if self.use_action_mask:
            masks_arr = _cat("masks", np.bool_)
        gs_arr = None
        if self.critic is not None:
            gs_arr = _cat("global_states", np.float32)

        return Batch(
            obs=torch.from_numpy(_cat("obs", np.float32)),
            actions=torch.from_numpy(_cat("actions", np.int64)),
            log_probs=torch.from_numpy(_cat("log_probs", np.float32)),
            values=torch.from_numpy(_cat("values", np.float32)),
            advantages=torch.from_numpy(_cat("advantages", np.float32)),
            returns=torch.from_numpy(_cat("returns", np.float32)),
            action_masks=torch.from_numpy(masks_arr) if masks_arr is not None else None,
            global_states=torch.from_numpy(gs_arr) if gs_arr is not None else None,
        )

    def close(self) -> None:
        for conn in self._parents:
            try:
                conn.send((CMD_CLOSE, None))
            except Exception:
                pass
        for p in self._workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
