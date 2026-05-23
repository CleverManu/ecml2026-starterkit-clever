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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.multiprocessing import set_start_method as _set_start_method

from submission.model import ActorCritic, NetConfig
from training.env_factory import make_training_env
from training.rollout import Batch, _PerAgentTrace, _compute_gae


# Workers communicate via simple (command, payload) tuples on a Pipe.
CMD_SET_WEIGHTS = "set_weights"
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
) -> None:
    """Entry point for each parallel rollout worker."""
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

            elif cmd == CMD_COLLECT:
                n_steps: int = payload["n_steps"]
                gamma: float = payload["gamma"]
                gae_lambda: float = payload["gae_lambda"]
                reward_scale: float = payload["reward_scale"]
                progress_coef: float = payload.get("progress_coef", 0.0)

                # Run the rollout and ship back the batch + episode metrics.
                result = _collect(
                    env=env, model=model, n_steps=n_steps,
                    gamma=gamma, gae_lambda=gae_lambda,
                    reward_scale=reward_scale, progress_coef=progress_coef,
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
    env, model, n_steps, gamma, gae_lambda, reward_scale, progress_coef,
    obs_dict, info, ep_return, ep_length,
) -> Dict[str, Any]:
    """Pure rollout function used by workers. Mirrors RolloutCollector.collect."""
    from collections import defaultdict

    traces = defaultdict(_PerAgentTrace)
    finished_advantages, finished_returns = [], []
    finished_obs, finished_actions, finished_log_probs, finished_values = [], [], [], []

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    success_rates: List[float] = []

    prev_distance: Dict[int, float] = {}
    if progress_coef > 0:
        prev_distance = _distances_to_target(env)

    inv_scale = 1.0 / reward_scale
    steps_done = 0

    while steps_done < n_steps:
        handles = env.get_agent_handles()
        need = [
            h for h in handles
            if obs_dict.get(h) is not None and info["action_required"][h]
        ]
        actions_dict: Dict[int, int] = {h: 0 for h in handles}

        if need:
            batch_obs = np.stack([np.asarray(obs_dict[h], dtype=np.float32) for h in need])
            obs_t = torch.from_numpy(batch_obs)
            with torch.no_grad():
                logits, values_t = model(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                sampled = dist.sample()
                log_probs = dist.log_prob(sampled)
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
                                  finished_advantages, finished_returns)
            traces = defaultdict(_PerAgentTrace)

            if steps_done < n_steps:
                obs_dict, info = env.reset()
                if progress_coef > 0:
                    prev_distance = _distances_to_target(env)

    # Bootstrap partial traces at end of rollout window.
    for h, tr in traces.items():
        if not tr.actions:
            continue
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
                          finished_advantages, finished_returns)

    # If the rollout window happened to end on the same step the episode
    # terminated, the env is now in a done state and would raise
    # "Episode is done, cannot call step()" on the next collect(). Reset
    # here so the carry-over obs_dict/info point at a fresh episode.
    # We only reset in this case; otherwise we keep mid-episode state so
    # the next collect continues the same episode (correct on-policy behaviour).
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

    return {
        "batch": {
            "obs": obs_arr, "actions": actions_arr.astype(np.int64),
            "log_probs": log_probs_arr.astype(np.float32),
            "values": values_arr.astype(np.float32),
            "advantages": advantages_arr.astype(np.float32),
            "returns": returns_arr.astype(np.float32),
        },
        "obs_dict": obs_dict, "info": info,
        "ep_return": ep_return, "ep_length": ep_length,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "success_rates": success_rates,
    }


def _flush_to_buffers(tr, last_value, gamma, gae_lambda,
                     b_obs, b_act, b_lp, b_val, b_adv, b_ret):
    while len(tr.rewards) < len(tr.actions):
        tr.rewards.append(0.0)
    adv, ret = _compute_gae(tr.rewards, tr.values, last_value, gamma, gae_lambda)
    b_obs.append(np.stack(tr.obs).astype(np.float32))
    b_act.append(np.asarray(tr.actions, dtype=np.int64))
    b_lp.append(np.asarray(tr.log_probs, dtype=np.float32))
    b_val.append(np.asarray(tr.values, dtype=np.float32))
    b_adv.append(adv.astype(np.float32))
    b_ret.append(ret.astype(np.float32))


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
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 100.0
    progress_reward_coef: float = 0.0
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
        for i in range(self.n_envs):
            parent, child = Pipe(duplex=True)
            p = Process(
                target=_worker_loop,
                args=(child, i, dict(self.env_kwargs), model_cfg, self.base_seed + i * 1000),
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

        # 2. Kick off parallel collection.
        per_worker = max(1, self.rollout_steps // self.n_envs)
        for conn in self._parents:
            conn.send((CMD_COLLECT, {
                "n_steps": per_worker,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "reward_scale": self.reward_scale,
                "progress_coef": self.progress_reward_coef,
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
            parts = [b[key] for b in all_batches if b[key].shape[0] > 0]
            if not parts:
                shape = (0, self.model.cfg.obs_dim) if key == "obs" else (0,)
                return np.zeros(shape, dtype=dtype)
            return np.concatenate(parts, axis=0)

        return Batch(
            obs=torch.from_numpy(_cat("obs", np.float32)),
            actions=torch.from_numpy(_cat("actions", np.int64)),
            log_probs=torch.from_numpy(_cat("log_probs", np.float32)),
            values=torch.from_numpy(_cat("values", np.float32)),
            advantages=torch.from_numpy(_cat("advantages", np.float32)),
            returns=torch.from_numpy(_cat("returns", np.float32)),
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
