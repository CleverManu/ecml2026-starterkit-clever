"""
Diagnostic eval: load a checkpoint and run N episodes to classify failure modes.

Usage
-----
::

    python -m training.diagnose --checkpoint runs/v4_stage0/checkpoints/best.pt \
                                --obs-version v4 --decision-points-only \
                                --use-action-mask \
                                --episodes 10 --scene scene_1 --line-length 2

Reports per-episode and aggregate statistics about *why* agents failed:
arrived, deadlocked (no movement in last N steps), never departed, etc.
Lets us tell the difference between "policy is learning but slowly" and
"policy is making structural mistakes that pile up agents".

Does not modify the running training job in any way -- this is purely an
offline analysis of a saved checkpoint.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from typing import Dict, List

import numpy as np
import torch

# Late imports inside main() so --help works without the rest of the env loaded.


# An agent is considered "stuck" if it hasn't moved for this many *recent*
# action-required ticks. 20 is conservative; small enough that we don't miss
# real deadlocks, large enough not to flag normal waiting-for-other-agent.
STUCK_WINDOW = 20


def classify_agent_endstate(agent, idle_streaks: Dict[int, int]) -> str:
    """Return one of: arrived, stuck, never_departed, on_map_at_timeout."""
    if int(agent.state) == 6:
        return "arrived"
    if agent.position is None:
        # State < 4 (WAITING / READY_TO_DEPART / MALFUNCTION_OFF_MAP) means
        # the agent never made it onto the map.
        return "never_departed"
    if idle_streaks.get(agent.handle, 0) >= STUCK_WINDOW:
        return "stuck"
    return "on_map_at_timeout"


def run_episode(env, policy_fn, max_steps: int):
    """Run one episode using the given policy_fn(env, handle, obs) -> int.

    Returns a dict of per-agent failure-mode counts plus episode-level
    metrics (length, success_rate, n_decision_points_visited).
    """
    obs, info = env.reset()
    handles = env.get_agent_handles()

    # Track movement: most recent positions per agent and how many recent
    # action-required ticks have passed without a position change.
    last_position: Dict[int, tuple] = {h: None for h in handles}
    idle_streak: Dict[int, int] = {h: 0 for h in handles}
    longest_idle: Dict[int, int] = {h: 0 for h in handles}

    ep_length = 0
    for step in range(max_steps):
        # Build actions per agent. For agents with no obs or with action_required=False,
        # send DO_NOTHING and the env handles them.
        actions = {h: 0 for h in handles}
        for h in handles:
            if obs.get(h) is not None and info["action_required"][h]:
                actions[h] = policy_fn(env, h, obs[h])

        # Track idle streaks for agents that DID get an action this step.
        for h in handles:
            a = env.agents[h]
            if a.position is None:
                continue
            if last_position[h] == a.position:
                idle_streak[h] += 1
                longest_idle[h] = max(longest_idle[h], idle_streak[h])
            else:
                idle_streak[h] = 0
            last_position[h] = a.position

        obs, _, dones, info = env.step(actions)
        ep_length += 1
        if dones["__all__"]:
            break

    classes = Counter()
    for a in env.agents:
        classes[classify_agent_endstate(a, longest_idle)] += 1

    return {
        "length": ep_length,
        "n_agents": len(handles),
        "classes": classes,
        "success_rate": classes["arrived"] / len(handles),
        "longest_idle_p95": float(np.percentile(list(longest_idle.values()), 95)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scene", default="scene_1")
    p.add_argument("--line-length", type=int, default=2)
    p.add_argument("--obs-version", default="v4", choices=["v1", "v2", "v3", "v4"])
    p.add_argument("--n-agents-range", nargs=2, type=int, default=None)
    p.add_argument("--use-action-mask", action="store_true",
                   help="Apply same masking at eval as during training.")
    p.add_argument("--decision-points-only", action="store_true",
                   help="Apply same decision-point gating as during training. "
                        "Non-decision cells get the hard-coded default action.")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="argmax action selection (default True).")
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    args = p.parse_args()

    # Late imports for fast --help.
    from submission.model import load_checkpoint
    from submission.my_observation_builder import extract_mask_from_obs
    from training.env_factory import make_training_env

    model = load_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    print(f"Loaded checkpoint: obs_dim={model.cfg.obs_dim}, "
          f"hidden={model.cfg.hidden}")

    # The decision-point utilities. Imported lazily so v1/v2 obs versions can
    # still use this script without requiring the v4 module.
    if args.decision_points_only:
        from submission.decision_point import is_decision_point, default_action

    def policy_fn(env, handle, observation):
        """Inference for one agent (mirrors training-time logic exactly)."""
        if args.decision_points_only and not is_decision_point(env, handle):
            return default_action(env, handle)
        obs_np = np.asarray(observation, dtype=np.float32)
        mask = extract_mask_from_obs(obs_np) if args.use_action_mask else None
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).unsqueeze(0)
            mask_t = torch.from_numpy(mask).unsqueeze(0) if mask is not None else None
            action, _, _ = model.act(obs_t, action_mask=mask_t,
                                     deterministic=args.deterministic)
        return int(action.item())

    n_agents_range = tuple(args.n_agents_range) if args.n_agents_range else None
    env = make_training_env(
        scene=args.scene, line_length=args.line_length,
        obs_version=args.obs_version, n_agents_range=n_agents_range,
    )

    # Aggregate stats across episodes.
    all_classes = Counter()
    all_lengths, all_success, all_idle_p95 = [], [], []

    print(f"\nRunning {args.episodes} eval episodes "
          f"(scene={args.scene}, line_length={args.line_length}, "
          f"deterministic={args.deterministic})...\n")
    t0 = time.time()
    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)
        torch.manual_seed(args.seed + ep)
        env.reset(random_seed=args.seed + ep)
        max_steps = env._max_episode_steps
        result = run_episode(env, policy_fn, max_steps=max_steps)
        all_classes.update(result["classes"])
        all_lengths.append(result["length"])
        all_success.append(result["success_rate"])
        all_idle_p95.append(result["longest_idle_p95"])
        c = result["classes"]
        print(f"  ep {ep+1:2d}: {result['length']:4d} steps  "
              f"succ={result['success_rate']*100:5.1f}%  "
              f"arrived={c['arrived']:3d}  stuck={c['stuck']:3d}  "
              f"never_dep={c['never_departed']:3d}  "
              f"timeout_on_map={c['on_map_at_timeout']:3d}  "
              f"idle_p95={result['longest_idle_p95']:.0f}")

    print(f"\nTotal: {time.time()-t0:.1f}s")
    n_total = sum(all_classes.values())
    print(f"\nAggregate over {args.episodes} episodes ({n_total} agent-trajectories):")
    print(f"  mean success rate: {np.mean(all_success)*100:.1f}% "
          f"(min {min(all_success)*100:.1f}%, max {max(all_success)*100:.1f}%)")
    print(f"  mean episode length: {np.mean(all_lengths):.0f}")
    print(f"  agent end-states (across all episodes):")
    for k in ("arrived", "stuck", "never_departed", "on_map_at_timeout"):
        n = all_classes[k]
        print(f"    {k:<22s}: {n:4d}  ({100*n/n_total:5.1f}%)")
    print(f"  longest-idle p95 across episodes: "
          f"{np.mean(all_idle_p95):.0f} +/- {np.std(all_idle_p95):.0f}")


if __name__ == "__main__":
    main()
