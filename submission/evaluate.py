"""
Evaluate a trained checkpoint against the official ``debug-environments`` test set.

This mimics what the competition evaluator does inside Docker -- runs the policy
on each test scenario and reports per-episode success rate and normalized reward.

Usage
-----
::

    # 1. Download the test set (once)
    wget https://data.flatland.cloud/benchmarks/Flatland3/debug-environments.zip
    unzip debug-environments.zip -d scenarios/

    # 2. Evaluate
    python -m training.evaluate --checkpoint runs/baseline/checkpoints/best.pt \\
        --scenarios scenarios/debug-environments

Pass ``--csv out.csv`` to also write a machine-readable summary.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rewards import ECML2026Rewards

from my_orga.model import load_checkpoint
from my_orga.my_observation_builder import MyObservationBuilder
from my_orga.my_policy import MyPolicy


def _read_metadata(metadata_csv: Path) -> List[dict]:
    """Read the test-set ``metadata.csv``."""
    rows = []
    with metadata_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def run_one(scenario_path: Path, policy: MyPolicy) -> Tuple[float, float, int]:
    """Run one episode. Returns (success_rate, normalized_reward, elapsed_steps)."""
    env, _ = RailEnvPersister.load_new(
        str(scenario_path),
        obs_builder=MyObservationBuilder(),
        rewards=ECML2026Rewards(),
    )
    obs_dict, info = env.reset()

    handles = env.get_agent_handles()
    n_agents = env.get_num_agents()
    cumulative = {h: 0.0 for h in handles}

    for _ in range(env._max_episode_steps):
        action_dict = policy.act_many(handles, observations=list(obs_dict.values()))
        obs_dict, rewards, dones, info = env.step(action_dict)
        for h in handles:
            cumulative[h] += float(rewards.get(h, 0.0))
        if dones["__all__"]:
            break

    elapsed = env._elapsed_steps
    arrived = sum(int(agent.state == 6) for agent in env.agents)  # TrainState.DONE = 6
    success_rate = arrived / max(n_agents, 1)
    normalized = float(env.rewards.normalize(
        *cumulative.values(),
        max_episode_steps=env._max_episode_steps,
        num_agents=n_agents,
    ))
    return success_rate, normalized, elapsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Flatland PPO checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pt file produced by training.")
    p.add_argument("--scenarios", type=str, required=True,
                   help="Folder containing debug-environments/ (with metadata.csv).")
    p.add_argument("--csv", type=str, default=None,
                   help="Optional output CSV path.")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap on episodes (useful for smoke runs).")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions instead of arg-max (useful for debugging).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenarios_dir = Path(args.scenarios)
    metadata = scenarios_dir / "metadata.csv"
    if not metadata.is_file():
        raise SystemExit(f"metadata.csv not found at {metadata}")

    policy = MyPolicy(checkpoint_path=args.checkpoint,
                      deterministic=not args.stochastic)
    rows = _read_metadata(metadata)
    if args.max_episodes is not None:
        rows = rows[: args.max_episodes]

    out_rows: List[Dict] = []
    print(f"Evaluating {len(rows)} episodes from {scenarios_dir} ...")
    for i, row in enumerate(rows):
        ep_id = row.get("episode_id") or row.get("ep_id") or f"row_{i}"
        pkl_path = scenarios_dir / row.get("env_path", f"{ep_id}.pkl")
        if not pkl_path.is_file():
            print(f"  [{i+1}/{len(rows)}] {ep_id}: SKIP (missing {pkl_path})")
            continue
        succ, norm_r, steps = run_one(pkl_path, policy)
        out_rows.append({
            "episode_id": ep_id, "env_time": steps,
            "success_rate": succ, "normalized_reward": norm_r,
        })
        print(f"  [{i+1}/{len(rows)}] {ep_id}: success={succ:.2%} "
              f"norm_reward={norm_r:.4f} steps={steps}")

    if out_rows:
        mean_succ = np.mean([r["success_rate"] for r in out_rows])
        mean_norm = np.mean([r["normalized_reward"] for r in out_rows])
        print(f"\nMean success rate: {mean_succ:.2%}")
        print(f"Mean normalized reward: {mean_norm:.4f}")
    else:
        print("No episodes evaluated.")

    if args.csv and out_rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
