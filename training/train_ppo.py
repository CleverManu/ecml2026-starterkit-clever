"""
Main PPO training script for the ECML 2026 Flatland Competition.

Usage
-----
::

    # Quick smoke run (a handful of updates, no TensorBoard)
    python -m training.train_ppo --total-steps 10000 --rollout-steps 512 --log-dir runs/smoke

    # Full CPU baseline (overnight on a laptop)
    python -m training.train_ppo --total-steps 5000000 --rollout-steps 1024 --log-dir runs/baseline

Run from the project root so the ``submission`` and ``training`` packages resolve.
"""
from __future__ import annotations

# Suppress Flatland's "Could not find path" spam (happens often with
# line_length > 2 when the line generator retries waypoint sampling).
# Set before any flatland import so the filter is in place when those modules load.
import warnings
warnings.filterwarnings("ignore", message=".*Could not find path.*")

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from submission.model import ActorCritic, NetConfig, save_checkpoint, load_checkpoint
from submission.my_observation_builder import get_obs_dim, get_obs_dim_v2, get_obs_dim_v3

from training.env_factory import make_training_env, DEFAULT_SCENARIO
from training.ppo import PPOConfig, PPOTrainer
from training.rollout import RolloutCollector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training for Flatland ECML 2026.")
    p.add_argument("--total-steps", type=int, default=5_000_000,
                   help="Total environment steps to train for.")
    p.add_argument("--rollout-steps", type=int, default=1024,
                   help="Environment steps per rollout (PPO update cycle).")
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO,
                   help="Competition scenario pickle path.")
    p.add_argument("--line-length", type=int, default=2,
                   help="Maximum waypoints per train (2 = simple A->B).")
    p.add_argument("--scene", type=str, default=None,
                   help="Station-set restriction: scene_1..scene_5 or omit for all.")
    p.add_argument("--obs-version", type=str, default="v1", choices=["v1", "v2", "v3"],
                   help="v1 = tree obs only (252 dims). "
                        "v2 = tree obs + 8 global features (260 dims). "
                        "v3 = v2 + 5 action-mask features at end (265 dims). "
                        "Pick v3 if you want action masking; requires fresh training.")
    p.add_argument("--use-action-mask", action="store_true",
                   help="Apply hard masking to invalid actions during training. "
                        "Recommended only with --obs-version v3 (so masking also "
                        "works at evaluation time).")
    p.add_argument("--arrival-bonus", type=float, default=0.0,
                   help="Raw-units reward given on the step an agent reaches its "
                        "target. Dense positive feedback for the actual success "
                        "metric. Typical: 5 to 50.")
    p.add_argument("--n-agents-range", nargs=2, type=int, default=None,
                   metavar=("LO", "HI"),
                   help="If set, randomize agent count uniformly in [LO, HI] on "
                        "every reset. Competition agent counts range 8-532, so a "
                        "range like '10 60' is sensible for training.")
    p.add_argument("--num-envs", type=int, default=1,
                   help="Number of parallel env workers. >1 enables multi-process "
                        "collection (recommended for CPU training).")
    p.add_argument("--shape-progress", type=float, default=0.0,
                   help="If > 0, add a small reward each step for shrinking "
                        "distance-to-target. Typical values: 0.3 to 1.0.")
    p.add_argument("--anneal-lr", action="store_true",
                   help="Linearly anneal learning rate from initial to 0 over "
                        "--total-steps. Standard PPO practice.")
    p.add_argument("--hidden", type=int, default=256, help="Hidden width.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--reward-scale", type=float, default=100.0,
                   help="Divide raw rewards by this. ECML2026 penalties are huge; "
                        "keeping value targets ~O(10) makes the value head trainable.")
    p.add_argument("--log-dir", type=str, default="runs/baseline")
    p.add_argument("--checkpoint-every", type=int, default=10,
                   help="Save a checkpoint every N updates.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None,
                   help="Optional checkpoint path to resume from.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Optional TensorBoard; degrade gracefully if not installed.
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(str(log_dir / "tb"))
        print(f"TensorBoard logs -> {log_dir / 'tb'}")
    except ImportError:
        print("(TensorBoard not installed; logging to stdout only.)")

    env_kwargs = dict(
        scenario_path=args.scenario,
        line_length=args.line_length,
        scene=args.scene,
        obs_version=args.obs_version,
        n_agents_range=tuple(args.n_agents_range) if args.n_agents_range else None,
    )

    # Resolve observation dim from the chosen builder version.
    if args.obs_version == "v3":
        obs_dim = get_obs_dim_v3()
    elif args.obs_version == "v2":
        obs_dim = get_obs_dim_v2()
    else:
        obs_dim = get_obs_dim()

    if args.use_action_mask and args.obs_version != "v3":
        print(f"WARNING: --use-action-mask with --obs-version {args.obs_version}: "
              f"masking will work during training but not at inference (the "
              f"docker policy can't access the env). Use --obs-version v3 for "
              f"consistent train/eval masking.")

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        model = load_checkpoint(args.resume, map_location="cpu")
        if model.cfg.obs_dim != obs_dim:
            raise SystemExit(
                f"Checkpoint obs_dim={model.cfg.obs_dim} doesn't match selected "
                f"--obs-version (gives {obs_dim}). To resume, match the obs "
                f"version used during the original training run."
            )
    else:
        model = ActorCritic(NetConfig(obs_dim=obs_dim, hidden=args.hidden))
    model.to("cpu").train()

    # Single env (for setup info) + collector selection.
    setup_env = make_training_env(**env_kwargs)
    setup_env.reset(random_seed=args.seed)
    print(f"Env: {setup_env.width}x{setup_env.height}, n_agents={setup_env.get_num_agents()}, "
          f"obs_dim={obs_dim}, max_episode_steps={setup_env._max_episode_steps}, "
          f"num_envs={args.num_envs}, n_agents_range={args.n_agents_range}, "
          f"action_mask={args.use_action_mask}")
    del setup_env

    if args.num_envs > 1:
        from training.parallel_rollout import ParallelRolloutCollector
        collector = ParallelRolloutCollector(
            n_envs=args.num_envs, env_kwargs=env_kwargs,
            model=model, rollout_steps=args.rollout_steps,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            reward_scale=args.reward_scale,
            progress_reward_coef=args.shape_progress,
            arrival_bonus=args.arrival_bonus,
            use_action_mask=args.use_action_mask,
            base_seed=args.seed,
        )
    else:
        collector = RolloutCollector(
            env=make_training_env(**env_kwargs),
            model=model, rollout_steps=args.rollout_steps,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            reward_scale=args.reward_scale,
            progress_reward_coef=args.shape_progress,
            arrival_bonus=args.arrival_bonus,
            use_action_mask=args.use_action_mask,
        )
    trainer = PPOTrainer(
        model=model,
        cfg=PPOConfig(
            lr=args.lr, clip_eps=args.clip_eps,
            value_coef=args.value_coef, entropy_coef=args.entropy_coef,
            epochs=args.epochs, minibatch_size=args.minibatch_size,
        ),
    )

    global_step = 0
    update_idx = 0
    start_time = time.time()
    best_success_rate = -1.0

    try:
        while global_step < args.total_steps:
            # Optional linear LR anneal from initial value to 0.
            if args.anneal_lr:
                frac = 1.0 - (global_step / args.total_steps)
                new_lr = args.lr * max(frac, 0.0)
                for g in trainer.optimizer.param_groups:
                    g["lr"] = new_lr

            t0 = time.time()
            batch = collector.collect()
            collect_time = time.time() - t0

            global_step += args.rollout_steps
            update_idx += 1

            t0 = time.time()
            metrics = trainer.update(batch)
            update_time = time.time() - t0

            ep_ret = float(np.mean(collector.episode_returns)) if collector.episode_returns else 0.0
            ep_len = float(np.mean(collector.episode_lengths)) if collector.episode_lengths else 0.0
            succ = float(np.mean(collector.success_rates)) if collector.success_rates else 0.0
            sps = args.rollout_steps / max(collect_time + update_time, 1e-6)

            elapsed = time.time() - start_time
            print(
                f"[upd {update_idx:5d} | step {global_step:>9d} | {elapsed/60:6.1f} min] "
                f"ep_ret={ep_ret:8.2f} ep_len={ep_len:6.1f} succ={succ:.2%} "
                f"| pl={metrics['policy_loss']:+.3f} vl={metrics['value_loss']:.3f} "
                f"H={metrics['entropy']:.3f} kl={metrics['approx_kl']:+.3f} "
                f"clip={metrics['clip_frac']:.2%} n={metrics['n_samples']:5d} "
                f"| sps={sps:5.1f}"
            )

            if tb_writer is not None:
                tb_writer.add_scalar("rollout/ep_return_mean", ep_ret, global_step)
                tb_writer.add_scalar("rollout/ep_length_mean", ep_len, global_step)
                tb_writer.add_scalar("rollout/success_rate", succ, global_step)
                tb_writer.add_scalar("rollout/n_episodes", len(collector.episode_returns), global_step)
                tb_writer.add_scalar("ppo/policy_loss", metrics["policy_loss"], global_step)
                tb_writer.add_scalar("ppo/value_loss", metrics["value_loss"], global_step)
                tb_writer.add_scalar("ppo/entropy", metrics["entropy"], global_step)
                tb_writer.add_scalar("ppo/approx_kl", metrics["approx_kl"], global_step)
                tb_writer.add_scalar("ppo/clip_fraction", metrics["clip_frac"], global_step)
                tb_writer.add_scalar("ppo/n_samples", metrics["n_samples"], global_step)
                tb_writer.add_scalar("perf/steps_per_second", sps, global_step)

            if update_idx % args.checkpoint_every == 0:
                path = ckpt_dir / f"ckpt_upd{update_idx:05d}.pt"
                save_checkpoint(model, str(path),
                                extra={"global_step": global_step, "update": update_idx})
                # Also overwrite latest.pt for convenience.
                save_checkpoint(model, str(ckpt_dir / "latest.pt"),
                                extra={"global_step": global_step, "update": update_idx})
                print(f"  saved {path}")

            if collector.success_rates and succ > best_success_rate:
                best_success_rate = succ
                save_checkpoint(
                    model, str(ckpt_dir / "best.pt"),
                    extra={"global_step": global_step, "update": update_idx,
                           "success_rate": succ},
                )

    except (KeyboardInterrupt, Exception) as e:
        # Emergency save: don't lose progress to OOM-killed workers, ctrl-C,
        # transient pipe errors, etc. The user can resume from emergency.pt.
        emergency_path = ckpt_dir / "emergency.pt"
        try:
            save_checkpoint(
                model, str(emergency_path),
                extra={"global_step": global_step, "update": update_idx,
                       "crashed": True, "reason": type(e).__name__},
            )
            print(f"\n!!! Training interrupted by {type(e).__name__}: {e}")
            print(f"!!! Emergency checkpoint saved -> {emergency_path}")
            print(f"!!! Resume with:  --resume {emergency_path}")
        except Exception as save_err:
            print(f"!!! Could not save emergency checkpoint: {save_err}")
        if hasattr(collector, "close"):
            try:
                collector.close()
            except Exception:
                pass
        raise

    # Final save
    save_checkpoint(model, str(ckpt_dir / "final.pt"),
                    extra={"global_step": global_step, "update": update_idx})
    print(f"Done. Final checkpoint -> {ckpt_dir / 'final.pt'}")
    if best_success_rate >= 0:
        print(f"Best success rate during training: {best_success_rate:.2%}")
    if tb_writer is not None:
        tb_writer.close()
    if hasattr(collector, "close"):
        collector.close()


if __name__ == "__main__":
    main()
