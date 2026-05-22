# Iterating beyond the baseline

This document covers what changed since the v1 baseline and how to use it.
For first-time setup, see [`README.md`](README.md).

## What's new

| Improvement | Flag | Expected payoff |
|---|---|---|
| Parallel env workers | `--num-envs N` | ~Nx wall-clock speedup on N-core CPU |
| Global obs features | `--obs-version v2` | Better state, ~ +5-10% success rate |
| Progress reward shaping | `--shape-progress 1.0` | Faster early learning |
| Linear LR annealing | `--anneal-lr` | More stable late training |
| Value loss clipping | enabled by default | Stability, no flag needed |

All flags are independent — combine freely.

## Why this set of improvements

After the first 2M-step run, the limiting factors were:

1. **Single-threaded env stepping at ~150 sps.** Measurement: the env step
   alone (tree-obs + shortest-path predictor + step) does 162 sps with no
   policy in the loop -- meaning the network forward pass is <10% of total
   time. **Conclusion: parallelism on the env side is the biggest win
   available.**
2. **Sparse, mostly-terminal rewards.** With 6 agents and ~400 max steps per
   episode, the policy sees only one informative reward signal at the end of
   each episode. Progress shaping fills the gap.
3. **Purely local observation.** The tree obs is rich locally but tells the
   agent nothing about how the episode as a whole is going.

## Parallel envs (biggest win)

Workers run separate Python processes with their own `RailEnv`. Each iteration:
1. Main sends the current network weights to every worker (≈600 KB transfer).
2. Each worker collects `rollout_steps / num_envs` env steps.
3. Workers send batches back; main concatenates them and runs the PPO update.

```bash
# 4 workers — good for a 4-core laptop
python -m training.train_ppo --num-envs 4 --rollout-steps 4096 --log-dir runs/parallel
```

Choose `--num-envs` to match the number of *physical* cores you have. Going
much beyond that adds context-switch overhead without speeding training.
Hyperthreading siblings (logical cores beyond physical) help a little but not
much because the env step is CPU-bound.

Also bump `--rollout-steps` proportionally — each worker contributes only
`rollout_steps / num_envs` steps, so you want the total ≥ `1024 * num_envs`
for stable advantage estimates.

**Multiprocessing start method.** Defaults to `fork` on Linux (fastest) and
`spawn` on macOS/Windows. Override with `MP_START_METHOD=spawn` env var if
fork causes problems (e.g. inherited CUDA state, although we're CPU-only).

**Debugging tip.** If a worker silently dies, the main loop will hang on
`conn.recv()`. Add print statements inside `_worker_loop` in
`training/parallel_rollout.py` and run with `python -u` to see them.

## Global observation features (v2)

The tree obs sees ~20 cells around the agent. The v2 builder appends 8 global
features so the policy also sees:

- **time pressure** — fraction of episode elapsed, plus this agent's
  earliest-departure and latest-arrival relative to now
- **traffic state** — fraction of agents arrived / on map / malfunctioning /
  still waiting to depart
- **this agent's malfunction** — time remaining if currently broken

Vector grows from 252 to 260 floats. Tiny compute cost, real signal.

```bash
python -m training.train_ppo --obs-version v2 --log-dir runs/v2
```

**Important when submitting:** if you trained with `v2`, your Docker image
must use the v2 builder at evaluation time too. Update the env var in
`Dockerfile`:

```dockerfile
ENV OBS_BUILDER=my_orga.my_observation_builder.MyObservationBuilderV2
```

(Note the `V2` suffix.) A checkpoint trained with v1 obs **will not work**
with v2 obs in the container — the input dimension would mismatch the network.
Training will refuse to `--resume` from a checkpoint with the wrong obs_dim,
but the Docker submission has no such guard.

## Reward shaping (progress)

ECML2026Rewards is mostly silence until the end of the episode, when a big
positive or big negative reward arrives. PPO struggles to assign credit
backwards through ~400 steps of zeros.

Solution: at each step, give each agent a small reward proportional to how
much its shortest-path-to-target shrunk this step. This gives a dense gradient
toward "drive toward the goal" without changing the optimal policy (the
shaping potential is path-length, so it's a difference of potentials --
Ng et al. 1999 proved this preserves optimality for any γ ≤ 1).

```bash
python -m training.train_ppo --shape-progress 1.0 --log-dir runs/shaped
```

Tuning:
- `0.5` → mild nudge, optimal policy almost unchanged
- `1.0` → recommended starting point
- `2.0+` → strong drag toward target; can make the policy ignore other agents

Set to `0.0` to disable (default).

## Linear LR annealing

```bash
python -m training.train_ppo --anneal-lr --total-steps 5000000 --log-dir runs/anneal
```

Linearly decays the learning rate from `--lr` (default 3e-4) down to 0 over
the full `--total-steps` budget. Standard PPO practice. Helps prevent late
training from undoing learned behaviour with noisy gradient steps.

## Value loss clipping (auto-on)

Already enabled by default with `--value-clip-eps 0.2` in
`training/ppo.py:PPOConfig`. Clamps the value head's per-update delta the same
way the policy is clamped. No CLI flag — set `PPOConfig.value_clip_eps=0` in
code to disable.

## Recommended recipe for your next run

Given your previous run (`--scene scene_4 --line-length 3 --total-steps 2000000`)
and that you're CPU-only on a laptop, here's what I'd try next:

```bash
# Stage 1: warm up on an easier scene, lots of parallelism
python -m training.train_ppo \
  --scene scene_1 --line-length 2 \
  --num-envs 4 \
  --rollout-steps 4096 \
  --obs-version v2 \
  --shape-progress 1.0 \
  --anneal-lr \
  --total-steps 3000000 \
  --hidden 256 --epochs 4 \
  --log-dir runs/v2_stage1

# Stage 2: scale to your real target, resume from stage 1
python -m training.train_ppo \
  --scene scene_4 --line-length 3 \
  --num-envs 4 \
  --rollout-steps 4096 \
  --obs-version v2 \
  --shape-progress 0.5 \
  --anneal-lr \
  --resume runs/v2_stage1/checkpoints/best.pt \
  --total-steps 5000000 \
  --log-dir runs/v2_stage2
```

Why this order:
- Stage 1 learns basic "go to the goal" on the easiest available scene with
  strong shaping. This is fast because line-length-2 episodes are short.
- Stage 2 takes the warmed-up policy and trains on the actual target
  distribution, with weaker shaping so the policy learns to handle conflicts
  rather than blindly racing to its target.

The `--resume` only loads weights, not optimizer state, so the first updates
in stage 2 may look noisy. That's fine.

## Submission with v2

```bash
# After stage 2 finishes:
cp runs/v2_stage2/checkpoints/best.pt submission/checkpoint.pt

# Edit Dockerfile to use V2 obs builder:
# ENV OBS_BUILDER=my_orga.my_observation_builder.MyObservationBuilderV2
# (your folder may be submission/ rather than my_orga/ — adjust accordingly)

git add submission/checkpoint.pt Dockerfile
git commit -m "v2 obs + shaping + parallel envs"
git push
# Then trigger the docker workflow as before.
```

## Things I deliberately did *not* add (and why)

- **LSTM/GRU policy.** Real benefit (memory across steps), but slows training
  by ~2x and makes hyperparameter tuning much harder. Wait until you've
  saturated the feedforward version first.
- **Graph neural network observation.** Major redesign. Worth it for the
  serious entries to the competition, not for an incremental improvement.
- **Action masking at switches.** Only some actions are valid at certain cells.
  Would speed up convergence, but the env already handles invalid actions
  internally, so the cost is "slower learning" rather than "broken training".
  Defer until you've squeezed everything out of the dense reward.
- **MAPPO (centralized critic).** Real gains in MARL theory, but requires a
  separate critic network seeing all agents' obs simultaneously. Significant
  code complexity for unclear gain on a 6-agent problem.

## Files changed in this update

```
my_orga/my_observation_builder.py    # + MyObservationBuilderV2 class
training/ppo.py                       # + value loss clipping
training/rollout.py                   # + progress reward shaping
training/env_factory.py               # + obs_version parameter
training/train_ppo.py                 # + new CLI flags + parallel-collector path
training/parallel_rollout.py          # NEW: ParallelRolloutCollector
```

No changes needed in `my_orga/my_policy.py` or `my_orga/model.py` — the policy
just sees whatever vector the obs builder produces, and the model's
`obs_dim` already came from the config.
