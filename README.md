# PPO baseline for the ECML 2026 Flatland Competition

A parameter-sharing Proximal Policy Optimization (PPO) agent that trains on the
official competition topology by re-sampling lines and timetables at every
episode. CPU-trainable on a laptop; ships as a Docker image via the starter-kit
GitHub Action.

## Repository layout

```
submission/                          # Submission code (copied into Docker image)
  __init__.py
  my_observation_builder.py       # Flattened+normalized TreeObs (depth=2, 252 features)
  my_policy.py                    # Policy.act_many implementation that loads checkpoint.pt
  model.py                        # Actor-critic network + checkpoint utilities
  requirements.txt                # torch (numpy comes from base image)
  checkpoint.pt                   # YOU PROVIDE THIS after training

Dockerfile                        # Builds the submission image

training/                         # Training pipeline (NOT shipped in Docker)
  train_ppo.py                    # Main training script
  ppo.py                          # PPO loss, clipped surrogate, value clipping
  rollout.py                      # Multi-agent rollout collector with GAE
  env_factory.py                  # Builds the training env with the sampler
  evaluate.py                     # Evaluation against debug-environments.zip
  requirements.txt                # torch, tensorboard, tqdm
  sampling/                       # Vendored from PR #7
    __init__.py
    sampling_env_generator.py
    stations.pkl                  # Competition topology stations
    level_0_scenario_1.pkl        # Competition grid snapshot

README.md                         # (this file)
```

## Approach in one paragraph

Every agent feeds a normalized, flattened depth-2 tree observation (`252` floats)
into a shared two-layer MLP (`252 -> 256 -> 256`) that outputs a `5`-way action
distribution and a scalar value. The same network is reused for every agent on
every step (parameter sharing), which is the standard MARL pattern for
homogeneous agents in Flatland. We optimize the policy with PPO (clipped
surrogate, GAE, advantage normalization) using rollouts collected from a single
env. The training env is the competition grid (loaded from
`level_0_scenario_1.pkl`) wrapped by the sampler from PR #7, so every
`env.reset()` produces fresh lines and timetables on the same fixed topology --
that gives us essentially unlimited training tasks on the right distribution.

Reward shaping: none, beyond a constant scale (`/100`) so the value head's
targets stay numerically small. ECML2026Rewards has large terminal penalties
(collision = -250, no-arrival = -100) which would otherwise dwarf the policy
gradient.

## Quick start

### 1. Install dependencies

```bash
# Submission-side (also enough to use the trained policy locally)
pip install -r submission/requirements.txt
pip install flatland-rl>=4.2.5

# Training-side (additionally)
pip install -r training/requirements.txt
```

### 2. Train

```bash
# Smoke run: ~5 minutes, just to verify everything wires up
python -m training.train_ppo \
  --total-steps 20000 --rollout-steps 1024 --epochs 2 \
  --hidden 128 --log-dir runs/smoke

# Full CPU baseline: overnight on a laptop
python -m training.train_ppo \
  --total-steps 5000000 --rollout-steps 1024 \
  --epochs 4 --hidden 256 --log-dir runs/baseline
```

Useful flags:
- `--scene scene_1` ... `scene_5` -- restrict to a station subset
- `--line-length 3` -- include trips with one intermediate stop
- `--resume runs/baseline/checkpoints/latest.pt` -- continue training
- `--reward-scale 100` -- divisor for raw rewards (default 100)

Checkpoints land in `<log-dir>/checkpoints/`. Three are written:
- `ckpt_upd00010.pt`, `ckpt_upd00020.pt`, ... -- one per `--checkpoint-every` updates
- `latest.pt` -- always points at the most recent
- `best.pt` -- highest rollout success rate seen so far
- `final.pt` -- written when training exits cleanly

If TensorBoard is installed, run `tensorboard --logdir runs/` to see live curves.

### 3. Evaluate locally (optional but recommended)

```bash
# Fetch the official debug test set
wget https://data.flatland.cloud/benchmarks/Flatland3/debug-environments.zip
unzip debug-environments.zip -d scenarios/

python -m training.evaluate \
  --checkpoint runs/baseline/checkpoints/best.pt \
  --scenarios scenarios/debug-environments \
  --csv eval_results.csv
```

Prints per-episode success rate + normalized reward, matching the metrics the
competition evaluator reports.

### 4. Prepare for submission

```bash
# Copy the checkpoint you want to submit into the submission folder.
cp runs/baseline/checkpoints/best.pt submission/checkpoint.pt
```

That's it -- `submission/checkpoint.pt` is the only artefact the policy needs at
inference time. The Docker image picks it up automatically.

### 5. Build & test the Docker image locally (optional)

```bash
docker build -t myorga/mysolution -f Dockerfile .

# Single-episode smoke test
docker run myorga/mysolution flatland-trajectory-generate-from-policy \
  --data-dir /tmp

# Full debug-set evaluation
docker run -v $PWD/scenarios/debug-environments/:/inputs \
  myorga/mysolution flatland-trajectory-generate-from-metadata \
  --metadata-csv /inputs/metadata.csv --data-dir /tmp
```

### 6. Submit to the competition

The starter-kit ships a `docker` GitHub Action that builds and pushes the image
to GHCR. Workflow:

1. Fork [`flatland-association/ecml2026-starterkit`](https://github.com/flatland-association/ecml2026-starterkit).
2. Drop the contents of this repo into the fork (replacing `submission/` and the
   `Dockerfile`). Train and copy `checkpoint.pt` into `submission/` as above, then
   commit and push.
3. Trigger the `docker` workflow under your fork's **Actions** tab.
4. After it succeeds, copy the image URL from
   `ghcr.io/<your-user>/<your-fork>:latest`.
5. Paste the URL into a new submission at
   <https://competition.flatland.cloud/>.

If your fork is private, grant the Flatland Competition account access to the
package under the repo's package settings.

## Design choices

**Observation builder (`MyObservationBuilder`).** Subclasses
`TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(max_depth=30))`.
A depth-2 tree gives `1 + 4 + 16 = 21` nodes and `12` features per node, for a
flat vector of `252` floats. Each node is split into three feature groups
(distance-to-event, min-distance-to-target, agent-counts) and normalized
independently before clipping to `[-1, 1]`. We did not pull in
`flatland.ml.observations` because it depends on `ray`, which isn't always
available in lightweight base images.

**Policy network.** Two-hidden-layer MLP with Tanh activations, orthogonal
init, separate policy and value heads. Defaults to width `256` which gives
about `140 k` parameters -- small enough to fit thousands of forward passes per
second on CPU.

**Parameter sharing.** A single network produces decisions for all `n` agents
simultaneously per step. Each `(agent, decision_step)` pair is treated as an
independent sample in the PPO update. This is the canonical MARL recipe for
homogeneous agents and avoids the credit-assignment headaches of fully
centralized methods.

**Decision-only sampling.** Flatland reports `info["action_required"][handle]`,
which is `True` only at switch cells where the action actually matters. We only
ask the policy for an action at those cells (otherwise we send
`DO_NOTHING`), which cuts per-step compute and gives cleaner training data --
the policy never has to learn that "between decisions, anything works".

**Reward credit assignment.** Rewards arriving between decisions are
accumulated onto the agent's *most recent* decision. This is the standard
"options-style" credit assignment for sparse decision-cell control.

**GAE per agent.** Each agent has its own trajectory of decisions and gets its
own GAE pass (lambda = 0.95, gamma = 0.99). Bootstrap value is `0` for
finished episodes and the network's value estimate of the latest obs otherwise.

**Sampler (PR #7).** Patches the env's `rail_generator`, `line_generator`, and
`timetable_generator` so that every `env.reset()` produces a fresh task on the
fixed competition grid. This is the only sane way to train -- without it we'd
overfit one scenario instantly.

## Hyperparameter notes

The defaults are tuned for CPU laptop training, not for benchmark-leading
performance. If you have a GPU (or just patience):
- Bump `--hidden` to `512` and `--epochs` to `8`.
- Increase `--rollout-steps` to `4096` for better advantage estimates.
- Try `--gamma 0.995` for a longer-horizon credit window (max_episode_steps is
  often a few hundred).
- Curriculum: start with `--line-length 2 --scene scene_1` for a small station
  set, then progressively widen to `--scene scene_3` and eventually
  `--scene scene_5` (all stations).

## Expected results

A random-init policy gets `0-20%` arrival rate on rollouts. With ~10M env steps
of training (rough overnight CPU budget), a PPO baseline of this size should
plateau around `30-60%` arrival rate depending on luck and the specific seeds
sampled. That's a reasonable submission floor; competitive entries will need
either much more compute or a smarter observation/architecture (graph nets,
attention, MAPPO, curriculum).

## Troubleshooting

**"checkpoint not found" warning at container start.** You forgot to copy
`checkpoint.pt` into `submission/`. The policy falls back to random weights so the
container still runs, but submissions will be useless.

**Docker build fails on "Cannot import submission.my_policy.MyPolicy".** Make sure
`submission/__init__.py` exists and that you didn't introduce a top-level import
that breaks under the container's Python version.

**Submission accepted but scores `0%`.** Check that the same observation builder
you trained with is the one specified in the `OBS_BUILDER` env var.

**Folder name confusion (`my_orga/` vs `submission/`).** The upstream
starter-kit has been mid-rename: the file tree in `main` says `my_orga/` but the
default `Dockerfile` says `submission/`. This repo uses `my_orga/` because that's
what's actually on disk in `main`. If the upstream standardizes on `submission/`,
just rename the folder and update the `POLICY`/`OBS_BUILDER` env vars in the
`Dockerfile`.

## License

MIT (matches the starter-kit).
