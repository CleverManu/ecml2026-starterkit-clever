"""
Proximal Policy Optimization update step.

Implements the clipped surrogate objective from
"Proximal Policy Optimization Algorithms" (Schulman et al., 2017).

Operates on a flat batch of (obs, action, log_prob, value, advantage, return)
samples assembled by :class:`training.rollout.RolloutCollector`. Each sample is
treated independently, which is exactly what we want for a parameter-shared
multi-agent policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from submission.model import ActorCritic
from training.rollout import Batch


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2  # Clip value updates the same way the policy is clipped.
                                 # Set to 0 to disable.
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 256
    normalize_advantages: bool = True


class PPOTrainer:
    def __init__(self, model: ActorCritic, cfg: PPOConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def update(self, batch: Batch) -> Dict[str, float]:
        """Run ``epochs`` passes over the batch and return averaged metrics."""
        if len(batch) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                    "approx_kl": 0.0, "clip_frac": 0.0, "n_samples": 0}

        advantages = batch.advantages
        if self.cfg.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(batch)
        indices = np.arange(n)

        metrics: Dict[str, List[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "approx_kl": [], "clip_frac": [],
        }

        for _ in range(self.cfg.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = indices[start:start + self.cfg.minibatch_size]
                mb_obs = batch.obs[mb]
                mb_act = batch.actions[mb]
                mb_logp_old = batch.log_probs[mb]
                mb_val_old = batch.values[mb]
                mb_adv = advantages[mb]
                mb_ret = batch.returns[mb]
                mb_mask = batch.action_masks[mb] if batch.action_masks is not None else None

                logp, entropy, value = self.model.evaluate(mb_obs, mb_act, action_mask=mb_mask)

                ratio = torch.exp(logp - mb_logp_old)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps,
                                      1 + self.cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value clipping: prevent the value function from changing too fast
                # in any single update step. Standard PPO trick.
                if self.cfg.value_clip_eps > 0:
                    v_clipped = mb_val_old + torch.clamp(
                        value - mb_val_old,
                        -self.cfg.value_clip_eps, self.cfg.value_clip_eps,
                    )
                    v_loss_unclipped = (value - mb_ret).pow(2)
                    v_loss_clipped = (v_clipped - mb_ret).pow(2)
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (value - mb_ret).pow(2).mean()

                entropy_mean = entropy.mean()

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_logp_old - logp).mean().item()
                    clip_frac = (
                        (ratio - 1).abs() > self.cfg.clip_eps
                    ).float().mean().item()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy_mean.item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_frac"].append(clip_frac)

        return {
            k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()
        } | {"n_samples": n}
