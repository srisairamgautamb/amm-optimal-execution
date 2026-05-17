"""PPO with optional CVaR penalty.

R-U dual: CVaR_alpha(L) = min_t [t + 1/(1-alpha) E[(L-t)^+]].
Refs: Rockafellar-Uryasev (2000), Chow-Ghavamzadeh (2014).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.amm import AMMEnv
from agent.ppo.buffer import RolloutBuffer, compute_gae, episode_returns_from_rewards
from agent.ppo.policy import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    obs_dim: int = 4
    act_dim: int = 1
    lr_policy: float = 3e-4
    lr_t: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    n_epochs: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    cvar_alpha: float = 0.95
    cvar_lambda: float = 0.0
    use_cvar_critic: bool = False
    device: str = "cpu"


class CVaRPPO:
    def __init__(self, config: PPOConfig) -> None:
        if not (0.5 < config.cvar_alpha < 1.0):
            raise ValueError(
                f"cvar_alpha must be in (0.5, 1.0), got {config.cvar_alpha}"
            )
        if config.cvar_lambda < 0:
            raise ValueError(f"cvar_lambda must be >= 0, got {config.cvar_lambda}")
        self.config = config
        self.device = torch.device(config.device)
        self.policy = ActorCritic(
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            with_cvar_head=config.use_cvar_critic,
        ).to(self.device)
        self._cvar_t = nn.Parameter(torch.tensor(0.0, device=self.device))
        self._opt_policy = optim.Adam(self.policy.parameters(), lr=config.lr_policy)
        self._opt_t = optim.Adam([self._cvar_t], lr=config.lr_t)

    def _to_tensor(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    @torch.no_grad()
    def collect_rollout(
        self,
        env: AMMEnv,
        n_steps: int,
        *,
        rng_seed: int,
    ) -> RolloutBuffer:
        use_cvar_critic = self.config.use_cvar_critic
        obs_buf, raw_act_buf, lp_buf, r_buf, v_buf, vc_buf, d_buf = (
            [], [], [], [], [], [], []
        )
        obs, _ = env.reset(seed=rng_seed)
        episode_seed = rng_seed
        for _ in range(n_steps):
            obs_t = self._to_tensor(obs).unsqueeze(0)
            out = self.policy(obs_t)
            if use_cvar_critic:
                dist, value, value_cvar = out
            else:
                dist, value = out
                value_cvar = torch.zeros_like(value)
            raw_action = dist.sample()
            log_prob = ActorCritic.log_prob_squashed(dist, raw_action)
            squashed = ActorCritic.squash(raw_action).squeeze(0).cpu().numpy()
            obs_next, r, term, trunc, _ = env.step(squashed)
            done = bool(term or trunc)
            obs_buf.append(obs)
            raw_act_buf.append(raw_action.squeeze(0).cpu().numpy())
            lp_buf.append(float(log_prob.item()))
            r_buf.append(float(r))
            v_buf.append(float(value.item()))
            vc_buf.append(float(value_cvar.item()))
            d_buf.append(1.0 if done else 0.0)
            if done:
                episode_seed += 1
                obs, _ = env.reset(seed=episode_seed)
            else:
                obs = obs_next
        obs_t = self._to_tensor(obs).unsqueeze(0)
        out = self.policy(obs_t)
        if use_cvar_critic:
            _, last_value, last_value_cvar = out
        else:
            _, last_value = out
            last_value_cvar = torch.zeros_like(last_value)
        last_value_f = float(last_value.item())
        last_value_cvar_f = float(last_value_cvar.item())

        rewards = self._to_tensor(r_buf)
        values = self._to_tensor(v_buf)
        values_cvar = self._to_tensor(vc_buf)
        dones = self._to_tensor(d_buf)
        advantages, returns = compute_gae(
            rewards, values, dones,
            self.config.gamma, self.config.gae_lambda, last_value_f,
        )
        ep_returns = episode_returns_from_rewards(rewards, dones)

        cvar_targets = torch.zeros_like(rewards)
        advantages_cvar = torch.zeros_like(rewards)
        if use_cvar_critic and ep_returns.numel() > 0:
            from agent.ppo.buffer import step_episode_ids
            losses = -ep_returns
            sorted_losses, _ = torch.sort(losses)
            cutoff = int(math.ceil(self.config.cvar_alpha * len(losses)))
            cutoff = min(max(cutoff, 1), len(losses))
            var_thresh = float(sorted_losses[cutoff - 1].item())
            mean_ret = float(ep_returns.mean().item())
            ep_ids = step_episode_ids(dones)
            step_ep_loss = losses[ep_ids]
            step_ep_return = ep_returns[ep_ids]
            in_tail = (step_ep_loss > var_thresh).float()
            cvar_targets = in_tail * step_ep_return + (1.0 - in_tail) * mean_ret
            advantages_cvar, _ = compute_gae(
                rewards, values_cvar, dones,
                self.config.gamma, self.config.gae_lambda, last_value_cvar_f,
            )

        return RolloutBuffer(
            obs=self._to_tensor(np.array(obs_buf)),
            raw_actions=self._to_tensor(np.array(raw_act_buf)),
            log_probs=self._to_tensor(lp_buf),
            rewards=rewards,
            values=values,
            dones=dones,
            returns=returns,
            advantages=advantages,
            episode_returns=ep_returns,
            values_cvar=values_cvar,
            cvar_targets=cvar_targets,
            advantages_cvar=advantages_cvar,
        )

    def _cvar_step_weight(self, buffer: RolloutBuffer) -> Tuple[torch.Tensor, float]:
        T = buffer.rewards.shape[0]
        device = self.device
        if (self.config.cvar_lambda <= 0.0
                or buffer.episode_returns.numel() == 0):
            return torch.ones(T, device=device), 0.0

        losses = -buffer.episode_returns.to(device)
        n_eps = losses.shape[0]
        cutoff = int(math.ceil(self.config.cvar_alpha * n_eps))
        cutoff = min(max(cutoff, 1), n_eps)
        sorted_losses, _ = torch.sort(losses)
        var_threshold = float(sorted_losses[cutoff - 1].item())

        ep_id = torch.zeros(T, dtype=torch.long, device=device)
        cur = 0
        dones_list = buffer.dones.tolist()
        for i in range(T):
            ep_id[i] = cur
            if dones_list[i] > 0.5:
                cur = min(cur + 1, n_eps - 1)
        step_loss = losses[ep_id]
        in_tail = (step_loss > var_threshold).float()
        weight = 1.0 + self.config.cvar_lambda * (1.0 / (1.0 - self.config.cvar_alpha)) * in_tail
        return weight, var_threshold

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        T = buffer.rewards.shape[0]
        use_cvar_critic = self.config.use_cvar_critic
        ret = buffer.returns
        old_log = buffer.log_probs
        obs = buffer.obs
        raw_act = buffer.raw_actions

        cvar_weight, var_threshold = self._cvar_step_weight(buffer)
        cvar_weight_max = float(cvar_weight.max().item())
        if use_cvar_critic and self.config.cvar_lambda > 0.0:
            mix = self.config.cvar_lambda / (1.0 + self.config.cvar_lambda)
            base_blend = (1.0 - mix) * buffer.advantages + mix * buffer.advantages_cvar
            adv_blend = base_blend * cvar_weight
        else:
            adv_blend = buffer.advantages * cvar_weight
        adv = (adv_blend - adv_blend.mean()) / (adv_blend.std() + 1e-8)

        metrics: Dict[str, float] = {
            "policy_loss": 0.0, "value_loss": 0.0,
            "value_loss_cvar": 0.0,
            "entropy": 0.0, "cvar_penalty": 0.0,
            "cvar_var_threshold": var_threshold,
            "cvar_weight_max": cvar_weight_max,
        }
        n_batches = 0

        idx = np.arange(T)
        for _ in range(self.config.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, T, self.config.minibatch_size):
                mb = idx[start:start + self.config.minibatch_size]
                if len(mb) < 2:
                    continue
                mb_t = torch.as_tensor(mb, device=self.device)

                out = self.policy(obs[mb_t])
                if use_cvar_critic:
                    dist, value, value_cvar = out
                else:
                    dist, value = out
                    value_cvar = None
                new_log = ActorCritic.log_prob_squashed(dist, raw_act[mb_t])
                ratio = torch.exp(new_log - old_log[mb_t])
                surr1 = ratio * adv[mb_t]
                surr2 = torch.clamp(
                    ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio,
                ) * adv[mb_t]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (value - ret[mb_t]).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()

                value_loss_cvar_term = torch.tensor(0.0, device=self.device)
                if use_cvar_critic and value_cvar is not None:
                    value_loss_cvar_term = (
                        value_cvar - buffer.cvar_targets[mb_t]
                    ).pow(2).mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.value_coef * value_loss_cvar_term
                    - self.config.entropy_coef * entropy
                )
                self._opt_policy.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm,
                )
                self._opt_policy.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["value_loss_cvar"] += float(value_loss_cvar_term.item())
                metrics["entropy"] += float(entropy.item())
                n_batches += 1

        if self.config.cvar_lambda > 0.0 and buffer.episode_returns.numel() > 0:
            losses = -buffer.episode_returns.to(self.device)
            for _ in range(self.config.n_epochs):
                self._opt_t.zero_grad()
                relu = torch.clamp(losses - self._cvar_t, min=0.0)
                cvar_est = self._cvar_t + (1.0 / (1.0 - self.config.cvar_alpha)) * relu.mean()
                t_loss = cvar_est
                t_loss.backward()
                self._opt_t.step()
            metrics["cvar_penalty"] = float(
                self.config.cvar_lambda * cvar_est.detach().item()
            )

        if n_batches > 0:
            for k in ("policy_loss", "value_loss", "value_loss_cvar", "entropy"):
                metrics[k] /= n_batches
        metrics["mean_return"] = float(buffer.episode_returns.mean().item())
        metrics["cvar_t"] = float(self._cvar_t.detach().item())
        if buffer.episode_returns.numel() >= 2:
            alpha = self.config.cvar_alpha
            losses_np = -buffer.episode_returns.numpy()
            losses_np.sort()
            cutoff = int(math.ceil(alpha * len(losses_np)))
            tail = losses_np[cutoff:] if cutoff < len(losses_np) else losses_np[-1:]
            metrics["cvar_return"] = float(np.mean(tail))
        else:
            metrics["cvar_return"] = float("nan")
        return metrics

    @torch.no_grad()
    def predict(self, obs: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        obs_t = self._to_tensor(obs).unsqueeze(0)
        out = self.policy(obs_t)
        dist = out[0]
        if deterministic:
            raw = dist.mean
        else:
            raw = dist.sample()
        action = ActorCritic.squash(raw).squeeze(0).cpu().numpy()
        return action.astype(np.float64)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "cvar_t": self._cvar_t.detach().cpu(),
                "config": self.config.__dict__,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "CVaRPPO":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = PPOConfig(**ckpt["config"])
        agent = cls(cfg)
        agent.policy.load_state_dict(ckpt["policy_state"])
        with torch.no_grad():
            agent._cvar_t.copy_(torch.as_tensor(ckpt["cvar_t"], device=agent.device))
        return agent
