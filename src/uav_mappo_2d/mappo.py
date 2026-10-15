from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import MAPPOConfig
from .models import CentralizedCritic, GaussianActor


class RolloutBuffer:
    """Fixed [time, environment] on-policy rollout storage."""

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        num_agents: int,
        observation_dim: int,
        state_dim: int,
    ) -> None:
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        shape = (self.horizon, self.num_envs)
        self.observations = np.zeros(
            shape + (num_agents, observation_dim), dtype=np.float32
        )
        self.states = np.zeros(shape + (state_dim,), dtype=np.float32)
        self.actions = np.zeros(shape + (num_agents, 2), dtype=np.float32)
        self.old_log_probabilities = np.zeros(
            shape + (num_agents,), dtype=np.float32
        )
        self.values = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=np.float32)
        self.active_masks = np.zeros(shape + (num_agents,), dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.position = 0

    def insert(
        self,
        observations: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        log_probabilities: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        active_masks: np.ndarray,
    ) -> None:
        if self.position >= self.horizon:
            raise RuntimeError("rollout buffer is full")
        index = self.position
        self.observations[index] = observations
        self.states[index] = states
        self.actions[index] = actions
        self.old_log_probabilities[index] = log_probabilities
        self.values[index] = values
        self.rewards[index] = rewards
        self.dones[index] = dones
        self.active_masks[index] = active_masks
        self.position += 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if self.position != self.horizon:
            raise RuntimeError("GAE requires a complete rollout")
        next_values = np.asarray(last_values, dtype=np.float32)
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for timestep in reversed(range(self.horizon)):
            continuation = 1.0 - self.dones[timestep]
            delta = (
                self.rewards[timestep]
                + gamma * next_values * continuation
                - self.values[timestep]
            )
            gae = delta + gamma * gae_lambda * continuation * gae
            self.advantages[timestep] = gae
            next_values = self.values[timestep]
        self.returns[:] = self.advantages + self.values


class MAPPO:
    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        n_agents: int,
        config: MAPPOConfig,
        device: str | torch.device,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.rng = np.random.default_rng(seed)
        self.actor = GaussianActor(observation_dim, config.hidden_size).to(self.device)
        self.critic = CentralizedCritic(state_dim, config.hidden_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate, eps=1e-5
        )

    @torch.no_grad()
    def act(
        self,
        observations: np.ndarray,
        states: np.ndarray,
        active_masks: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations_t = torch.as_tensor(
            observations, device=self.device, dtype=torch.float32
        )
        states_t = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        masks_t = torch.as_tensor(
            active_masks, device=self.device, dtype=torch.float32
        )
        actions, log_probabilities = self.actor.sample(
            observations_t, deterministic=deterministic
        )
        actions = actions * masks_t[..., None]
        log_probabilities = log_probabilities * masks_t
        values = self.critic(states_t)
        return (
            actions.cpu().numpy(),
            log_probabilities.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def policy_actions(
        self,
        observations: np.ndarray,
        active_masks: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        observations_t = torch.as_tensor(
            observations, device=self.device, dtype=torch.float32
        )
        masks_t = torch.as_tensor(
            active_masks, device=self.device, dtype=torch.float32
        )
        actions, _ = self.actor.sample(observations_t, deterministic=deterministic)
        return (actions * masks_t[..., None]).cpu().numpy()

    @torch.no_grad()
    def values(self, states: np.ndarray) -> np.ndarray:
        states_t = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        return self.critic(states_t).cpu().numpy()

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        cfg = self.config
        to_tensor = lambda value: torch.as_tensor(
            value, device=self.device, dtype=torch.float32
        )
        observations = to_tensor(buffer.observations)
        states = to_tensor(buffer.states)
        actions = to_tensor(buffer.actions)
        old_log_probabilities = to_tensor(buffer.old_log_probabilities)
        old_values = to_tensor(buffer.values)
        returns = to_tensor(buffer.returns)
        active_masks = to_tensor(buffer.active_masks)
        team_advantages = to_tensor(buffer.advantages)

        actor_advantages = team_advantages[..., None].expand(-1, -1, self.n_agents)
        valid_advantages = actor_advantages[active_masks > 0.5]
        actor_advantages = (
            actor_advantages - valid_advantages.mean()
        ) / valid_advantages.std(unbiased=False).clamp_min(1e-8)

        transition_count = buffer.horizon * buffer.num_envs
        observations = observations.reshape(
            transition_count, self.n_agents, observations.shape[-1]
        )
        states = states.reshape(transition_count, states.shape[-1])
        actions = actions.reshape(transition_count, self.n_agents, 2)
        old_log_probabilities = old_log_probabilities.reshape(
            transition_count, self.n_agents
        )
        old_values = old_values.reshape(transition_count)
        returns = returns.reshape(transition_count)
        active_masks = active_masks.reshape(transition_count, self.n_agents)
        actor_advantages = actor_advantages.reshape(
            transition_count, self.n_agents
        )

        rows: list[dict[str, float]] = []
        for _ in range(cfg.update_epochs):
            order = self.rng.permutation(transition_count)
            for start in range(0, transition_count, cfg.minibatch_size):
                indices = torch.as_tensor(
                    order[start : start + cfg.minibatch_size],
                    device=self.device,
                    dtype=torch.long,
                )
                obs_batch = observations[indices]
                action_batch = actions[indices]
                old_log_batch = old_log_probabilities[indices]
                mask_batch = active_masks[indices]
                advantage_batch = actor_advantages[indices]

                new_log_probability, entropy = self.actor.evaluate_actions(
                    obs_batch, action_batch
                )
                log_ratio = new_log_probability - old_log_batch
                ratio = log_ratio.exp()
                unclipped = ratio * advantage_batch
                clipped = ratio.clamp(
                    1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio
                ) * advantage_batch
                actor_count = mask_batch.sum().clamp_min(1.0)
                actor_loss = -(
                    torch.minimum(unclipped, clipped) * mask_batch
                ).sum() / actor_count
                entropy_mean = (entropy * mask_batch).sum() / actor_count

                self.actor_optimizer.zero_grad(set_to_none=True)
                (actor_loss - cfg.entropy_coefficient * entropy_mean).backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), cfg.max_grad_norm
                )
                self.actor_optimizer.step()

                state_batch = states[indices]
                return_batch = returns[indices]
                old_value_batch = old_values[indices]
                values = self.critic(state_batch)
                clipped_values = old_value_batch + (values - old_value_batch).clamp(
                    -cfg.value_clip, cfg.value_clip
                )
                critic_loss = 0.5 * torch.maximum(
                    (values - return_batch).square(),
                    (clipped_values - return_batch).square(),
                ).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (cfg.value_coefficient * critic_loss).backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), cfg.max_grad_norm
                )
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (
                        ((ratio - 1.0) - log_ratio) * mask_batch
                    ).sum() / actor_count
                    clip_fraction = (
                        ((ratio - 1.0).abs() > cfg.clip_ratio).float() * mask_batch
                    ).sum() / actor_count
                rows.append(
                    {
                        "actor_loss": float(actor_loss.detach().cpu()),
                        "critic_loss": float(critic_loss.detach().cpu()),
                        "entropy": float(entropy_mean.detach().cpu()),
                        "approx_kl": float(approx_kl.detach().cpu()),
                        "clip_fraction": float(clip_fraction.detach().cpu()),
                        "actor_grad_norm": float(actor_grad_norm.detach().cpu()),
                        "critic_grad_norm": float(critic_grad_norm.detach().cpu()),
                    }
                )

        with torch.no_grad():
            new_values = self.critic(states)
            return_variance = torch.var(returns, unbiased=False)
            if float(return_variance) < 1e-8:
                explained_variance = torch.zeros((), device=self.device)
            else:
                explained_variance = 1.0 - torch.var(
                    returns - new_values, unbiased=False
                ) / return_variance
        metrics = {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
        }
        metrics["explained_variance"] = float(explained_variance.cpu())
        if not np.all(np.isfinite(list(metrics.values()))):
            raise FloatingPointError(f"Non-finite MAPPO metrics: {metrics}")
        return metrics

    def save(
        self,
        path: str | Path,
        *,
        global_step: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "global_step": int(global_step),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "extra": extra or {},
            },
            output,
        )

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        return checkpoint
