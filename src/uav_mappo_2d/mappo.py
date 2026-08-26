from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import MAPPOConfig
from .models import RecurrentActor, RecurrentCritic


@dataclass
class EpisodeBuffer:
    actor_hidden_size: int
    n_agents: int
    observations: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    log_probabilities: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)
    active_masks: list[np.ndarray] = field(default_factory=list)

    def append(
        self,
        *,
        observation: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        log_probability: np.ndarray,
        value: float,
        reward: float,
        done: bool,
        active_mask: np.ndarray,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.log_probabilities.append(np.asarray(log_probability, dtype=np.float32).copy())
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.active_masks.append(np.asarray(active_mask, dtype=np.float32).copy())

    def __len__(self) -> int:
        return len(self.rewards)


class MAPPO:
    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        n_agents: int,
        config: MAPPOConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.actor = RecurrentActor(observation_dim, 2, config.hidden_size).to(self.device)
        self.critic = RecurrentCritic(state_dim, config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
            eps=1e-5,
        )

    def initial_actor_hidden(self) -> np.ndarray:
        return np.zeros((self.n_agents, self.config.hidden_size), dtype=np.float32)

    def initial_critic_hidden(self) -> np.ndarray:
        return np.zeros((1, self.config.hidden_size), dtype=np.float32)

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        state: np.ndarray,
        actor_hidden: np.ndarray,
        critic_hidden: np.ndarray,
        active_mask: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        action, log_probability, value, next_actor_h, next_critic_h = self.act_batch(
            observation[None],
            state[None],
            actor_hidden[None],
            critic_hidden[None],
            active_mask[None],
            deterministic=deterministic,
        )
        return (
            action[0],
            log_probability[0],
            float(value[0]),
            next_actor_h[0],
            next_critic_h[0],
        )

    @torch.no_grad()
    def act_batch(
        self,
        observations: np.ndarray,
        states: np.ndarray,
        actor_hidden: np.ndarray,
        critic_hidden: np.ndarray,
        active_masks: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Infer actions and values for all environments in one device batch."""
        env_count, agent_count, observation_dim = observations.shape
        obs_t = torch.as_tensor(
            observations.reshape(env_count * agent_count, observation_dim),
            device=self.device,
            dtype=torch.float32,
        )
        state_t = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        actor_h = torch.as_tensor(
            actor_hidden.reshape(env_count * agent_count, -1),
            device=self.device,
            dtype=torch.float32,
        )
        critic_h = torch.as_tensor(
            critic_hidden[:, 0], device=self.device, dtype=torch.float32
        )
        mask_t = torch.as_tensor(
            active_masks.reshape(env_count * agent_count, 1),
            device=self.device,
            dtype=torch.float32,
        )
        action, log_probability, next_actor_h = self.actor.act(
            obs_t, actor_h, mask_t, deterministic=deterministic
        )
        value, next_critic_h = self.critic(
            state_t, critic_h, torch.ones((env_count, 1), device=self.device)
        )
        action = action * mask_t
        return (
            action.reshape(env_count, agent_count, -1).cpu().numpy(),
            log_probability.reshape(env_count, agent_count).cpu().numpy(),
            value.cpu().numpy(),
            next_actor_h.reshape(env_count, agent_count, -1).cpu().numpy(),
            next_critic_h[:, None].cpu().numpy(),
        )

    @torch.no_grad()
    def act_policy(
        self,
        observation: np.ndarray,
        actor_hidden: np.ndarray,
        active_mask: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        action, next_hidden = self.act_policy_batch(
            observation[None],
            actor_hidden[None],
            active_mask[None],
            deterministic=deterministic,
        )
        return action[0], next_hidden[0]

    @torch.no_grad()
    def act_policy_batch(
        self,
        observations: np.ndarray,
        actor_hidden: np.ndarray,
        active_masks: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        env_count, agent_count, observation_dim = observations.shape
        obs_t = torch.as_tensor(
            observations.reshape(env_count * agent_count, observation_dim),
            device=self.device,
            dtype=torch.float32,
        )
        hidden_t = torch.as_tensor(
            actor_hidden.reshape(env_count * agent_count, -1),
            device=self.device,
            dtype=torch.float32,
        )
        mask_t = torch.as_tensor(
            active_masks.reshape(env_count * agent_count, 1),
            device=self.device,
            dtype=torch.float32,
        )
        action, _, next_hidden = self.actor.act(
            obs_t, hidden_t, mask_t, deterministic=deterministic
        )
        return (
            (action * mask_t).reshape(env_count, agent_count, -1).cpu().numpy(),
            next_hidden.reshape(env_count, agent_count, -1).cpu().numpy(),
        )

    def update(self, buffer: EpisodeBuffer) -> dict[str, float]:
        return self.update_many([buffer])

    def update_many(self, buffers: list[EpisodeBuffer]) -> dict[str, float]:
        """Update from padded recurrent episodes batched across environments."""
        if not buffers or any(len(buffer) == 0 for buffer in buffers):
            raise ValueError("Cannot update from an empty episode buffer batch")
        cfg = self.config
        env_count = len(buffers)
        max_length = max(len(buffer) for buffer in buffers)
        observation_dim = buffers[0].observations[0].shape[-1]
        state_dim = buffers[0].states[0].shape[-1]
        action_dim = buffers[0].actions[0].shape[-1]

        observations_np = np.zeros(
            (max_length, env_count, self.n_agents, observation_dim), np.float32
        )
        states_np = np.zeros((max_length, env_count, state_dim), np.float32)
        actions_np = np.zeros(
            (max_length, env_count, self.n_agents, action_dim), np.float32
        )
        old_log_probabilities_np = np.zeros(
            (max_length, env_count, self.n_agents), np.float32
        )
        old_values_np = np.zeros((max_length, env_count), np.float32)
        active_masks_np = np.zeros(
            (max_length, env_count, self.n_agents), np.float32
        )
        advantages_np = np.zeros(
            (max_length, env_count, self.n_agents), np.float32
        )
        returns_np = np.zeros((max_length, env_count), np.float32)
        valid_steps_np = np.zeros((max_length, env_count), np.float32)

        for env_index, buffer in enumerate(buffers):
            length = len(buffer)
            advantages, returns = self._gae(buffer)
            observations_np[:length, env_index] = np.stack(buffer.observations)
            states_np[:length, env_index] = np.stack(buffer.states)
            actions_np[:length, env_index] = np.stack(buffer.actions)
            old_log_probabilities_np[:length, env_index] = np.stack(
                buffer.log_probabilities
            )
            old_values_np[:length, env_index] = buffer.values
            active_masks_np[:length, env_index] = np.stack(buffer.active_masks)
            advantages_np[:length, env_index] = advantages[:, None]
            returns_np[:length, env_index] = returns
            valid_steps_np[:length, env_index] = 1.0

        to_tensor = lambda value: torch.as_tensor(
            value, device=self.device, dtype=torch.float32
        )
        observations = to_tensor(observations_np)
        states = to_tensor(states_np)
        actions = to_tensor(actions_np)
        old_log_probabilities = to_tensor(old_log_probabilities_np)
        old_values = to_tensor(old_values_np)
        active_masks = to_tensor(active_masks_np)
        advantages = to_tensor(advantages_np)
        returns = to_tensor(returns_np)
        valid_steps = to_tensor(valid_steps_np)

        all_valid_advantages = advantages[active_masks > 0.0]
        advantage_mean = all_valid_advantages.mean()
        advantage_std = all_valid_advantages.std(unbiased=False)
        advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)
        actor_count = active_masks.sum().clamp_min(1.0)
        value_count = valid_steps.sum().clamp_min(1.0)

        metrics: dict[str, float] = {}
        for _ in range(cfg.update_epochs):
            actor_hidden = torch.zeros(
                (env_count, self.n_agents, cfg.hidden_size), device=self.device
            )
            critic_hidden = torch.zeros(
                (env_count, cfg.hidden_size), device=self.device
            )
            new_log_probabilities = []
            entropies = []
            new_values = []
            for timestep in range(max_length):
                actor_mask = active_masks[timestep, :, :, None]
                log_probability, entropy, next_actor_hidden = (
                    self.actor.evaluate_actions(
                        observations[timestep].reshape(
                            env_count * self.n_agents, observation_dim
                        ),
                        actions[timestep].reshape(
                            env_count * self.n_agents, action_dim
                        ),
                        actor_hidden.reshape(
                            env_count * self.n_agents, cfg.hidden_size
                        ),
                        actor_mask.reshape(env_count * self.n_agents, 1),
                    )
                )
                actor_hidden = next_actor_hidden.reshape(
                    env_count, self.n_agents, cfg.hidden_size
                )
                value, critic_hidden = self.critic(
                    states[timestep],
                    critic_hidden,
                    valid_steps[timestep, :, None],
                )
                new_log_probabilities.append(
                    log_probability.reshape(env_count, self.n_agents)
                )
                entropies.append(entropy.reshape(env_count, self.n_agents))
                new_values.append(value)

            new_log_probabilities_t = torch.stack(new_log_probabilities)
            entropies_t = torch.stack(entropies)
            new_values_t = torch.stack(new_values)
            ratio = torch.exp(new_log_probabilities_t - old_log_probabilities)
            unclipped = ratio * advantages
            clipped = torch.clamp(
                ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio
            ) * advantages
            policy_loss = -(
                torch.minimum(unclipped, clipped) * active_masks
            ).sum() / actor_count
            entropy = (entropies_t * active_masks).sum() / actor_count
            approximate_kl = (
                (old_log_probabilities - new_log_probabilities_t) * active_masks
            ).sum() / actor_count

            value_clipped = old_values + (new_values_t - old_values).clamp(
                -cfg.value_clip, cfg.value_clip
            )
            value_loss_unclipped = (new_values_t - returns).pow(2)
            value_loss_clipped = (value_clipped - returns).pow(2)
            value_loss = 0.5 * (
                torch.maximum(value_loss_unclipped, value_loss_clipped)
                * valid_steps
            ).sum() / value_count

            total_loss = (
                policy_loss
                + cfg.value_coefficient * value_loss
                - cfg.entropy_coefficient * entropy
            )
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                cfg.max_grad_norm,
            )
            self.optimizer.step()
            metrics = {
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "approx_kl": float(approximate_kl.detach().cpu()),
                "episodes_per_update": float(len(buffers)),
            }
        return metrics

    def _gae(self, buffer: EpisodeBuffer) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        values = np.asarray(buffer.values, dtype=np.float32)
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for timestep in reversed(range(len(buffer))):
            continuation = 1.0 - dones[timestep]
            delta = (
                rewards[timestep]
                + cfg.gamma * next_value * continuation
                - values[timestep]
            )
            gae = delta + cfg.gamma * cfg.gae_lambda * continuation * gae
            advantages[timestep] = gae
            next_value = values[timestep]
        return advantages, advantages + values

    def actor_snapshot(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (key, value.detach().cpu().clone()) for key, value in self.actor.state_dict().items()
        )

    def load_actor_snapshot(self, state: OrderedDict[str, torch.Tensor]) -> None:
        self.actor.load_state_dict(state)

    def save(self, path: str | Path, *, episode: int, extra: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "episode": episode,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "extra": extra or {},
            },
            output,
        )

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint
