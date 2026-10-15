from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


def _init_layer(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class CategoricalActor(nn.Module):
    def __init__(self, observation_dim: int, hidden_size: int = 128, n_actions: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _init_layer(nn.Linear(observation_dim, hidden_size), 2**0.5),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, hidden_size), 2**0.5),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, n_actions), 0.01),
        )
    def distribution(self, observations: torch.Tensor, action_masks: torch.Tensor) -> Categorical:
        logits = self.network(observations)
        action_masks = action_masks.bool()
        if action_masks.shape != logits.shape or not action_masks.any(dim=-1).all():
            raise ValueError("Each categorical action mask must match logits and allow an action")
        logits = logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min)
        return Categorical(logits=logits)

    def sample(
        self, observations: torch.Tensor, action_masks: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations, action_masks)
        actions = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions)

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor, action_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations, action_masks)
        return distribution.log_prob(actions.long()), distribution.entropy()


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _init_layer(nn.Linear(state_dim, hidden_size), 2**0.5),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, hidden_size), 2**0.5),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, 1), 1.0),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)
