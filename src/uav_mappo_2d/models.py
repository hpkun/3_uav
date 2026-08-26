from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def _init_layer(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class GaussianActor(nn.Module):
    def __init__(self, observation_dim: int = 40, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _init_layer(nn.Linear(observation_dim, hidden_size), 2**0.5),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, hidden_size), 2**0.5),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, 2), 0.01),
        )
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.network(observations)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _log_probability(
        distribution: Normal, raw_actions: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        correction = torch.log(1.0 - actions.square() + 1e-6)
        return (distribution.log_prob(raw_actions) - correction).sum(dim=-1)

    def sample(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        raw_actions = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw_actions)
        return actions, self._log_probability(distribution, raw_actions, actions)

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bounded = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_actions = torch.atanh(bounded)
        distribution = self.distribution(observations)
        log_probability = self._log_probability(distribution, raw_actions, bounded)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probability, entropy


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim: int = 36, hidden_size: int = 128) -> None:
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
