from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def _init_layer(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class RecurrentActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            _init_layer(nn.Linear(observation_dim, hidden_size), gain=2**0.5),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, hidden_size), gain=2**0.5),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.constant_(parameter, 0.0)
        self.mean = _init_layer(nn.Linear(hidden_size, action_dim), gain=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def _distribution(
        self, observation: torch.Tensor, hidden: torch.Tensor, mask: torch.Tensor
    ) -> tuple[Normal, torch.Tensor]:
        hidden = hidden * mask
        encoded = self.encoder(observation)
        next_hidden = self.gru(encoded, hidden)
        mean = self.mean(next_hidden)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std), next_hidden

    def act(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, next_hidden = self._distribution(observation, hidden, mask)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        log_probability = distribution.log_prob(raw_action) - torch.log(
            1.0 - action.pow(2) + 1e-6
        )
        return action, log_probability.sum(-1), next_hidden

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, next_hidden = self._distribution(observation, hidden, mask)
        clipped = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_action = torch.atanh(clipped)
        log_probability = distribution.log_prob(raw_action) - torch.log(
            1.0 - clipped.pow(2) + 1e-6
        )
        entropy = distribution.entropy().sum(-1)
        return log_probability.sum(-1), entropy, next_hidden


class RecurrentCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            _init_layer(nn.Linear(state_dim, hidden_size), gain=2**0.5),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, hidden_size), gain=2**0.5),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.constant_(parameter, 0.0)
        self.value = _init_layer(nn.Linear(hidden_size, 1), gain=1.0)

    def forward(
        self, state: torch.Tensor, hidden: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden * mask
        encoded = self.encoder(state)
        next_hidden = self.gru(encoded, hidden)
        return self.value(next_hidden).squeeze(-1), next_hidden
