"""Independent deterministic actors and centralized Q networks."""
import torch
from torch import nn


def mlp(inputs, outputs, hidden):
    network = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(),
                            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return network


class MADDPGActor(nn.Module):
    def __init__(self, observation_dim=68, hidden_size=64):
        super().__init__()
        self.network = mlp(observation_dim, 2, hidden_size)

    def pre_tanh(self, observation):
        return self.network(observation)

    def forward(self, observation):
        return torch.tanh(self.pre_tanh(observation))


class MADDPGCritic(nn.Module):
    def __init__(self, state_dim=60, n_agents=5, hidden_size=64):
        super().__init__()
        self.network = mlp(state_dim + n_agents * 2, 1, hidden_size)

    def forward(self, state, joint_actions):
        return self.network(torch.cat([state, joint_actions.flatten(start_dim=1)], dim=-1)).squeeze(-1)
