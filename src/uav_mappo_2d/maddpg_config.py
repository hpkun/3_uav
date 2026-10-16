"""Algorithm-only MADDPG reference profile; environment stays in base.yaml."""
from dataclasses import dataclass, fields
from pathlib import Path
import yaml


@dataclass
class MADDPGConfig:
    hidden_size: int = 64
    actor_learning_rate: float = .01
    critic_learning_rate: float = .01
    gamma: float = .95
    tau: float = .01
    buffer_capacity: int = 1_000_000
    batch_size: int = 1024
    update_interval_steps: int = 100
    max_grad_norm: float = .5
    actor_regularization: float = .001
    ou_mu: float = 0.
    ou_theta: float = .15
    ou_sigma: float = .20
    learning_starts: int | None = None

    def __post_init__(self):
        for name in ('hidden_size', 'buffer_capacity', 'batch_size', 'update_interval_steps'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.batch_size > self.buffer_capacity:
            raise ValueError('batch_size exceeds replay capacity')
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError('Invalid gamma/tau')
        if min(self.actor_learning_rate, self.critic_learning_rate, self.max_grad_norm) <= 0:
            raise ValueError('Learning rates and clipping threshold must be positive')
        if min(self.ou_theta, self.ou_sigma, self.actor_regularization) < 0:
            raise ValueError('Noise and regularization must be nonnegative')

    def warmup(self, episode_steps):
        value = self.batch_size * episode_steps if self.learning_starts is None else self.learning_starts
        if not isinstance(value, int) or not self.batch_size <= value <= self.buffer_capacity:
            raise ValueError('learning_starts must be between batch_size and replay capacity')
        return value


def load_maddpg_config(path: str | Path):
    with Path(path).open(encoding='utf-8') as stream:
        raw = yaml.safe_load(stream) or {}
    unknown = set(raw) - {f.name for f in fields(MADDPGConfig)}
    if unknown:
        raise ValueError(f'Unknown MADDPG configuration: {sorted(unknown)}')
    return MADDPGConfig(**raw)
