from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class EnvironmentConfig:
    map_size: float = 10_000.0
    dt: float = 1.0
    episode_steps: int = 300
    speed_min: float = 50.0
    speed_max: float = 150.0
    initial_speed: float = 100.0
    spawn_x: float = 3_500.0
    spawn_y_spacing: float = 1_500.0
    spawn_jitter: float = 200.0
    attack_range: float = 600.0
    radar_range: float = 2_000.0
    attack_ata_degrees: float = 15.0
    attack_aspect_degrees: float = 50.0
    collision_distance: float = 150.0
    collision_avoidance_distance: float = 450.0
    collision_reward_distance: float = 600.0


@dataclass
class RewardConfig:
    all_enemy_destroyed_reward: float = 100.0
    timeout_advantage_reward: float = 80.0
    enemy_kill_reward: float = 20.0
    all_own_destroyed_penalty: float = 100.0
    timeout_disadvantage_penalty: float = 80.0
    own_loss_penalty: float = 20.0


@dataclass
class RenderConfig:
    width: int = 800
    height: int = 800
    margin: int = 45
    show_trails: bool = True
    trail_length: int = 80
    icon_source: str = "对抗环境图.png"


@dataclass
class MAPPOConfig:
    hidden_size: int = 128
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 512


@dataclass
class TrainingConfig:
    blue_strategy: int = 1
    total_env_steps: int = 2_000_000
    parallel_envs: int = 16
    rollout_length: int = 128
    log_interval_steps: int = 20_000
    checkpoint_interval_steps: int = 100_000
    evaluation_interval_steps: int = 100_000
    evaluation_episodes: int = 100
    output_dir: str = "runs/xu_course1_seed7"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.blue_strategy not in (1, 2, 3):
            raise ValueError("blue_strategy must be 1, 2, or 3")


@dataclass
class ProjectConfig:
    seed: int = 7
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any] | None) -> T:
    values = values or {}
    valid = {item.name for item in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    valid = {item.name for item in fields(ProjectConfig)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown ProjectConfig fields: {sorted(unknown)}")
    return ProjectConfig(
        seed=int(raw.get("seed", 7)),
        environment=_construct(EnvironmentConfig, raw.get("environment")),
        reward=_construct(RewardConfig, raw.get("reward")),
        render=_construct(RenderConfig, raw.get("render")),
        mappo=_construct(MAPPOConfig, raw.get("mappo")),
        training=_construct(TrainingConfig, raw.get("training")),
    )
