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
    initial_speed_min: float = 95.0
    initial_speed_max: float = 105.0
    max_acceleration: float = 10.0
    max_turn_rate_degrees: float = 10.0
    spawn_x_min: float = 3_200.0
    spawn_x_max: float = 3_800.0
    spawn_y_min: float = -3_000.0
    spawn_y_max: float = 3_000.0
    min_spawn_y_separation: float = 500.0
    heading_jitter_degrees: float = 5.0
    attack_range: float = 1_000.0
    radar_range: float = 3_000.0
    attack_ata_degrees: float = 30.0
    attack_aspect_degrees: float = 45.0
    collision_distance: float = 50.0
    blue_escape_distance: float | None = None

    def __post_init__(self) -> None:
        if self.spawn_y_max - self.spawn_y_min < 4 * self.min_spawn_y_separation:
            raise ValueError("Spawn y range cannot fit five separated UAVs")


@dataclass
class RewardConfig:
    weapon_kill_reward: float = 10.0
    weapon_loss_penalty: float = 10.0
    oob_penalty: float = 20.0
    collision_penalty: float = 5.0
    terminal_reward: float = 20.0


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
    total_env_steps: int = 2_000_000
    parallel_envs: int = 16
    rollout_length: int = 128
    log_interval_steps: int = 20_000
    checkpoint_interval_steps: int = 100_000
    evaluation_interval_steps: int = 100_000
    evaluation_episodes: int = 100
    output_dir: str = "runs/v2_1_5v5_seed7"
    device: str = "cuda"


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
