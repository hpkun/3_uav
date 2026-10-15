from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class EnvironmentConfig:
    map_size: float = 10_000.0
    dt: float = 1.0
    episode_steps: int = 150
    speed_min: float = 150.0
    speed_max: float = 250.0
    initial_speed_min: float = 190.0
    initial_speed_max: float = 210.0
    max_acceleration: float = 20.0
    max_turn_rate: float = 0.2617993878
    spawn_x: float = 3_500.0
    spawn_y_spacing: float = 1_500.0
    spawn_jitter: float = 200.0
    heading_jitter_degrees: float = 5.0
    attack_range: float = 1_200.0
    radar_range: float = 4_000.0
    attack_ata_degrees: float = 15.0
    attack_aspect_degrees: float = 50.0
    collision_distance: float = 150.0
    collision_warning_distance: float = 450.0
    potential_distance_scale: float = 3_000.0
    blue_assignment_interval: int = 5
    blue_tail_distance: float = 800.0
    blue_boundary_margin: float = 500.0


@dataclass
class RewardConfig:
    weapon_kill_reward: float = 1.0
    weapon_loss_penalty: float = 1.0
    oob_penalty: float = 1.0
    collision_penalty: float = 2.0
    near_collision_scale: float = 0.05
    terminal_reward: float = 5.0
    shaping_scale: float = 0.1
    shaping_gamma: float = 0.99


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
    output_dir: str = "runs/fixed_5v5"
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
