from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class EnvironmentConfig:
    agents_per_team: int = 3
    map_size: float = 10_000.0
    center_size: float = 3_000.0
    physics_dt: float = 0.2
    decision_dt: float = 1.0
    episode_seconds: float = 150.0
    speed_min: float = 150.0
    speed_max: float = 250.0
    initial_speed: float = 200.0
    max_acceleration: float = 20.0
    max_turn_rate: float = 0.2617993878
    agent_radius: float = 25.0
    spawn_x_fraction: float = 0.42
    spawn_y_fraction: float = 0.28
    spawn_jitter: float = 120.0
    heading_jitter_degrees: float = 8.0
    boundary_grace_seconds: float = 2.0
    collision_enabled: bool = False
    radar_range: float = 3_000.0
    radar_fov_degrees: float = 120.0
    weapon_range: float = 1_200.0
    weapon_fov_degrees: float = 60.0
    tail_cone_degrees: float = 120.0
    lock_seconds: float = 1.5
    lock_decay_rate: float = 2.0
    ammunition: int = 2


@dataclass
class RewardConfig:
    kill: float = 20.0
    loss: float = -20.0
    win: float = 100.0
    lose: float = -100.0
    boundary: float = -5.0
    collision: float = -2.0
    step: float = -0.001
    shaping_scale: float = 0.2
    shaping_gamma: float = 0.99
    team_fraction: float = 0.8
    local_fraction: float = 0.2


@dataclass
class RenderConfig:
    width: int = 800
    height: int = 800
    margin: int = 45
    show_center: bool = False
    show_radar: bool = False
    show_weapon: bool = False
    show_trails: bool = True
    trail_length: int = 80
    icon_source: str = "对抗环境图.png"


@dataclass
class MAPPOConfig:
    hidden_size: int = 128
    learning_rate: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.1
    value_clip: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 5


@dataclass
class TrainingConfig:
    episodes: int = 2_000
    parallel_envs: int = 16
    rule_opponent_episodes: int = 300
    snapshot_interval: int = 100
    snapshot_pool_size: int = 8
    checkpoint_interval: int = 100
    log_interval: int = 10
    output_dir: str = "runs/default"
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
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    valid_top = {item.name for item in fields(ProjectConfig)}
    unknown = set(raw) - valid_top
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
