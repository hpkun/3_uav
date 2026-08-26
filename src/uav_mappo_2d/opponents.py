from __future__ import annotations

import math

import numpy as np

from .env import UAVCombatEnv, wrap_angle


class RuleBasedOpponent:
    """Deterministic pursuit controller used for curriculum warm-up and evaluation."""

    def __init__(self, cruise_fraction: float = 0.65) -> None:
        self.cruise_fraction = cruise_fraction

    def act(self, env: UAVCombatEnv, team: int) -> np.ndarray:
        cfg = env.config
        actions = np.zeros((env.n_agents, 2), dtype=np.float32)
        enemy = 1 - team
        half = cfg.map_size / 2.0
        desired_speed = cfg.speed_min + self.cruise_fraction * (
            cfg.speed_max - cfg.speed_min
        )
        for agent in range(env.n_agents):
            if not env.alive[team, agent]:
                continue
            position = env.positions[team, agent]
            alive_enemy = np.flatnonzero(env.alive[enemy])
            if alive_enemy.size == 0:
                continue

            boundary_margin = half - np.max(np.abs(position))
            if boundary_margin < 0.12 * cfg.map_size:
                target_point = np.zeros(2)
            else:
                target = min(
                    alive_enemy,
                    key=lambda idx: np.linalg.norm(env.positions[enemy, idx] - position),
                )
                target_point = env.positions[enemy, target]
            delta = target_point - position
            desired_heading = math.atan2(delta[1], delta[0])
            heading_error = float(wrap_angle(desired_heading - env.headings[team, agent]))
            actions[agent, 1] = np.clip(
                heading_error / max(cfg.max_turn_rate * cfg.decision_dt, 1e-6), -1.0, 1.0
            )
            actions[agent, 0] = np.clip(
                (desired_speed - env.speeds[team, agent])
                / max(cfg.max_acceleration * cfg.decision_dt, 1e-6),
                -1.0,
                1.0,
            )
        return actions
