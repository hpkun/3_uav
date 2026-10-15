from __future__ import annotations

import itertools
import math

import numpy as np

from .env import UAVCombatEnv, wrap_angle


class RuleBasedOpponent:
    """Deterministic target assignment, tail pursuit, and boundary protection."""

    def __init__(self) -> None:
        self.targets = np.full(UAVCombatEnv.N_AGENTS, -1, dtype=np.int32)
        self._last_assignment_step = -10**9

    def reset(self) -> None:
        self.targets.fill(-1)
        self._last_assignment_step = -10**9

    @staticmethod
    def _distance(env: UAVCombatEnv, team: int, agent: int, target: int) -> float:
        return float(
            np.linalg.norm(env.positions[1 - team, target] - env.positions[team, agent])
        )

    def _assign_targets(self, env: UAVCombatEnv, team: int) -> None:
        friends = [int(index) for index in np.flatnonzero(env.alive[team])]
        enemies = [int(index) for index in np.flatnonzero(env.team_detected_enemy_mask(team))]
        self.targets.fill(-1)
        if not friends or not enemies:
            self._last_assignment_step = env.step_count
            return

        if len(friends) <= len(enemies):
            choices = []
            for assignment in itertools.permutations(enemies, len(friends)):
                cost = sum(
                    self._distance(env, team, agent, target)
                    for agent, target in zip(friends, assignment)
                )
                choices.append((cost, assignment))
            _, assignment = min(choices, key=lambda item: (item[0], item[1]))
            for agent, target in zip(friends, assignment):
                self.targets[agent] = target
        else:
            choices = []
            for selected_friends in itertools.permutations(friends, len(enemies)):
                cost = sum(
                    self._distance(env, team, agent, target)
                    for agent, target in zip(selected_friends, enemies)
                )
                choices.append((cost, selected_friends))
            _, selected_friends = min(choices, key=lambda item: (item[0], item[1]))
            assigned = set(selected_friends)
            for agent, target in zip(selected_friends, enemies):
                self.targets[agent] = target
            for agent in friends:
                if agent in assigned:
                    continue
                self.targets[agent] = min(
                    enemies,
                    key=lambda target: (
                        self._distance(env, team, agent, target),
                        target,
                    ),
                )
        self._last_assignment_step = env.step_count

    def _needs_assignment(self, env: UAVCombatEnv, team: int) -> bool:
        if env.step_count - self._last_assignment_step >= env.config.blue_assignment_interval:
            return True
        detected = env.team_detected_enemy_mask(team)
        for agent in np.flatnonzero(env.alive[team]):
            target = int(self.targets[int(agent)])
            if (target < 0 and detected.any()) or (target >= 0 and not detected[target]):
                return True
        return False

    @staticmethod
    def _flying_outward(env: UAVCombatEnv, team: int, agent: int) -> bool:
        cfg = env.config
        half = cfg.map_size / 2.0
        x, y = env.positions[team, agent]
        heading = env.headings[team, agent]
        vx, vy = math.cos(heading), math.sin(heading)
        margin = cfg.blue_boundary_margin
        return bool(
            (x >= half - margin and vx > 0.0)
            or (x <= -half + margin and vx < 0.0)
            or (y >= half - margin and vy > 0.0)
            or (y <= -half + margin and vy < 0.0)
        )

    def act(self, env: UAVCombatEnv, team: int = 1) -> np.ndarray:
        if self._needs_assignment(env, team):
            self._assign_targets(env, team)
        cfg = env.config
        enemy = 1 - team
        actions = np.zeros((env.N_AGENTS, env.ACTION_DIM), dtype=np.float32)
        for agent in np.flatnonzero(env.alive[team]):
            agent = int(agent)
            target = int(self.targets[agent])
            has_target = target >= 0
            desired_speed = cfg.speed_max
            if self._flying_outward(env, team, agent):
                target_point = np.zeros(2, dtype=np.float64)
            elif has_target:
                target_heading = env.headings[enemy, target]
                target_point = env.positions[enemy, target] - cfg.blue_tail_distance * np.array(
                    [math.cos(target_heading), math.sin(target_heading)]
                )
            else:
                target_point = None
            if target_point is not None:
                delta = target_point - env.positions[team, agent]
                desired_heading = math.atan2(float(delta[1]), float(delta[0]))
                heading_error = float(wrap_angle(desired_heading - env.headings[team, agent]))
                actions[agent, 1] = np.clip(
                    heading_error / (cfg.max_turn_rate * cfg.dt), -1.0, 1.0
                )

            if has_target:
                target_distance = self._distance(env, team, agent, target)
                if target_distance <= 2.0 * cfg.attack_range:
                    desired_speed = float(
                        np.clip(
                            env.speeds[enemy, target] + 20.0,
                            cfg.speed_min,
                            cfg.speed_max,
                        )
                    )
            actions[agent, 0] = np.clip(
                (desired_speed - env.speeds[team, agent])
                / (cfg.max_acceleration * cfg.dt),
                -1.0,
                1.0,
            )
        return actions
