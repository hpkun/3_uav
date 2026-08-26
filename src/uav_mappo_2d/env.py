from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvironmentConfig, RenderConfig, RewardConfig


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


class UAVCombatEnv(gym.Env):
    """Symmetric two-team, fixed-wing UAV combat environment.

    Arrays use the layout ``[team, agent, ...]``. The environment exposes both
    teams so training code can freely combine learned, scripted, or snapshot
    policies. Actors receive local observations; ``global_state(team)`` returns
    a team-centric centralized-critic state.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}
    SELF_FEATURES = 12
    ENTITY_FEATURES = 14

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_config: RenderConfig | None = None,
        render_mode: str | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvironmentConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_config = render_config or RenderConfig()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only render_mode=None or 'rgb_array' is supported")
        self.render_mode = render_mode
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])

        cfg = self.config
        if cfg.agents_per_team < 1:
            raise ValueError("agents_per_team must be positive")
        ratio = cfg.decision_dt / cfg.physics_dt
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("decision_dt must be an integer multiple of physics_dt")
        self.physics_substeps = int(round(ratio))
        self.max_steps = int(math.ceil(cfg.episode_seconds / cfg.decision_dt))
        self.n_agents = cfg.agents_per_team
        self.entity_slots = (self.n_agents - 1) + self.n_agents
        self.obs_dim = self.SELF_FEATURES + self.entity_slots * self.ENTITY_FEATURES
        self.state_unit_features = 10
        self.state_dim = 2 * self.n_agents * self.state_unit_features + 1

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2, self.n_agents, 2), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2, self.n_agents, self.obs_dim),
            dtype=np.float32,
        )

        self.positions = np.zeros((2, self.n_agents, 2), dtype=np.float64)
        self.speeds = np.zeros((2, self.n_agents), dtype=np.float64)
        self.headings = np.zeros((2, self.n_agents), dtype=np.float64)
        self.alive = np.ones((2, self.n_agents), dtype=bool)
        self.ammo = np.full((2, self.n_agents), cfg.ammunition, dtype=np.int32)
        self.lock_target = np.full((2, self.n_agents), -1, dtype=np.int32)
        self.lock_progress = np.zeros((2, self.n_agents), dtype=np.float64)
        self.outside_time = np.zeros((2, self.n_agents), dtype=np.float64)
        self.step_count = 0
        self.elapsed_time = 0.0
        self.last_winner: int | None = None
        self.kills = np.zeros(2, dtype=np.int32)
        self.first_kill_time: float | None = None
        self._last_potential = np.zeros(2, dtype=np.float64)
        self._renderer = None
        trail_capacity = self.render_config.trail_length * self.physics_substeps
        self._trails = [
            [deque(maxlen=trail_capacity) for _ in range(self.n_agents)]
            for _ in range(2)
        ]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        cfg = self.config
        half = cfg.map_size / 2.0
        spawn_x = half * cfg.spawn_x_fraction * 2.0
        y_span = half * cfg.spawn_y_fraction * 2.0
        base_y = np.linspace(-y_span, y_span, self.n_agents)
        mirror_y = -1.0 if self.np_random.random() < 0.5 else 1.0

        for team in range(2):
            sign = -1.0 if team == 0 else 1.0
            y_order = self.np_random.permutation(base_y) * mirror_y
            self.positions[team, :, 0] = sign * spawn_x
            self.positions[team, :, 1] = y_order
            self.positions[team] += self.np_random.uniform(
                -cfg.spawn_jitter, cfg.spawn_jitter, size=(self.n_agents, 2)
            )
            nominal_heading = 0.0 if team == 0 else np.pi
            jitter = np.deg2rad(cfg.heading_jitter_degrees)
            self.headings[team] = nominal_heading + self.np_random.uniform(
                -jitter, jitter, size=self.n_agents
            )

        self.speeds.fill(cfg.initial_speed)
        self.alive.fill(True)
        self.ammo.fill(cfg.ammunition)
        self.lock_target.fill(-1)
        self.lock_progress.fill(0.0)
        self.outside_time.fill(0.0)
        self.step_count = 0
        self.elapsed_time = 0.0
        self.last_winner = None
        self.kills.fill(0)
        self.first_kill_time = None
        for team in range(2):
            for agent in range(self.n_agents):
                self._trails[team][agent].clear()
                self._trails[team][agent].append(self.positions[team, agent].copy())
        self._last_potential = np.array([self._potential(0), self._potential(1)])
        observation = self.observe_all()
        return observation, self._info()

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != self.action_space.shape:
            raise ValueError(f"Expected actions {self.action_space.shape}, got {actions.shape}")
        actions = np.clip(actions, -1.0, 1.0)

        local_events = np.zeros((2, self.n_agents), dtype=np.float64)
        team_events = np.full(2, self.reward_config.step, dtype=np.float64)

        for _ in range(self.physics_substeps):
            self._integrate(actions)
            self._apply_boundaries(team_events, local_events)
            if self.config.collision_enabled:
                self._apply_collisions(team_events, local_events)
            self._update_weapons(team_events, local_events)
            self.elapsed_time += self.config.physics_dt

        self.step_count += 1
        terminated = not self.alive[0].any() or not self.alive[1].any()
        truncated = self.step_count >= self.max_steps and not terminated

        potential = np.array([self._potential(0), self._potential(1)])
        shaping = self.reward_config.shaping_scale * (
            self.reward_config.shaping_gamma * potential - self._last_potential
        )
        self._last_potential = potential
        team_reward = team_events + shaping

        if terminated or truncated:
            self.last_winner = self._determine_winner()
            if self.last_winner is not None:
                team_reward[self.last_winner] += self.reward_config.win
                team_reward[1 - self.last_winner] += self.reward_config.lose

        rewards = np.zeros((2, self.n_agents), dtype=np.float32)
        for team in range(2):
            rewards[team] = (
                self.reward_config.team_fraction * team_reward[team]
                + self.reward_config.local_fraction * local_events[team]
            )
            rewards[team, ~self.alive[team]] = team_reward[team]

        return self.observe_all(), rewards, terminated, truncated, self._info()

    def _integrate(self, actions: np.ndarray) -> None:
        cfg = self.config
        dt = cfg.physics_dt
        for team in range(2):
            active = self.alive[team]
            acceleration = actions[team, :, 0] * cfg.max_acceleration
            turn_rate = actions[team, :, 1] * cfg.max_turn_rate
            self.speeds[team, active] = np.clip(
                self.speeds[team, active] + acceleration[active] * dt,
                cfg.speed_min,
                cfg.speed_max,
            )
            self.headings[team, active] = wrap_angle(
                self.headings[team, active] + turn_rate[active] * dt
            )
            self.positions[team, active, 0] += (
                self.speeds[team, active] * np.cos(self.headings[team, active]) * dt
            )
            self.positions[team, active, 1] += (
                self.speeds[team, active] * np.sin(self.headings[team, active]) * dt
            )
            for agent in np.flatnonzero(active):
                self._trails[team][int(agent)].append(
                    self.positions[team, int(agent)].copy()
                )

    def _apply_boundaries(self, team_events: np.ndarray, local_events: np.ndarray) -> None:
        half = self.config.map_size / 2.0
        dt = self.config.physics_dt
        for team in range(2):
            for agent in range(self.n_agents):
                if not self.alive[team, agent]:
                    continue
                outside = bool(np.any(np.abs(self.positions[team, agent]) > half))
                if outside:
                    self.outside_time[team, agent] += dt
                else:
                    self.outside_time[team, agent] = max(
                        0.0, self.outside_time[team, agent] - 2.0 * dt
                    )
                if self.outside_time[team, agent] >= self.config.boundary_grace_seconds:
                    self._destroy(team, agent)
                    team_events[team] += self.reward_config.loss + self.reward_config.boundary
                    local_events[team, agent] += (
                        self.reward_config.loss + self.reward_config.boundary
                    )

    def _apply_collisions(self, team_events: np.ndarray, local_events: np.ndarray) -> None:
        threshold = 2.0 * self.config.agent_radius
        victims: set[tuple[int, int]] = set()
        units = [(t, a) for t in range(2) for a in range(self.n_agents) if self.alive[t, a]]
        for index, first in enumerate(units):
            for second in units[index + 1 :]:
                if np.linalg.norm(self.positions[first] - self.positions[second]) < threshold:
                    victims.update((first, second))
        for team, agent in sorted(victims):
            if not self.alive[team, agent]:
                continue
            self._destroy(team, agent)
            penalty = self.reward_config.loss + self.reward_config.collision
            team_events[team] += penalty
            local_events[team, agent] += penalty

    def _update_weapons(self, team_events: np.ndarray, local_events: np.ndarray) -> None:
        cfg = self.config
        pending: list[tuple[int, int, int]] = []
        dt = cfg.physics_dt
        for team in range(2):
            enemy = 1 - team
            for agent in range(self.n_agents):
                if not self.alive[team, agent] or self.ammo[team, agent] <= 0:
                    self.lock_target[team, agent] = -1
                    self.lock_progress[team, agent] = 0.0
                    continue
                target = int(self.lock_target[team, agent])
                if target < 0 or not self._valid_weapon_solution(team, agent, target):
                    candidates = [
                        idx
                        for idx in range(self.n_agents)
                        if self.alive[enemy, idx]
                        and self._valid_weapon_solution(team, agent, idx)
                    ]
                    target = (
                        min(
                            candidates,
                            key=lambda idx: np.linalg.norm(
                                self.positions[enemy, idx] - self.positions[team, agent]
                            ),
                        )
                        if candidates
                        else -1
                    )
                    if target != self.lock_target[team, agent]:
                        self.lock_progress[team, agent] = max(
                            0.0,
                            self.lock_progress[team, agent] - cfg.lock_decay_rate * dt,
                        )
                    self.lock_target[team, agent] = target

                if target >= 0:
                    self.lock_progress[team, agent] += dt
                    if self.lock_progress[team, agent] + 1e-9 >= cfg.lock_seconds:
                        pending.append((team, agent, target))
                        self.ammo[team, agent] -= 1
                        self.lock_target[team, agent] = -1
                        self.lock_progress[team, agent] = 0.0
                else:
                    self.lock_progress[team, agent] = max(
                        0.0, self.lock_progress[team, agent] - cfg.lock_decay_rate * dt
                    )

        claimed: set[tuple[int, int]] = set()
        for team, attacker, target in pending:
            enemy = 1 - team
            key = (enemy, target)
            if key in claimed or not self.alive[enemy, target]:
                continue
            claimed.add(key)
            self._destroy(enemy, target)
            self.kills[team] += 1
            if self.first_kill_time is None:
                self.first_kill_time = self.elapsed_time + dt
            team_events[team] += self.reward_config.kill
            team_events[enemy] += self.reward_config.loss
            local_events[team, attacker] += self.reward_config.kill
            local_events[enemy, target] += self.reward_config.loss

    def _destroy(self, team: int, agent: int) -> None:
        self.alive[team, agent] = False
        self.lock_target[team, agent] = -1
        self.lock_progress[team, agent] = 0.0

    def _relative_geometry(
        self, team: int, agent: int, other_team: int, other: int
    ) -> tuple[float, float, float]:
        delta = self.positions[other_team, other] - self.positions[team, agent]
        distance = float(np.linalg.norm(delta))
        bearing = math.atan2(delta[1], delta[0])
        own_bearing = float(wrap_angle(bearing - self.headings[team, agent]))
        other_bearing = float(
            wrap_angle(bearing + np.pi - self.headings[other_team, other])
        )
        return distance, own_bearing, other_bearing

    def _visible(self, team: int, agent: int, target: int) -> bool:
        if not self.alive[1 - team, target]:
            return False
        distance, own_bearing, _ = self._relative_geometry(team, agent, 1 - team, target)
        return distance <= self.config.radar_range and abs(own_bearing) <= math.radians(
            self.config.radar_fov_degrees / 2.0
        )

    def _valid_weapon_solution(self, team: int, agent: int, target: int) -> bool:
        if not self.alive[team, agent] or not self.alive[1 - team, target]:
            return False
        distance, own_bearing, target_bearing = self._relative_geometry(
            team, agent, 1 - team, target
        )
        forward = abs(own_bearing) <= math.radians(self.config.weapon_fov_degrees / 2.0)
        behind_target = abs(target_bearing) >= math.pi - math.radians(
            self.config.tail_cone_degrees / 2.0
        )
        return distance <= self.config.weapon_range and forward and behind_target

    def _potential(self, team: int) -> float:
        enemy = 1 - team
        attack_score = 0.0
        threat_score = 0.0
        for agent in range(self.n_agents):
            if not self.alive[team, agent]:
                continue
            best_attack = 0.0
            best_threat = 0.0
            for target in range(self.n_agents):
                if not self.alive[enemy, target]:
                    continue
                distance, own_bearing, target_bearing = self._relative_geometry(
                    team, agent, enemy, target
                )
                proximity = math.exp(-distance / max(self.config.weapon_range, 1.0))
                forward = max(0.0, math.cos(own_bearing))
                tail = max(0.0, -math.cos(target_bearing))
                best_attack = max(best_attack, proximity * forward * tail)

                _, enemy_bearing, own_aspect = self._relative_geometry(
                    enemy, target, team, agent
                )
                threat = proximity * max(0.0, math.cos(enemy_bearing)) * max(
                    0.0, -math.cos(own_aspect)
                )
                best_threat = max(best_threat, threat)
            attack_score += best_attack
            threat_score += best_threat

        scale = max(self.n_agents, 1)
        return (attack_score - threat_score) / scale

    def observe_all(self) -> np.ndarray:
        return np.stack([self.observe_team(0), self.observe_team(1)], axis=0)

    def observe_team(self, team: int) -> np.ndarray:
        return np.stack(
            [self._observe_agent(team, agent) for agent in range(self.n_agents)], axis=0
        ).astype(np.float32)

    def _observe_agent(self, team: int, agent: int) -> np.ndarray:
        cfg = self.config
        half = cfg.map_size / 2.0
        alive_value = 1.0 if self.alive[team, agent] else 0.0
        frame_rotation = 0.0 if team == 0 else np.pi
        position = rotate(self.positions[team, agent], frame_rotation) / half
        heading = float(wrap_angle(self.headings[team, agent] + frame_rotation))
        speed_norm = 2.0 * (
            (self.speeds[team, agent] - cfg.speed_min)
            / max(cfg.speed_max - cfg.speed_min, 1e-6)
        ) - 1.0
        boundary = np.clip((half - np.abs(self.positions[team, agent])) / half, 0.0, 1.0)
        time_left = 1.0 - self.step_count / max(self.max_steps, 1)
        lock_fraction = self.lock_progress[team, agent] / max(cfg.lock_seconds, 1e-6)
        own = np.array(
            [
                position[0],
                position[1],
                speed_norm,
                math.sin(heading),
                math.cos(heading),
                self.ammo[team, agent] / max(cfg.ammunition, 1),
                np.clip(lock_fraction, 0.0, 1.0),
                boundary[0],
                boundary[1],
                time_left,
                alive_value,
                self.outside_time[team, agent] / max(cfg.boundary_grace_seconds, 1e-6),
            ],
            dtype=np.float64,
        )

        entities: list[np.ndarray] = []
        for ally in range(self.n_agents):
            if ally != agent:
                entities.append(self._entity_features(team, agent, team, ally, is_ally=True))
        for enemy in range(self.n_agents):
            entities.append(
                self._entity_features(team, agent, 1 - team, enemy, is_ally=False)
            )
        observation = np.concatenate([own, *entities])
        if not self.alive[team, agent]:
            observation.fill(0.0)
        return np.clip(observation, -1.0, 1.0)

    def _entity_features(
        self, team: int, agent: int, other_team: int, other: int, *, is_ally: bool
    ) -> np.ndarray:
        other_alive = bool(self.alive[other_team, other])
        visible = other_alive and (
            is_ally or self._visible(team, agent, other)
        )
        result = np.zeros(self.ENTITY_FEATURES, dtype=np.float64)
        result[9] = float(other_alive)
        result[10] = float(visible)
        result[11] = 1.0 if is_ally else -1.0
        if not visible:
            return result

        delta = self.positions[other_team, other] - self.positions[team, agent]
        relative_body = rotate(delta, -self.headings[team, agent])
        own_velocity = self.speeds[team, agent] * np.array(
            [math.cos(self.headings[team, agent]), math.sin(self.headings[team, agent])]
        )
        other_velocity = self.speeds[other_team, other] * np.array(
            [math.cos(self.headings[other_team, other]), math.sin(self.headings[other_team, other])]
        )
        relative_velocity = rotate(other_velocity - own_velocity, -self.headings[team, agent])
        distance = float(np.linalg.norm(delta))
        bearing = math.atan2(relative_body[1], relative_body[0])
        relative_heading = float(
            wrap_angle(self.headings[other_team, other] - self.headings[team, agent])
        )
        normalizer = self.config.radar_range if not is_ally else self.config.map_size
        result[0:2] = relative_body / max(normalizer, 1.0)
        result[2:4] = relative_velocity / max(2.0 * self.config.speed_max, 1.0)
        result[4] = distance / max(normalizer, 1.0)
        result[5:7] = [math.sin(relative_heading), math.cos(relative_heading)]
        result[7:9] = [math.sin(bearing), math.cos(bearing)]
        if not is_ally:
            result[12] = float(self._valid_weapon_solution(team, agent, other))
            result[13] = float(self._valid_weapon_solution(other_team, other, agent))
        return result

    def global_state(self, team: int) -> np.ndarray:
        half = self.config.map_size / 2.0
        order = (team, 1 - team)
        chunks: list[np.ndarray] = []
        rotation = 0.0 if team == 0 else np.pi
        for side in order:
            for agent in range(self.n_agents):
                pos = rotate(self.positions[side, agent], rotation) / half
                heading = float(wrap_angle(self.headings[side, agent] + rotation))
                velocity = self.speeds[side, agent] * np.array(
                    [math.cos(self.headings[side, agent]), math.sin(self.headings[side, agent])]
                )
                velocity = rotate(velocity, rotation) / max(self.config.speed_max, 1.0)
                chunks.append(
                    np.array(
                        [
                            pos[0],
                            pos[1],
                            velocity[0],
                            velocity[1],
                            math.sin(heading),
                            math.cos(heading),
                            self.speeds[side, agent] / max(self.config.speed_max, 1.0),
                            float(self.alive[side, agent]),
                            self.ammo[side, agent] / max(self.config.ammunition, 1),
                            self.lock_progress[side, agent]
                            / max(self.config.lock_seconds, 1e-6),
                        ],
                        dtype=np.float64,
                    )
                )
        tail = np.array([1.0 - self.step_count / max(self.max_steps, 1)])
        return np.clip(np.concatenate([*chunks, tail]), -1.0, 1.0).astype(np.float32)

    def _determine_winner(self) -> int | None:
        alive_counts = self.alive.sum(axis=1)
        if alive_counts[0] != alive_counts[1]:
            return int(np.argmax(alive_counts))
        return None

    def _info(self) -> dict[str, Any]:
        return {
            "time": self.elapsed_time,
            "step": self.step_count,
            "alive": self.alive.sum(axis=1).astype(int).tolist(),
            "ammo": np.sum(self.ammo * self.alive, axis=1).astype(int).tolist(),
            "winner": self.last_winner,
            "kills": self.kills.astype(int).tolist(),
            "first_kill_time": self.first_kill_time,
            "states": np.stack([self.global_state(0), self.global_state(1)]),
            "active_masks": self.alive.astype(np.float32).copy(),
        }

    def render(self) -> np.ndarray:
        if self._renderer is None:
            from .rendering import CombatRenderer

            self._renderer = CombatRenderer(
                self.config, self.render_config, self.project_root
            )
        return self._renderer.render(self)

    def close(self) -> None:
        self._renderer = None
