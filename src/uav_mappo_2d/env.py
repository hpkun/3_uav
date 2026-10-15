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


class UAVCombatEnv(gym.Env):
    """Fixed 5v5 Continuous 2D UAV Combat Baseline V2."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}
    N_AGENTS = 5
    OBS_DIM = 68
    STATE_DIM = 60
    ACTION_DIM = 2
    REWARD_COMPONENTS = (
        "weapon_reward", "oob_reward", "collision_reward",
        "terminal_reward", "shaping_reward",
    )
    EPISODE_METRICS = (
        "red_alive", "blue_alive", "red_combat_kill", "blue_combat_kill",
        "red_weapon_loss", "blue_weapon_loss", "red_oob", "blue_oob",
        "red_collision", "blue_collision", "red_attack_window", "blue_attack_window",
        "red_radar_contact_steps", "blue_radar_contact_steps",
    )

    def __init__(self, config=None, reward_config=None, render_config=None,
                 render_mode=None, project_root=None) -> None:
        super().__init__()
        self.config = config or EnvironmentConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_config = render_config or RenderConfig()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only render_mode=None or 'rgb_array' is supported")
        self.render_mode = render_mode
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.n_agents, self.obs_dim, self.state_dim = self.N_AGENTS, self.OBS_DIM, self.STATE_DIM
        self.action_space = spaces.Box(-1., 1., (2, self.N_AGENTS, self.ACTION_DIM), np.float32)
        self.observation_space = spaces.Box(-1., 1., (2, self.N_AGENTS, self.OBS_DIM), np.float32)
        self.positions = np.zeros((2, self.N_AGENTS, 2), dtype=np.float64)
        self.speeds = np.zeros((2, self.N_AGENTS), dtype=np.float64)
        self.headings = np.zeros((2, self.N_AGENTS), dtype=np.float64)
        self.alive = np.ones((2, self.N_AGENTS), dtype=bool)
        self.combat_kills = np.zeros(2, dtype=np.int32)
        self.weapon_losses = np.zeros(2, dtype=np.int32)
        self.oob_losses = np.zeros(2, dtype=np.int32)
        self.collision_losses = np.zeros(2, dtype=np.int32)
        self.attack_window_count = np.zeros(2, dtype=np.int32)
        self.radar_contact_steps = np.zeros(2, dtype=np.int32)
        self.step_count = 0
        self.last_winner = None
        self._potential = 0.0
        self._episode_return = 0.0
        self._episode_rewards = {key: 0.0 for key in self.REWARD_COMPONENTS}
        self._renderer = None
        self._trails = [[deque(maxlen=self.render_config.trail_length)
                         for _ in range(self.N_AGENTS)] for _ in range(2)]

    @property
    def elapsed_time(self) -> float:
        return self.step_count * self.config.dt

    def _sample_spawn_y(self) -> np.ndarray:
        cfg = self.config
        for _ in range(1000):
            values = self.np_random.uniform(cfg.spawn_y_min, cfg.spawn_y_max, self.N_AGENTS)
            if np.all(np.diff(np.sort(values)) >= cfg.min_spawn_y_separation):
                return values
        # Independent jitter and ID permutation preserve team independence in fallback.
        slack = (cfg.spawn_y_max - cfg.spawn_y_min) / 4 - cfg.min_spawn_y_separation
        jitter = min(100.0, slack / 4)
        values = np.linspace(cfg.spawn_y_min + jitter, cfg.spawn_y_max - jitter, self.N_AGENTS)
        values += self.np_random.uniform(-jitter, jitter, self.N_AGENTS)
        return self.np_random.permutation(values)

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        cfg = self.config
        for team in range(2):
            x = self.np_random.uniform(cfg.spawn_x_min, cfg.spawn_x_max, self.N_AGENTS)
            self.positions[team, :, 0] = -x if team == 0 else x
            self.positions[team, :, 1] = self._sample_spawn_y()
        jitter = self.np_random.uniform(-math.radians(cfg.heading_jitter_degrees),
                                       math.radians(cfg.heading_jitter_degrees), (2, self.N_AGENTS))
        self.headings[0] = jitter[0]
        self.headings[1] = wrap_angle(math.pi + jitter[1])
        self.speeds[:] = self.np_random.uniform(cfg.initial_speed_min, cfg.initial_speed_max, (2, self.N_AGENTS))
        self.alive.fill(True)
        for values in (self.combat_kills, self.weapon_losses, self.oob_losses,
                       self.collision_losses, self.attack_window_count, self.radar_contact_steps):
            values.fill(0)
        self.step_count, self.last_winner = 0, None
        self._episode_return = 0.0
        self._episode_rewards = {key: 0.0 for key in self.REWARD_COMPONENTS}
        for team in range(2):
            for agent in range(self.N_AGENTS):
                self._trails[team][agent].clear()
                self._trails[team][agent].append(self.positions[team, agent].copy())
        self._potential = self._compute_potential()
        return self._get_actor_obs(), self._info(
            {key: np.zeros(2) for key in self.REWARD_COMPONENTS}, False
        )

    @staticmethod
    def validate_actions(actions, expected_shape) -> np.ndarray:
        values = np.asarray(actions)
        if values.shape != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {values.shape}")
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError("Continuous actions must be finite numbers")
        if np.any(np.abs(values) > 1.0 + 1e-6):
            raise ValueError("Continuous actions must be in [-1,1]")
        # Clip only floating-point tolerance; invalid policy outputs are rejected above.
        return np.clip(values, -1., 1.).astype(np.float32)

    def step(self, actions):
        actions = self.validate_actions(actions, self.action_space.shape)
        previous_positions = self.positions.copy()
        old_potential = self._potential
        self._update_kinematics(actions)
        oob = self._resolve_out_of_bounds()
        collisions = self._resolve_collisions(previous_positions)
        kills = self._resolve_attacks()
        self.step_count += 1
        terminated = self._is_terminal()
        self.last_winner = self._determine_winner() if terminated else None
        terminal = np.zeros(2)
        if terminated and self.last_winner is not None:
            terminal[self.last_winner] = self.reward_config.terminal_reward
            terminal[1 - self.last_winner] = -self.reward_config.terminal_reward
        self._potential = 0.0 if terminated else self._compute_potential()
        cfg = self.reward_config
        shaping = cfg.shaping_scale * (cfg.shaping_gamma * self._potential - old_potential)
        components = {
            "weapon_reward": cfg.weapon_kill_reward * kills - cfg.weapon_loss_penalty * kills[::-1],
            "oob_reward": -cfg.oob_penalty * oob,
            "collision_reward": -cfg.collision_penalty * collisions,
            "terminal_reward": terminal,
            "shaping_reward": np.array([shaping, -shaping]),
        }
        totals = sum(components.values())
        for key, values in components.items():
            self._episode_rewards[key] += float(values[0])
        self._episode_return += float(totals[0])
        rewards = np.repeat(totals[:, None], self.N_AGENTS, axis=1).astype(np.float32)
        return self._get_actor_obs(), rewards, terminated, False, self._info(components, terminated)

    def _update_kinematics(self, actions):
        cfg = self.config
        for team in range(2):
            active = self.alive[team]
            self.speeds[team, active] = np.clip(
                self.speeds[team, active] + actions[team, active, 0] * cfg.max_acceleration * cfg.dt,
                cfg.speed_min, cfg.speed_max,
            )
            self.headings[team, active] = wrap_angle(
                self.headings[team, active] + actions[team, active, 1] * math.radians(cfg.max_turn_rate_degrees) * cfg.dt
            )
            self.positions[team, active, 0] += self.speeds[team, active] * np.cos(self.headings[team, active]) * cfg.dt
            self.positions[team, active, 1] += self.speeds[team, active] * np.sin(self.headings[team, active]) * cfg.dt
            for agent in np.flatnonzero(active):
                self._trails[team][agent].append(self.positions[team, agent].copy())

    def _resolve_out_of_bounds(self):
        outside = self.alive & np.any(np.abs(self.positions) > self.config.map_size / 2, axis=-1)
        losses = outside.sum(axis=1).astype(np.int32)
        self.alive[outside] = False
        self.oob_losses += losses
        return losses

    def _resolve_collisions(self, previous_positions):
        first, second = np.triu_indices(2 * self.N_AGENTS, k=1)
        valid = self.alive.ravel()[first] & self.alive.ravel()[second]
        first, second = first[valid], second[valid]
        previous, current = previous_positions.reshape(-1, 2), self.positions.reshape(-1, 2)
        r0 = previous[first] - previous[second]
        displacement = current - previous
        dr = displacement[first] - displacement[second]
        denominator = np.sum(dr * dr, axis=1)
        tau = np.zeros(len(first))
        np.divide(-np.sum(r0 * dr, axis=1), denominator, out=tau, where=denominator > 1e-12)
        closest = r0 + np.clip(tau, 0, 1)[:, None] * dr
        collided = np.sum(closest * closest, axis=1) <= self.config.collision_distance**2
        victims = np.unique(np.concatenate((first[collided], second[collided])))
        losses = np.bincount(victims // self.N_AGENTS, minlength=2).astype(np.int32)
        self.alive.ravel()[victims] = False
        self.collision_losses += losses
        return losses

    def team_detected_enemy_mask(self, team):
        detected = np.zeros(self.N_AGENTS, dtype=bool)
        friends, enemies = np.flatnonzero(self.alive[team]), np.flatnonzero(self.alive[1 - team])
        if friends.size and enemies.size:
            delta = self.positions[team, friends][:, None] - self.positions[1 - team, enemies][None]
            detected[enemies] = np.any(np.sum(delta**2, axis=-1) <= self.config.radar_range**2, axis=0)
        return detected

    def _compute_pair_geometry(self, attacker_team, attacker, target_team, target):
        delta = self.positions[target_team, target] - self.positions[attacker_team, attacker]
        los = math.atan2(float(delta[1]), float(delta[0]))
        return (float(np.linalg.norm(delta)),
                float(wrap_angle(los - self.headings[attacker_team, attacker])),
                float(wrap_angle(los - self.headings[target_team, target])))

    def _attack_valid(self, attacker_team, attacker, target_team, target):
        if not (self.alive[attacker_team, attacker] and self.alive[target_team, target]):
            return False
        distance, alpha, beta = self._compute_pair_geometry(attacker_team, attacker, target_team, target)
        return bool(distance <= self.config.attack_range
                    and abs(alpha) <= math.radians(self.config.attack_ata_degrees) + 1e-12
                    and abs(beta) <= math.radians(self.config.attack_aspect_degrees) + 1e-12)

    def _resolve_attacks(self):
        victims = [set(), set()]
        for team in range(2):
            # Both diagnostics refer to the post-OOB/collision, pre-weapon snapshot.
            self.radar_contact_steps[team] += int(self.team_detected_enemy_mask(team).any())
            for attacker in np.flatnonzero(self.alive[team]):
                valid = [int(j) for j in np.flatnonzero(self.alive[1 - team])
                         if self._attack_valid(team, attacker, 1 - team, j)]
                self.attack_window_count[team] += len(valid)
                if valid:
                    chosen = min(valid, key=lambda j: (self._compute_pair_geometry(team, attacker, 1 - team, j)[0], j))
                    victims[1 - team].add(chosen)
        kills = np.zeros(2, dtype=np.int32)
        for team in range(2):
            for agent in victims[team]:
                self.alive[team, agent] = False
            kills[1 - team] = len(victims[team])
        self.combat_kills += kills
        self.weapon_losses += kills[::-1]
        return kills

    def _pair_quality(self, team, attacker, target_team, target):
        distance, alpha, beta = self._compute_pair_geometry(team, attacker, target_team, target)
        cfg, reward = self.config, self.reward_config
        q_d = np.clip((cfg.radar_range - distance) / (cfg.radar_range - cfg.attack_range), 0, 1)
        return float(reward.potential_distance_weight * q_d
                     + reward.potential_ata_weight * (1 + math.cos(alpha)) / 2
                     + reward.potential_aspect_weight * (1 + math.cos(beta)) / 2)

    def _team_attack_quality(self, team):
        attackers = np.flatnonzero(self.alive[team])
        targets = np.flatnonzero(self.team_detected_enemy_mask(team))
        if not len(attackers) or not len(targets):
            return 0.0
        return float(np.mean([max(self._pair_quality(team, i, 1 - team, j) for i in attackers) for j in targets]))

    def _compute_potential(self):
        return self._team_attack_quality(0) - self._team_attack_quality(1)

    def _is_terminal(self):
        return bool(not self.alive[0].any() or not self.alive[1].any() or self.step_count >= self.config.episode_steps)

    def _determine_winner(self):
        red, blue = self.alive.sum(axis=1)
        return 0 if red > blue else (1 if blue > red else None)

    def _get_actor_obs(self):
        cfg = self.config
        observations = np.zeros((2, self.N_AGENTS, self.OBS_DIM), dtype=np.float32)
        speed_scale = cfg.speed_max - cfg.speed_min
        for team in range(2):
            detected = self.team_detected_enemy_mask(team)
            for agent in np.flatnonzero(self.alive[team]):
                own = observations[team, agent]
                own[:5] = (*self.positions[team, agent] / (cfg.map_size / 2),
                           2 * (self.speeds[team, agent] - cfg.speed_min) / speed_scale - 1,
                           math.sin(self.headings[team, agent]), math.cos(self.headings[team, agent]))
                entities = [(team, j) for j in range(self.N_AGENTS) if j != agent] + [(1 - team, j) for j in range(self.N_AGENTS)]
                for slot, (other_team, other) in enumerate(entities):
                    if not self.alive[other_team, other] or (other_team != team and not detected[other]):
                        continue
                    distance, alpha, beta = self._compute_pair_geometry(team, agent, other_team, other)
                    start = 5 + 7 * slot
                    own[start:start + 7] = (
                        distance / (math.sqrt(2) * cfg.map_size), math.sin(alpha), math.cos(alpha),
                        math.sin(beta), math.cos(beta),
                        np.clip((self.speeds[other_team, other] - self.speeds[team, agent]) / speed_scale, -1, 1), 1.,
                    )
        return observations

    def global_state(self):
        cfg = self.config
        state = np.zeros((2, self.N_AGENTS, 6), dtype=np.float32)
        state[..., :2] = self.positions / (cfg.map_size / 2)
        state[..., 2] = 2 * (self.speeds - cfg.speed_min) / (cfg.speed_max - cfg.speed_min) - 1
        state[..., 3], state[..., 4], state[..., 5] = np.sin(self.headings), np.cos(self.headings), self.alive
        state[~self.alive] = 0
        return state.ravel()

    def _counts(self):
        red, blue = self.alive.sum(axis=1)
        return dict(zip(self.EPISODE_METRICS, (
            int(red), int(blue), *map(int, self.combat_kills), *map(int, self.weapon_losses),
            *map(int, self.oob_losses), *map(int, self.collision_losses),
            *map(int, self.attack_window_count), *map(int, self.radar_contact_steps),
        )))

    def _info(self, components, terminated):
        result = ("draw" if self.last_winner is None else ("win" if self.last_winner == 0 else "loss")) if terminated else None
        totals = sum(components.values())
        info = {
            "step": self.step_count, "winner": self.last_winner, "result": result,
            "alive": self.alive.sum(axis=1).tolist(), "active_masks": self.alive.astype(np.float32).copy(),
            "global_state": self.global_state(), "combat_kills": self.combat_kills.tolist(),
            "weapon_losses": self.weapon_losses.tolist(), "oob": self.oob_losses.tolist(),
            "collision": self.collision_losses.tolist(), "attack_window_count": self.attack_window_count.tolist(),
            "radar_contact_steps": self.radar_contact_steps.tolist(), **self._counts(),
            **{key: float(value[0]) for key, value in components.items()},
            "team_reward_components": {key: value.tolist() for key, value in components.items()},
            "total_reward": float(totals[0]), "team_rewards": totals.tolist(), "potential": self._potential,
        }
        if terminated:
            info["episode_summary"] = {
                "return": self._episode_return, "result": result, **self._counts(),
                **self._episode_rewards, "episode_length": self.step_count,
            }
        return info

    def render(self):
        if self._renderer is None:
            from .rendering import CombatRenderer
            self._renderer = CombatRenderer(self.config, self.render_config, self.project_root)
        return self._renderer.render(self)

    def close(self):
        self._renderer = None
