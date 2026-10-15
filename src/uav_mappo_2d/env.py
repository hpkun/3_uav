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
    """Xu-thesis-aligned fixed 5v5 baseline; conventions are documented separately."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}
    N_AGENTS = 5
    OBS_DIM = 68
    STATE_DIM = 60
    N_ACTIONS = 10
    NO_OP = 2
    SPEED_DELTAS = np.array([-1, -.5, 0, .5, 1, 0, 0, 0, 0, 0], dtype=np.float64)
    HEADING_DELTAS = np.deg2rad([0, 0, 0, 0, 0, -5, -2, 0, 2, 5])
    RIGHT_ACTIONS = (5, 6, 7)
    LEFT_ACTIONS = (7, 8, 9)
    REWARD_COMPONENTS = (
        "kill_reward", "own_loss_reward", "terminal_reward", "timeout_reward",
        "distance_reward", "attack_angle_reward", "collision_avoidance_reward",
    )

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
        self.n_agents, self.obs_dim, self.state_dim = self.N_AGENTS, self.OBS_DIM, self.STATE_DIM
        self.action_space = spaces.MultiDiscrete(np.full((2, self.N_AGENTS), self.N_ACTIONS))
        # No physical boundary clamp: normalized x/y and distances may exceed one.
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(2, self.N_AGENTS, self.OBS_DIM), dtype=np.float32
        )
        self.positions = np.zeros((2, self.N_AGENTS, 2), dtype=np.float64)
        self.speeds = np.zeros((2, self.N_AGENTS), dtype=np.float64)
        self.headings = np.zeros((2, self.N_AGENTS), dtype=np.float64)
        self.alive = np.ones((2, self.N_AGENTS), dtype=bool)
        self.combat_kills = np.zeros(2, dtype=np.int32)
        self.collision_losses = np.zeros(2, dtype=np.int32)
        self._ever_outside = np.zeros((2, self.N_AGENTS), dtype=bool)
        self.step_count = 0
        self.last_winner: int | None = None
        self._episode_return = 0.0
        self._episode_rewards = {key: 0.0 for key in self.REWARD_COMPONENTS}
        self._renderer = None
        self._trails = [
            [deque(maxlen=self.render_config.trail_length) for _ in range(self.N_AGENTS)]
            for _ in range(2)
        ]

    @property
    def elapsed_time(self) -> float:
        return self.step_count * self.config.dt

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        super().reset(seed=seed)
        cfg = self.config
        base_y = (np.arange(self.N_AGENTS) - (self.N_AGENTS - 1) / 2) * cfg.spawn_y_spacing
        jitter = self.np_random.uniform(-cfg.spawn_jitter, cfg.spawn_jitter, (2, self.N_AGENTS, 2))
        self.positions[0, :, 0] = -cfg.spawn_x + jitter[0, :, 0]
        self.positions[1, :, 0] = cfg.spawn_x + jitter[1, :, 0]
        self.positions[:, :, 1] = base_y + jitter[:, :, 1]
        self.headings[0].fill(0.0)
        self.headings[1].fill(math.pi)
        self.speeds.fill(cfg.initial_speed)
        self.alive.fill(True)
        self.combat_kills.fill(0)
        self.collision_losses.fill(0)
        self._ever_outside.fill(False)
        self.step_count = 0
        self.last_winner = None
        self._episode_return = 0.0
        self._episode_rewards = {key: 0.0 for key in self.REWARD_COMPONENTS}
        for team in range(2):
            for agent in range(self.N_AGENTS):
                self._trails[team][agent].clear()
                self._trails[team][agent].append(self.positions[team, agent].copy())
        return self._get_actor_obs(), self._info(
            {key: np.zeros(2) for key in self.REWARD_COMPONENTS}, terminated=False
        )

    def available_action_masks(self, team: int = 0) -> np.ndarray:
        """Nearest-aircraft 450m constraint; ties use flattened team/agent ID."""
        masks = np.ones((self.N_AGENTS, self.N_ACTIONS), dtype=bool)
        flat_positions, flat_alive = self.positions.reshape(-1, 2), self.alive.ravel()
        for agent in range(self.N_AGENTS):
            if not self.alive[team, agent]:
                masks[agent] = False
                masks[agent, self.NO_OP] = True
                continue
            own_id = team * self.N_AGENTS + agent
            candidates = np.flatnonzero(flat_alive).tolist()
            candidates.remove(own_id)
            if not candidates:
                continue
            delta = flat_positions[candidates] - flat_positions[own_id]
            distances = np.linalg.norm(delta, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= self.config.collision_avoidance_distance:
                los = math.atan2(float(delta[nearest, 1]), float(delta[nearest, 0]))
                alpha = float(wrap_angle(los - self.headings[team, agent]))
                subset = self.LEFT_ACTIONS if alpha < -1e-12 else self.RIGHT_ACTIONS
                masks[agent] = False
                masks[agent, list(subset)] = True
        return masks

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions)
        if actions.shape != self.action_space.shape:
            raise ValueError(f"Expected actions {self.action_space.shape}, got {actions.shape}")
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("Actions must be integers in [0,9]")
        if np.any((actions < 0) | (actions >= self.N_ACTIONS)):
            raise ValueError("Actions must be integers in [0,9]")
        masks = np.stack([self.available_action_masks(team) for team in range(2)])
        if not np.all(np.take_along_axis(masks, actions[..., None], axis=-1)):
            raise ValueError("Action violates the current collision-avoidance mask")
        previous_positions = self.positions.copy()
        previous_metrics = self._enemy_metrics()
        self._update_kinematics(actions)
        self._ever_outside |= self.alive & self.outside_map_mask()
        collisions = self._resolve_collisions(previous_positions)
        kills = self._resolve_attacks()
        self.step_count += 1
        terminated = self._is_terminal()
        self.last_winner = self._determine_winner() if terminated else None
        components = self._reward_components(previous_positions, previous_metrics, kills, collisions, terminated)
        totals = sum(components.values())
        for key in self.REWARD_COMPONENTS:
            self._episode_rewards[key] += float(components[key][0])
        self._episode_return += float(totals[0])
        rewards = np.repeat(totals[:, None], self.N_AGENTS, axis=1).astype(np.float32)
        return self._get_actor_obs(), rewards, terminated, False, self._info(components, terminated)

    def _update_kinematics(self, actions: np.ndarray) -> None:
        cfg = self.config
        for team in range(2):
            active = self.alive[team]
            # Eq. (3.1): position uses v_t and rho_t, actions affect the next state.
            self.positions[team, active, 0] += self.speeds[team, active] * np.cos(self.headings[team, active]) * cfg.dt
            self.positions[team, active, 1] += self.speeds[team, active] * np.sin(self.headings[team, active]) * cfg.dt
            self.speeds[team, active] = np.clip(
                self.speeds[team, active] + self.SPEED_DELTAS[actions[team, active]], cfg.speed_min, cfg.speed_max
            )
            self.headings[team, active] = wrap_angle(
                self.headings[team, active] + self.HEADING_DELTAS[actions[team, active]]
            )
            for agent in np.flatnonzero(active):
                self._trails[team][int(agent)].append(self.positions[team, agent].copy())

    def outside_map_mask(self) -> np.ndarray:
        """Diagnostic only; never changes physical state, reward or terminal status."""
        return np.any(np.abs(self.positions) > self.config.map_size / 2, axis=-1)

    def _alive_pairs(self):
        first, second = np.triu_indices(2 * self.N_AGENTS, k=1)
        valid = self.alive.ravel()[first] & self.alive.ravel()[second]
        return first[valid], second[valid]

    def _resolve_collisions(self, previous_positions: np.ndarray) -> np.ndarray:
        first, second = self._alive_pairs()
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

    def team_detected_enemy_mask(self, team: int) -> np.ndarray:
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

    def _attack_valid(self, attacker_team, attacker, target_team, target) -> bool:
        if not (self.alive[attacker_team, attacker] and self.alive[target_team, target]):
            return False
        distance, alpha, beta = self._compute_pair_geometry(attacker_team, attacker, target_team, target)
        return bool(distance <= self.config.attack_range
                    and abs(alpha) <= math.radians(self.config.attack_ata_degrees) + 1e-12
                    and abs(beta) <= math.radians(self.config.attack_aspect_degrees) + 1e-12)

    def _resolve_attacks(self) -> np.ndarray:
        victims: list[set[int]] = [set(), set()]
        for team in range(2):
            for attacker in np.flatnonzero(self.alive[team]):
                valid = [int(j) for j in np.flatnonzero(self.alive[1 - team])
                         if self._attack_valid(team, attacker, 1 - team, j)]
                if valid:
                    chosen = min(valid, key=lambda j: (self._compute_pair_geometry(team, attacker, 1 - team, j)[0], j))
                    victims[1 - team].add(chosen)
        kills = np.zeros(2, dtype=np.int32)
        for team in range(2):
            for agent in victims[team]:
                self.alive[team, agent] = False
            kills[1 - team] = len(victims[team])
        self.combat_kills += kills
        return kills

    def _enemy_metrics(self) -> tuple[np.ndarray, np.ndarray]:
        distances = np.full((2, self.N_AGENTS), np.inf)
        angles = np.full_like(distances, np.inf)
        for team in range(2):
            targets = np.flatnonzero(self.team_detected_enemy_mask(team))
            if not len(targets):
                continue
            for agent in np.flatnonzero(self.alive[team]):
                geometry = [self._compute_pair_geometry(team, agent, 1 - team, j) for j in targets]
                distances[team, agent] = min(row[0] for row in geometry)
                angles[team, agent] = min(abs(row[1]) for row in geometry)
        return distances, angles

    def _reward_components(self, previous_positions, previous_metrics, kills, collisions, terminated):
        cfg = self.reward_config
        result = {key: np.zeros(2) for key in self.REWARD_COMPONENTS}
        result["kill_reward"] = cfg.enemy_kill_reward * kills
        result["own_loss_reward"] = -cfg.own_loss_penalty * (kills[::-1] + collisions)
        if terminated and self.last_winner is not None:
            winner, loser = self.last_winner, 1 - self.last_winner
            if not self.alive[loser].any():
                result["terminal_reward"][winner] = cfg.all_enemy_destroyed_reward
                result["terminal_reward"][loser] = -cfg.all_own_destroyed_penalty
            else:
                result["timeout_reward"][winner] = cfg.timeout_advantage_reward
                result["timeout_reward"][loser] = -cfg.timeout_disadvantage_penalty
        old_distances, old_angles = previous_metrics
        distances, angles = self._enemy_metrics()
        flat_current = self.positions.reshape(-1, 2)
        flat_previous = previous_positions.reshape(-1, 2)
        for team in range(2):
            survivors = np.flatnonzero(self.alive[team])
            if not len(survivors):
                continue
            distance_rewards, angle_rewards, collision_rewards = [], [], []
            for agent in survivors:
                before_d, now_d = old_distances[team, agent], distances[team, agent]
                before_a, now_a = old_angles[team, agent], angles[team, agent]
                distance_rewards.append((before_d - now_d) / 1000 if np.isfinite(before_d + now_d) and before_d > 1500 else 0.0)
                angle_rewards.append(before_a - now_a if np.isfinite(before_a + now_a) else 0.0)
                own_id = team * self.N_AGENTS + agent
                others = np.flatnonzero(self.alive.ravel()).tolist()
                others.remove(own_id)
                value = 0.0
                if others:
                    nearest = min(others, key=lambda j: (float(np.linalg.norm(flat_current[j] - flat_current[own_id])), j))
                    d_curr = float(np.linalg.norm(flat_current[nearest] - flat_current[own_id]))
                    if d_curr < self.config.collision_reward_distance:
                        d_prev = float(np.linalg.norm(flat_previous[nearest] - flat_previous[own_id]))
                        value = (d_curr - d_prev) / 1000
                collision_rewards.append(value)
            result["distance_reward"][team] = np.mean(distance_rewards)
            result["attack_angle_reward"][team] = np.mean(angle_rewards)
            result["collision_avoidance_reward"][team] = np.mean(collision_rewards)
        return result

    def _is_terminal(self) -> bool:
        return bool(not self.alive[0].any() or not self.alive[1].any() or self.step_count >= self.config.episode_steps)

    def _determine_winner(self) -> int | None:
        red, blue = self.alive.sum(axis=1)
        return 0 if red > blue else (1 if blue > red else None)

    def _get_actor_obs(self) -> np.ndarray:
        cfg = self.config
        observations = np.zeros((2, self.N_AGENTS, self.OBS_DIM), dtype=np.float32)
        for team in range(2):
            detected = self.team_detected_enemy_mask(team)
            for agent in np.flatnonzero(self.alive[team]):
                own = observations[team, agent]
                own[:5] = (* (self.positions[team, agent] / (cfg.map_size / 2)),
                           (self.speeds[team, agent] - cfg.speed_min) / (cfg.speed_max - cfg.speed_min),
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
                        (self.speeds[other_team, other] - self.speeds[team, agent]) / (cfg.speed_max - cfg.speed_min), 1.0,
                    )
        return observations

    def global_state(self) -> np.ndarray:
        cfg = self.config
        state = np.zeros((2, self.N_AGENTS, 6), dtype=np.float32)
        state[..., :2] = self.positions / (cfg.map_size / 2)
        state[..., 2] = (self.speeds - cfg.speed_min) / (cfg.speed_max - cfg.speed_min)
        state[..., 3] = np.sin(self.headings)
        state[..., 4] = np.cos(self.headings)
        state[..., 5] = self.alive
        state[~self.alive] = 0
        return state.ravel()

    def _info(self, components, terminated: bool) -> dict[str, Any]:
        result = ("draw" if self.last_winner is None else ("win" if self.last_winner == 0 else "loss")) if terminated else None
        totals = sum(components.values())
        outside = (self.outside_map_mask() & self.alive).sum(axis=1)
        info = {
            "step": self.step_count, "winner": self.last_winner, "result": result,
            "alive": self.alive.sum(axis=1).tolist(), "active_masks": self.alive.astype(np.float32).copy(),
            "available_action_masks": self.available_action_masks(0),
            "global_state": self.global_state(), "combat_kills": self.combat_kills.tolist(),
            "collision": self.collision_losses.tolist(), "outside_map": outside.tolist(),
            **{key: float(value[0]) for key, value in components.items()},
            "total_reward": float(totals[0]), "team_rewards": totals.tolist(),
        }
        if terminated:
            info["episode_summary"] = {
                "return": self._episode_return, "total_return": self._episode_return, "result": result,
                "red_alive": int(self.alive[0].sum()), "blue_alive": int(self.alive[1].sum()),
                "red_combat_kill": int(self.combat_kills[0]), "blue_combat_kill": int(self.combat_kills[1]),
                "red_collision": int(self.collision_losses[0]), "blue_collision": int(self.collision_losses[1]),
                "red_outside_map": int(outside[0]), "blue_outside_map": int(outside[1]),
                "red_ever_outside": int(self._ever_outside[0].sum()), "blue_ever_outside": int(self._ever_outside[1].sum()),
                **self._episode_rewards, "episode_length": self.step_count,
            }
        return info

    def render(self) -> np.ndarray:
        if self._renderer is None:
            from .rendering import CombatRenderer
            self._renderer = CombatRenderer(self.config, self.render_config, self.project_root)
        return self._renderer.render(self)

    def close(self) -> None:
        self._renderer = None
