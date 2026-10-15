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
    """Fixed homogeneous 5v5, two-dimensional, constant-altitude air combat."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}
    N_AGENTS = 5
    OBS_DIM = 68
    STATE_DIM = 60
    ACTION_DIM = 2
    REWARD_COMPONENTS = (
        "weapon_reward", "oob_reward", "collision_reward",
        "near_collision_reward", "terminal_reward", "shaping_reward",
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
        self.n_agents = self.N_AGENTS
        self.obs_dim = self.OBS_DIM
        self.state_dim = self.STATE_DIM
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(2, self.N_AGENTS, self.ACTION_DIM), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(2, self.N_AGENTS, self.OBS_DIM), dtype=np.float32
        )
        self.positions = np.zeros((2, self.N_AGENTS, 2), dtype=np.float64)
        self.speeds = np.zeros((2, self.N_AGENTS), dtype=np.float64)
        self.headings = np.zeros((2, self.N_AGENTS), dtype=np.float64)
        self.alive = np.ones((2, self.N_AGENTS), dtype=bool)
        self.step_count = 0
        self.last_winner: int | None = None
        self.combat_kills = np.zeros(2, dtype=np.int32)
        self.oob_losses = np.zeros(2, dtype=np.int32)
        self.collision_losses = np.zeros(2, dtype=np.int32)
        self._potential = 0.0
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

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        cfg = self.config
        base_y = (
            np.arange(self.N_AGENTS, dtype=np.float64) - (self.N_AGENTS - 1) / 2
        ) * cfg.spawn_y_spacing
        position_jitter = self.np_random.uniform(
            -cfg.spawn_jitter, cfg.spawn_jitter, size=(2, self.N_AGENTS, 2)
        )
        heading_jitter = self.np_random.uniform(
            -math.radians(cfg.heading_jitter_degrees),
            math.radians(cfg.heading_jitter_degrees),
            size=(2, self.N_AGENTS),
        )
        initial_speeds = self.np_random.uniform(
            cfg.initial_speed_min, cfg.initial_speed_max, size=(2, self.N_AGENTS)
        )

        self.positions[0, :, 0] = -cfg.spawn_x + position_jitter[0, :, 0]
        self.positions[1, :, 0] = cfg.spawn_x + position_jitter[1, :, 0]
        self.positions[0, :, 1] = base_y + position_jitter[0, :, 1]
        self.positions[1, :, 1] = base_y + position_jitter[1, :, 1]
        self.headings[0] = heading_jitter[0]
        self.headings[1] = wrap_angle(np.pi + heading_jitter[1])
        self.speeds[:] = initial_speeds
        self.alive.fill(True)
        self.step_count = 0
        self.last_winner = None
        self.combat_kills.fill(0)
        self.oob_losses.fill(0)
        self.collision_losses.fill(0)
        self._episode_return = 0.0
        self._episode_rewards = {key: 0.0 for key in self.REWARD_COMPONENTS}
        for team in range(2):
            for agent in range(self.N_AGENTS):
                self._trails[team][agent].clear()
                self._trails[team][agent].append(self.positions[team, agent].copy())
        self._potential = self._compute_potential()
        return self._get_actor_obs(), self._info(
            components={key: np.zeros(2) for key in self.REWARD_COMPONENTS},
            terminated=False,
        )

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {actions.shape}"
            )
        actions = np.clip(actions, -1.0, 1.0)
        previous_positions = self.positions.copy()
        self._update_kinematics(actions)
        oob_this_step = self._resolve_out_of_bounds()
        collision_this_step = self._resolve_collisions(previous_positions)
        combat_this_step = self._resolve_attacks()
        near_collision_reward = self._near_collision_reward(previous_positions)
        self.step_count += 1

        terminated = self._is_terminal()
        self.last_winner = self._determine_winner() if terminated else None
        terminal_reward = np.zeros(2, dtype=np.float64)
        if terminated and self.last_winner is not None:
            terminal_reward[self.last_winner] = self.reward_config.terminal_reward
            terminal_reward[1 - self.last_winner] = -self.reward_config.terminal_reward

        next_potential = 0.0 if terminated else self._compute_potential()
        shaping_reward = self.reward_config.shaping_scale * (
            self.reward_config.shaping_gamma * next_potential - self._potential
        )
        self._potential = next_potential
        rcfg = self.reward_config
        components = {
            "weapon_reward": rcfg.weapon_kill_reward * combat_this_step
            - rcfg.weapon_loss_penalty * combat_this_step[::-1],
            "oob_reward": -rcfg.oob_penalty * oob_this_step,
            "collision_reward": -rcfg.collision_penalty * collision_this_step,
            "near_collision_reward": near_collision_reward,
            "terminal_reward": terminal_reward,
            "shaping_reward": np.array([shaping_reward, -shaping_reward]),
        }
        team_rewards = sum(components.values())
        for key, value in components.items():
            self._episode_rewards[key] += float(value[0])
        self._episode_return += float(team_rewards[0])
        rewards = np.repeat(team_rewards[:, None], self.N_AGENTS, axis=1).astype(np.float32)
        info = self._info(
            components=components,
            terminated=terminated,
        )
        return self._get_actor_obs(), rewards, terminated, False, info

    def _update_kinematics(self, actions: np.ndarray) -> None:
        cfg = self.config
        for team in range(2):
            active = self.alive[team]
            acceleration = actions[team, :, 0] * cfg.max_acceleration
            turn_rate = actions[team, :, 1] * cfg.max_turn_rate
            self.speeds[team, active] = np.clip(
                self.speeds[team, active] + acceleration[active] * cfg.dt,
                cfg.speed_min,
                cfg.speed_max,
            )
            self.headings[team, active] = wrap_angle(
                self.headings[team, active] + turn_rate[active] * cfg.dt
            )
            self.positions[team, active, 0] += (
                self.speeds[team, active]
                * np.cos(self.headings[team, active])
                * cfg.dt
            )
            self.positions[team, active, 1] += (
                self.speeds[team, active]
                * np.sin(self.headings[team, active])
                * cfg.dt
            )
            for agent in np.flatnonzero(active):
                self._trails[team][int(agent)].append(
                    self.positions[team, int(agent)].copy()
                )

    def _resolve_out_of_bounds(self) -> np.ndarray:
        half = self.config.map_size / 2.0
        losses = np.zeros(2, dtype=np.int32)
        for team in range(2):
            outside = self.alive[team] & np.any(
                np.abs(self.positions[team]) > half, axis=1
            )
            losses[team] = int(outside.sum())
            self.alive[team, outside] = False
        self.oob_losses += losses
        return losses

    def _alive_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Unordered pairs from the current surviving snapshot, across both teams."""
        first, second = np.triu_indices(2 * self.N_AGENTS, k=1)
        alive = self.alive.ravel()
        valid = alive[first] & alive[second]
        return first[valid], second[valid]

    def _resolve_collisions(self, previous_positions: np.ndarray) -> np.ndarray:
        first, second = self._alive_pairs()
        previous = previous_positions.reshape(-1, 2)
        current = self.positions.reshape(-1, 2)
        r0 = previous[first] - previous[second]
        displacement = current - previous
        dr = displacement[first] - displacement[second]
        denominator = np.sum(dr * dr, axis=1)
        tau = np.zeros(len(first), dtype=np.float64)
        np.divide(-np.sum(r0 * dr, axis=1), denominator,
                  out=tau, where=denominator > 1e-12)
        tau = np.clip(tau, 0.0, 1.0)
        closest = r0 + tau[:, None] * dr
        collided = np.sum(closest * closest, axis=1) <= self.config.collision_distance**2
        victims = np.unique(np.concatenate((first[collided], second[collided])))
        # Collect every pair before removing anyone, so multi-aircraft events are synchronous.
        losses = np.bincount(victims // self.N_AGENTS, minlength=2).astype(np.int32)
        self.alive.ravel()[victims] = False
        self.collision_losses += losses
        return losses

    def _near_collision_reward(self, previous_positions: np.ndarray) -> np.ndarray:
        first, second = self._alive_pairs()
        previous = previous_positions.reshape(-1, 2)
        current = self.positions.reshape(-1, 2)
        d_prev = np.linalg.norm(previous[first] - previous[second], axis=1)
        d_curr = np.linalg.norm(current[first] - current[second], axis=1)
        warning = self.config.collision_warning_distance
        approaching = (d_curr < warning) & (d_curr < d_prev)
        penalty = -self.reward_config.near_collision_scale * np.clip(
            (d_prev[approaching] - d_curr[approaching]) / warning, 0.0, 1.0
        )
        rewards = np.zeros(2, dtype=np.float64)
        np.add.at(rewards, first[approaching] // self.N_AGENTS, penalty)
        np.add.at(rewards, second[approaching] // self.N_AGENTS, penalty)
        return rewards

    def team_detected_enemy_mask(self, team: int) -> np.ndarray:
        """Current circular radar union; no contact memory or hidden enemy access."""
        detected = np.zeros(self.N_AGENTS, dtype=bool)
        friends = np.flatnonzero(self.alive[team])
        enemies = np.flatnonzero(self.alive[1 - team])
        if friends.size and enemies.size:
            delta = (self.positions[team, friends][:, None, :]
                     - self.positions[1 - team, enemies][None, :, :])
            detected[enemies] = np.any(
                np.sum(delta * delta, axis=-1) <= self.config.radar_range**2, axis=0
            )
        return detected

    def _compute_pair_geometry(
        self, attacker_team: int, attacker: int, target_team: int, target: int
    ) -> tuple[float, float, float]:
        delta = self.positions[target_team, target] - self.positions[attacker_team, attacker]
        distance = float(np.linalg.norm(delta))
        los = math.atan2(float(delta[1]), float(delta[0]))
        alpha = float(wrap_angle(los - self.headings[attacker_team, attacker]))
        beta = float(wrap_angle(los - self.headings[target_team, target]))
        return distance, alpha, beta

    def _attack_valid(
        self, attacker_team: int, attacker: int, target_team: int, target: int
    ) -> bool:
        if not self.alive[attacker_team, attacker] or not self.alive[target_team, target]:
            return False
        distance, alpha, beta = self._compute_pair_geometry(
            attacker_team, attacker, target_team, target
        )
        return (
            distance <= self.config.attack_range
            and abs(alpha) <= math.radians(self.config.attack_ata_degrees) + 1e-12
            and abs(beta) <= math.radians(self.config.attack_aspect_degrees) + 1e-12
        )

    def _resolve_attacks(self) -> np.ndarray:
        victims: list[set[int]] = [set(), set()]
        for attacker_team in range(2):
            target_team = 1 - attacker_team
            for attacker in np.flatnonzero(self.alive[attacker_team]):
                valid_targets = [
                    int(target)
                    for target in np.flatnonzero(self.alive[target_team])
                    if self._attack_valid(
                        attacker_team, int(attacker), target_team, int(target)
                    )
                ]
                if valid_targets:
                    chosen = min(
                        valid_targets,
                        key=lambda target: (
                            self._compute_pair_geometry(
                                attacker_team, int(attacker), target_team, target
                            )[0],
                            target,
                        ),
                    )
                    victims[target_team].add(chosen)

        combat_this_step = np.zeros(2, dtype=np.int32)
        for victim_team in range(2):
            for target in sorted(victims[victim_team]):
                self.alive[victim_team, target] = False
            attacker_team = 1 - victim_team
            combat_this_step[attacker_team] = len(victims[victim_team])
        self.combat_kills += combat_this_step
        return combat_this_step

    def _pair_quality(
        self, attacker_team: int, attacker: int, target_team: int, target: int
    ) -> float:
        distance, alpha, beta = self._compute_pair_geometry(
            attacker_team, attacker, target_team, target
        )
        q_distance = math.exp(-distance / self.config.potential_distance_scale)
        q_alpha = (1.0 + math.cos(alpha)) / 2.0
        q_beta = (1.0 + math.cos(beta)) / 2.0
        return 0.30 * q_distance + 0.35 * q_alpha + 0.35 * q_beta

    def _team_attack_quality(self, attacker_team: int) -> float:
        target_team = 1 - attacker_team
        attackers = np.flatnonzero(self.alive[attacker_team])
        targets = np.flatnonzero(self.team_detected_enemy_mask(attacker_team))
        if attackers.size == 0 or targets.size == 0:
            return 0.0
        target_scores = [
            max(
                self._pair_quality(
                    attacker_team, int(attacker), target_team, int(target)
                )
                for attacker in attackers
            )
            for target in targets
        ]
        return float(np.mean(target_scores))

    def _compute_potential(self) -> float:
        return self._team_attack_quality(0) - self._team_attack_quality(1)

    def _is_terminal(self) -> bool:
        return (
            not self.alive[0].any()
            or not self.alive[1].any()
            or self.step_count >= self.config.episode_steps
        )

    def _determine_winner(self) -> int | None:
        red_alive = int(self.alive[0].sum())
        blue_alive = int(self.alive[1].sum())
        if red_alive > blue_alive:
            return 0
        if blue_alive > red_alive:
            return 1
        return None

    def _get_actor_obs(self) -> np.ndarray:
        cfg = self.config
        observations = np.zeros((2, self.N_AGENTS, self.OBS_DIM), dtype=np.float32)
        distance_scale = math.sqrt(2.0) * cfg.map_size
        speed_scale = cfg.speed_max - cfg.speed_min
        half = cfg.map_size / 2.0
        for team in range(2):
            detected = self.team_detected_enemy_mask(team)
            for agent in range(self.N_AGENTS):
                if not self.alive[team, agent]:
                    continue
                own = observations[team, agent]
                own[:5] = (
                    self.positions[team, agent, 0] / half,
                    self.positions[team, agent, 1] / half,
                    (self.speeds[team, agent] - cfg.speed_min) / speed_scale,
                    math.sin(self.headings[team, agent]),
                    math.cos(self.headings[team, agent]),
                )
                entities = [
                    (team, other)
                    for other in range(self.N_AGENTS)
                    if other != agent
                ] + [(1 - team, other) for other in range(self.N_AGENTS)]
                for slot, (other_team, other) in enumerate(entities):
                    if not self.alive[other_team, other]:
                        continue
                    if other_team != team and not detected[other]:
                        continue
                    distance, alpha, beta = self._compute_pair_geometry(
                        team, agent, other_team, other
                    )
                    start = 5 + 7 * slot
                    own[start : start + 7] = (
                        distance / distance_scale,
                        math.sin(alpha),
                        math.cos(alpha),
                        math.sin(beta),
                        math.cos(beta),
                        (self.speeds[other_team, other] - self.speeds[team, agent])
                        / speed_scale,
                        1.0,
                    )
        return observations

    def global_state(self) -> np.ndarray:
        cfg = self.config
        state = np.zeros(self.STATE_DIM, dtype=np.float32)
        half = cfg.map_size / 2.0
        speed_scale = cfg.speed_max - cfg.speed_min
        for team in range(2):
            for agent in range(self.N_AGENTS):
                if not self.alive[team, agent]:
                    continue
                start = (team * self.N_AGENTS + agent) * 6
                state[start : start + 6] = (
                    self.positions[team, agent, 0] / half,
                    self.positions[team, agent, 1] / half,
                    (self.speeds[team, agent] - cfg.speed_min) / speed_scale,
                    math.sin(self.headings[team, agent]),
                    math.cos(self.headings[team, agent]),
                    1.0,
                )
        return state

    def _info(
        self,
        *,
        components: dict[str, np.ndarray],
        terminated: bool,
    ) -> dict[str, Any]:
        result = None
        if terminated:
            result = "draw" if self.last_winner is None else (
                "win" if self.last_winner == 0 else "loss"
            )
        info: dict[str, Any] = {
            "step": self.step_count,
            "winner": self.last_winner,
            "result": result,
            "alive": self.alive.sum(axis=1).astype(int).tolist(),
            "active_masks": self.alive.astype(np.float32).copy(),
            "global_state": self.global_state(),
            "combat_kills": self.combat_kills.astype(int).tolist(),
            "oob": self.oob_losses.astype(int).tolist(),
            "collision": self.collision_losses.astype(int).tolist(),
            **{key: float(value[0]) for key, value in components.items()},
            "event_reward": float(sum(components[key][0] for key in self.REWARD_COMPONENTS[:4])),
            "total_reward": float(sum(value[0] for value in components.values())),
            "team_rewards": sum(components.values()).tolist(),
            "potential": float(self._potential),
        }
        if terminated:
            info["episode_summary"] = {
                "return": float(self._episode_return),
                "result": result,
                "red_alive": int(self.alive[0].sum()),
                "blue_alive": int(self.alive[1].sum()),
                "red_combat_kill": int(self.combat_kills[0]),
                "blue_combat_kill": int(self.combat_kills[1]),
                "red_oob": int(self.oob_losses[0]),
                "blue_oob": int(self.oob_losses[1]),
                "red_collision": int(self.collision_losses[0]),
                "blue_collision": int(self.collision_losses[1]),
                **self._episode_rewards,
                "event_reward": sum(self._episode_rewards[key] for key in self.REWARD_COMPONENTS[:4]),
                "episode_length": self.step_count,
            }
        return info

    def render(self) -> np.ndarray:
        if self._renderer is None:
            from .rendering import CombatRenderer

            self._renderer = CombatRenderer(
                self.config, self.render_config, self.project_root
            )
        return self._renderer.render(self)

    def close(self) -> None:
        self._renderer = None
