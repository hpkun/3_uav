from __future__ import annotations

import math
import numpy as np

from .env import UAVCombatEnv, wrap_angle


class RuleBasedOpponent:
    """V2 continuous direct / assigned-tail / threat-aware pursuit controllers."""

    def __init__(self, strategy=2):
        if strategy not in (1, 2, 3):
            raise ValueError("strategy must be 1, 2 or 3")
        self.strategy = strategy
        self.targets = np.full(UAVCombatEnv.N_AGENTS, -1, dtype=np.int32)
        self.reset()

    def reset(self):
        self.targets.fill(-1)
        self._last_assignment_step = -10**9
        self._lane_y = None
        self._waypoint_x = None

    @staticmethod
    def _distance(env, team, agent, target):
        return float(np.linalg.norm(env.positions[1 - team, target] - env.positions[team, agent]))

    @staticmethod
    def control(env, team, agent, desired_heading, desired_speed):
        cfg = env.config
        return np.array([
            np.clip((desired_speed - env.speeds[team, agent]) / (cfg.max_acceleration * cfg.dt), -1, 1),
            np.clip(wrap_angle(desired_heading - env.headings[team, agent])
                    / (math.radians(cfg.max_turn_rate_degrees) * cfg.dt), -1, 1),
        ], dtype=np.float32)

    @staticmethod
    def boundary_point(env, team, agent):
        half = env.config.map_size / 2
        x, y = env.positions[team, agent]
        vx, vy = math.cos(env.headings[team, agent]), math.sin(env.headings[team, agent])
        margin = env.config.blue_boundary_margin
        outward = ((x >= half - margin and vx > 0) or (x <= -half + margin and vx < 0)
                   or (y >= half - margin and vy > 0) or (y <= -half + margin and vy < 0))
        if not outward:
            return None
        point = np.clip(env.positions[team, agent], -3500., 3500.)
        return np.zeros(2) if np.linalg.norm(point - env.positions[team, agent]) < 1e-6 else point

    @staticmethod
    def _heading_to(env, team, agent, point):
        delta = point - env.positions[team, agent]
        return math.atan2(float(delta[1]), float(delta[0]))

    def _assign_targets(self, env, team):
        friends = [int(i) for i in np.flatnonzero(env.alive[team])]
        enemies = [int(i) for i in np.flatnonzero(env.team_detected_enemy_mask(team))]
        self.targets.fill(-1)
        pairs = sorted((self._distance(env, team, i, j), i, j) for i in friends for j in enemies)
        matched = set()
        for _, agent, target in pairs:
            if self.targets[agent] < 0 and target not in matched:
                self.targets[agent] = target
                matched.add(target)
        for agent in friends:
            if self.targets[agent] < 0 and enemies:
                self.targets[agent] = min(enemies, key=lambda j: (self._distance(env, team, agent, j), j))
        self._last_assignment_step = env.step_count

    def _needs_assignment(self, env, team):
        if env.step_count - self._last_assignment_step >= env.config.blue_assignment_interval:
            return True
        detected = env.team_detected_enemy_mask(team)
        for agent in np.flatnonzero(env.alive[team]):
            target = int(self.targets[agent])
            if (target >= 0 and not detected[target]) or (target < 0 and detected.any()):
                return True
        return False

    def _avoidance(self, env, team, agent, detected):
        # Only known entities are consulted; any enemy within 300m is radar-visible.
        entities = [(team, int(i)) for i in np.flatnonzero(env.alive[team]) if i != agent]
        entities += [(1 - team, int(j)) for j in np.flatnonzero(detected)]
        if not entities:
            return None
        other_team, other = min(entities, key=lambda pair: (
            float(np.linalg.norm(env.positions[pair] - env.positions[team, agent])), pair,
        ))
        delta = env.positions[other_team, other] - env.positions[team, agent]
        distance = float(np.linalg.norm(delta))
        own_velocity = env.speeds[team, agent] * np.array([math.cos(env.headings[team, agent]), math.sin(env.headings[team, agent])])
        other_velocity = env.speeds[other_team, other] * np.array([math.cos(env.headings[other_team, other]), math.sin(env.headings[other_team, other])])
        if distance < env.config.blue_collision_avoidance_distance and np.dot(delta, other_velocity - own_velocity) < 0:
            heading = math.atan2(float(-delta[1]), float(-delta[0]))
            speed = max(env.config.speed_min, env.speeds[team, agent] - 20)
            return heading, speed
        return None

    def _break_action(self, env, team, agent, detected):
        threats = []
        for target in np.flatnonzero(detected):
            distance, alpha, beta = env._compute_pair_geometry(1 - team, target, team, agent)
            if distance <= 1500 and abs(alpha) <= math.radians(45) + 1e-12 and abs(beta) <= math.radians(70) + 1e-12:
                threats.append((distance, int(target)))
        if not threats:
            return None
        _, target = min(threats)
        cfg = env.config
        enemy_heading = env.headings[1 - team, target]
        enemy_next = env.positions[1 - team, target] + cfg.dt * env.speeds[1 - team, target] * np.array([math.cos(enemy_heading), math.sin(enemy_heading)])
        speed = min(cfg.speed_max, env.speeds[team, agent] + cfg.max_acceleration * cfg.dt)
        choices = []
        for turn in (-1., 1.):
            heading = float(wrap_angle(env.headings[team, agent] + turn * math.radians(cfg.max_turn_rate_degrees) * cfg.dt))
            own_next = env.positions[team, agent] + speed * cfg.dt * np.array([math.cos(heading), math.sin(heading)])
            delta = own_next - enemy_next
            beta = abs(float(wrap_angle(math.atan2(float(delta[1]), float(delta[0])) - heading)))
            choices.append((beta, turn))
        # Single-step constant-velocity enemy forecast; equal scores choose right.
        _, turn = max(choices, key=lambda pair: (pair[0], -pair[1]))
        action = self.control(env, team, agent, env.headings[team, agent], cfg.speed_max)
        action[1] = turn
        return action

    def _patrol_point(self, env, team, agent):
        point = np.array([self._waypoint_x[agent], self._lane_y[agent]])
        if np.linalg.norm(point - env.positions[team, agent]) < 300:
            self._waypoint_x[agent] *= -1
            point[0] = self._waypoint_x[agent]
        return point

    def act(self, env, team=1):
        if self._lane_y is None:
            self._lane_y = env.positions[team, :, 1].copy()
            self._waypoint_x = np.where(env.positions[team, :, 0] >= 0, 1000., -1000.)
        detected = env.team_detected_enemy_mask(team)
        if self.strategy == 1:
            self.targets.fill(-1)
            for agent in np.flatnonzero(env.alive[team]):
                candidates = np.flatnonzero(detected)
                if len(candidates):
                    self.targets[agent] = min(candidates, key=lambda j: (self._distance(env, team, agent, j), j))
        elif self._needs_assignment(env, team):
            self._assign_targets(env, team)
        actions = np.zeros((env.N_AGENTS, env.ACTION_DIM), dtype=np.float32)
        for agent in np.flatnonzero(env.alive[team]):
            point = self.boundary_point(env, team, agent)
            speed = 100.0
            if point is not None:
                actions[agent] = self.control(env, team, agent, self._heading_to(env, team, agent, point), speed)
                continue
            avoidance = self._avoidance(env, team, agent, detected)
            if avoidance is not None:
                actions[agent] = self.control(env, team, agent, *avoidance)
                continue
            if self.strategy == 3:
                action = self._break_action(env, team, agent, detected)
                if action is not None:
                    actions[agent] = action
                    continue
            target = int(self.targets[agent])
            if target >= 0:
                point = env.positions[1 - team, target].copy()
                if self.strategy == 1:
                    speed = 120.0
                else:
                    heading = env.headings[1 - team, target]
                    point -= env.config.blue_tail_distance * np.array([math.cos(heading), math.sin(heading)])
                    speed = (env.config.speed_max if self._distance(env, team, agent, target) > env.config.blue_far_distance
                             else np.clip(env.speeds[1 - team, target] + env.config.blue_speed_advantage, env.config.speed_min, env.config.speed_max))
            else:
                point = self._patrol_point(env, team, agent)
            actions[agent] = self.control(env, team, agent, self._heading_to(env, team, agent, point), speed)
        return actions
