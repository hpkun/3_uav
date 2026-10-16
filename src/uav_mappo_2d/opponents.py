from __future__ import annotations

import math
import numpy as np

from .env import UAVCombatEnv, wrap_angle


class RuleBasedOpponent:
    """Stateless Greedy Shooter: nearest detected target, direct constant-speed pursuit."""

    @staticmethod
    def control(env, team, agent, desired_heading):
        error = wrap_angle(desired_heading - env.headings[team, agent])
        maximum = math.radians(env.config.max_turn_rate_degrees) * env.config.dt
        return np.array([0., np.clip(error / maximum, -1., 1.)], dtype=np.float32)

    @staticmethod
    def boundary_heading(env, team, agent):
        heading = env.headings[team, agent]
        predicted = env.positions[team, agent] + env.speeds[team, agent] * env.config.dt * np.array([
            math.cos(heading), math.sin(heading),
        ])
        if np.any(np.abs(predicted) > env.config.map_size / 2):
            # Continuous finite-turn adaptation, not an instantaneous physical reversal.
            return float(wrap_angle(heading + math.pi))
        return None

    @staticmethod
    def nearest_target(env, team, agent, detected):
        candidates = np.flatnonzero(detected)
        if not len(candidates):
            return None
        return min(candidates, key=lambda j: (
            float(np.linalg.norm(env.positions[1-team, j] - env.positions[team, agent])), int(j),
        ))

    def act(self, env, team=1):
        actions = np.zeros((UAVCombatEnv.N_AGENTS, UAVCombatEnv.ACTION_DIM), dtype=np.float32)
        detected = env.team_detected_enemy_mask(team)
        for agent in np.flatnonzero(env.alive[team]):
            heading = self.boundary_heading(env, team, agent)
            if heading is None:
                target = self.nearest_target(env, team, agent, detected)
                if target is None:
                    continue
                delta = env.positions[1-team, target] - env.positions[team, agent]
                heading = math.atan2(float(delta[1]), float(delta[0]))
            actions[agent] = self.control(env, team, agent, heading)
        return actions
