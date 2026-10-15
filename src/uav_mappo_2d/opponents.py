from __future__ import annotations

import math

import numpy as np

from .env import UAVCombatEnv, wrap_angle


class RuleBasedOpponent:
    """Xu Strategies 1/2/3 with shared radar and a highest-priority avoidance rule."""

    def __init__(self, strategy: int = 1) -> None:
        if strategy not in (1, 2, 3):
            raise ValueError("strategy must be 1, 2 or 3")
        self.strategy = strategy
        self.targets = np.full(UAVCombatEnv.N_AGENTS, -1, dtype=np.int32)

    def reset(self) -> None:
        self.targets.fill(-1)

    @staticmethod
    def _distance(env: UAVCombatEnv, team: int, agent: int, target: int) -> float:
        return float(np.linalg.norm(env.positions[1 - team, target] - env.positions[team, agent]))

    def _assign_targets(self, env: UAVCombatEnv, team: int) -> None:
        self.targets.fill(-1)
        friends = [int(i) for i in np.flatnonzero(env.alive[team])]
        enemies = [int(i) for i in np.flatnonzero(env.team_detected_enemy_mask(team))]
        if self.strategy == 1 or not friends or not enemies:
            return
        preferences = {
            i: sorted(enemies, key=lambda j: (self._distance(env, team, i, j), j))
            for i in friends
        }
        if self.strategy == 2:
            for i in friends:
                self.targets[i] = preferences[i][0]
            return
        capacity = min(len(friends), 2 + int((~env.alive[1 - team]).sum()))
        # Nearest-first proposals; a target retains its closest capacity proposers.
        # Rejected agents try their next nearest target until no proposals remain.
        next_choice = {i: 0 for i in friends}
        retained: dict[int, list[int]] = {j: [] for j in enemies}
        pending = friends
        while pending:
            proposed = {j: list(retained[j]) for j in enemies}
            for i in pending:
                if next_choice[i] < len(preferences[i]):
                    target = preferences[i][next_choice[i]]
                    next_choice[i] += 1
                    proposed[target].append(i)
            pending = []
            for j in enemies:
                ordered = sorted(proposed[j], key=lambda i: (self._distance(env, team, i, j), i))
                retained[j] = ordered[:capacity]
                pending.extend(i for i in ordered[capacity:] if next_choice[i] < len(preferences[i]))
            pending.sort()
        for target, agents in retained.items():
            for agent in agents:
                self.targets[agent] = target

    @staticmethod
    def _steering_action(error: float) -> int:
        # Quantization to the closest allowed turn increment is a project convention.
        return min(range(5, 10), key=lambda a: (abs(UAVCombatEnv.HEADING_DELTAS[a] - error), a))

    def act(self, env: UAVCombatEnv, team: int = 1) -> np.ndarray:
        self._assign_targets(env, team)
        masks = env.available_action_masks(team)
        actions = np.full(env.N_AGENTS, env.NO_OP, dtype=np.int64)
        friends = [int(i) for i in np.flatnonzero(env.alive[team])]
        for agent in friends:
            if not masks[agent].all():
                actions[agent] = 5 if masks[agent, 5] else 9
                continue
            desired_heading = 0.0 if team == 0 else math.pi
            target = int(self.targets[agent])
            if target >= 0:
                delta = env.positions[1 - team, target] - env.positions[team, agent]
                desired_heading = math.atan2(float(delta[1]), float(delta[0]))
            elif self.strategy == 3:
                others = [i for i in friends if i != agent]
                if others:
                    farthest = min(others, key=lambda i: (-float(np.linalg.norm(env.positions[team, i] - env.positions[team, agent])), i))
                    delta = env.positions[team, farthest] - env.positions[team, agent]
                    desired_heading = math.atan2(float(delta[1]), float(delta[0]))
            error = float(wrap_angle(desired_heading - env.headings[team, agent]))
            action = self._steering_action(error)
            if action == 7:
                difference = env.config.initial_speed - env.speeds[team, agent]
                action = min(range(5), key=lambda a: (abs(env.SPEED_DELTAS[a] - difference), a))
            actions[agent] = action
        return actions
