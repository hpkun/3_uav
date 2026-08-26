from __future__ import annotations

import numpy as np

from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.opponents import RuleBasedOpponent


def test_rule_assignment_reassignment_and_boundary_protection() -> None:
    env = UAVCombatEnv()
    env.reset(seed=7)
    rule = RuleBasedOpponent()
    rule.act(env, 1)
    assert sorted(rule.targets.tolist()) == [0, 1, 2]

    previous = int(rule.targets[0])
    env.alive[0, previous] = False
    rule.act(env, 1)
    assert rule.targets[0] >= 0
    assert rule.targets[0] != previous
    assert env.alive[0, rule.targets[0]]

    env.positions[1, 0] = (4_600.0, 100.0)
    env.headings[1, 0] = 0.0
    actions = rule.act(env, 1)
    assert actions[0, 1] < -0.9
