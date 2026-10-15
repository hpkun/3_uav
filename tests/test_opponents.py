from __future__ import annotations

import numpy as np

from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.opponents import RuleBasedOpponent


def test_no_contact_advance_and_contact_assignment() -> None:
    env = UAVCombatEnv()
    env.reset(seed=7)
    rule = RuleBasedOpponent()
    actions = rule.act(env, 1)
    assert np.all(rule.targets == -1)
    assert np.all(actions[:, 1] == 0.0)
    assert np.all(actions[:, 0] > 0.0)
    env.positions[0, :, 0] = -500.0
    env.positions[1, :, 0] = 500.0
    rule.act(env, 1)
    assert sorted(rule.targets.tolist()) == list(range(5))
    assert rule.targets.tolist() == list(range(5))
    previous = rule.targets.copy()
    rule.reset()
    rule.act(env, 1)
    np.testing.assert_array_equal(rule.targets, previous)


def test_lost_or_dead_contacts_reassign_immediately() -> None:
    env = UAVCombatEnv()
    env.reset(seed=7)
    env.positions[1, :, 0] = 3000.0
    env.positions[0, :, 0] = -4000.0
    env.positions[0, 0] = (1000.0, 0.0)
    rule = RuleBasedOpponent()
    rule.act(env, 1)
    assert np.all(rule.targets == 0)
    env.positions[0, 0] = (-4000.0, 0.0)
    rule.act(env, 1)
    assert np.all(rule.targets == -1)
    env.positions[0, 2] = (1000.0, 0.0)
    rule.act(env, 1)
    assert np.all(rule.targets == 2)
    env.alive[0, 2] = False
    rule.act(env, 1)
    assert np.all(rule.targets == -1)


def test_more_friends_cover_each_detected_enemy() -> None:
    env = UAVCombatEnv()
    env.reset(seed=3)
    env.positions[1, :, 0] = 2500.0
    env.positions[0, :, 0] = -4500.0
    env.positions[0, 1] = (1000.0, -1500.0)
    env.positions[0, 3] = (1000.0, 1500.0)
    rule = RuleBasedOpponent()
    rule.act(env, 1)
    assert set(rule.targets) == {1, 3}
    assert rule.targets[1] == 1
    assert rule.targets[3] == 3


def test_periodic_reassignment_and_more_contacts_than_friends() -> None:
    env = UAVCombatEnv()
    env.reset(seed=7)
    env.positions[0, :, 0] = -500.0
    env.positions[1, :, 0] = 500.0
    rule = RuleBasedOpponent()
    rule.act(env, 1)
    assert rule.targets.tolist() == list(range(5))
    env.positions[0, [0, 4]] = env.positions[0, [4, 0]]
    env.step_count = 4
    rule.act(env, 1)
    assert rule.targets.tolist() == list(range(5))
    env.step_count = 5
    rule.act(env, 1)
    assert rule.targets.tolist() == [4, 1, 2, 3, 0]
    env.alive[1, 1:4] = False
    rule._assign_targets(env, 1)
    assert rule.targets.tolist() == [4, -1, -1, -1, 0]


def test_boundary_protection_without_contact() -> None:
    env = UAVCombatEnv()
    env.reset(seed=7)
    env.alive[0] = False

    env.positions[1, 0] = (4_600.0, 100.0)
    env.headings[1, 0] = 0.0
    rule = RuleBasedOpponent()
    actions = rule.act(env, 1)
    assert np.all(rule.targets == -1)
    assert actions[0, 1] < -0.9
