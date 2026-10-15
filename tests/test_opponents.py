import math

import numpy as np
import pytest

from uav_mappo_2d.opponents import RuleBasedOpponent


def test_strategy1_straight_speed_avoidance_and_recovery(make_env):
    env = make_env((0,0),(1,0))
    rule = RuleBasedOpponent(1)
    assert rule.act(env)[0] == 2
    env.positions[1,0],env.positions[0,0] = (0,0),(-400,0)
    assert rule.act(env)[0] == 5
    env.positions[0,0] = (-4000,-4000)
    env.headings[1,0] = math.pi - math.radians(10)
    assert rule.act(env)[0] == 9
    env.headings[1,0] = math.pi
    env.speeds[1,0] = 99
    assert rule.act(env)[0] == 4


def test_strategy2_nearest_contact_direct_pursuit_and_no_contact(make_env):
    env = make_env((0,0),(0,1),(1,0))
    env.positions[1,0] = (0,0)
    env.headings[1,0] = 0
    env.positions[0,0],env.positions[0,1] = (1000,100),(1500,-1000)
    rule = RuleBasedOpponent(2)
    action = rule.act(env)[0]
    assert rule.targets[0] == 0 and action == 9
    # Target's heading is irrelevant to direct pursuit (no tail point).
    env.headings[0,0] = math.pi
    assert rule.act(env)[0] == action
    env.positions[0] = (4500,-4500)
    assert rule.act(env)[0] == 5  # wrap(pi)=-pi: deterministic right turn toward initial heading.
    assert np.all(rule.targets == -1)


def test_strategy3_capacity_and_farther_reassignment(make_env):
    env = make_env()
    env.positions[1,:,0] = np.arange(5)*10
    env.positions[1,:,1] = 0
    env.positions[0] = (4500,4500)
    env.positions[0,0],env.positions[0,1] = (1000,0),(1500,0)
    rule = RuleBasedOpponent(3)
    rule.act(env)
    assert rule.targets.tolist() == [-1,1,1,0,0]
    assert all(t in (-1,0,1) for t in rule.targets)
    env.alive[0,2] = False
    rule.act(env)
    assert rule.targets.tolist() == [1,1,0,0,0]
    old = rule.targets.copy()
    rule.reset()
    rule.act(env)
    np.testing.assert_array_equal(old,rule.targets)


def test_strategy3_no_contact_farthest_teammate(make_env):
    env = make_env((1,0),(1,1),(1,2),(0,0))
    env.positions[0,0] = (4500,-4500)
    env.positions[1,0],env.positions[1,1],env.positions[1,2] = (0,0),(1000,0),(-3000,1000)
    env.headings[1,0] = 0
    rule = RuleBasedOpponent(3)
    assert rule.act(env)[0] == 9
    assert np.all(rule.targets == -1)


@pytest.mark.parametrize('strategy',[1,2,3])
def test_all_strategies_obey_avoidance_and_dead_masks(make_env,strategy):
    env = make_env((1,0),(0,0))
    env.positions[1,0],env.positions[0,0] = (0,0),(400,50)
    env.headings[1,0] = 0
    actions = RuleBasedOpponent(strategy).act(env)
    assert actions[0] == 5
    assert np.all(actions[1:] == 2)
    masks = env.available_action_masks(1)
    assert np.all(masks[np.arange(5),actions])
    env.positions[0,0,1] = -50
    assert RuleBasedOpponent(strategy).act(env)[0] == 9


@pytest.mark.parametrize('strategy',[2,3])
def test_target_lost_or_dead_is_not_used(make_env,strategy):
    env = make_env((1,0),(0,0))
    env.positions[1,0],env.positions[0,0] = (0,0),(1000,0)
    rule = RuleBasedOpponent(strategy)
    rule.act(env)
    assert rule.targets[0] == 0
    env.positions[0,0] = (2000.001,0)
    rule.act(env)
    assert rule.targets[0] == -1
    env.positions[0,0] = (1000,0)
    env.alive[0,0] = False
    rule.act(env)
    assert rule.targets[0] == -1
