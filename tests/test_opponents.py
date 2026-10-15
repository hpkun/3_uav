import math
import numpy as np
import pytest

from uav_mappo_2d.opponents import RuleBasedOpponent


def test_boundary_has_priority_over_avoidance_and_does_not_mutate_state(make_env):
    env = make_env((1,0),(0,0))
    env.positions[1,0],env.positions[0,0] = (4500,100),(4300,100)
    env.headings[1,0],env.headings[0,0] = 0,0
    env.speeds[1,0],env.speeds[0,0] = 100,150
    old = env.positions.copy()
    rule = RuleBasedOpponent(2)
    point = rule.boundary_point(env,1,0)
    np.testing.assert_allclose(point,[3500,100])
    actions = rule.act(env)
    assert actions.dtype == np.float32 and actions.shape == (5,2)
    assert actions[0,1] < -.99 and np.all(np.abs(actions) <= 1)
    np.testing.assert_array_equal(old,env.positions)
    assert np.all(actions[1:] == 0)


@pytest.mark.parametrize('distance,closing,triggered',[(299,True,True),(300,True,False),(299,False,False)])
def test_avoidance_closing_threshold_and_away_heading(make_env,distance,closing,triggered):
    env = make_env((1,0),(0,0))
    env.positions[1,0],env.positions[0,0] = (0,0),(distance,0)
    env.speeds.fill(100)
    env.headings[1,0] = 0 if closing else math.pi
    env.headings[0,0] = math.pi if closing else 0
    rule = RuleBasedOpponent(2)
    avoidance = rule._avoidance(env,1,0,env.team_detected_enemy_mask(1))
    assert (avoidance is not None) == triggered
    if triggered:
        heading,speed = avoidance
        assert abs(heading) == pytest.approx(math.pi) and speed == 80
        action = rule.act(env)[0]
        np.testing.assert_allclose(action,[-1,-1])


def test_s1_direct_nearest_and_hidden_enemy_independence(make_env):
    env = make_env((1,0),(0,0),(0,1))
    env.positions[1,0],env.positions[0,0],env.positions[0,1] = (0,0),(1000,100),(2000,-100)
    env.headings[1,0] = 0
    env.headings[0,0] = math.pi/2
    rule = RuleBasedOpponent(1)
    actions = rule.act(env)
    assert rule.targets[0] == 0 and actions[0,1] > 0
    env.positions[0] = (4500,4500)
    rule.reset()
    a = rule.act(env)
    assert rule.targets[0] == -1
    env.positions[0] = (-4500,-4500)
    b = rule.act(env)
    np.testing.assert_array_equal(a,b)


def test_patrol_lane_switch_and_reset(make_env):
    env = make_env((1,0),(0,0))
    env.positions[1,0],env.positions[0,0] = (3500,600),(-4500,-4500)
    rule = RuleBasedOpponent(2)
    rule.act(env)
    assert rule._lane_y[0] == 600 and rule._waypoint_x[0] == 1000
    env.positions[1,0] = (1000,600)
    rule.act(env)
    assert rule._waypoint_x[0] == -1000 and rule._lane_y[0] == 600
    rule.reset()
    assert rule._lane_y is None and rule._waypoint_x is None


def test_greedy_assignment_periodic_refresh_and_lost_dead_targets(make_env):
    env = make_env()
    env.positions[0,:,0],env.positions[1,:,0] = -500,1000
    env.positions[:,:,1] = np.arange(5)*600-1200
    rule = RuleBasedOpponent(2)
    rule.act(env)
    assert rule.targets.tolist() == list(range(5))
    env.positions[0,[0,4]] = env.positions[0,[4,0]]
    env.step_count = 4
    rule.act(env)
    assert rule.targets.tolist() == list(range(5))
    env.step_count = 5
    rule.act(env)
    assert rule.targets.tolist() == [4,1,2,3,0]
    env.alive[0,4] = False
    rule.act(env)
    assert 4 not in rule.targets
    env.positions[0] = (-4500,-4500)
    rule.act(env)
    assert np.all(rule.targets == -1)
    env.positions[0,0] = (500,0)
    rule.act(env)
    assert np.all(rule.targets == 0)


def test_greedy_matches_shortest_pairs_and_reuses_when_targets_fewer(make_env):
    env = make_env((1,0),(1,1),(0,0),(0,1))
    env.positions[1,0],env.positions[1,1] = (0,0),(200,0)
    env.positions[0,0],env.positions[0,1] = (1000,0),(2000,0)
    rule = RuleBasedOpponent(2)
    rule._assign_targets(env,1)
    assert rule.targets[:2].tolist() == [1,0]
    env.alive[0,1] = False
    rule._assign_targets(env,1)
    assert rule.targets[:2].tolist() == [0,0]


@pytest.mark.parametrize('distance,target_speed,current_speed,acceleration',[(2001,100,140,1),(1000,100,140,-1),(1000,145,140,1)])
def test_s2_tail_point_and_speed_controller(make_env,distance,target_speed,current_speed,acceleration):
    env = make_env((1,0),(0,0))
    env.positions[1,0],env.positions[0,0] = (0,0),(distance,100)
    env.headings[1,0],env.headings[0,0] = 0,math.pi/2
    env.speeds[1,0],env.speeds[0,0] = current_speed,target_speed
    action = RuleBasedOpponent(2).act(env)[0]
    assert action[0] == acceleration
    assert action[1] < 0  # tail point=(distance,-600), unlike direct positive-y pursuit.


def test_s3_no_threat_matches_s2_and_threat_uses_break(make_env):
    env = make_env((1,0),(0,0))
    env.positions[1,0],env.positions[0,0] = (1000,200),(0,0)
    env.headings[1,0],env.headings[0,0] = math.pi,0
    a = RuleBasedOpponent(2).act(env)
    b = RuleBasedOpponent(3).act(env)
    np.testing.assert_array_equal(a,b)
    env.headings[1,0] = 0
    env.speeds[1,0],env.speeds[0,0] = 140,80
    rule = RuleBasedOpponent(3)
    action = rule.act(env)[0]
    assert action[0] == 1 and abs(action[1]) == 1
    assert RuleBasedOpponent(2).act(env)[0,0] == -1
    # Independent forecast: chosen turn must maximize next enemy->own aspect error.
    scores = {}
    for turn in (-1,1):
        heading = turn*math.radians(10)
        position = env.positions[1,0]+150*np.array([math.cos(heading),math.sin(heading)])
        enemy = env.positions[0,0]+np.array([80,0])
        los = math.atan2(position[1]-enemy[1],position[0]-enemy[0])
        scores[turn] = abs((los-heading+math.pi)%(2*math.pi)-math.pi)
    assert scores[int(action[1])] == max(scores.values())
