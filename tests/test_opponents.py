import math
import numpy as np
import pytest

from uav_mappo_2d.opponents import RuleBasedOpponent
from uav_mappo_2d.env import wrap_angle


def setup_pair(make_env,distance):
    env = make_env((1,0),(0,0))
    env.positions[1,0],env.positions[0,0] = (0,0),(distance,0)
    env.headings[1,0] = math.radians(5)
    env.headings[0,0] = math.pi/2
    env.speeds.fill(100)
    return env


@pytest.mark.parametrize('distance',[1500,1000,500])
def test_rear_pursuit_at_all_target_distances_without_acceleration(make_env,distance):
    env = setup_pair(make_env,distance)
    delta = np.array([distance,-1000.])
    desired = math.atan2(delta[1],delta[0])
    action = RuleBasedOpponent().act(env)[0]
    expected = RuleBasedOpponent.control(env,1,0,desired)
    np.testing.assert_allclose(action,expected,atol=1e-6)
    # Both direct pursuit and moving away have different turn semantics here.
    assert action[0] == 0 and action[1] == -1


@pytest.mark.parametrize('heading,expected',[(0,[-1000,0]),(math.pi/2,[0,-1000])])
def test_rear_point_geometry_uses_existing_attack_range(make_env,heading,expected):
    env = setup_pair(make_env,800)
    env.positions[0,0] = (0,0)
    env.headings[0,0] = heading
    np.testing.assert_allclose(RuleBasedOpponent.rear_point(env,1,0),expected,atol=1e-12)
    env.config.attack_range = 750
    np.testing.assert_allclose(RuleBasedOpponent.rear_point(env,1,0),np.array(expected)*.75,atol=1e-12)
    assert not hasattr(env.config,'blue_escape_distance')


def test_rear_heading_is_not_direct_or_away_and_has_no_speed_control(make_env):
    env = setup_pair(make_env,2000)
    env.headings[1,0] = math.atan2(-1000,2000)+math.radians(2)
    actual = RuleBasedOpponent().act(env)[0]
    np.testing.assert_allclose(actual,[0,-.2],atol=1e-6)
    direct = RuleBasedOpponent.control(env,1,0,0.)
    away = RuleBasedOpponent.control(env,1,0,math.pi)
    assert not np.allclose(actual,direct) and not np.allclose(actual,away)


@pytest.mark.parametrize('offset',[0.,5e-7])
def test_rear_point_numerical_fallback_is_target_heading(make_env,offset):
    env = setup_pair(make_env,1000)
    env.positions[0,0],env.headings[0,0] = (0,0),0.
    env.positions[1,0],env.headings[1,0] = (-1000+offset,0),math.radians(5)
    action = RuleBasedOpponent().act(env)[0]
    np.testing.assert_allclose(action,[0,-.5],atol=1e-6)
    assert np.isfinite(action).all()


def test_nearest_detected_target_per_agent_tie_id_and_reselection(make_env):
    env = make_env((1,0),(1,1),(0,0),(0,1))
    env.positions[1,0],env.positions[1,1] = (0,0),(0,100)
    env.positions[0,0],env.positions[0,1] = (1500,0),(-1500,0)
    env.headings.fill(0)
    rule = RuleBasedOpponent()
    mask = env.team_detected_enemy_mask(1)
    assert rule.nearest_target(env,1,0,mask) == 0
    assert rule.nearest_target(env,1,1,mask) == 0
    env.positions[0,1] = (0,-1200)
    assert rule.nearest_target(env,1,0,env.team_detected_enemy_mask(1)) == 1
    desired = math.atan2(*((rule.rear_point(env,1,1)-env.positions[1,0])[::-1]))
    np.testing.assert_allclose(rule.act(env)[0],rule.control(env,1,0,desired))
    assert not vars(rule)


def test_no_contact_straight_flight_hidden_independence_and_dead_zero(make_env):
    env = make_env((1,0),(0,0))
    env.positions[1,0],env.positions[0,0] = (0,0),(4500,4500)
    env.headings[1,0] = .3
    rule = RuleBasedOpponent()
    a = rule.act(env)
    env.positions[0,0] = (-4500,-4500)
    b = rule.act(env)
    np.testing.assert_array_equal(a,np.zeros((5,2),np.float32))
    np.testing.assert_array_equal(a,b)
    assert a.dtype == np.float32


def test_old_policy_mechanisms_are_absent():
    rule = RuleBasedOpponent()
    assert not vars(rule)
    for name in ['strategy','targets','_assign_targets','_needs_assignment',
                 '_avoidance','_break_action','_patrol_point','boundary_point','reset']:
        assert not hasattr(rule,name)
    with pytest.raises(TypeError):
        RuleBasedOpponent(strategy=2)


def test_team_sharing_target_visible_through_other_friend(make_env):
    env = make_env((1,0),(1,1),(0,0))
    env.positions[1,0],env.positions[1,1],env.positions[0,0] = (-4000,0),(-1500,0),(1400,0)
    env.headings[1,0] = math.pi/2
    assert env.team_detected_enemy_mask(1)[0]
    np.testing.assert_allclose(RuleBasedOpponent().act(env)[0],[0,-1])


@pytest.mark.parametrize('axis,sign',[(0,1),(0,-1),(1,1),(1,-1)])
def test_boundary_nominal_prediction_priority_finite_turn_no_state_mutation(make_env,axis,sign):
    env = make_env((1,0),(0,0))
    env.positions[1,0] = (0,0)
    env.positions[1,0,axis] = sign*4950
    heading = {(0,1):0.,(0,-1):math.pi,(1,1):math.pi/2,(1,-1):-math.pi/2}[axis,sign]
    env.headings[1,0] = heading
    env.speeds[1,0] = 100
    env.positions[0,0] = env.positions[1,0] - np.array([0,300])
    rule = RuleBasedOpponent()
    before = [v.copy() for v in [env.positions,env.speeds,env.headings,env.alive]]
    boundary = rule.boundary_heading(env,1,0)
    assert wrap_angle(boundary-heading) == pytest.approx(-math.pi)
    expected = rule.control(env,1,0,boundary)
    np.testing.assert_allclose(rule.act(env)[0],expected)
    assert abs(expected[1]) == 1 and expected[0] == 0
    for old,current in zip(before,[env.positions,env.speeds,env.headings,env.alive]):
        np.testing.assert_array_equal(old,current)


def test_nominal_endpoint_exact_boundary_not_triggered(make_env):
    env = setup_pair(make_env,1200)
    env.positions[1,0] = (4900,0)
    env.headings[1,0] = 0
    env.positions[0,0] = (4900,1200)
    assert RuleBasedOpponent.boundary_heading(env,1,0) is None
    np.testing.assert_allclose(RuleBasedOpponent().act(env)[0],[0,1])


def test_boundary_is_not_shield_and_can_still_oob(make_env,zero_actions):
    env = setup_pair(make_env,1200)
    env.positions[1,0] = (4990,0)
    env.headings[1,0] = 0
    zero_actions[1] = RuleBasedOpponent().act(env)
    env.step(zero_actions)
    assert not env.alive[1,0] and env.oob_losses[1] == 1


def test_no_contact_speed_and_heading_remain_initial_over_steps(make_env,zero_actions):
    env = make_env((1,0),(0,0))
    env.positions[1,0],env.positions[0,0] = (2500,2500),(-2500,-2500)
    env.speeds[1,0],env.headings[1,0] = 103,.1
    rule = RuleBasedOpponent()
    for _ in range(3):
        zero_actions[1] = rule.act(env)
        env.step(zero_actions)
    assert env.speeds[1,0] == 103 and env.headings[1,0] == pytest.approx(.1)
