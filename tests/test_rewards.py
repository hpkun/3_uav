import math

import numpy as np
import pytest


def components(env, previous_positions, previous_metrics):
    return env._reward_components(previous_positions,previous_metrics,np.zeros(2,dtype=int),np.zeros(2,dtype=int),False)


@pytest.mark.parametrize('attacker',[0,1])
def test_kill_loss_terminal_rewards(make_env,noop,attacker):
    env = make_env((0,0),(1,0))
    env.positions[attacker,0],env.positions[1-attacker,0] = (0,0),(400,0)
    env.headings.fill(0)
    _,reward,done,_,info = env.step(noop(env))
    assert done
    np.testing.assert_allclose(reward[attacker],120)
    np.testing.assert_allclose(reward[1-attacker],-120)
    assert info['kill_reward'] == (20 if attacker == 0 else 0)
    assert info['own_loss_reward'] == (0 if attacker == 0 else -20)
    assert info['terminal_reward'] == (100 if attacker == 0 else -100)


@pytest.mark.parametrize('red,blue,value',[(3,2,80),(2,3,-80),(3,3,0)])
def test_timeout_reward(make_env,noop,red,blue,value):
    env = make_env(episode_steps=1)
    env.alive[0,red:] = False
    env.alive[1,blue:] = False
    _,_,done,truncated,info = env.step(noop(env))
    assert done and not truncated
    assert info['timeout_reward'] == value
    assert info['terminal_reward'] == 0


def test_mutual_total_destruction_is_draw_and_collision_own_loss(make_env,noop):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (0,0),(150,0)
    env.headings.fill(0)
    _,reward,done,_,info = env.step(noop(env))
    assert done and info['result'] == 'draw'
    assert info['terminal_reward'] == info['timeout_reward'] == 0
    assert info['kill_reward'] == 0 and info['own_loss_reward'] == -20
    np.testing.assert_allclose(reward,-20)


@pytest.mark.parametrize('before,after,expected',[
    (1900,1800,.1),(1800,1900,-.1),(1500,1400,0),(1400,1300,0),
    (2100,1900,0),(1900,2100,0),
])
def test_distance_aux_threshold_and_contact_continuity(make_env,before,after,expected):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (0,0),(before,0)
    old,metrics = env.positions.copy(),env._enemy_metrics()
    env.positions[1,0,0] = after
    r = components(env,old,metrics)
    assert r['distance_reward'][0] == pytest.approx(expected)


@pytest.mark.parametrize('old_angle,new_angle,expected',[(30,10,20),(10,30,-20)])
def test_attack_angle_aux_radians(make_env,old_angle,new_angle,expected):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (0,0),(1000,0)
    env.headings[0,0] = math.radians(old_angle)
    old,metrics = env.positions.copy(),env._enemy_metrics()
    env.headings[0,0] = math.radians(new_angle)
    assert components(env,old,metrics)['attack_angle_reward'][0] == pytest.approx(math.radians(expected))


@pytest.mark.parametrize('before,after,expected',[(500,400,-.1),(400,500,.1),(500,600,0),(600,700,0)])
def test_collision_aux_distance_and_sign(make_env,before,after,expected):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (0,0),(before,0)
    old,metrics = env.positions.copy(),env._enemy_metrics()
    env.positions[1,0,0] = after
    assert components(env,old,metrics)['collision_avoidance_reward'][0] == pytest.approx(expected)


def test_collision_aux_only_current_nearest_same_aircraft_difference(make_env):
    env = make_env((0,0),(1,0),(1,1))
    env.positions[0,0],env.positions[1,0],env.positions[1,1] = (0,0),(500,0),(200,0)
    old,metrics = env.positions.copy(),env._enemy_metrics()
    env.positions[1,0],env.positions[1,1] = (300,0),(400,0)
    assert components(env,old,metrics)['collision_avoidance_reward'][0] == pytest.approx(-.2)


def test_full_episode_component_and_death_accounting(make_env):
    env = make_env()
    rng = np.random.default_rng(17)
    accumulated = {key:0.0 for key in env.REWARD_COMPONENTS}
    done = False
    while not done:
        masks = np.stack([env.available_action_masks(t) for t in range(2)])
        actions = np.array([[rng.choice(np.flatnonzero(masks[t,i])) for i in range(5)] for t in range(2)])
        _,reward,done,_,info = env.step(actions)
        for key in accumulated:
            accumulated[key] += info[key]
        assert sum(info[k] for k in accumulated) == pytest.approx(info['total_reward'])
        np.testing.assert_allclose(reward[0],info['total_reward'],atol=1e-5)
        for t in range(2):
            assert env.alive[t].sum()+env.collision_losses[t]+env.combat_kills[1-t] == 5
    summary = info['episode_summary']
    for key in accumulated:
        assert summary[key] == pytest.approx(accumulated[key])
    assert summary['total_return'] == pytest.approx(sum(accumulated.values()))
