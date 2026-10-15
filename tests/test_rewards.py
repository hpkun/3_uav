import math
import numpy as np
import pytest

from uav_mappo_2d.config import RewardConfig


@pytest.mark.parametrize('attacker',[0,1])
def test_separate_weapon_kill_loss_and_terminal(make_env,zero_actions,attacker):
    env = make_env((0,0),(1,0))
    env.positions[attacker,0],env.positions[1-attacker,0] = (0,0),(500,0)
    env.headings.fill(0)
    env.speeds.fill(100)
    _,reward,done,_,info = env.step(zero_actions)
    assert done
    assert info['team_reward_components']['kill_reward'] == ([10,0] if attacker == 0 else [0,10])
    assert info['team_reward_components']['weapon_loss_reward'] == ([0,-10] if attacker == 0 else [-10,0])
    assert info['terminal_reward'] == (20 if attacker == 0 else -20)
    np.testing.assert_allclose(reward[attacker],30)
    np.testing.assert_allclose(reward[1-attacker],-30)


@pytest.mark.parametrize('cause',['oob','collision'])
@pytest.mark.parametrize('team',[0,1])
def test_self_penalty_no_positive_opponent_event(make_env,zero_actions,cause,team):
    env = make_env((0,0),(1,0),(team,1),(team,2))
    env.positions[0,0],env.positions[1,0] = (-3000,-3000),(3000,3000)
    env.headings.fill(0)
    env.speeds.fill(100)
    if cause == 'oob':
        env.positions[team,1],env.positions[team,2] = (4990,0),(0,3000)
        expected = -20
    else:
        env.positions[team,1],env.positions[team,2] = (0,0),(40,0)
        expected = -10
    _,reward,done,_,info = env.step(zero_actions)
    assert not done and info['kill_reward'] == info['weapon_loss_reward'] == 0
    np.testing.assert_allclose(reward[team],expected)
    np.testing.assert_allclose(reward[1-team],0)


def test_cross_team_collision_penalty_without_death_or_terminal(make_env,zero_actions):
    env = make_env((0,0),(1,0),attack_range=1)
    env.positions[0,0],env.positions[1,0] = (0,0),(50,0)
    env.headings.fill(0)
    env.speeds.fill(100)
    _,reward,done,_,info = env.step(zero_actions)
    assert not done and env.alive.sum() == 2
    assert info['collision_events'] == [1,1]
    assert info['kill_reward'] == info['weapon_loss_reward'] == info['terminal_reward'] == 0
    np.testing.assert_allclose(reward,-5)


def test_collision_and_weapon_reward_can_coexist(make_env,zero_actions):
    env = make_env((0,0),(0,1),(1,0))
    env.positions[0,0],env.positions[0,1],env.positions[1,0] = (0,0),(40,0),(700,0)
    env.headings.fill(0)
    env.speeds.fill(100)
    _,reward,done,_,info = env.step(zero_actions)
    assert done
    assert info['kill_reward'] == 10 and info['collision_reward'] == -10
    assert info['terminal_reward'] == 20 and info['total_reward'] == 20
    np.testing.assert_allclose(reward[1],-30)


@pytest.mark.parametrize('red,blue,value',[(3,2,20),(2,3,-20),(3,3,0)])
def test_timeout_terminal_reward(make_env,zero_actions,red,blue,value):
    env = make_env(episode_steps=1)
    env.alive[0,red:] = False
    env.alive[1,blue:] = False
    _,_,done,truncated,info = env.step(zero_actions)
    assert done and not truncated and info['terminal_reward'] == value


def test_simultaneous_oob_draw(make_env,zero_actions):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (4990,0),(-4990,0)
    env.headings[0,0],env.headings[1,0] = 0,math.pi
    env.speeds.fill(100)
    _,reward,done,_,info = env.step(zero_actions)
    assert done and info['result'] == 'draw' and info['terminal_reward'] == 0
    assert env.oob_losses.tolist() == [1,1]
    np.testing.assert_allclose(reward,-20)


def test_no_potential_state_or_reward(make_env,zero_actions):
    env = make_env()
    for name in ['_potential','_compute_potential','_pair_quality','_team_attack_quality']:
        assert not hasattr(env,name)
    assert set(vars(RewardConfig())) == {
        'weapon_kill_reward','weapon_loss_penalty','collision_penalty','oob_penalty','terminal_reward'}
    assert set(env.REWARD_COMPONENTS) == {
        'kill_reward','weapon_loss_reward','collision_reward','oob_reward','terminal_reward'}
    _,_,_,_,info = env.step(zero_actions)
    assert 'potential' not in info and 'shaping_reward' not in info
    assert info['total_reward'] == 0


def test_episode_components_death_accounting_and_counter_reset(make_env):
    env = make_env()
    rng = np.random.default_rng(17)
    accumulated = {k:0. for k in env.REWARD_COMPONENTS}
    done = False
    while not done:
        _,reward,done,_,info = env.step(rng.uniform(-1,1,(2,5,2)))
        for key in accumulated:
            accumulated[key] += info[key]
        assert sum(info[k] for k in accumulated) == pytest.approx(info['total_reward'])
        np.testing.assert_allclose(reward[0],info['total_reward'],atol=1e-6)
        for team in range(2):
            assert env.alive[team].sum()+env.weapon_losses[team]+env.oob_losses[team] == 5
            assert env.weapon_losses[team] == env.combat_kills[1-team]
            assert env.radar_contact_steps[team] <= env.step_count
    summary = info['episode_summary']
    assert summary['return'] == pytest.approx(sum(accumulated.values()))
    for key in accumulated:
        assert summary[key] == pytest.approx(accumulated[key])
    env.reset(seed=17)
    for values in (env.weapon_losses,env.oob_losses,env.combat_kills,env.collision_events,env.attack_window_count,env.radar_contact_steps):
        assert not values.any()
