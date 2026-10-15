import math
import numpy as np
import pytest


@pytest.mark.parametrize('attacker',[0,1])
def test_weapon_event_and_terminal_rewards(make_env,zero_actions,attacker):
    env = make_env((0,0),(1,0))
    env.positions[attacker,0],env.positions[1-attacker,0] = (0,0),(500,0)
    env.headings.fill(0)
    env.speeds.fill(100)
    _,reward,done,_,info = env.step(zero_actions)
    assert done
    assert info['weapon_reward'] == (5 if attacker == 0 else -5)
    assert info['terminal_reward'] == (20 if attacker == 0 else -20)
    np.testing.assert_allclose(reward[attacker],25)
    np.testing.assert_allclose(reward[1-attacker],-25)


@pytest.mark.parametrize('cause',['oob','collision'])
@pytest.mark.parametrize('team',[0,1])
def test_self_penalty_no_positive_opponent_event(make_env,zero_actions,cause,team):
    env = make_env((0,0),(1,0),(team,1),(team,2))
    env.positions[0,0],env.positions[1,0] = (-3000,-3000),(3000,3000)
    env.headings.fill(0)
    env.speeds.fill(100)
    if cause == 'oob':
        env.positions[team,1],env.positions[team,2] = (4990,0),(0,3000)
        expected = -5
    else:
        env.positions[team,1],env.positions[team,2] = (0,0),(40,0)
        expected = -10
    _,reward,done,_,info = env.step(zero_actions)
    assert not done and info['weapon_reward'] == 0
    np.testing.assert_allclose(reward[team],expected)
    np.testing.assert_allclose(reward[1-team],0)


def test_cross_team_collision_draw_rewards(make_env,zero_actions):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (0,0),(50,0)
    env.headings.fill(0)
    env.speeds.fill(100)
    _,reward,done,_,info = env.step(zero_actions)
    assert done and info['result'] == 'draw'
    assert info['weapon_reward'] == info['terminal_reward'] == 0
    np.testing.assert_allclose(reward,-5)


@pytest.mark.parametrize('red,blue,value',[(3,2,20),(2,3,-20),(3,3,0)])
def test_timeout_terminal_reward(make_env,zero_actions,red,blue,value):
    env = make_env(episode_steps=1)
    env.alive[0,red:] = False
    env.alive[1,blue:] = False
    _,_,done,truncated,info = env.step(zero_actions)
    assert done and not truncated and info['terminal_reward'] == value


def test_radar_constrained_potential_quality_and_improvement(make_env):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (0,0),(3000.001,0)
    env.headings.fill(0)
    assert env._pair_quality(0,0,1,0) > 0
    assert env._team_attack_quality(0) == env._team_attack_quality(1) == 0
    env.positions[1,0] = (3000,0)
    assert env._team_attack_quality(0) > 0
    env.positions[1,0] = (2000,0)
    env.headings[0,0] = math.radians(60)
    earlier = env._compute_potential()
    env.positions[1,0] = (1000,0)
    env.headings[0,0] = 0
    assert env._compute_potential() > earlier
    tail = env._pair_quality(0,0,1,0)
    assert tail == pytest.approx(1)
    env.headings[1,0] = math.pi
    assert env._pair_quality(0,0,1,0) < tail


def test_terminal_zero_potential_and_zero_sum_shaping(make_env,zero_actions):
    env = make_env((0,0),(1,0),shaping_scale=.5,episode_steps=1)
    env.positions[0,0],env.positions[1,0] = (0,0),(2000,0)
    env.headings.fill(0)
    env.speeds.fill(100)
    env._potential = env._compute_potential()
    before = env._potential
    _,_,done,_,info = env.step(zero_actions)
    assert done and info['potential'] == 0
    assert info['shaping_reward'] == pytest.approx(-.5*before)
    red,blue = info['team_reward_components']['shaping_reward']
    assert red == -blue


def test_episode_components_death_accounting_and_counter_reset(make_env):
    env = make_env(shaping_scale=.5)
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
            assert env.alive[team].sum()+env.weapon_losses[team]+env.oob_losses[team]+env.collision_losses[team] == 5
            assert env.weapon_losses[team] == env.combat_kills[1-team]
            assert env.radar_contact_steps[team] <= env.step_count
    summary = info['episode_summary']
    assert summary['return'] == pytest.approx(sum(accumulated.values()))
    for key in accumulated:
        assert summary[key] == pytest.approx(accumulated[key])
    env.reset(seed=17)
    for values in (env.weapon_losses,env.oob_losses,env.combat_kills,env.collision_losses,env.attack_window_count,env.radar_contact_steps):
        assert not values.any()
