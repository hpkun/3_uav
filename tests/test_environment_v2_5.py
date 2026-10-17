"""V2.5 contract: geometry, one seeded trial, synchronous victims and frozen reward."""
from dataclasses import asdict
import math
import random

import numpy as np
import pytest

from uav_mappo_2d.config import EnvironmentConfig, RewardConfig, load_config
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.opponents import RuleBasedOpponent


class TrialRNG:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self.values)


def pair(make_env, **overrides):
    env = make_env((0, 0), (1, 0), **overrides)
    env.positions[0, 0], env.positions[1, 0] = (0, 0), (500, 0)
    env.headings.fill(0)
    env.speeds.fill(300)
    return env


def test_v25_formal_config_and_reset():
    cfg = load_config('configs/base.yaml').environment
    assert cfg == EnvironmentConfig()
    assert (cfg.speed_min, cfg.speed_max, cfg.initial_speed_min, cfg.initial_speed_max) == (200, 400, 300, 300)
    assert (cfg.attack_range, cfg.attack_angle_degrees, cfg.hit_probability, cfg.radar_range) == (1000, 90, .7, 3000)
    assert (cfg.dt, cfg.max_acceleration, cfg.max_turn_rate_degrees) == (1, 10, 10)
    assert not {'attack_ata_degrees', 'attack_aspect_degrees', 'ammo', 'radar_angle_degrees'} & asdict(cfg).keys()
    assert asdict(RewardConfig()) == dict(weapon_kill_reward=10., weapon_loss_penalty=10.,
                                        oob_penalty=20., collision_penalty=.2, terminal_reward=20.)
    env = UAVCombatEnv(cfg)
    obs, info = env.reset(seed=31)
    assert np.all(env.speeds == 300)
    assert obs.shape == (2, 5, 68) and info['global_state'].shape == (60,)
    assert env.action_space.shape == (2, 5, 2)
    env.close()


@pytest.mark.parametrize('overrides', [
    dict(speed_min=0), dict(speed_min=400), dict(speed_max=199),
    dict(initial_speed_min=199), dict(initial_speed_max=401),
    dict(initial_speed_min=301), dict(attack_range=0),
    dict(attack_angle_degrees=0), dict(attack_angle_degrees=181),
    dict(hit_probability=-.01), dict(hit_probability=1.01),
    dict(hit_probability=float('nan')), dict(attack_angle_degrees=float('nan')),
])
def test_invalid_config(overrides):
    with pytest.raises(ValueError):
        EnvironmentConfig(**overrides)


@pytest.mark.parametrize('probability', [0., 1.])
def test_probability_endpoints_accepted(probability):
    assert EnvironmentConfig(hit_probability=probability).hit_probability == probability


def test_repeated_acceleration_clips_and_normalizes(make_env, zero_actions):
    env = make_env()
    for acceleration, expected in [(1, 400), (-1, 200)]:
        zero_actions[..., 0] = acceleration
        for _ in range(50):
            env._update_kinematics(zero_actions)
        assert np.all(env.speeds == expected)
        # Reset positions only to inspect valid observation values after long motion.
        env.positions.fill(0)
        normalized = 1 if acceleration == 1 else -1
        assert np.all(env._get_actor_obs()[..., 2] == normalized)
        assert np.all(env.global_state().reshape(2, 5, 6)[..., 2] == normalized)


@pytest.mark.parametrize('distance,alpha,valid', [
    (1000, 0, True), (1000 + 1e-6, 0, False),
    (800, 90, True), (800, -90, True),
    (800, 90 + 1e-6, False), (800, -90 - 1e-6, False), (800, 180, False),
])
def test_geometry_inclusive_and_independent_of_aspect(make_env, distance, alpha, valid):
    env = pair(make_env)
    env.positions[1, 0] = (distance, 0)
    env.headings[0, 0] = -math.radians(alpha)
    for beta in [-180, -91, -45, 0, 45, 91, 180]:
        env.headings[1, 0] = math.radians(beta)
        assert env._attack_valid(0, 0, 1, 0) == valid


@pytest.mark.parametrize('draw,hit', [(0., True), (.699999, True), (.70, False), (.999999, False)])
def test_hit_miss_and_reward(make_env, zero_actions, draw, hit):
    env = pair(make_env)
    rng = TrialRNG([draw])
    env.np_random = rng
    _, reward, done, _, info = env.step(zero_actions)
    assert rng.calls == 1
    assert env.attack_window_count.tolist() == [1, 0]
    assert env.combat_kills.tolist() == [int(hit), 0]
    assert env.weapon_losses.tolist() == [0, int(hit)]
    assert done == hit
    np.testing.assert_allclose(reward[0], 30 if hit else 0)
    np.testing.assert_allclose(reward[1], -30 if hit else 0)
    assert info['kill_reward'] == (10 if hit else 0)
    assert info['weapon_loss_reward'] == 0
    assert info['team_reward_components']['weapon_loss_reward'][1] == (-10 if hit else 0)
    assert sum(info[k] for k in env.REWARD_COMPONENTS) == info['total_reward']


def test_one_trial_nearest_target_no_retry_after_miss(make_env):
    env = pair(make_env)
    env.alive[1, 1] = True
    env.positions[1, 1] = (600, 0)
    env.headings[1, 1] = 0
    env.np_random = TrialRNG([.7])
    assert env._resolve_attacks().tolist() == [0, 0]
    assert env.np_random.calls == 1 and env.alive[1, :2].all()
    assert env.attack_window_count.tolist() == [2, 0]


@pytest.mark.parametrize('draws', [[.1, .2, .3], [.9, .2, .3]])
def test_snapshot_mutual_kills_and_deduplicated_rewards(make_env, zero_actions, draws):
    env = make_env((0, 0), (0, 1), (1, 0))
    # Parallel flight keeps both red attackers equidistant; blue points back.
    # Call attack resolution directly so heading changes cannot alter this geometry.
    env.positions[0, 0], env.positions[0, 1], env.positions[1, 0] = (0, 100), (0, -100), (500, 0)
    env.headings[0, :2] = 0
    env.headings[1, 0] = math.pi
    env.np_random = TrialRNG(draws)
    kills = env._resolve_attacks()
    assert env.np_random.calls == 3  # Blue fires despite already being hit; both red trials occur.
    assert kills.tolist() == [1, 1]
    assert env.combat_kills.tolist() == [1, 1] and env.weapon_losses.tolist() == [1, 1]
    assert env.attack_window_count.tolist() == [2, 2]
    assert not env.alive[0, 0] and env.alive[0, 1] and not env.alive[1, 0]


def test_same_target_hits_count_and_reward_once(make_env, zero_actions):
    env = pair(make_env)
    env.alive[0, 1] = True
    env.positions[0, 1] = (-100, 0)
    env.np_random = TrialRNG([.1, .2])
    _, _, done, _, info = env.step(zero_actions)
    assert done and env.np_random.calls == 2
    assert info['combat_kills'] == [1, 0] and info['weapon_losses'] == [0, 1]
    assert info['team_reward_components']['kill_reward'] == [10, 0]
    assert info['team_reward_components']['weapon_loss_reward'] == [0, -10]


def test_no_window_consumes_no_attack_randomness(make_env):
    env = pair(make_env)
    env.positions[1, 0] = (2000, 0)
    env.np_random = TrialRNG([])
    assert env._resolve_attacks().tolist() == [0, 0]
    assert env.np_random.calls == 0


@pytest.mark.parametrize('seed', [9, 41])
def test_seeded_full_rollout_without_global_random(make_env, monkeypatch, seed):
    def forbidden(*args, **kwargs):
        raise AssertionError('Global random source used for environment attacks')
    monkeypatch.setattr(np.random, 'random', forbidden)
    monkeypatch.setattr(random, 'random', forbidden)
    envs = [make_env(), make_env()]
    for env in envs:
        env.reset(seed=seed)
    rule = RuleBasedOpponent()
    trials = 0
    while True:
        # Replay the exact same actions on the second independent seeded env.
        actions = np.stack([rule.act(envs[0], t) for t in range(2)])
        a, b = [env.step(actions) for env in envs]
        for attr in ['positions', 'speeds', 'headings', 'alive', 'combat_kills', 'attack_window_count']:
            np.testing.assert_array_equal(getattr(envs[0], attr), getattr(envs[1], attr))
        assert envs[0].np_random.bit_generator.state == envs[1].np_random.bit_generator.state
        np.testing.assert_array_equal(a[1], b[1])
        assert a[2] == b[2]
        trials += int(envs[0].attack_window_count.sum())
        if a[2]:
            break
    assert trials > 0
