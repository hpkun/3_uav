from __future__ import annotations

from dataclasses import replace

import numpy as np

from uav_mappo_2d.config import EnvironmentConfig, RenderConfig, RewardConfig
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.opponents import RuleBasedOpponent


def make_env(**overrides) -> UAVCombatEnv:
    config = replace(
        EnvironmentConfig(),
        agents_per_team=overrides.pop("agents_per_team", 2),
        episode_seconds=overrides.pop("episode_seconds", 10.0),
        **overrides,
    )
    return UAVCombatEnv(config, RewardConfig(), RenderConfig(width=320, height=320))


def test_reset_is_seeded_and_spaces_match() -> None:
    env = make_env(agents_per_team=3)
    first, first_info = env.reset(seed=17)
    positions = env.positions.copy()
    second, second_info = env.reset(seed=17)

    assert first.shape == env.observation_space.shape
    assert env.action_space.shape == (2, 3, 2)
    assert env.global_state(0).shape == (env.state_dim,)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(positions, env.positions)
    np.testing.assert_allclose(first_info["states"], second_info["states"])


def test_rule_actions_and_dynamics_stay_finite() -> None:
    env = make_env()
    observation, _ = env.reset(seed=2)
    rule = RuleBasedOpponent()
    initial_trail_points = len(env._trails[0][0])
    for _ in range(5):
        actions = np.stack([rule.act(env, 0), rule.act(env, 1)])
        observation, rewards, terminated, truncated, info = env.step(actions)
        assert np.isfinite(observation).all()
        assert np.isfinite(rewards).all()
        assert np.all(env.speeds[env.alive] >= env.config.speed_min)
        assert np.all(env.speeds[env.alive] <= env.config.speed_max)
        if terminated or truncated:
            break
    assert info["step"] > 0
    assert len(env._trails[0][0]) >= initial_trail_points + env.physics_substeps


def test_tail_lock_destroys_target_and_consumes_ammo() -> None:
    env = make_env(agents_per_team=1, lock_seconds=0.8, decision_dt=0.4)
    env.reset(seed=1)
    env.positions[0, 0] = [0.0, 0.0]
    env.positions[1, 0] = [500.0, 0.0]
    env.headings[:, 0] = 0.0
    env.speeds[:, 0] = env.config.speed_min
    env._last_potential = np.array([env._potential(0), env._potential(1)])

    actions = np.zeros(env.action_space.shape, dtype=np.float32)
    terminated = False
    total_reward = 0.0
    for _ in range(3):
        _, reward, terminated, _, info = env.step(actions)
        total_reward += float(reward[0, 0])
        if terminated:
            break

    assert terminated
    assert not env.alive[1, 0]
    assert env.ammo[0, 0] == env.config.ammunition - 1
    assert info["winner"] == 0
    assert info["kills"] == [1, 0]
    assert info["first_kill_time"] is not None
    assert total_reward > 0.0


def test_renderer_returns_rgb_frame() -> None:
    env = make_env()
    env.reset(seed=3)
    frame = env.render()
    assert frame.shape == (320, 320, 3)
    assert frame.dtype == np.uint8
    assert frame.var() > 0


def test_timeout_with_equal_survivors_is_draw() -> None:
    env = make_env(agents_per_team=2, episode_seconds=1.0)
    env.reset(seed=9)
    env.positions[0] = [[0.0, 0.0], [100.0, 100.0]]
    env.positions[1] = [[4000.0, 4000.0], [4200.0, 4200.0]]
    env.ammo[0] = 0
    env.ammo[1] = env.config.ammunition
    actions = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, terminated, truncated, info = env.step(actions)
    assert not terminated
    assert truncated
    assert info["alive"] == [2, 2]
    assert info["kills"] == [0, 0]
    assert info["winner"] is None
