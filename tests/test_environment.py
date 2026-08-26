from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from uav_mappo_2d.config import EnvironmentConfig, RenderConfig, RewardConfig
from uav_mappo_2d.env import UAVCombatEnv


def make_env(**overrides: float | int) -> UAVCombatEnv:
    return UAVCombatEnv(
        replace(EnvironmentConfig(), **overrides),
        RewardConfig(),
        RenderConfig(width=320, height=320),
    )


def put_pair(
    env: UAVCombatEnv,
    attacker_position: tuple[float, float],
    target_position: tuple[float, float],
    attacker_heading: float,
    target_heading: float,
) -> None:
    env.positions[0, 0] = attacker_position
    env.positions[1, 0] = target_position
    env.headings[0, 0] = attacker_heading
    env.headings[1, 0] = target_heading
    env.speeds[:, 0] = 200.0


def test_reset_is_independent_distributionally_symmetric_and_reproducible() -> None:
    env = make_env()
    observations, info = env.reset(seed=17)
    positions = env.positions.copy()
    headings = env.headings.copy()
    speeds = env.speeds.copy()
    state = info["global_state"].copy()
    assert observations.shape == (2, 3, 40)
    assert env.observation_space.shape == (2, 3, 40)
    assert env.action_space.shape == (2, 3, 2)
    assert info["global_state"].shape == (36,)
    assert np.isfinite(observations).all()
    assert np.isfinite(info["global_state"]).all()
    assert np.max(np.abs(observations)) <= 1.0 + 1e-6
    assert np.all((-3_700.0 <= positions[0, :, 0]) & (positions[0, :, 0] <= -3_300.0))
    assert np.all((3_300.0 <= positions[1, :, 0]) & (positions[1, :, 0] <= 3_700.0))
    base_y = np.array([-1_500.0, 0.0, 1_500.0])
    assert np.all(np.abs(positions[0, :, 1] - base_y) <= 200.0)
    assert np.all(np.abs(positions[1, :, 1] - base_y) <= 200.0)
    assert np.all(np.abs(headings[0]) <= math.radians(5.0))
    blue_offsets = (headings[1] - math.pi + math.pi) % (2.0 * math.pi) - math.pi
    assert np.all(np.abs(blue_offsets) <= math.radians(5.0))
    assert np.all((190.0 <= speeds) & (speeds <= 210.0))
    assert not (
        np.allclose(positions[0, :, 0], -positions[1, :, 0])
        and np.allclose(positions[0, :, 1], positions[1, :, 1])
        and np.allclose(speeds[0], speeds[1])
    )

    second_observations, second_info = env.reset(seed=17)
    np.testing.assert_allclose(second_observations, observations)
    np.testing.assert_allclose(second_info["global_state"], state)
    np.testing.assert_allclose(env.positions, positions)
    np.testing.assert_allclose(env.headings, headings)
    np.testing.assert_allclose(env.speeds, speeds)


def test_pair_geometry_tail_and_attack_conditions() -> None:
    env = make_env()
    env.reset(seed=1)
    put_pair(env, (0.0, 0.0), (500.0, 0.0), 0.0, 0.0)
    distance, alpha, beta = env._compute_pair_geometry(0, 0, 1, 0)
    assert distance == 500.0
    assert abs(alpha) < 1e-9
    assert abs(beta) < 1e-9
    assert env._attack_valid(0, 0, 1, 0)

    env.headings[1, 0] = math.pi
    _, alpha, beta = env._compute_pair_geometry(0, 0, 1, 0)
    assert abs(alpha) < 1e-9
    assert math.isclose(abs(beta), math.pi)
    assert not env._attack_valid(0, 0, 1, 0)

    env.headings[1, 0] = 0.0
    env.headings[0, 0] = math.radians(45.0)
    assert not env._attack_valid(0, 0, 1, 0)

    put_pair(env, (0.0, 0.0), (1201.0, 0.0), 0.0, 0.0)
    assert not env._attack_valid(0, 0, 1, 0)


def test_synchronous_mutual_kill() -> None:
    env = make_env()
    env.reset(seed=2)
    env.alive[:] = False
    env.alive[0, :2] = True
    env.alive[1, :2] = True
    env.positions[0, 0] = (0.0, 0.0)
    env.positions[1, 0] = (500.0, 0.0)
    env.headings[0, 0] = env.headings[1, 0] = 0.0
    env.positions[1, 1] = (1000.0, 1000.0)
    env.positions[0, 1] = (500.0, 1000.0)
    env.headings[0, 1] = env.headings[1, 1] = math.pi
    env.speeds.fill(150.0)
    _, _, _, _, info = env.step(np.zeros((2, 3, 2), dtype=np.float32))
    assert not env.alive[1, 0]
    assert not env.alive[0, 1]
    assert info["combat_kills"] == [1, 1]


def test_multiple_attackers_count_one_target_once() -> None:
    env = make_env()
    env.reset(seed=3)
    env.alive[:] = False
    env.alive[0, :2] = True
    env.alive[1, 0] = True
    env.positions[0, 0] = (0.0, 0.0)
    env.positions[0, 1] = (-100.0, 0.0)
    env.positions[1, 0] = (500.0, 0.0)
    env.headings.fill(0.0)
    env.speeds.fill(150.0)
    _, _, done, _, info = env.step(np.zeros((2, 3, 2), dtype=np.float32))
    assert done
    assert info["combat_kills"] == [1, 0]
    assert info["episode_summary"]["red_combat_kill"] == 1


def test_out_of_bounds_dies_immediately_and_cannot_attack() -> None:
    env = make_env()
    env.reset(seed=4)
    env.positions[0, 0] = (4_900.0, 0.0)
    env.headings[0, 0] = 0.0
    env.speeds[0, 0] = 150.0
    env.positions[1, 0] = (4_990.0, 0.0)
    env.headings[1, 0] = math.pi
    _, rewards, _, _, info = env.step(np.zeros((2, 3, 2), dtype=np.float32))
    assert not env.alive[0, 0]
    assert info["oob"][0] == 1
    assert info["combat_kills"][0] == 0
    assert info["event_reward"] <= -1.0
    np.testing.assert_allclose(rewards[0], rewards[0, 0])


def test_reward_components_and_terminal_timeout() -> None:
    env = make_env(episode_steps=1)
    env.reset(seed=5)
    env.positions[0, :, 0] = -4_000.0
    env.positions[1, :, 0] = 4_000.0
    _, rewards, terminated, truncated, info = env.step(
        np.zeros((2, 3, 2), dtype=np.float32)
    )
    assert terminated and not truncated
    assert info["result"] == "draw"
    assert info["terminal_reward"] == 0.0
    assert math.isclose(
        info["event_reward"] + info["terminal_reward"] + info["shaping_reward"],
        info["total_reward"],
        abs_tol=1e-7,
    )
    assert info["potential"] == 0.0
    np.testing.assert_allclose(rewards[0], info["total_reward"])
    np.testing.assert_allclose(rewards[1], -info["total_reward"])


def test_dead_slots_and_renderer() -> None:
    env = make_env()
    env.reset(seed=6)
    env.alive[0, 1] = False
    observations = env._get_actor_obs()
    state = env.global_state()
    assert np.all(observations[0, 1] == 0.0)
    assert np.all(state[6:12] == 0.0)
    frame = env.render()
    assert frame.shape == (320, 320, 3)
    assert frame.dtype == np.uint8
    assert frame.var() > 0
