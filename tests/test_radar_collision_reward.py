from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from uav_mappo_2d.config import EnvironmentConfig, RewardConfig
from uav_mappo_2d.env import UAVCombatEnv


ZERO_ACTIONS = np.zeros((2, 5, 2), dtype=np.float32)


def isolated_env(*entities: tuple[int, int], **environment_overrides) -> UAVCombatEnv:
    env = UAVCombatEnv(
        replace(EnvironmentConfig(), **environment_overrides),
        replace(RewardConfig(), shaping_scale=0.0, near_collision_scale=0.0),
    )
    env.reset(seed=15)
    env.alive.fill(False)
    for team, agent in entities:
        env.alive[team, agent] = True
    env.speeds.fill(150.0)
    env.headings.fill(0.0)
    return env


def enemy_slot(observations: np.ndarray, team: int, enemy: int) -> np.ndarray:
    start = 5 + 7 * (4 + enemy)
    return observations[team, :, start:start + 7]


@pytest.mark.parametrize("team", [0, 1])
def test_radar_union_sharing_loss_and_full_critic_state(team: int) -> None:
    env = UAVCombatEnv()
    env.reset(seed=5)
    env.alive[1 - team] = False
    env.alive[1 - team, 2] = True
    env.positions[team] = (-3000.0, 0.0)
    env.positions[1 - team, 2] = (2000.0, 0.0)
    assert not env.team_detected_enemy_mask(team).any()
    assert np.all(enemy_slot(env._get_actor_obs(), team, 2) == 0.0)
    state = env.global_state()
    index = ((1 - team) * 5 + 2) * 6
    assert state[index + 5] == 1.0
    assert state[index] == pytest.approx(0.4)
    # Only one teammate at exactly 4km: all five living actors share the contact.
    env.positions[team, 4] = (-2000.0, 0.0)
    assert env.team_detected_enemy_mask(team)[2]
    assert np.all(enemy_slot(env._get_actor_obs(), team, 2)[:, -1] == 1.0)
    env.positions[team, 4, 0] -= 0.001
    assert np.all(enemy_slot(env._get_actor_obs(), team, 2) == 0.0)
    env.positions[team, 4, 0] += 0.001
    env.alive[team, 4] = False  # Dead radar platforms do not contribute.
    assert not env.team_detected_enemy_mask(team).any()
    env.alive[team, 4] = True
    env.alive[1 - team, 2] = False
    assert not env.team_detected_enemy_mask(team).any()
    assert np.all(enemy_slot(env._get_actor_obs(), team, 2) == 0.0)


def test_friendly_slots_are_known_outside_radar() -> None:
    env = isolated_env((0, 0), (0, 4), (1, 0))
    env.positions[0, 0] = (-4500.0, -4000.0)
    env.positions[0, 4] = (4500.0, 4000.0)
    obs = env._get_actor_obs()
    assert obs[0, 0, 5 + 3 * 7 + 6] == 1.0
    assert obs[0, 4, 5 + 6] == 1.0
    assert np.all(obs[0, 1:4] == 0.0)


@pytest.mark.parametrize("distance,alpha,beta,valid", [
    (1200.0, 0.0, 0.0, True),
    (1200.001, 0.0, 0.0, False),
    (600.0, 15.0, 0.0, True),
    (600.0, -15.0, 0.0, True),
    (600.0, 15.001, 0.0, False),
    (600.0, -15.001, 0.0, False),
    (600.0, 0.0, 50.0, True),
    (600.0, 0.0, -50.0, True),
    (600.0, 0.0, 50.001, False),
    (600.0, 0.0, -50.001, False),
    (600.0, 0.0, 180.0, False),
])
def test_weapon_geometry_boundaries(distance, alpha, beta, valid) -> None:
    env = isolated_env((0, 0), (1, 0))
    env.positions[0, 0] = (0.0, 0.0)
    env.positions[1, 0] = (distance, 0.0)
    env.headings[0, 0] = -math.radians(alpha)
    env.headings[1, 0] = -math.radians(beta)
    assert env._attack_valid(0, 0, 1, 0) == valid


@pytest.mark.parametrize("same_team", [False, True])
def test_endpoint_collision_and_symmetric_ramming_penalty(same_team: bool) -> None:
    second = (0, 1) if same_team else (1, 0)
    env = isolated_env((0, 0), second)
    env.positions[0, 0] = (-200.0, 0.0)
    env.positions[second] = (200.0, 0.0)
    env.headings[second] = math.pi
    env.reward_config.near_collision_scale = 0.05
    _, rewards, _, _, info = env.step(ZERO_ACTIONS)
    assert not env.alive.any()
    assert info["combat_kills"] == [0, 0]
    assert info["oob"] == [0, 0]
    assert info["near_collision_reward"] == 0.0
    assert info["collision"] == ([2, 0] if same_team else [1, 1])
    assert info["collision_reward"] == (-4.0 if same_team else -2.0)
    if not same_team:
        np.testing.assert_allclose(rewards, -2.0)
        assert info["result"] == "draw"
        assert info["terminal_reward"] == 0.0


def test_high_speed_crossing_detects_swept_collision() -> None:
    env = isolated_env((0, 0), (1, 0))
    env.positions[0, 0] = (-125.0, 0.0)
    env.positions[1, 0] = (0.0, -125.0)
    env.headings[1, 0] = math.pi / 2
    env.speeds.fill(250.0)
    previous_distance = np.linalg.norm(env.positions[0, 0] - env.positions[1, 0])
    env.step(ZERO_ACTIONS)
    current_distance = np.linalg.norm(env.positions[0, 0] - env.positions[1, 0])
    # Both endpoints exceed 150m, but the paths intersect at t=.5.
    assert previous_distance > 150.0
    assert current_distance > 150.0
    assert not env.alive.any()


def test_tunneling_with_both_endpoints_outside_radius() -> None:
    env = isolated_env((0, 0), (1, 0))
    env.positions[0, 0] = (-125.0, 0.0)
    env.positions[1, 0] = (125.0, 0.0)
    env.headings[1, 0] = math.pi
    env.speeds.fill(250.0)
    assert np.linalg.norm(env.positions[0, 0] - env.positions[1, 0]) == 250.0
    env.step(ZERO_ACTIONS)
    assert np.linalg.norm(env.positions[0, 0] - env.positions[1, 0]) == pytest.approx(250.0)
    assert env.collision_losses.tolist() == [1, 1]
    assert env.combat_kills.tolist() == [0, 0]


def test_simultaneous_three_aircraft_pairs_count_each_victim_once() -> None:
    env = isolated_env((0, 0), (0, 1), (1, 0))
    for entity, x in zip(((0, 0), (0, 1), (1, 0)), (0.0, 100.0, 200.0)):
        env.positions[entity] = (x, 0.0)
    _, _, _, _, info = env.step(ZERO_ACTIONS)
    assert info["collision"] == [2, 1]
    assert info["combat_kills"] == [0, 0]
    env.reset(seed=15)
    assert env.collision_losses.tolist() == [0, 0]


def test_collision_victims_cannot_fire_and_death_causes_are_exclusive() -> None:
    env = isolated_env((0, 0), (0, 1), (1, 0))
    env.positions[0, 0] = (0.0, 0.0)
    env.positions[0, 1] = (100.0, 0.0)
    env.positions[1, 0] = (900.0, 0.0)
    _, _, _, _, info = env.step(ZERO_ACTIONS)
    assert info["collision"] == [2, 0]
    assert info["combat_kills"] == [0, 0]
    assert env.alive[1, 0]


def test_oob_precedes_collision_and_weapons() -> None:
    env = isolated_env((0, 0), (1, 0))
    env.positions[0, 0] = (4900.0, 0.0)
    env.positions[1, 0] = (4800.0, 0.0)
    env.headings[1, 0] = 0.0
    _, _, _, _, info = env.step(ZERO_ACTIONS)
    # Blue could shoot red's tail and both are within collision radius,
    # but red's OOB takes precedence over both other death causes.
    assert info["oob"] == [1, 0]
    assert info["collision"] == [0, 0]
    assert info["combat_kills"] == [0, 0]
    assert env.alive[1, 0]
    assert info["oob_reward"] == -1.0


@pytest.mark.parametrize("attacker_team", [0, 1])
def test_weapon_reward_and_terminal_weights(attacker_team: int) -> None:
    env = isolated_env((0, 0), (1, 0))
    env.positions[attacker_team, 0] = (0.0, 0.0)
    env.positions[1 - attacker_team, 0] = (800.0, 0.0)
    env.reward_config.near_collision_scale = 0.05
    _, rewards, done, _, info = env.step(ZERO_ACTIONS)
    assert done
    sign = 1.0 if attacker_team == 0 else -1.0
    assert info["weapon_reward"] == sign
    assert info["near_collision_reward"] == 0.0
    assert info["terminal_reward"] == 5.0 * sign
    np.testing.assert_allclose(rewards[attacker_team], 6.0)
    np.testing.assert_allclose(rewards[1 - attacker_team], -6.0)


@pytest.mark.parametrize("offset", [0.0, 200.0])
def test_weapon_selects_nearest_target_then_id(offset: float) -> None:
    env = isolated_env((0, 0), (1, 0), (1, 1))
    env.positions[0, 0] = (0.0, 0.0)
    env.positions[1, 0] = (800.0 + offset, 100.0)
    env.positions[1, 1] = (800.0, -100.0)
    env._resolve_attacks()
    victim = 0 if offset == 0.0 else 1
    assert not env.alive[1, victim]
    assert env.alive[1, 1 - victim]
    assert env.combat_kills.tolist() == [1, 0]


@pytest.mark.parametrize("cause", ["oob", "collision"])
@pytest.mark.parametrize("team", [0, 1])
def test_self_losses_do_not_reward_opponent(cause: str, team: int) -> None:
    # Other survivors keep this nonterminal, isolating immediate event rewards.
    env = isolated_env((0, 0), (1, 0), (team, 1), (team, 2))
    env.positions[0, 0] = (-3500.0, -3500.0)
    env.positions[1, 0] = (3500.0, 3500.0)
    if cause == "oob":
        env.positions[team, 1] = (4900.0, 0.0)
        env.positions[team, 2] = (0.0, 3000.0)
        expected = -1.0
    else:
        env.positions[team, 1] = (0.0, 0.0)
        env.positions[team, 2] = (100.0, 0.0)
        expected = -4.0
    _, rewards, done, _, info = env.step(ZERO_ACTIONS)
    assert not done
    np.testing.assert_allclose(rewards[team], expected)
    np.testing.assert_allclose(rewards[1 - team], 0.0)
    assert info["weapon_reward"] == 0.0


@pytest.mark.parametrize("same_team", [False, True])
@pytest.mark.parametrize("approaching", [False, True])
def test_near_collision_only_penalizes_approaching_survivors(same_team, approaching) -> None:
    second = (0, 1) if same_team else (1, 0)
    env = isolated_env((0, 0), second, attack_range=0.0)
    env.reward_config.near_collision_scale = 0.05
    env.positions[0, 0] = (-300.0 if approaching else -150.0, 0.0)
    env.positions[second] = (300.0 if approaching else 150.0, 0.0)
    env.headings[0, 0] = 0.0
    env.headings[second] = math.pi if approaching else 0.0
    if not approaching:
        env.speeds[second] = 250.0  # 300 -> 400m, still inside the warning zone.
    _, _, _, _, info = env.step(ZERO_ACTIONS)
    assert env.alive.sum() == 2
    expected = -0.05 * (300.0 / 450.0) * (2 if same_team else 1) if approaching else 0.0
    assert info["near_collision_reward"] == pytest.approx(expected)


def test_potential_uses_only_current_radar_targets_and_terminal_zero() -> None:
    env = isolated_env((0, 0), (1, 0))
    env.positions[0, 0] = (0.0, 0.0)
    env.positions[1, 0] = (4100.0, 0.0)
    assert env._pair_quality(0, 0, 1, 0) > 0.7
    assert env._team_attack_quality(0) == 0.0
    assert env._team_attack_quality(1) == 0.0
    env.positions[1, 0] = (3900.0, 0.0)
    assert env._team_attack_quality(0) == pytest.approx(env._pair_quality(0, 0, 1, 0))
    assert env._compute_potential() > 0.0
    env.config.episode_steps = 1
    env.reward_config.shaping_scale = 0.1
    env._potential = env._compute_potential()
    previous = env._potential
    _, _, done, _, info = env.step(ZERO_ACTIONS)
    assert done and info["potential"] == 0.0
    assert info["shaping_reward"] == pytest.approx(-0.1 * previous)


def test_full_episode_reward_accounting_and_death_accounting() -> None:
    env = UAVCombatEnv()
    env.reset(seed=27)
    rng = np.random.default_rng(27)
    accumulated = {key: 0.0 for key in env.REWARD_COMPONENTS}
    done = False
    while not done:
        _, rewards, done, _, info = env.step(rng.uniform(-1, 1, (2, 5, 2)))
        for key in accumulated:
            accumulated[key] += info[key]
        assert sum(info[key] for key in accumulated) == pytest.approx(info["total_reward"])
        assert sum(info[key] for key in env.REWARD_COMPONENTS[:4]) == pytest.approx(info["event_reward"])
        np.testing.assert_allclose(rewards[0], info["total_reward"], atol=1e-6)
        np.testing.assert_allclose(rewards[1], info["team_rewards"][1], atol=1e-6)
        for team in range(2):
            deaths = env.oob_losses[team] + env.collision_losses[team] + env.combat_kills[1 - team]
            assert env.alive[team].sum() + deaths == 5
    summary = info["episode_summary"]
    for key, value in accumulated.items():
        assert summary[key] == pytest.approx(value)
    assert sum(accumulated.values()) == pytest.approx(summary["return"])


def test_swapping_team_labels_preserves_detection_rewards_and_events() -> None:
    original = UAVCombatEnv()
    swapped = UAVCombatEnv()
    original.reset(seed=51)
    swapped.reset(seed=51)
    for key in ("positions", "speeds", "headings", "alive"):
        getattr(swapped, key)[:] = getattr(original, key)[::-1]
    swapped._potential = swapped._compute_potential()
    rng = np.random.default_rng(51)
    for _ in range(30):
        actions = rng.uniform(-1, 1, (2, 5, 2))
        _, ra, da, _, ia = original.step(actions)
        _, rb, db, _, ib = swapped.step(actions[::-1])
        np.testing.assert_allclose(ra, rb[::-1], atol=1e-6)
        np.testing.assert_array_equal(original.alive, swapped.alive[::-1])
        for team in range(2):
            np.testing.assert_array_equal(original.team_detected_enemy_mask(team),
                                          swapped.team_detected_enemy_mask(1 - team))
        assert da == db
        if da:
            break
