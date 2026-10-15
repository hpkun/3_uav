import math

import numpy as np
import pytest

from uav_mappo_2d.env import UAVCombatEnv


def test_scene_shapes_and_reproducible_independent_spawn(make_env):
    env = make_env()
    obs, info = env.reset(seed=41)
    positions = env.positions.copy()
    assert env.N_AGENTS == 5 and env.N_ACTIONS == 10
    assert env.config.map_size == 10000 and env.config.episode_steps == 300
    assert env.config.speed_min == 50 and env.config.speed_max == 150
    assert obs.shape == (2, 5, 68) and info['global_state'].shape == (60,)
    assert env.action_space.shape == (2, 5)
    assert np.all(env.action_space.nvec == 10)
    assert info['available_action_masks'].shape == (5, 10)
    assert np.all(env.speeds == 100)
    np.testing.assert_array_equal(env.headings[0], 0)
    np.testing.assert_array_equal(env.headings[1], math.pi)
    assert np.all(np.abs(env.positions[0, :, 0] + 3500) <= 200)
    assert np.all(np.abs(env.positions[1, :, 0] - 3500) <= 200)
    assert np.all(np.abs(env.positions[:, :, 1] - np.array([-3000,-1500,0,1500,3000])) <= 200)
    assert not np.allclose(env.positions[0, :, 0], -env.positions[1, :, 0])
    again, more = env.reset(seed=41)
    np.testing.assert_array_equal(env.positions, positions)
    np.testing.assert_array_equal(obs, again)
    np.testing.assert_array_equal(info['global_state'], more['global_state'])


@pytest.mark.parametrize('action,dv,degrees', [
    (0,-1,0),(1,-.5,0),(2,0,0),(3,.5,0),(4,1,0),
    (5,0,-5),(6,0,-2),(7,0,0),(8,0,2),(9,0,5),
])
def test_action_mapping_and_old_state_movement(make_env, action, dv, degrees):
    env = make_env()
    env.positions[0, 0] = (0, 0)
    env.headings[0, 0] = math.pi / 2
    a = np.full((2,5), 2, dtype=np.int64)
    a[0,0] = action
    env._update_kinematics(a)
    np.testing.assert_allclose(env.positions[0,0], [0,100], atol=1e-10)
    assert env.speeds[0,0] == 100 + dv
    assert env.headings[0,0] == pytest.approx(math.pi/2 + math.radians(degrees))


def test_speed_clip_and_heading_wrap(make_env):
    env = make_env()
    env.speeds[0,:2] = [50,150]
    env.headings[0,2] = math.radians(179)
    a = np.full((2,5), 2, dtype=np.int64)
    a[0,:3] = [0,4,9]
    env._update_kinematics(a)
    assert env.speeds[0,:2].tolist() == [50,150]
    assert env.headings[0,2] == pytest.approx(math.radians(-176))


@pytest.mark.parametrize('team',[0,1])
def test_radar_boundary_sharing_lost_dead_and_critic(make_env,team):
    env = make_env()
    env.positions[team] = (-3000,0)
    env.alive[1-team] = False
    env.alive[1-team,2] = True
    env.positions[1-team,2] = (1000,0)
    slot = slice(5+7*(4+2), 5+7*(4+3))
    assert not env.team_detected_enemy_mask(team).any()
    assert np.all(env._get_actor_obs()[team,:,slot] == 0)
    state = env.global_state().reshape(2,5,6)
    assert state[1-team,2,0] == pytest.approx(.2) and state[1-team,2,5] == 1
    env.positions[team,4] = (-1000,0)
    assert env.team_detected_enemy_mask(team)[2]
    assert np.all(env._get_actor_obs()[team,:,slot][:,-1] == 1)
    env.positions[team,4,0] -= .001
    assert np.all(env._get_actor_obs()[team,:,slot] == 0)
    env.positions[team,4,0] += .001
    env.alive[team,4] = False
    assert not env.team_detected_enemy_mask(team).any()
    env.alive[team,4] = True
    env.alive[1-team,2] = False
    assert not env.team_detected_enemy_mask(team).any()


def test_dead_slots_and_unbounded_observation_space_render(make_env):
    env = make_env()
    env.alive[0,1] = False
    env.positions[0,0] = (7000,8000)
    obs = env._get_actor_obs()
    assert np.all(obs[0,1] == 0)
    assert np.all(env.global_state().reshape(2,5,6)[0,1] == 0)
    assert env.observation_space.contains(obs)
    # Friendly slots remain known even outside radar.
    assert obs[0,0,5+7+6] == 1
    frame = env.render()
    assert frame.shape == (320,320,3) and frame.dtype == np.uint8


@pytest.mark.parametrize('distance,alpha,beta,valid',[
    (600,0,0,True),(600.001,0,0,False),
    (500,15,0,True),(500,-15,0,True),(500,15.001,0,False),(500,-15.001,0,False),
    (500,0,50,True),(500,0,-50,True),(500,0,50.001,False),(500,0,-50.001,False),
    (500,0,180,False),
])
def test_weapon_geometry(make_env,distance,alpha,beta,valid):
    env = make_env((0,0),(1,0))
    env.positions[0,0] = (0,0)
    env.positions[1,0] = (distance,0)
    env.headings[0,0] = -math.radians(alpha)
    env.headings[1,0] = -math.radians(beta)
    assert env._attack_valid(0,0,1,0) == valid


def test_synchronous_weapon_kills(make_env,noop):
    env = make_env((0,0),(0,1),(1,0),(1,1))
    env.positions[0,0], env.positions[1,0] = (0,0),(400,0)
    env.positions[1,1], env.positions[0,1] = (1000,2000),(600,2000)
    env.headings[:,0] = 0
    env.headings[:,1] = math.pi
    _,_,_,_,info = env.step(noop(env))
    assert info['combat_kills'] == [1,1]
    assert not env.alive[1,0] and not env.alive[0,1]
    assert info['collision'] == [0,0]


def test_duplicate_weapon_victim_and_nearest_tie_break(make_env,noop):
    env = make_env((0,0),(0,1),(1,0))
    env.positions[0,0],env.positions[0,1],env.positions[1,0] = (0,0),(-200,0),(300,0)
    env.headings.fill(0)
    _,_,done,_,info = env.step(noop(env))
    assert done and info['combat_kills'] == [1,0]
    assert info['kill_reward'] == 20 and info['terminal_reward'] == 100


@pytest.mark.parametrize('offset,victim',[(0,0),(100,1)])
def test_nearest_then_id_target_selection(make_env,offset,victim):
    env = make_env((0,0),(1,0),(1,1))
    env.positions[0,0] = (0,0)
    env.positions[1,0],env.positions[1,1] = (400+offset,70),(400,-70)
    env.headings.fill(0)
    env._resolve_attacks()
    assert not env.alive[1,victim] and env.alive[1,1-victim]


@pytest.mark.parametrize('team',[0,1])
def test_collision_threshold_same_or_opposite_team(make_env,noop,team):
    other = (team,1) if team == 0 else (1,0)
    env = make_env((0,0),other)
    env.positions[0,0],env.positions[other] = (0,0),(150,0)
    env.headings.fill(0)
    _,_,_,_,info = env.step(noop(env))
    assert not env.alive.any()
    assert info['collision'] == ([2,0] if team == 0 else [1,1])
    assert info['combat_kills'] == [0,0]


def test_swept_crossing_and_multi_pair_dedup(make_env,noop):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (-75,50),(75,-50)
    env.speeds.fill(150)
    env.headings[0,0],env.headings[1,0] = 0,math.pi
    assert np.linalg.norm(env.positions[0,0]-env.positions[1,0]) > 150
    env.step(noop(env))
    assert np.linalg.norm(env.positions[0,0]-env.positions[1,0]) > 150
    assert env.collision_losses.tolist() == [1,1]
    env = make_env((0,0),(0,1),(1,0))
    env.positions[0,0],env.positions[0,1],env.positions[1,0] = (0,0),(100,0),(200,0)
    env.headings.fill(0)
    env.step(noop(env))
    assert env.collision_losses.tolist() == [2,1]
    assert env.combat_kills.tolist() == [0,0]


def test_collision_victim_cannot_fire(make_env,noop):
    env = make_env((0,0),(0,1),(1,0))
    env.positions[0,0],env.positions[0,1],env.positions[1,0] = (0,0),(100,0),(500,0)
    env.headings.fill(0)
    env.step(noop(env))
    assert env.alive[1,0] and not env.alive[0].any()
    assert env.combat_kills.tolist() == [0,0]


@pytest.mark.parametrize('x,y,expected',[
    (451,0,list(range(10))), (450,10,list(range(10))),
    (400,50,[5,6,7]),(400,-50,[7,8,9]),(450,0,[5,6,7]),
    (400,0,[5,6,7]),(400,-1e-13,[5,6,7]),
])
def test_avoidance_direction_and_inclusive_buffer(make_env,x,y,expected):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (0,0),(x,y)
    env.headings[0,0] = 0
    assert np.flatnonzero(env.available_action_masks(0)[0]).tolist() == expected


def test_avoidance_only_nearest_and_dead_canonical_noop(make_env):
    env = make_env((0,0),(1,0),(1,1))
    env.positions[0,0] = (0,0)
    env.positions[1,0],env.positions[1,1] = (300,-50),(400,100)
    env.headings[0,0] = 0
    assert np.flatnonzero(env.available_action_masks(0)[0]).tolist() == [7,8,9]
    assert np.flatnonzero(env.available_action_masks(0)[1]).tolist() == [2]


def test_step_rejects_invalid_or_noninteger_actions(make_env,noop):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (0,0),(400,0)
    a = noop(env)
    a[0,0] = 4
    with pytest.raises(ValueError,match='mask'):
        env.step(a)
    with pytest.raises(ValueError,match='integers'):
        env.step(a.astype(np.float32))


def test_outside_nominal_map_is_diagnostic_only(make_env,noop):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (4990,0),(-4000,0)
    env.headings[0,0] = 0
    _,rewards,done,_,info = env.step(noop(env))
    assert env.positions[0,0,0] == 5090 and env.alive[0,0]
    assert not done and info['outside_map'] == [1,0]
    np.testing.assert_array_equal(rewards,0)
    assert 'oob_reward' not in info and 'potential' not in info
