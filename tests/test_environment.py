import math
import numpy as np
import pytest

from uav_mappo_2d.env import UAVCombatEnv, wrap_angle


def test_v2_reset_shapes_independent_spacing_and_reproducibility(make_env):
    env = make_env()
    obs, info = env.reset(seed=41)
    positions, speeds, headings = env.positions.copy(), env.speeds.copy(), env.headings.copy()
    assert obs.shape == (2,5,68) and info['global_state'].shape == (60,)
    assert env.action_space.shape == (2,5,2) and env.action_space.dtype == np.float32
    assert env.config.episode_steps == 300 and env.config.map_size == 10000
    assert np.all((-3800 <= positions[0,:,0]) & (positions[0,:,0] <= -3200))
    assert np.all((3200 <= positions[1,:,0]) & (positions[1,:,0] <= 3800))
    assert np.all(np.abs(positions[:,:,1]) <= 3000)
    assert np.all(np.diff(np.sort(positions[:,:,1],axis=1),axis=1) >= 500)
    assert not np.allclose(positions[0,:,1],positions[1,:,1])
    assert not np.allclose(positions[0,:,0],-positions[1,:,0])
    assert np.all((95 <= speeds) & (speeds <= 105))
    assert np.all(np.abs(headings[0]) <= math.radians(5))
    assert np.all(np.abs(wrap_angle(headings[1]-math.pi)) <= math.radians(5))
    flat = positions.reshape(-1,2)
    first,second = np.triu_indices(10,k=1)
    assert np.all(np.linalg.norm(flat[first]-flat[second],axis=1) > 50)
    again,more = env.reset(seed=41)
    np.testing.assert_array_equal(obs,again)
    np.testing.assert_array_equal(info['global_state'],more['global_state'])
    np.testing.assert_array_equal(env.positions,positions)
    np.testing.assert_array_equal(env.speeds,speeds)
    np.testing.assert_array_equal(env.headings,headings)
    assert env.observation_space.contains(obs)


def test_spawn_fallback_keeps_spacing_bounds_and_independence(make_env):
    env = make_env()
    class RejectionRNG:
        def __init__(self):
            self.rng = np.random.default_rng(8)
        def uniform(self, low, high, size):
            return np.zeros(size) if (low,high) == (-3000.,3000.) else self.rng.uniform(low,high,size)
        def permutation(self, values):
            return self.rng.permutation(values)
    env._np_random = RejectionRNG()
    a,b = env._sample_spawn_y(),env._sample_spawn_y()
    for values in (a,b):
        assert np.all(np.abs(values) <= 3000)
        assert np.all(np.diff(np.sort(values)) >= 500)
    assert not np.allclose(a,b)


@pytest.mark.parametrize('action,speed,degrees',[([-1,-1],90,-10),([0,0],100,0),([1,1],110,10)])
def test_continuous_mapping_semi_implicit_movement(make_env,zero_actions,action,speed,degrees):
    env = make_env()
    env.positions[0,0],env.speeds[0,0],env.headings[0,0] = (0,0),100,0
    zero_actions[0,0] = action
    env._update_kinematics(zero_actions)
    assert env.speeds[0,0] == speed
    assert env.headings[0,0] == pytest.approx(math.radians(degrees))
    np.testing.assert_allclose(env.positions[0,0],speed*np.array([math.cos(math.radians(degrees)),math.sin(math.radians(degrees))]),atol=1e-5)


def test_speed_clipping_heading_wrap_and_dead_freeze(make_env,zero_actions):
    env = make_env()
    env.speeds[0,:2] = [50,150]
    zero_actions[0,0,0],zero_actions[0,1,0] = -1,1
    env.headings[0,2] = math.radians(175)
    zero_actions[0,2,1] = 1
    env.alive[0,3] = False
    old = env.positions[0,3].copy()
    env._update_kinematics(zero_actions)
    assert env.speeds[0,:2].tolist() == [50,150]
    assert env.headings[0,2] == pytest.approx(math.radians(-175))
    np.testing.assert_array_equal(env.positions[0,3],old)


@pytest.mark.parametrize('value',[1.01,-1.01,np.nan,np.inf])
def test_invalid_actions_rejected_without_changing_state(make_env,zero_actions,value):
    env = make_env()
    old = env.positions.copy()
    zero_actions[0,0,0] = value
    with pytest.raises(ValueError):
        env.step(zero_actions)
    np.testing.assert_array_equal(env.positions,old)


@pytest.mark.parametrize('team',[0,1])
def test_radar_3000_inclusive_sharing_loss_dead_and_full_state(make_env,team):
    env = make_env()
    env.positions[team] = (-4000,0)
    env.alive[1-team] = False
    env.alive[1-team,2] = True
    env.positions[1-team,2] = (1000,0)
    slot = slice(5+7*6,5+7*7)
    assert not env.team_detected_enemy_mask(team).any()
    assert np.all(env._get_actor_obs()[team,:,slot] == 0)
    assert env.global_state().reshape(2,5,6)[1-team,2,5] == 1
    env.positions[team,4] = (-2000,0)
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


def test_observation_speed_normalization_geometry_slots_and_render(make_env):
    env = make_env((0,0),(0,2),(1,0))
    env.positions[0,0],env.positions[0,2],env.positions[1,0] = (0,0),(500,0),(0,1000)
    env.speeds[0,0],env.speeds[0,2],env.speeds[1,0] = 50,150,100
    env.headings.fill(0)
    obs = env._get_actor_obs()
    assert obs[0,0,2] == -1 and obs[0,2,2] == 1 and obs[1,0,2] == 0
    assert np.all(obs[0,1] == 0) and np.all(obs[0,0,5:12] == 0)
    friend = obs[0,0,12:19]
    np.testing.assert_allclose(friend[1:],[0,1,0,1,1,1],atol=1e-6)
    enemy = obs[0,0,33:40]
    np.testing.assert_allclose(enemy[1:],[1,0,1,0,.5,1],atol=1e-6)
    state = env.global_state().reshape(2,5,6)
    assert state[0,0,2] == -1 and state[0,2,2] == 1
    assert np.all(state[0,1] == 0)
    assert env.render().shape == (320,320,3)


@pytest.mark.parametrize('distance,alpha,beta,valid',[
    (1000,0,0,True),(1000.001,0,0,False),
    (800,30,0,True),(800,-30,0,True),(800,30.001,0,False),(800,-30.001,0,False),
    (800,0,45,True),(800,0,-45,True),(800,0,45.001,False),(800,0,-45.001,False),
    (800,0,180,False),
])
def test_weapon_boundaries_tail_and_head_on(make_env,distance,alpha,beta,valid):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (0,0),(distance,0)
    env.headings[0,0],env.headings[1,0] = -math.radians(alpha),-math.radians(beta)
    assert env._attack_valid(0,0,1,0) == valid


def test_synchronous_kills_window_contact_and_unique_victims(make_env,zero_actions):
    env = make_env((0,0),(0,1),(1,0),(1,1))
    env.positions[0,0],env.positions[1,0] = (0,0),(400,0)
    env.positions[1,1],env.positions[0,1] = (1000,2000),(600,2000)
    env.headings[:,0] = 0
    env.headings[:,1] = math.pi
    env.speeds.fill(100)
    _,_,_,_,info = env.step(zero_actions)
    assert info['combat_kills'] == info['weapon_losses'] == [1,1]
    assert info['attack_window_count'] == [1,1]
    assert info['radar_contact_steps'] == [1,1]
    env = make_env((0,0),(0,1),(1,0))
    env.positions[0,0],env.positions[0,1],env.positions[1,0] = (0,0),(-100,0),(500,0)
    env.headings.fill(0)
    env.speeds.fill(100)
    _,_,_,_,info = env.step(zero_actions)
    assert info['combat_kills'] == [1,0] and info['weapon_losses'] == [0,1]
    assert info['attack_window_count'] == [2,0]


@pytest.mark.parametrize('offset,victim',[(0,0),(100,1)])
def test_weapon_target_nearest_then_id(make_env,offset,victim):
    env = make_env((0,0),(1,0),(1,1))
    env.positions[0,0] = (0,0)
    env.positions[1,0],env.positions[1,1] = (500+offset,100),(500,-100)
    env.headings.fill(0)
    env._resolve_attacks()
    assert not env.alive[1,victim] and env.alive[1,1-victim]


@pytest.mark.parametrize('first,second,events',[
    ((0,0),(0,1),[2,0]), ((1,0),(1,1),[0,2]), ((0,0),(1,0),[1,1]),
])
@pytest.mark.parametrize('distance,collided',[(49.9,True),(50,True),(50.001,False)])
def test_collision_threshold_all_team_combinations(make_env,zero_actions,first,second,events,distance,collided):
    env = make_env(first,second,attack_range=1)
    env.positions[first],env.positions[second] = (0,0),(distance,0)
    env.headings.fill(0)
    env.speeds.fill(100)
    env.step(zero_actions)
    assert env.alive[first] and env.alive[second]
    assert env.collision_events.tolist() == (events if collided else [0,0])
    assert not env.combat_kills.any() and not env.weapon_losses.any()


def test_swept_only_crossing_is_not_production_collision(make_env,zero_actions):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (-100,0),(100,0)
    env.headings[0,0],env.headings[1,0] = 0,math.pi
    env.speeds.fill(150)
    env.step(zero_actions)
    assert np.linalg.norm(env.positions[0,0]-env.positions[1,0]) == pytest.approx(100)
    assert env.collision_events.tolist() == [0,0] and env.combat_kills.tolist() == [0,0]
    assert env.alive[0,0] and env.alive[1,0]
    old = env.positions.copy()
    env.step(zero_actions)
    assert np.linalg.norm(env.positions[0,0]-old[0,0]) == pytest.approx(150)


def test_endpoint_collision_deduplicates_participants_each_step(make_env,zero_actions):
    env = make_env((0,0),(0,1),(1,0),attack_range=1)
    env.positions[0,0],env.positions[0,1],env.positions[1,0] = (0,0),(40,0),(20,0)
    env.headings.fill(0)
    env.speeds.fill(100)
    env.step(zero_actions)
    assert env.collision_events.tolist() == [2,1] and env.combat_kills.tolist() == [0,0]
    assert env.alive.sum() == 3
    env.step(zero_actions)
    assert env.collision_events.tolist() == [4,2]  # Per-UAV per-step events, not once per episode.


def test_collision_event_does_not_block_same_step_weapon(make_env,zero_actions):
    env = make_env((0,0),(0,1),(1,0))
    env.positions[0,0],env.positions[0,1],env.positions[1,0] = (0,0),(40,0),(700,0)
    env.headings.fill(0)
    env.speeds.fill(100)
    env.step(zero_actions)
    assert not env.alive[1,0] and env.alive[0,:2].all()
    assert env.collision_events.tolist() == [2,0]
    assert env.combat_kills.tolist() == [1,0]
    assert env.weapon_losses.tolist() == [0,1]
    assert env.attack_window_count.tolist() == [2,0]


def test_collision_events_can_exceed_team_size_without_any_death(make_env,zero_actions):
    env = make_env((0,0),(1,0),attack_range=1)
    env.positions[0,0],env.positions[1,0] = (0,0),(40,0)
    env.speeds.fill(100)
    env.headings.fill(0)
    for _ in range(7):
        _,_,done,_,info = env.step(zero_actions)
        assert not done and env.alive.sum() == 2
    assert info['red_collision_events'] == info['blue_collision_events'] == 7
    assert not env.weapon_losses.any() and not env.oob_losses.any()


@pytest.mark.parametrize('axis,sign,heading',[(0,1,0),(0,-1,math.pi),(1,1,math.pi/2),(1,-1,-math.pi/2)])
def test_oob_death_each_boundary_only_once(make_env,zero_actions,axis,sign,heading):
    env = make_env((0,0),(1,0))
    env.positions[0,0] = (0,0)
    env.positions[0,0,axis] = sign*4990
    env.positions[1,0] = (-2000,-2000)
    env.headings[0,0] = heading
    _,_,done,_,info = env.step(zero_actions)
    assert done and not env.alive[0,0] and info['oob'] == [1,0]
    assert info['weapon_losses'] == info['collision_events'] == [0,0]
    env._resolve_out_of_bounds()
    assert env.oob_losses.tolist() == [1,0]


def test_oob_precedes_collision_and_weapon(make_env,zero_actions):
    env = make_env((0,0),(1,0))
    # Endpoints remain <50m, but the first UAV is already OOB and excluded.
    env.positions[0,0],env.positions[1,0] = (4910,0),(4870,0)
    env.headings[0,0],env.headings[1,0] = 0,0
    env.speeds.fill(100)
    env.step(zero_actions)
    assert env.oob_losses.tolist() == [1,0]
    assert env.collision_events.tolist() == env.weapon_losses.tolist() == [0,0]
    assert env.alive[1,0]


def test_collision_empty_and_singleton_alive_sets(make_env):
    env = make_env()
    env.alive.fill(False)
    assert env._resolve_collisions().tolist() == [0,0]
    env.alive[0,0] = True
    assert env._resolve_collisions().tolist() == [0,0]


def test_same_endpoint_pair_can_collide_and_fire(make_env,zero_actions):
    env = make_env((0,0),(1,0))
    env.positions[0,0],env.positions[1,0] = (0,0),(40,0)
    env.headings.fill(0)
    env.speeds.fill(100)
    _,_,done,_,info = env.step(zero_actions)
    assert done and info['collision_events'] == [1,1]
    assert info['combat_kills'] == [1,0] and info['weapon_losses'] == [0,1]
    assert env.alive[0,0] and not env.alive[1,0]
