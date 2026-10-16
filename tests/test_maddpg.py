from dataclasses import asdict, replace
import numpy as np
import pytest
import torch

from uav_mappo_2d.maddpg import (MADDPG, JointReplayBuffer, OUNoise, td_target, clip_parameter_gradients)
from uav_mappo_2d.maddpg_config import MADDPGConfig, load_maddpg_config
from uav_mappo_2d.train_maddpg import due_updates, better_evaluation, run_training
from uav_mappo_2d.train import _is_better_evaluation
from uav_mappo_2d.config import ProjectConfig, EnvironmentConfig, TrainingConfig, RewardConfig, RenderConfig
from uav_mappo_2d.parallel_env import ParallelUAVEnvs
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.evaluate_maddpg import evaluate


@pytest.fixture
def algorithm():
    return replace(MADDPGConfig(),hidden_size=8,batch_size=4,buffer_capacity=32,learning_starts=4)


def batch(n=4):
    rng = np.random.default_rng(9)
    return {k: (np.ones((n,*shape),np.float32) if 'mask' in k else
                np.zeros((n,*shape),np.float32) if k=='dones' else
                rng.uniform(.1,.9,(n,*shape)).astype(np.float32))
            for k,shape in JointReplayBuffer.SHAPES.items()}


def copy_params(model):
    return [p.detach().clone() for p in model.parameters()]


def unchanged(model, old):
    return all(torch.equal(p,v) for p,v in zip(model.parameters(),old))


def test_reference_config_and_dimensions():
    c = load_maddpg_config('configs/maddpg.yaml')
    assert c == MADDPGConfig() and c.warmup(300)==307200
    p = MADDPG(c)
    assert len(p.actors)==len(p.critics)==len(p.target_actors)==len(p.target_critics)==5
    assert [sum(v.numel() for v in a.parameters()) for a in p.actors] == [8706]*5
    assert [sum(v.numel() for v in q.parameters()) for q in p.critics] == [8769]*5
    assert p.actors[0].network[0].in_features == 68
    assert p.critics[0].network[0].in_features == 70


def test_all_online_and_target_storages_independent(algorithm):
    p = MADDPG(algorithm)
    models = [*p.actors,*p.critics,*p.target_actors,*p.target_critics]
    assert len({id(m) for m in models})==20
    pointers = [v.data_ptr() for m in models for v in m.parameters()]
    assert len(set(pointers))==len(pointers)
    for online,target in [(p.actors,p.target_actors),(p.critics,p.target_critics)]:
        for a,t in zip(online.parameters(),target.parameters()):
            assert torch.equal(a,t) and not t.requires_grad


def test_actor_update_isolation_and_q_gradient(algorithm):
    p = MADDPG(replace(algorithm,actor_regularization=0.))
    b = p.tensor_batch(batch())
    with torch.no_grad():
        for v in p.critics[0].parameters():
            v.fill_(.1)  # guaranteed nonzero dQ/da, with positive inputs
    others = [copy_params(m) for m in p.actors[1:]]
    critics, targets = copy_params(p.critics),copy_params(p.target_actors)
    initial = copy_params(p.actors[0])
    result = p.update_actor(0,b)
    assert not unchanged(p.actors[0],initial) and result['actor_grad_norm']>0
    assert all(unchanged(m,v) for m,v in zip(p.actors[1:],others))
    assert unchanged(p.critics,critics) and unchanged(p.target_actors,targets)
    assert all(v.requires_grad and v.grad is None for v in p.critics[0].parameters())


def test_critic_update_isolation(algorithm):
    p = MADDPG(algorithm)
    b = p.tensor_batch(batch())
    initial,others,actors = copy_params(p.critics[0]),copy_params(p.critics[1:]),copy_params(p.actors)
    p.update_critic(0,b,torch.ones(4)*10)
    assert not unchanged(p.critics[0],initial)
    assert unchanged(p.critics[1:],others) and unchanged(p.actors,actors)


def test_soft_update_exact_tensor_formula(algorithm):
    p = MADDPG(algorithm)
    old_targets = [copy_params(p.target_actors),copy_params(p.target_critics)]
    b = p.tensor_batch(batch())
    p.update_critic(0,b,torch.ones(4))
    p.update_actor(0,b)
    assert not torch.equal(next(p.actors[0].parameters()),next(p.target_actors[0].parameters()))
    p.soft_update()
    for online,target,previous in zip((p.actors,p.critics),(p.target_actors,p.target_critics),old_targets):
        for a,t,old in zip(online.parameters(),target.parameters(),previous):
            torch.testing.assert_close(t,(1-algorithm.tau)*old+algorithm.tau*a,rtol=0,atol=1e-7)


@pytest.mark.parametrize('done,next_active,expected',[(0,1,2.9),(1,1,1.),(0,0,1.)])
def test_td_target_mask_and_detach(done,next_active,expected):
    q = torch.tensor([2.],requires_grad=True)
    y = td_target(torch.tensor([1.]),torch.tensor([done]),torch.tensor([next_active]),q,.95)
    assert y.item()==pytest.approx(expected) and not y.requires_grad


def test_target_joint_action_masks_and_all_targets_pre_update(algorithm):
    p = MADDPG(algorithm)
    b = p.tensor_batch(batch())
    b['next_active_masks'][:,2]=0
    seen=[]
    hooks=[q.register_forward_pre_hook(lambda m,args: seen.append(args[1].clone())) for q in p.target_critics]
    y = p.targets(b)
    for h in hooks:
        h.remove()
    assert len(seen)==5 and all(torch.equal(a,seen[0]) for a in seen)
    assert torch.equal(seen[0][:,2],torch.zeros(4,2))
    assert torch.equal(y[:,2],b['rewards']) and not y.requires_grad


def test_current_dead_samples_skip_own_losses(algorithm):
    p = MADDPG(algorithm)
    b = batch()
    b['active_masks'][:,0]=0
    b['next_active_masks'][:,0]=0
    actor,critic = copy_params(p.actors[0]),copy_params(p.critics[0])
    metrics = p.update(b)
    assert unchanged(p.actors[0],actor) and unchanged(p.critics[0],critic)
    assert metrics['agent0_updated']==0 and metrics['updated_agents']==4


def test_all_dead_batch_safely_skips_without_nonfinite_losses(algorithm):
    p = MADDPG(algorithm)
    b = batch()
    b['active_masks'].fill(0)
    b['next_active_masks'].fill(0)
    actors,critics = copy_params(p.actors),copy_params(p.critics)
    metrics = p.update(b)
    assert metrics['updated_agents']==0 and np.isfinite(list(metrics.values())).all()
    assert unchanged(p.actors,actors) and unchanged(p.critics,critics)


def test_round_order_and_same_joint_batch_for_all_agents(algorithm,monkeypatch):
    p = MADDPG(algorithm)
    order, batches = [],[]
    target,critic,actor,soft = p.targets,p.update_critic,p.update_actor,p.soft_update
    def targets(b):
        order.append('targets')
        return target(b)
    def update_critic(i,b,y):
        order.append(f'critic{i}')
        batches.append(id(b))
        return critic(i,b,y)
    def update_actor(i,b):
        order.append(f'actor{i}')
        batches.append(id(b))
        return actor(i,b)
    def update_targets():
        order.append('soft')
        return soft()
    monkeypatch.setattr(p,'targets',targets)
    monkeypatch.setattr(p,'update_critic',update_critic)
    monkeypatch.setattr(p,'update_actor',update_actor)
    monkeypatch.setattr(p,'soft_update',update_targets)
    p.update(batch())
    assert order==['targets',*[f'critic{i}' for i in range(5)],*[f'actor{i}' for i in range(5)],'soft']
    assert len(set(batches))==1


def test_losses_only_use_current_active_rows(algorithm):
    p = MADDPG(algorithm)
    b = p.tensor_batch(batch())
    b['active_masks'][1:,0]=0
    target = torch.tensor([2.,1e8,1e8,1e8])
    q = p.critics[0](b['states'][:1],b['actions'][:1]).detach()
    result = p.update_critic(0,b,target)
    assert result['critic_loss']==pytest.approx(float((q-2).square().mean()))


def test_actor_uses_other_replay_actions_not_other_current_policies(algorithm):
    p = MADDPG(algorithm)
    b = p.tensor_batch(batch())
    seen=[]
    hook = p.critics[2].register_forward_pre_hook(lambda m,args: seen.append(args[1].detach().clone()))
    old = b['actions'].clone()
    p.update_actor(2,b)
    hook.remove()
    assert torch.equal(b['actions'],old)
    assert torch.equal(seen[0][:,[0,1,3,4]],old[:,[0,1,3,4]])


def test_actor_loss_regularization_formula_on_active_samples(algorithm):
    p = MADDPG(algorithm)
    b = p.tensor_batch(batch())
    b['active_masks'][2:,0]=0
    with torch.no_grad():
        action = p.actors[0](b['observations'][:2,0])
        joint = b['actions'][:2].clone()
        joint[:,0]=action
        expected = -p.critics[0](b['states'][:2],joint).mean()+algorithm.actor_regularization*action.square().mean()
    result = p.update_actor(0,b)
    assert result['actor_loss']==pytest.approx(float(expected))


def test_replay_ring_and_joint_alignment():
    replay = JointReplayBuffer(3,seed=8)
    for v in range(5):
        replay.add(observations=np.full((5,68),v),states=np.full(60,v+10),actions=np.full((5,2),v/10),
                   rewards=v+20,next_observations=np.full((5,68),v+1),next_states=np.full(60,v+11),
                   dones=v%2,active_masks=np.full(5,v%2),next_active_masks=np.full(5,1-v%2))
    assert len(replay)==3 and replay.position==2
    assert set(replay.data['observations'][:,0,0])=={2,3,4}
    b = replay.sample(3)
    for key,shape in replay.SHAPES.items():
        assert b[key].shape==(3,*shape)
    for n in range(3):
        v = b['observations'][n,0,0]
        assert (b['observations'][n]==v).all() and (b['states'][n]==v+10).all()
        assert np.allclose(b['actions'][n],v/10) and b['rewards'][n]==v+20
        assert (b['next_observations'][n]==v+1).all() and (b['next_states'][n]==v+11).all()
        assert b['dones'][n]==v%2 and (b['active_masks'][n]==v%2).all()
        assert (b['next_active_masks'][n]==1-v%2).all()
    b['observations'].fill(999)
    assert 999 not in replay.data['observations']


def test_actions_deterministic_local_and_dead_zero_with_noise(algorithm):
    p = MADDPG(algorithm)
    obs = batch(2)['observations']
    mask = np.ones((2,5),np.float32)
    mask[:,1]=0
    a = p.policy_actions(obs,mask)
    assert np.array_equal(a,p.policy_actions(obs,mask)) and np.abs(a).max()<=1
    assert not a[:,1].any()
    changed = obs.copy()
    changed[:,3]+=10
    assert np.array_equal(a[:,0],p.policy_actions(changed,mask)[:,0])
    noise = OUNoise(2,replace(algorithm,ou_sigma=100),seed=5)
    sampled = p.actions(obs,mask,noise)
    assert np.abs(sampled).max()<=1 and not sampled[:,1].any()


def test_ou_independent_seeded_and_environment_reset(algorithm):
    a,b = OUNoise(16,algorithm,seed=3),OUNoise(16,algorithm,seed=3)
    values = a.sample()
    assert np.array_equal(values,b.sample()) and values.shape==(16,5,2)
    assert not np.array_equal(values[0],values[1]) and not np.array_equal(values[0,0],values[0,1])
    a.reset([2,4])
    assert (a.state[[2,4]]==algorithm.ou_mu).all() and np.array_equal(a.state[0],values[0])


def test_official_per_parameter_gradient_clipping():
    a,b = torch.nn.Parameter(torch.zeros(2)),torch.nn.Parameter(torch.zeros(3))
    a.grad,b.grad = torch.tensor([3.,4.]),torch.tensor([0.,0.,12.])
    norm = clip_parameter_gradients([a,b],.5)
    assert norm==pytest.approx(13.)
    assert float(a.grad.norm())<=.500001 and float(b.grad.norm())<=.500001


def test_parallel_update_schedule_uses_individual_transitions():
    next_step,at = 100,[]
    for step in range(16,321,16):
        count,next_step = due_updates(step,next_step,100,True)
        at.extend([step]*count)
    assert at==[112,208,304]
    assert due_updates(256,100,100,True)==(2,300)
    assert due_updates(256,100,100,False)==(0,300)


def test_official_warmup_does_not_backlog_thousands_of_updates():
    c = MADDPGConfig()
    nxt,updates = 100,0
    for step in range(16,307201,16):
        count,nxt = due_updates(step,nxt,100,step>=c.warmup(300))
        updates+=count
    assert updates==1 and nxt==307300


def test_save_load_all_networks_optimizers_and_outputs(tmp_path,algorithm):
    p = MADDPG(algorithm)
    p.update(batch())
    path = tmp_path/'maddpg.pt'
    p.save(path,global_step=123,environment_config={'environment':asdict(EnvironmentConfig())},best_evaluation={'win_rate':.2})
    q = MADDPG(algorithm,seed=99)
    payload = q.load(path)
    assert payload['algorithm']=='maddpg' and payload['global_step']==123 and q.update_count==1
    for key in ('actors','critics','target_actors','target_critics'):
        assert unchanged(getattr(q,key),copy_params(getattr(p,key)))
    obs = batch()['observations']
    mask = np.ones((4,5))
    assert np.array_equal(p.policy_actions(obs,mask),q.policy_actions(obs,mask))
    for a,b in zip(p.critic_optimizers,q.critic_optimizers):
        assert len(a.state)==len(b.state)>0
    b = p.tensor_batch(batch())
    for i in range(5):
        torch.testing.assert_close(p.critics[i](b['states'],b['actions']),q.critics[i](b['states'],b['actions']),rtol=0,atol=0)
    wrong = tmp_path/'mappo.pt'
    torch.save({'actor':{}},wrong)
    with pytest.raises(ValueError,match='Not a MADDPG'):
        q.load(wrong)


def test_best_selection_same_as_mappo():
    for current in [None,dict(win_rate=.5,survival_difference=1)]:
        for candidate in [dict(win_rate=.4,survival_difference=5),dict(win_rate=.5,survival_difference=2),
                          dict(win_rate=.5,survival_difference=1)]:
            assert better_evaluation(candidate,current)==_is_better_evaluation(candidate,current)


def test_terminal_autoreset_transition_has_no_bootstrap(tmp_path,algorithm):
    environment = replace(EnvironmentConfig(),episode_steps=1)
    with ParallelUAVEnvs(1,environment,RewardConfig(),RenderConfig(),tmp_path,base_seed=17) as vector:
        obs,state,mask,_ = vector.reset()
        result = vector.step(np.zeros((1,5,2),np.float32))
        assert result.dones.tolist()==[1.]
        reference = UAVCombatEnv(environment)
        expected,info = reference.reset(seed=17+1_000_003)
        np.testing.assert_array_equal(result.observations[0],expected[0])
        np.testing.assert_array_equal(result.states[0],info['global_state'])
        replay = JointReplayBuffer(4)
        replay.add(observations=obs[0],states=state[0],actions=np.zeros((5,2)),rewards=result.rewards[0],
                   next_observations=result.observations[0],next_states=result.states[0],dones=result.dones[0],
                   active_masks=mask[0],next_active_masks=result.active_masks[0])
        p = MADDPG(algorithm)
        b = p.tensor_batch(replay.sample(1))
        assert torch.equal(p.targets(b),b['rewards'][:,None].expand(-1,5))
        b['next_observations']+=10000
        b['next_states']+=10000
        assert torch.equal(p.targets(b),b['rewards'][:,None].expand(-1,5))
        reference.close()


def test_short_training_files_metrics_and_evaluation(tmp_path,algorithm):
    cfg = ProjectConfig(environment=replace(EnvironmentConfig(),episode_steps=2),
        training=TrainingConfig(total_env_steps=32,parallel_envs=2,device='cpu',output_dir=str(tmp_path/'run'),
                                log_interval_steps=16,evaluation_interval_steps=16,evaluation_episodes=1,
                                checkpoint_interval_steps=16))
    final = run_training(cfg,replace(algorithm,update_interval_steps=8))
    payload = torch.load(final,weights_only=False)
    assert payload['global_step']==32 and payload['update_count']==4
    assert (final.parent/'best.pt').exists() and (final.parent/'step_000000032.pt').exists()
    p = MADDPG(replace(algorithm,update_interval_steps=8))
    p.load(final)
    result = evaluate(cfg,p,episodes=1,seed_start=3000000)
    assert result['mean_episode_length']==2 and np.isfinite(list(result.values())).all()
