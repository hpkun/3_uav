"""Single-variable V1 contract and exact legacy V0 regression."""
from dataclasses import asdict, replace
import csv
import hashlib

import numpy as np
import pytest
import torch

from uav_mappo_2d.maddpg import MADDPG, JointReplayBuffer, actor_diagnostics, clip_parameter_gradients, ACTOR_DIAGNOSTICS, ACTOR_AGGREGATES
from uav_mappo_2d.maddpg_config import MADDPGConfig, load_maddpg_config
from uav_mappo_2d.train_maddpg import log_line, run_training
from uav_mappo_2d.config import ProjectConfig, EnvironmentConfig, TrainingConfig


def config(mode='bounded_action'):
    return MADDPGConfig(hidden_size=8, batch_size=4, buffer_capacity=32,
                        learning_starts=4, actor_regularization_mode=mode)


def batch():
    rng = np.random.default_rng(29)
    b = {k: rng.uniform(-.8,.8,(4,*shape)).astype(np.float32)
         for k,shape in JointReplayBuffer.SHAPES.items()}
    b['active_masks'] = np.ones((4,5),np.float32)
    b['next_active_masks'] = np.ones((4,5),np.float32)
    b['dones'] = np.zeros(4,np.float32)
    b['active_masks'][2:,0] = 0
    return b


def assert_exact(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            assert_exact(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert_exact(x,y)
    else:
        assert a==b


def test_profiles_differ_only_in_mode():
    a,b = [asdict(load_maddpg_config(p)) for p in ('configs/maddpg.yaml','configs/maddpg_pretanh_reg.yaml')]
    assert a.pop('actor_regularization_mode')=='bounded_action'
    assert b.pop('actor_regularization_mode')=='pre_tanh'
    assert a==b
    assert MADDPGConfig().actor_regularization_mode=='bounded_action'


@pytest.mark.parametrize('mode',['invalid','',None,1])
def test_invalid_mode(mode):
    with pytest.raises(ValueError,match='actor_regularization_mode'):
        config(mode)


def test_forward_and_inference_exact_across_modes():
    a,b = MADDPG(config(),seed=7), MADDPG(config('pre_tanh'),seed=8)
    b.actors.load_state_dict(a.actors.state_dict())
    obs = torch.tensor(batch()['observations'])
    for x,y in zip(a.actors,b.actors):
        assert_exact(x(obs[:,0]),torch.tanh(x.network(obs[:,0])))
        assert_exact(x(obs[:,0]),y(obs[:,0]))
        assert_exact(x.pre_tanh(obs[:,0]),x.network(obs[:,0]))
    assert np.array_equal(a.policy_actions(obs,np.ones((4,5))),b.policy_actions(obs,np.ones((4,5))))


def legacy_update(p,i,b):
    """Frozen pre-V1 implementation, including clipping/Adam/order."""
    active=b['active_masks'][:,i].bool()
    critic=p.critics[i]
    critic.requires_grad_(False)
    try:
        joint=b['actions'][active].detach().clone()
        action=torch.tanh(p.actors[i].network(b['observations'][active,i]))
        joint[:,i]=action
        pg=-critic(b['states'][active],joint).mean()
        reg=p.config.actor_regularization*action.square().mean()
        loss=pg+reg
        opt=p.actor_optimizers[i]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad=clip_parameter_gradients(p.actors[i].parameters(),p.config.max_grad_norm)
        opt.step()
        return dict(actor_loss=float(loss.detach()),actor_grad_norm=grad,
                    policy_gradient_loss=float(pg.detach()),actor_reg=float(reg.detach()))
    finally:
        critic.requires_grad_(True)


def test_v0_exact_legacy_loss_parameters_optimizer_and_rng():
    a,b=MADDPG(config(),seed=7),MADDPG(config(),seed=7)
    data=a.tensor_batch(batch())
    for _ in range(3):
        state=torch.random.get_rng_state().clone()
        expected=legacy_update(a,0,data)
        actual=b.update_actor(0,data)
        assert_exact(state,torch.random.get_rng_state())
        assert_exact(expected,{k:actual[k] for k in expected})
        for group in ('actors','critics','target_actors','target_critics'):
            assert_exact(getattr(a,group).state_dict(),getattr(b,group).state_dict())
        assert_exact(a.actor_optimizers[0].state_dict(),b.actor_optimizers[0].state_dict())


@pytest.mark.parametrize('mode',['bounded_action','pre_tanh'])
def test_formulas_bounded_critic_input_and_active_diagnostics(mode):
    p=MADDPG(config(mode),seed=7)
    data=p.tensor_batch(batch())
    # Explicit logits also distinguish raw z from bounded action at the Critic.
    with torch.no_grad():
        p.actors[0].network[-1].bias.copy_(torch.tensor([3.,-4.]))
        z=p.actors[0].pre_tanh(data['observations'][:2,0])
        action=z.tanh()
        joint=data['actions'][:2].clone()
        joint[:,0]=action
        pg=-p.critics[0](data['states'][:2],joint).mean()
        reg=.001*(z if mode=='pre_tanh' else action).square().mean()
        expected=actor_diagnostics(z,action,reg)
    seen=[]
    h=p.critics[0].register_forward_pre_hook(lambda m,args: seen.append(args[1].detach().clone()))
    result=p.update_actor(0,data)
    h.remove()
    assert result['actor_loss']==float(pg+reg)
    assert result['actor_reg']==float(reg)
    assert_exact(seen[0],joint)
    assert seen[0].abs().max()<=1
    assert result['pre_tanh_abs_mean']==float(z.abs().mean())
    assert result['pre_tanh_abs_max']==float(z.abs().max())
    assert result['tanh_derivative_mean']==float((1-action.square()).mean())
    assert result['accel_saturation']==float((action[:,0].abs()>=.95).float().mean())
    assert result['turn_saturation']==float((action[:,1].abs()>=.95).float().mean())
    assert_exact(expected,{k:result[k] for k in ACTOR_DIAGNOSTICS})


def test_explicit_saturation_regularizer_gradients():
    z=torch.tensor([10.,-10.],dtype=torch.float64,requires_grad=True)
    bounded=.001*z.tanh().square().mean()
    pretanh=.001*z.square().mean()
    g0=torch.autograd.grad(bounded,z)[0]
    g1=torch.autograd.grad(pretanh,z)[0]
    assert g0.abs().max()<1e-10
    torch.testing.assert_close(g1,torch.tensor([.01,-.01],dtype=torch.float64),rtol=0,atol=0)
    assert float(g1.norm()/g0.norm())>1e8


def test_diagnostics_readonly_no_rng_or_grad_effect():
    z=torch.tensor([[3.,-4.],[0.,2.]],requires_grad=True)
    a=z.tanh()
    reg=.001*z.square().mean()
    baseline=torch.autograd.grad(reg+a.mean(),z,retain_graph=True)[0]
    rng=torch.random.get_rng_state().clone()
    before=z.detach().clone()
    result=actor_diagnostics(z,a,reg)
    assert all(isinstance(v,float) for v in result.values())
    assert_exact(before,z.detach())
    assert_exact(rng,torch.random.get_rng_state())
    assert_exact(baseline,torch.autograd.grad(reg+a.mean(),z)[0])


def test_aggregation_only_includes_updated_agents():
    p=MADDPG(config('pre_tanh'),seed=7)
    b=batch()
    b['active_masks'][:,0]=0
    metrics=p.update(b)
    assert metrics['agent0_updated']==0 and metrics['updated_agents']==4
    for aggregate,key in ACTOR_AGGREGATES.items():
        values=[metrics[f'agent{i}_{key}'] for i in range(1,5)]
        expected=max(values) if aggregate=='max_pre_tanh_abs' else np.mean(values)
        assert metrics[aggregate]==expected
        assert metrics[f'agent0_{key}']==0


def test_legacy_checkpoint_missing_mode_is_v0_and_not_rewritten(tmp_path):
    p=MADDPG(config(),seed=7)
    p.update(batch())
    path=tmp_path/'legacy.pt'
    p.save(path,global_step=2000000,environment_config={})
    payload=torch.load(path,weights_only=False)
    payload['maddpg_config'].pop('actor_regularization_mode')
    torch.save(payload,path)
    digest=hashlib.sha256(path.read_bytes()).digest()
    q=MADDPG(MADDPGConfig(**payload['maddpg_config']),seed=99)
    normalized=q.load(path)
    assert normalized['maddpg_config']['actor_regularization_mode']=='bounded_action'
    assert np.array_equal(p.policy_actions(batch()['observations'],np.ones((4,5))),q.policy_actions(batch()['observations'],np.ones((4,5))))
    with pytest.raises(ValueError,match='config must match'):
        MADDPG(config('pre_tanh')).load(path)
    assert hashlib.sha256(path.read_bytes()).digest()==digest


def test_v1_checkpoint_exact_roundtrip_and_reject_v0(tmp_path):
    p=MADDPG(config('pre_tanh'),seed=7)
    p.update(batch())
    path=tmp_path/'v1.pt'
    p.save(path,global_step=1234,environment_config={},best_evaluation={'win_rate':0.})
    q=MADDPG(config('pre_tanh'),seed=99)
    saved=q.load(path)
    assert saved['maddpg_config']['actor_regularization_mode']=='pre_tanh'
    assert saved['global_step']==1234 and q.update_count==p.update_count==1
    for group in ('actors','critics','target_actors','target_critics'):
        assert_exact(getattr(p,group).state_dict(),getattr(q,group).state_dict())
    for group in ('actor_optimizers','critic_optimizers'):
        for a,b in zip(getattr(p,group),getattr(q,group)):
            assert_exact(a.state_dict(),b.state_dict())
    assert np.array_equal(p.policy_actions(batch()['observations'],np.ones((4,5))),q.policy_actions(batch()['observations'],np.ones((4,5))))
    with pytest.raises(ValueError,match='config must match'):
        MADDPG(config()).load(path)


def test_v1_training_csv_tensorboard_and_warmup(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    output=tmp_path/'run'
    cfg=ProjectConfig(environment=replace(EnvironmentConfig(),episode_steps=2),
        training=TrainingConfig(total_env_steps=16,parallel_envs=2,device='cpu',output_dir=str(output),
            log_interval_steps=8,evaluation_interval_steps=16,evaluation_episodes=1,checkpoint_interval_steps=16))
    run_training(cfg,replace(config('pre_tanh'),update_interval_steps=8))
    with (output/'updates.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==2
    keys=set(ACTOR_AGGREGATES)|{f'agent{i}_{k}' for i in range(5) for k in ACTOR_DIAGNOSTICS}
    assert keys<=rows[0].keys()
    assert all(np.isfinite(float(v)) for r in rows for v in r.values())
    events=EventAccumulator(str(output/'tensorboard')).Reload()
    tags=events.Tags()['scalars']
    assert {'maddpg/'+k for k in ('actor_reg_mean','pre_tanh_abs_mean','pre_tanh_abs_max',
        'tanh_derivative_mean','accel_saturation_mean','turn_saturation_mean')}<=set(tags)
    assert {f'maddpg/agent{i}_{k}' for i in range(5) for k in ACTOR_DIAGNOSTICS}<=set(tags)
    warmup=log_line(2,16,0,2,0,[],{})
    assert 'warmup (loss=N/A)' in warmup and 'Replay-batch' not in warmup
    message=log_line(16,16,0,16,2,[],{k:float(v) for k,v in rows[-1].items()})
    assert 'Replay-batch[' in message and 'sat(A/W)' in message
