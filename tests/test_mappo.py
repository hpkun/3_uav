from dataclasses import replace
import numpy as np
import pytest
import torch

from uav_mappo_2d.config import MAPPOConfig
from uav_mappo_2d.mappo import MAPPO,RolloutBuffer
from uav_mappo_2d.models import ContinuousGaussianActor


def test_gaussian_tanh_corrected_logprob_and_determinism():
    torch.manual_seed(7)
    actor = ContinuousGaussianActor(68,32)
    obs = torch.randn(2,5,68)
    actions,logs = actor.sample(obs)
    deterministic,_ = actor.sample(obs,deterministic=True)
    assert actions.shape == (2,5,2) and logs.shape == (2,5)
    assert torch.max(actions.abs()) <= 1
    torch.testing.assert_close(deterministic,torch.tanh(actor.network(obs)))
    recomputed,entropy = actor.evaluate_actions(obs,actions)
    torch.testing.assert_close(logs,recomputed,atol=2e-5,rtol=2e-5)
    bounded = actions.clamp(-1+1e-6,1-1e-6)
    distribution = actor.distribution(obs)
    expected = (distribution.log_prob(torch.atanh(bounded))-torch.log(1-bounded.square()+1e-6)).sum(-1)
    torch.testing.assert_close(recomputed,expected)
    assert torch.isfinite(entropy).all() and torch.isfinite(logs).all()
    saturated,_ = actor.evaluate_actions(obs,torch.ones_like(actions))
    assert torch.isfinite(saturated).all()
    with torch.no_grad():
        actor.log_std.fill_(10)
    torch.testing.assert_close(actor.distribution(obs).stddev,torch.full_like(actions,math_exp_one()))


def math_exp_one():
    return float(np.e)


def test_rollout_continuous_ppo_dead_mask_finite_and_checkpoint(tmp_path):
    config = replace(MAPPOConfig(),hidden_size=32,update_epochs=2,minibatch_size=4)
    agent = MAPPO(68,60,5,config,'cpu',seed=8)
    rng = np.random.default_rng(8)
    obs = rng.normal(size=(2,5,68)).astype(np.float32)
    states = rng.normal(size=(2,60)).astype(np.float32)
    active = np.ones((2,5),np.float32)
    active[0,-1] = 0
    buffer = RolloutBuffer(4,2,5,68,60)
    assert buffer.actions.shape == (4,2,5,2) and buffer.actions.dtype == np.float32
    assert not hasattr(buffer,'action_masks')
    for step in range(4):
        actions,logs,values = agent.act(obs,states,active)
        assert np.all(actions[0,-1] == 0) and logs[0,-1] == 0
        buffer.insert(obs,states,actions,logs,values,np.array([.1,-.1],np.float32),
                      np.array([step==3,step==2],np.float32),active)
    buffer.compute_returns_and_advantages(agent.values(states),.99,.95)
    before = [p.detach().clone() for p in agent.actor.parameters()]
    metrics = agent.update(buffer)
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a,b) for a,b in zip(before,agent.actor.parameters()))
    with torch.no_grad():
        values = agent.critic(torch.as_tensor(buffer.states.reshape(-1,60))).numpy()
    returns = buffer.returns.ravel()
    assert metrics['explained_variance'] == pytest.approx(1-np.var(returns-values)/np.var(returns),abs=1e-5)
    path = tmp_path/'continuous.pt'
    agent.save(path,global_step=12)
    loaded = MAPPO(68,60,5,config,'cpu')
    loaded.load(path)
    np.testing.assert_array_equal(agent.policy_actions(obs,active),loaded.policy_actions(obs,active))


def test_dead_samples_do_not_contribute_to_actor_gradient():
    config = replace(MAPPOConfig(),hidden_size=16,update_epochs=1,minibatch_size=4)
    a,b = MAPPO(68,60,5,config,'cpu',seed=2),MAPPO(68,60,5,config,'cpu',seed=2)
    b.actor.load_state_dict(a.actor.state_dict())
    b.critic.load_state_dict(a.critic.state_dict())
    buffers = []
    for agent in (a,b):
        buf = RolloutBuffer(2,1,5,68,60)
        buf.position = 2
        buf.active_masks.fill(1)
        buf.active_masks[:,:,4] = 0
        buf.advantages[:,0] = [1,-1]
        buf.returns[:,0] = [1,-1]
        buffers.append(buf)
    buffers[1].observations[:,:,4] = 100
    buffers[1].actions[:,:,4] = .99
    buffers[1].old_log_probabilities[:,:,4] = 20
    a.update(buffers[0]); b.update(buffers[1])
    for p,q in zip(a.actor.parameters(),b.actor.parameters()):
        torch.testing.assert_close(p,q)


def test_gae_terminal_mask():
    buffer = RolloutBuffer(2,1,5,68,60)
    buffer.position = 2
    buffer.rewards[:,0] = [1,2]
    buffer.dones[:,0] = [1,0]
    buffer.compute_returns_and_advantages(np.array([3],np.float32),1,1)
    np.testing.assert_allclose(buffer.returns[:,0],[1,5])
