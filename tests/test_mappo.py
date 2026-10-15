from dataclasses import replace

import numpy as np
import pytest
import torch

from uav_mappo_2d.config import MAPPOConfig
from uav_mappo_2d.mappo import MAPPO, RolloutBuffer
from uav_mappo_2d.models import CategoricalActor


def test_categorical_mask_sampling_argmax_logprob_entropy():
    actor = CategoricalActor(68,32)
    obs = torch.zeros(1000,68)
    masks = torch.zeros(1000,10,dtype=torch.bool)
    masks[:,[5,6,7]] = True
    dist = actor.distribution(obs,masks)
    assert torch.all(dist.probs[:,[0,1,2,3,4,8,9]] == 0)
    actions,logs = actor.sample(obs,masks)
    assert actions.dtype == torch.int64 and actions.shape == (1000,)
    assert set(actions.tolist()) <= {5,6,7}
    deterministic,_ = actor.sample(obs,masks,deterministic=True)
    assert torch.all(masks.gather(-1,deterministic[:,None]))
    recomputed,entropy = actor.evaluate_actions(obs,actions,masks)
    torch.testing.assert_close(logs,recomputed)
    assert torch.isfinite(logs).all() and torch.isfinite(entropy).all()
    with pytest.raises(ValueError):
        actor.distribution(obs,torch.zeros_like(masks))


def build_rollout(agent):
    rng = np.random.default_rng(8)
    obs = rng.normal(size=(2,5,68)).astype(np.float32)
    obs[:,0,0] = [0,1]
    states = rng.normal(size=(2,60)).astype(np.float32)
    active = np.ones((2,5),dtype=np.float32)
    active[0,-1] = 0
    masks = np.ones((2,5,10),dtype=bool)
    masks[0,0] = False
    masks[0,0,[5,6,7]] = True
    masks[0,-1] = False
    masks[0,-1,2] = True
    buffer = RolloutBuffer(4,2,5,68,60,10)
    for step in range(4):
        actions,logs,values = agent.act(obs,states,active,masks)
        assert actions.shape == (2,5) and actions.dtype == np.int64
        assert actions[0,-1] == 2 and logs[0,-1] == 0
        assert np.all(np.take_along_axis(masks,actions[...,None],axis=-1))
        buffer.insert(obs,states,actions,logs,values,np.array([.1,-.1],np.float32),
                      np.array([step==3,step==2],np.float32),active,masks)
    buffer.compute_returns_and_advantages(agent.values(states),.99,.95)
    return buffer,obs,active,masks


def test_ppo_uses_stored_masks_and_updates_finitely(monkeypatch,tmp_path):
    config = replace(MAPPOConfig(),hidden_size=32,update_epochs=2,minibatch_size=4)
    agent = MAPPO(68,60,5,config,'cpu',seed=8)
    buffer,obs,active,masks = build_rollout(agent)
    old_weights = [p.detach().clone() for p in agent.actor.parameters()]
    evaluate = agent.actor.evaluate_actions
    def checked(observations,actions,action_masks):
        env_ids = observations[:,0,0].long()
        expected = torch.as_tensor(masks)[env_ids]
        torch.testing.assert_close(action_masks,expected)
        return evaluate(observations,actions,action_masks)
    monkeypatch.setattr(agent.actor,'evaluate_actions',checked)
    metrics = agent.update(buffer)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(before,after) for before,after in zip(old_weights,agent.actor.parameters()))
    assert buffer.actions.shape == (4,2,5) and buffer.actions.dtype == np.int64
    assert buffer.action_masks.shape == (4,2,5,10)
    with torch.no_grad():
        values = agent.critic(torch.as_tensor(buffer.states.reshape(-1,60))).numpy()
    returns = buffer.returns.ravel()
    assert metrics['explained_variance'] == pytest.approx(1-np.var(returns-values)/np.var(returns),abs=1e-5)
    path = tmp_path/'categorical.pt'
    agent.save(path,global_step=12,extra={'episode_index':4,'best_evaluation':{'win_rate':.5,'survival_difference':1}})
    loaded = MAPPO(68,60,5,config,'cpu',seed=9)
    checkpoint = loaded.load(path)
    assert checkpoint['global_step'] == 12
    np.testing.assert_array_equal(agent.policy_actions(obs,active,masks),loaded.policy_actions(obs,active,masks))
    assert loaded.actor_optimizer.state_dict()['state']
    initialized = MAPPO(68,60,5,config,'cpu',seed=10)
    initialized.load(path,load_optimizers=False)
    assert not initialized.actor_optimizer.state_dict()['state']
    assert not initialized.critic_optimizer.state_dict()['state']


def test_team_gae_stops_at_terminal():
    buffer = RolloutBuffer(2,1,5,68,60,10)
    buffer.position = 2
    buffer.rewards[:,0] = [1,2]
    buffer.dones[:,0] = [1,0]
    buffer.compute_returns_and_advantages(np.array([3],dtype=np.float32),1,1)
    np.testing.assert_allclose(buffer.returns[:,0],[1,5])
