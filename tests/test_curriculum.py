from dataclasses import replace

import pytest
import torch

from uav_mappo_2d.config import ProjectConfig,MAPPOConfig,TrainingConfig,load_config
from uav_mappo_2d.mappo import MAPPO
from uav_mappo_2d.train import build_parser,run_training


def test_curriculum_configs_and_initialization_flags():
    for course,steps in [(1,2000000),(2,2000000),(3,6000000)]:
        cfg = load_config(f'configs/xu_course{course}.yaml')
        assert cfg.training.blue_strategy == course and cfg.training.total_env_steps == steps
        assert cfg.training.output_dir == f'runs/xu_course{course}_seed7'
        assert cfg.environment == load_config('configs/base.yaml').environment
        assert cfg.reward == load_config('configs/base.yaml').reward
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--resume','x','--init-checkpoint','y'])
    with pytest.raises(ValueError):
        TrainingConfig(blue_strategy=4)


def test_course_initialization_resets_metadata_and_optimizers(tmp_path):
    mappo = replace(MAPPOConfig(),hidden_size=32)
    agent = MAPPO(68,60,5,mappo,'cpu',seed=2)
    # Give Adam real state, so restoring versus resetting it is observable.
    for network,optimizer in [(agent.actor,agent.actor_optimizer),(agent.critic,agent.critic_optimizer)]:
        optimizer.zero_grad()
        sum(p.sum() for p in network.parameters()).backward()
        optimizer.step()
    source = tmp_path/'source.pt'
    extra = {'episode_index':9,'best_evaluation':{'win_rate':.5,'survival_difference':1}}
    agent.save(source,global_step=12,extra=extra)
    cfg = ProjectConfig(mappo=mappo,training=TrainingConfig(total_env_steps=0,parallel_envs=1,device='cpu',output_dir=str(tmp_path/'init')))
    path = run_training(cfg,init_checkpoint=str(source))
    new = torch.load(path,weights_only=False)
    assert new['global_step'] == 0 and new['extra']['episode_index'] == 0
    assert new['extra']['best_evaluation'] is None
    assert not new['actor_optimizer']['state'] and not new['critic_optimizer']['state']
    for key,value in agent.actor.state_dict().items():
        torch.testing.assert_close(new['actor'][key],value)
    for key,value in agent.critic.state_dict().items():
        torch.testing.assert_close(new['critic'][key],value)
    cfg.training.total_env_steps = 12
    cfg.training.output_dir = str(tmp_path/'resume')
    resumed = torch.load(run_training(cfg,resume=str(source)),weights_only=False)
    assert resumed['global_step'] == 12 and resumed['extra'] == extra
    assert resumed['actor_optimizer']['state'] and resumed['critic_optimizer']['state']
