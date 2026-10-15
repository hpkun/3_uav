from dataclasses import replace
from pathlib import Path
import pytest
import torch

from uav_mappo_2d.config import ProjectConfig,MAPPOConfig,TrainingConfig,load_config
from uav_mappo_2d.mappo import MAPPO
from uav_mappo_2d.train import build_parser,run_training


def test_v2_config_defaults_and_no_active_old_courses():
    cfg = load_config('configs/base.yaml')
    defaults = ProjectConfig()
    assert cfg.environment == defaults.environment and cfg.reward == defaults.reward
    assert cfg.mappo == defaults.mappo and cfg.training.blue_strategy == 2
    assert not list(Path('configs').glob('xu_course*.yaml'))
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--resume','x','--init-checkpoint','y'])


def test_weights_only_initialization_and_resume_semantics(tmp_path):
    mappo = replace(MAPPOConfig(),hidden_size=32)
    agent = MAPPO(68,60,5,mappo,'cpu',seed=2)
    for network,optimizer in [(agent.actor,agent.actor_optimizer),(agent.critic,agent.critic_optimizer)]:
        optimizer.zero_grad()
        sum(p.sum() for p in network.parameters()).backward()
        optimizer.step()
    source = tmp_path/'source.pt'
    extra = {'episode_index':9,'best_evaluation':{'win_rate':.5,'survival_difference':1}}
    agent.save(source,global_step=12,extra=extra)
    cfg = ProjectConfig(mappo=mappo,training=TrainingConfig(total_env_steps=0,parallel_envs=1,device='cpu',output_dir=str(tmp_path/'init')))
    new = torch.load(run_training(cfg,init_checkpoint=str(source)),weights_only=False)
    assert new['global_step'] == 0 and new['extra']['episode_index'] == 0
    assert new['extra']['best_evaluation'] is None and not new['actor_optimizer']['state']
    for key,value in agent.actor.state_dict().items():
        torch.testing.assert_close(new['actor'][key],value)
    cfg.training.total_env_steps = 12
    cfg.training.output_dir = str(tmp_path/'resume')
    resumed = torch.load(run_training(cfg,resume=str(source)),weights_only=False)
    assert resumed['global_step'] == 12 and resumed['extra'] == extra
    assert resumed['actor_optimizer']['state'] and resumed['critic_optimizer']['state']
