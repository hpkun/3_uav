from dataclasses import replace
import os
import numpy as np
import pytest

from uav_mappo_2d.config import EnvironmentConfig,RenderConfig,RewardConfig
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.parallel_env import ParallelUAVEnvs


def test_parallel_continuous_shapes_and_exact_terminal_reset(tmp_path):
    cfg = replace(EnvironmentConfig(),episode_steps=1)
    with ParallelUAVEnvs(2,cfg,RewardConfig(),RenderConfig(),tmp_path,base_seed=9) as environments:
        assert len(set(environments.worker_pids)) == 2 and os.getpid() not in environments.worker_pids
        obs,states,active,infos = environments.reset()
        assert obs.shape == (2,5,68) and states.shape == (2,60) and active.shape == (2,5)
        result = environments.step(np.zeros((2,5,2),np.float32))
        assert not hasattr(result,'action_masks')
        assert np.all(result.dones == 1) and np.all(result.active_masks == 1)
        for i in range(2):
            reference = UAVCombatEnv(cfg)
            expected,info = reference.reset(seed=9+i+1000003)
            np.testing.assert_array_equal(result.observations[i],expected[0])
            np.testing.assert_array_equal(result.states[i],info['global_state'])
            assert 'episode_summary' in result.infos[i]
        again = environments.step(np.zeros((2,5,2),np.float32))
        assert np.all(again.dones == 1)
        replay,*_ = environments.reset()
        np.testing.assert_array_equal(replay,obs)
        for actions in [np.zeros((2,5)),np.full((2,5,2),np.nan),np.full((2,5,2),2.)]:
            with pytest.raises(ValueError):
                environments.step(actions)
