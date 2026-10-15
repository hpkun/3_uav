from dataclasses import replace
import os

import numpy as np
import pytest

from uav_mappo_2d.config import EnvironmentConfig,RenderConfig,RewardConfig
from uav_mappo_2d.parallel_env import ParallelUAVEnvs


@pytest.mark.parametrize('strategy',[1,2,3])
def test_workers_shapes_masks_and_terminal_auto_reset(tmp_path,strategy):
    with ParallelUAVEnvs(2,replace(EnvironmentConfig(),episode_steps=1),RewardConfig(),RenderConfig(),tmp_path,
                         base_seed=9,blue_strategy=strategy) as environments:
        assert len(set(environments.worker_pids)) == 2
        assert os.getpid() not in environments.worker_pids
        obs,states,active,masks,infos = environments.reset()
        assert obs.shape == (2,5,68) and states.shape == (2,60)
        assert active.shape == (2,5) and masks.shape == (2,5,10)
        result = environments.step(np.full((2,5),2,dtype=np.int64))
        assert np.all(result.dones == 1)
        assert result.observations.shape == (2,5,68) and result.states.shape == (2,60)
        assert result.action_masks.shape == (2,5,10) and np.all(result.action_masks)
        assert np.all(result.active_masks == 1)
        assert all('episode_summary' in i for i in result.infos)
        # Check exact reset state (including its mask), not just compatible shapes.
        from uav_mappo_2d.env import UAVCombatEnv
        for i in range(2):
            reference = UAVCombatEnv()
            expected,info = reference.reset(seed=9+i+1000003)
            np.testing.assert_array_equal(result.observations[i],expected[0])
            np.testing.assert_array_equal(result.states[i],info['global_state'])
            np.testing.assert_array_equal(result.action_masks[i],info['available_action_masks'])
        again = environments.step(np.full((2,5),2,dtype=np.int64))
        assert np.all(again.dones == 1)
        replay,*_ = environments.reset()
        np.testing.assert_array_equal(replay,obs)
        with pytest.raises(ValueError):
            environments.step(np.zeros((2,5),dtype=np.float32))
