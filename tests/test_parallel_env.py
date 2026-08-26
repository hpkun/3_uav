from __future__ import annotations

from dataclasses import replace
import os

import numpy as np

from uav_mappo_2d.config import EnvironmentConfig, RenderConfig, RewardConfig
from uav_mappo_2d.parallel_env import ParallelUAVEnvs


def test_parallel_workers_and_terminal_auto_reset(tmp_path) -> None:
    environments = ParallelUAVEnvs(
        2,
        replace(EnvironmentConfig(), episode_steps=1),
        RewardConfig(),
        RenderConfig(),
        tmp_path,
        base_seed=9,
    )
    try:
        assert len(set(environments.worker_pids)) == 2
        assert os.getpid() not in environments.worker_pids
        observations, states, masks, _ = environments.reset()
        assert observations.shape == (2, 3, 40)
        assert states.shape == (2, 36)
        assert masks.shape == (2, 3)
        result = environments.step(np.zeros((2, 3, 2), dtype=np.float32))
        assert np.all(result.dones == 1.0)
        assert all("episode_summary" in info for info in result.infos)
        assert np.all(result.active_masks == 1.0)
        assert result.observations.shape == (2, 3, 40)
    finally:
        environments.close()
