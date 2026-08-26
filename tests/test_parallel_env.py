from __future__ import annotations

import os
from dataclasses import replace

import numpy as np

from uav_mappo_2d.config import EnvironmentConfig, RenderConfig, RewardConfig
from uav_mappo_2d.parallel_env import ParallelUAVEnvs


def test_parallel_environments_use_distinct_processes(tmp_path) -> None:
    environment = replace(
        EnvironmentConfig(), agents_per_team=2, episode_seconds=2.0
    )
    parallel = ParallelUAVEnvs(
        2, environment, RewardConfig(), RenderConfig(), tmp_path
    )
    try:
        assert len(set(parallel.worker_pids)) == 2
        assert os.getpid() not in parallel.worker_pids
        resets = parallel.reset([0, 1], [31, 32], [0, 1])
        assert resets[0][1]["worker_pid"] != resets[1][1]["worker_pid"]

        learner_actions = [
            np.zeros((environment.agents_per_team, 2), dtype=np.float32)
            for _ in range(2)
        ]
        results = parallel.step([0, 1], learner_actions, [None, None])
        assert results[0][0].shape[0] == 2
        assert results[1][0].shape[0] == 2
        assert results[0][-1]["worker_pid"] != results[1][-1]["worker_pid"]
    finally:
        parallel.close()

