from dataclasses import replace

import numpy as np
import pytest

from uav_mappo_2d.config import EnvironmentConfig, RenderConfig
from uav_mappo_2d.env import UAVCombatEnv


@pytest.fixture
def make_env():
    def make(*entities, **overrides):
        env = UAVCombatEnv(replace(EnvironmentConfig(), **overrides), render_config=RenderConfig(width=320, height=320))
        env.reset(seed=17)
        if entities:
            env.alive.fill(False)
            for team, agent in entities:
                env.alive[team, agent] = True
        return env
    return make


@pytest.fixture
def noop():
    def actions(env):
        masks = np.stack([env.available_action_masks(t) for t in range(2)])
        # 2 and 7 are the two physical no-ops; 7 is valid in both steering subsets.
        return np.where(masks[..., 2], 2, 7).astype(np.int64)
    return actions
