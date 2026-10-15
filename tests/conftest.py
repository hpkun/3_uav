from dataclasses import replace
import numpy as np
import pytest

from uav_mappo_2d.config import EnvironmentConfig, RewardConfig, RenderConfig
from uav_mappo_2d.env import UAVCombatEnv


@pytest.fixture
def make_env():
    def make(*entities, shaping_scale=0., **overrides):
        env = UAVCombatEnv(replace(EnvironmentConfig(), **overrides),
                           replace(RewardConfig(), shaping_scale=shaping_scale),
                           RenderConfig(width=320, height=320))
        env.reset(seed=17)
        if entities:
            env.alive.fill(False)
            for entity in entities:
                env.alive[entity] = True
        return env
    return make


@pytest.fixture
def zero_actions():
    return np.zeros((2, 5, 2), dtype=np.float32)
