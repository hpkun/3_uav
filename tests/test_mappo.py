from __future__ import annotations

from dataclasses import replace

import numpy as np

from uav_mappo_2d.config import MAPPOConfig
from uav_mappo_2d.mappo import MAPPO, RolloutBuffer


def test_actor_critic_and_small_ppo_update_are_finite() -> None:
    config = replace(MAPPOConfig(), hidden_size=32, update_epochs=2, minibatch_size=4)
    agent = MAPPO(40, 36, 3, config, "cpu", seed=8)
    rng = np.random.default_rng(8)
    observations = rng.normal(size=(2, 3, 40)).astype(np.float32)
    states = rng.normal(size=(2, 36)).astype(np.float32)
    masks = np.ones((2, 3), dtype=np.float32)
    stochastic, log_probabilities, values = agent.act(
        observations, states, masks
    )
    deterministic = agent.policy_actions(observations, masks, deterministic=True)
    assert stochastic.shape == deterministic.shape == (2, 3, 2)
    assert log_probabilities.shape == (2, 3)
    assert values.shape == (2,)
    assert np.max(np.abs(stochastic)) <= 1.0
    assert np.max(np.abs(deterministic)) <= 1.0
    assert np.isfinite(log_probabilities).all()
    assert np.isfinite(values).all()

    buffer = RolloutBuffer(4, 2)
    for timestep in range(4):
        actions, log_probs, step_values = agent.act(observations, states, masks)
        buffer.insert(
            observations,
            states,
            actions,
            log_probs,
            step_values,
            np.array([0.1, -0.1], dtype=np.float32),
            np.array([timestep == 3, timestep == 2], dtype=np.float32),
            masks,
        )
    buffer.compute_returns_and_advantages(
        agent.values(states), config.gamma, config.gae_lambda
    )
    metrics = agent.update(buffer)
    expected = {
        "actor_loss",
        "critic_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "actor_grad_norm",
        "critic_grad_norm",
        "explained_variance",
    }
    assert expected == set(metrics)
    assert all(np.isfinite(value) for value in metrics.values())
