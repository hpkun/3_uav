from __future__ import annotations

from dataclasses import replace

import numpy as np

from uav_mappo_2d.config import EnvironmentConfig, MAPPOConfig, RewardConfig
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.mappo import EpisodeBuffer, MAPPO
from uav_mappo_2d.opponents import RuleBasedOpponent


def test_mappo_episode_update_is_finite() -> None:
    env_config = replace(
        EnvironmentConfig(), agents_per_team=2, episode_seconds=3.0, decision_dt=1.0
    )
    env = UAVCombatEnv(env_config, RewardConfig())
    observation, _ = env.reset(seed=5)
    config = replace(MAPPOConfig(), hidden_size=32, update_epochs=1)
    agent = MAPPO(env.obs_dim, env.state_dim, env.n_agents, config, "cpu")
    rule = RuleBasedOpponent()
    actor_hidden = agent.initial_actor_hidden()
    critic_hidden = agent.initial_critic_hidden()
    buffer = EpisodeBuffer(config.hidden_size, env.n_agents)

    terminated = truncated = False
    while not (terminated or truncated):
        active = env.alive[0].astype(np.float32)
        state = env.global_state(0)
        action, logp, value, next_actor_hidden, next_critic_hidden = agent.act(
            observation[0], state, actor_hidden, critic_hidden, active
        )
        joint_action = np.stack([action, rule.act(env, 1)])
        next_observation, rewards, terminated, truncated, _ = env.step(joint_action)
        buffer.append(
            observation=observation[0],
            state=state,
            action=action,
            log_probability=logp,
            value=value,
            reward=float(np.mean(rewards[0])),
            done=terminated or truncated,
            active_mask=active,
        )
        observation = next_observation
        actor_hidden = next_actor_hidden
        critic_hidden = next_critic_hidden

    short_buffer = EpisodeBuffer(config.hidden_size, env.n_agents)
    for timestep in range(2):
        short_buffer.append(
            observation=buffer.observations[timestep],
            state=buffer.states[timestep],
            action=buffer.actions[timestep],
            log_probability=buffer.log_probabilities[timestep],
            value=buffer.values[timestep],
            reward=buffer.rewards[timestep],
            done=timestep == 1,
            active_mask=buffer.active_masks[timestep],
        )

    metrics = agent.update_many([buffer, short_buffer])
    assert len(buffer) == 3
    assert metrics
    assert metrics["episodes_per_update"] == 2.0
    assert all(np.isfinite(value) for value in metrics.values())
