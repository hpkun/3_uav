from __future__ import annotations

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .config import ProjectConfig, load_config
from .env import UAVCombatEnv
from .mappo import EpisodeBuffer, MAPPO
from .parallel_env import ParallelUAVEnvs


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _progress_line(
    *,
    episode: int,
    total_episodes: int,
    stage: str,
    result: str,
    episode_return: float,
    mean_return: float,
    win_rate: float,
    length: int,
    alive: list[int],
    kills: list[int],
    steps_per_second: float,
    eta_seconds: float,
    sample_seconds: float,
    update_seconds: float,
) -> str:
    percent = 100.0 * episode / max(total_episodes, 1)
    return (
        f"[TRAIN {episode:>5}/{total_episodes:<5} {percent:6.2f}%] "
        f"stage={stage:<8} result={result} | "
        f"return={episode_return:8.2f} avg100={mean_return:8.2f} "
        f"win100={win_rate:6.1%} | steps={length:3d} "
        f"kills(R:B)={kills[0]}:{kills[1]} alive(R:B)={alive[0]}:{alive[1]} "
        f"sps={steps_per_second:6.0f} "
        f"time(S/U)={sample_seconds:.1f}/{update_seconds:.1f}s "
        f"eta={_duration(eta_seconds)}"
    )


def run_training(config: ProjectConfig, *, resume: str | None = None) -> Path:
    set_seed(config.seed)
    device = resolve_device(config.training.device)
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / "tensorboard")

    prototype_env = UAVCombatEnv(
        config.environment,
        config.reward,
        config.render,
        project_root=Path.cwd(),
    )
    learner = MAPPO(
        prototype_env.obs_dim,
        prototype_env.state_dim,
        prototype_env.n_agents,
        config.mappo,
        device,
    )
    opponent_policy = MAPPO(
        prototype_env.obs_dim,
        prototype_env.state_dim,
        prototype_env.n_agents,
        config.mappo,
        device,
    )
    n_agents = prototype_env.n_agents
    prototype_env.close()
    snapshots: deque = deque(maxlen=config.training.snapshot_pool_size)
    start_episode = 1
    if resume:
        checkpoint = learner.load(resume)
        start_episode = int(checkpoint.get("episode", 0)) + 1
    snapshots.append(learner.actor_snapshot())

    recent_returns: deque[float] = deque(maxlen=100)
    recent_wins: deque[float] = deque(maxlen=100)
    training_started = time.perf_counter()
    total_environment_steps = 0
    last_checkpoint = checkpoint_dir / "final.pt"
    sampling_rng = np.random.default_rng(config.seed + 91_337)
    parallel_envs = ParallelUAVEnvs(
        config.training.parallel_envs,
        config.environment,
        config.reward,
        config.render,
        Path.cwd(),
    )
    try:
        print(
            f"[START] episodes={config.training.episodes} agents={n_agents}v{n_agents} "
            f"parallel={config.training.parallel_envs} pids={parallel_envs.worker_pids} "
            f"device={device} rule_warmup={config.training.rule_opponent_episodes} "
            f"log_every={config.training.log_interval}",
            flush=True,
        )
        episode_cursor = start_episode
        while episode_cursor <= config.training.episodes:
            batch_episodes = list(
                range(
                    episode_cursor,
                    min(
                        episode_cursor + config.training.parallel_envs,
                        config.training.episodes + 1,
                    ),
                )
            )
            indices = list(range(len(batch_episodes)))
            training_teams = [int((episode + config.seed) % 2) for episode in batch_episodes]
            use_rule = [
                episode <= config.training.rule_opponent_episodes
                for episode in batch_episodes
            ]
            if not all(use_rule):
                snapshot = snapshots[int(sampling_rng.integers(0, len(snapshots)))]
                opponent_policy.load_actor_snapshot(snapshot)

            sample_started = time.perf_counter()
            reset_results = parallel_envs.reset(
                indices,
                [config.seed + episode for episode in batch_episodes],
                training_teams,
            )
            observations = {index: reset_results[index][0] for index in indices}
            infos = {index: reset_results[index][1] for index in indices}
            actor_hidden = {
                index: learner.initial_actor_hidden() for index in indices
            }
            critic_hidden = {
                index: learner.initial_critic_hidden() for index in indices
            }
            opponent_hidden = {
                index: opponent_policy.initial_actor_hidden() for index in indices
            }
            buffers = {
                index: EpisodeBuffer(config.mappo.hidden_size, n_agents)
                for index in indices
            }
            episode_returns = {index: 0.0 for index in indices}
            active_indices = set(indices)

            while active_indices:
                step_indices = sorted(active_indices)
                learner_observations = np.stack(
                    [observations[index][training_teams[index]] for index in step_indices]
                )
                learner_states = np.stack(
                    [infos[index]["states"][training_teams[index]] for index in step_indices]
                )
                learner_masks = np.stack(
                    [infos[index]["active_masks"][training_teams[index]] for index in step_indices]
                )
                batched_actions, batched_log_probabilities, batched_values, batched_actor_h, batched_critic_h = learner.act_batch(
                    learner_observations,
                    learner_states,
                    np.stack([actor_hidden[index] for index in step_indices]),
                    np.stack([critic_hidden[index] for index in step_indices]),
                    learner_masks,
                )

                opponent_actions_by_index: dict[int, np.ndarray | None] = {
                    index: None for index in step_indices
                }
                selfplay_indices = [index for index in step_indices if not use_rule[index]]
                if selfplay_indices:
                    selfplay_observations = np.stack(
                        [observations[index][1 - training_teams[index]] for index in selfplay_indices]
                    )
                    selfplay_masks = np.stack(
                        [infos[index]["active_masks"][1 - training_teams[index]] for index in selfplay_indices]
                    )
                    selfplay_actions, next_selfplay_hidden = opponent_policy.act_policy_batch(
                        selfplay_observations,
                        np.stack([opponent_hidden[index] for index in selfplay_indices]),
                        selfplay_masks,
                        deterministic=False,
                    )
                    for position, index in enumerate(selfplay_indices):
                        opponent_actions_by_index[index] = selfplay_actions[position]
                        opponent_hidden[index] = next_selfplay_hidden[position]

                pending: dict[
                    int,
                    tuple[
                        np.ndarray,
                        np.ndarray,
                        float,
                        np.ndarray,
                        np.ndarray,
                        np.ndarray,
                        np.ndarray,
                    ],
                ] = {}
                for position, index in enumerate(step_indices):
                    team = training_teams[index]
                    pending[index] = (
                        learner_states[position],
                        batched_actions[position],
                        float(batched_values[position]),
                        batched_log_probabilities[position],
                        batched_actor_h[position],
                        batched_critic_h[position],
                        learner_masks[position],
                    )

                step_results = parallel_envs.step(
                    step_indices,
                    [batched_actions[position] for position in range(len(step_indices))],
                    [opponent_actions_by_index[index] for index in step_indices],
                )
                for index in step_indices:
                    next_observation, rewards, terminated, truncated, next_info = (
                        step_results[index]
                    )
                    team = training_teams[index]
                    (
                        state,
                        action,
                        value,
                        log_probability,
                        next_actor_h,
                        next_critic_h,
                        active_mask,
                    ) = pending[index]
                    team_reward = float(np.mean(rewards[team]))
                    done = terminated or truncated
                    buffers[index].append(
                        observation=observations[index][team],
                        state=state,
                        action=action,
                        log_probability=log_probability,
                        value=value,
                        reward=team_reward,
                        done=done,
                        active_mask=active_mask,
                    )
                    episode_returns[index] += team_reward
                    observations[index] = next_observation
                    infos[index] = next_info
                    actor_hidden[index] = next_actor_h
                    critic_hidden[index] = next_critic_h
                    if done:
                        active_indices.remove(index)

            sample_seconds = time.perf_counter() - sample_started
            update_started = time.perf_counter()
            update_metrics = learner.update_many(
                [buffers[index] for index in indices]
            )
            update_seconds = time.perf_counter() - update_started
            batch_steps = sum(len(buffers[index]) for index in indices)
            total_environment_steps += batch_steps

            for index, episode in enumerate(batch_episodes):
                team = training_teams[index]
                opponent_team = 1 - team
                info = infos[index]
                episode_return = episode_returns[index]
                win = float(info["winner"] == team)
                recent_returns.append(episode_return)
                recent_wins.append(win)
                writer.add_scalar("train/episode_return", episode_return, episode)
                writer.add_scalar("train/win", win, episode)
                writer.add_scalar("train/episode_length", len(buffers[index]), episode)
                writer.add_scalar("train/alive_own", info["alive"][team], episode)
                writer.add_scalar(
                    "train/alive_enemy", info["alive"][opponent_team], episode
                )
                should_log = (
                    episode == start_episode
                    or episode == config.training.episodes
                    or episode % config.training.log_interval == 0
                )
                if should_log:
                    elapsed = max(time.perf_counter() - training_started, 1e-6)
                    completed = episode - start_episode + 1
                    rate = completed / elapsed
                    remaining = max(config.training.episodes - episode, 0)
                    winner = info["winner"]
                    result = (
                        "D" if winner is None else ("W" if winner == team else "L")
                    )
                    print(
                        _progress_line(
                            episode=episode,
                            total_episodes=config.training.episodes,
                            stage="rule" if use_rule[index] else "selfplay",
                            result=result,
                            episode_return=episode_return,
                            mean_return=float(np.mean(recent_returns)),
                            win_rate=float(np.mean(recent_wins)),
                            length=len(buffers[index]),
                            alive=info["alive"],
                            kills=info["kills"],
                            steps_per_second=total_environment_steps / elapsed,
                            eta_seconds=remaining / max(rate, 1e-9),
                            sample_seconds=sample_seconds,
                            update_seconds=update_seconds,
                        ),
                        flush=True,
                    )

            batch_end = batch_episodes[-1]
            batch_start = batch_episodes[0]
            for key, value in update_metrics.items():
                writer.add_scalar(f"loss/{key}", value, batch_end)

            if (
                batch_end // config.training.snapshot_interval
                > (batch_start - 1) // config.training.snapshot_interval
            ):
                snapshots.append(learner.actor_snapshot())
            if (
                batch_end // config.training.checkpoint_interval
                > (batch_start - 1) // config.training.checkpoint_interval
            ):
                last_checkpoint = checkpoint_dir / f"episode_{batch_end:06d}.pt"
                learner.save(
                    last_checkpoint,
                    episode=batch_end,
                    extra={"mean_return_100": float(np.mean(recent_returns)), "device": device},
                )
            episode_cursor = batch_end + 1

        last_checkpoint = checkpoint_dir / "final.pt"
        learner.save(last_checkpoint, episode=config.training.episodes, extra={"device": device})
    finally:
        writer.close()
        parallel_envs.close()
    return last_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train recurrent MAPPO in the UAV combat environment")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.episodes is not None:
        config.training.episodes = args.episodes
    if args.output is not None:
        config.training.output_dir = args.output
    if args.device is not None:
        config.training.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    checkpoint = run_training(config, resume=args.resume)
    print(f"[DONE] checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
