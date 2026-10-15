from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .config import ProjectConfig, load_config
from .env import UAVCombatEnv
from .evaluate import evaluate_policy, resolve_device
from .mappo import MAPPO, RolloutBuffer
from .parallel_env import ParallelUAVEnvs


CSV_FIELDS = [
    "global_step",
    "episode",
    "return",
    "result",
    *UAVCombatEnv.EPISODE_METRICS,
    *UAVCombatEnv.REWARD_COMPONENTS,
    "episode_length",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean(rows: deque[dict[str, float | str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _training_log(
    global_step: int,
    episode_index: int,
    rows: deque[dict[str, float | str]],
    metrics: dict[str, float],
) -> str:
    count = max(len(rows), 1)
    wins = sum(row["result"] == "win" for row in rows) / count
    draws = sum(row["result"] == "draw" for row in rows) / count
    losses = sum(row["result"] == "loss" for row in rows) / count
    return (
        f"[TRAIN] step={global_step} episodes={episode_index} "
        f"avg_return={_mean(rows, 'return'):.3f} "
        f"W/D/L={wins:.1%}/{draws:.1%}/{losses:.1%} "
        f"alive(R/B)={_mean(rows, 'red_alive'):.2f}/{_mean(rows, 'blue_alive'):.2f} "
        f"combat_kill(R/B)={_mean(rows, 'red_combat_kill'):.2f}/"
        f"{_mean(rows, 'blue_combat_kill'):.2f} "
        f"weapon_loss(R/B)={_mean(rows, 'red_weapon_loss'):.2f}/{_mean(rows, 'blue_weapon_loss'):.2f} "
        f"oob(R/B)={_mean(rows, 'red_oob'):.2f}/{_mean(rows, 'blue_oob'):.2f} "
        f"collision_events(R/B)={_mean(rows, 'red_collision_events'):.2f}/{_mean(rows, 'blue_collision_events'):.2f} "
        f"window(R/B)={_mean(rows, 'red_attack_window'):.2f}/{_mean(rows, 'blue_attack_window'):.2f} "
        f"contact(R/B)={_mean(rows, 'red_radar_contact_steps'):.1f}/{_mean(rows, 'blue_radar_contact_steps'):.1f} "
        f"reward(kill/loss/collision/oob/terminal)="
        f"{_mean(rows, 'kill_reward'):.2f}/{_mean(rows, 'weapon_loss_reward'):.2f}/"
        f"{_mean(rows, 'collision_reward'):.2f}/{_mean(rows, 'oob_reward'):.2f}/"
        f"{_mean(rows, 'terminal_reward'):.2f} "
        f"length={_mean(rows, 'episode_length'):.1f} "
        f"actor={metrics['actor_loss']:.4f} critic={metrics['critic_loss']:.4f} "
        f"entropy={metrics['entropy']:.4f} kl={metrics['approx_kl']:.5f} "
        f"clip={metrics['clip_fraction']:.3f} "
        f"grad(A/C)={metrics['actor_grad_norm']:.3f}/{metrics['critic_grad_norm']:.3f} "
        f"expl_var={metrics['explained_variance']:.3f}"
    )


def _write_episode(
    writer: csv.DictWriter,
    stream: object,
    summary: dict[str, float | str],
    global_step: int,
    episode_index: int,
) -> None:
    writer.writerow(
        {
            "global_step": global_step,
            "episode": episode_index,
            **{key: summary[key] for key in CSV_FIELDS[2:]},
        }
    )
    stream.flush()


def _is_better_evaluation(
    candidate: dict[str, float], current: dict[str, float] | None
) -> bool:
    if current is None:
        return True
    return (
        candidate["win_rate"],
        candidate["survival_difference"],
    ) > (
        current["win_rate"],
        current["survival_difference"],
    )


def run_training(config: ProjectConfig, *, resume: str | None = None, init_checkpoint: str | None = None) -> Path:
    if resume and init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    set_seed(config.seed)
    training = config.training
    if training.total_env_steps % training.parallel_envs:
        raise ValueError("total_env_steps must be divisible by parallel_envs")
    device = resolve_device(training.device)
    output_dir = Path(training.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard = SummaryWriter(output_dir / "tensorboard")
    learner = MAPPO(
        UAVCombatEnv.OBS_DIM, UAVCombatEnv.STATE_DIM, UAVCombatEnv.N_AGENTS,
        config.mappo, device, seed=config.seed,
    )
    global_step = 0
    episode_index = 0
    best_evaluation: dict[str, float] | None = None
    if init_checkpoint:
        learner.load(init_checkpoint, load_optimizers=False)
    if resume:
        checkpoint = learner.load(resume)
        global_step = int(checkpoint.get("global_step", 0))
        extra = checkpoint.get("extra", {})
        episode_index = int(extra.get("episode_index", 0))
        saved_best = extra.get("best_evaluation")
        best_evaluation = dict(saved_best) if saved_best is not None else None

    recent_episodes: deque[dict[str, float | str]] = deque(maxlen=100)
    last_metrics = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "actor_grad_norm": 0.0,
        "critic_grad_norm": 0.0,
        "explained_variance": 0.0,
    }
    next_log = (
        global_step // training.log_interval_steps + 1
    ) * training.log_interval_steps
    next_checkpoint = (
        global_step // training.checkpoint_interval_steps + 1
    ) * training.checkpoint_interval_steps
    next_evaluation = (
        global_step // training.evaluation_interval_steps + 1
    ) * training.evaluation_interval_steps
    episode_csv = output_dir / "episodes.csv"
    csv_exists = episode_csv.exists() and episode_csv.stat().st_size > 0
    csv_stream = episode_csv.open("a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS)
    if not csv_exists:
        csv_writer.writeheader()

    update_csv = output_dir / "updates.csv"
    update_exists = update_csv.exists() and update_csv.stat().st_size > 0
    update_stream = update_csv.open("a", newline="", encoding="utf-8")
    summary_keys = [key for key in CSV_FIELDS[2:] if key != "result"]
    update_writer = csv.DictWriter(update_stream, fieldnames=[
        "global_step", "episodes", "win_rate", "draw_rate", "loss_rate",
        *[f"mean_{key}" for key in summary_keys], *last_metrics,
    ])
    if not update_exists:
        update_writer.writeheader()

    environments = ParallelUAVEnvs(
        training.parallel_envs,
        config.environment,
        config.reward,
        config.render,
        Path.cwd(),
        base_seed=config.seed + global_step,
    )
    observations, states, active_masks, _ = environments.reset()
    print(
        f"[START] total_env_steps={training.total_env_steps} "
        f"rollout={training.rollout_length} parallel={training.parallel_envs} "
        f"device={device} pids={environments.worker_pids}",
        flush=True,
    )
    try:
        while global_step < training.total_env_steps:
            remaining_vector_steps = (
                training.total_env_steps - global_step
            ) // training.parallel_envs
            horizon = min(training.rollout_length, remaining_vector_steps)
            buffer = RolloutBuffer(
                horizon, training.parallel_envs, UAVCombatEnv.N_AGENTS,
                UAVCombatEnv.OBS_DIM, UAVCombatEnv.STATE_DIM,
            )
            for _ in range(horizon):
                actions, log_probabilities, values = learner.act(
                    observations, states, active_masks
                )
                result = environments.step(actions)
                buffer.insert(
                    observations,
                    states,
                    actions,
                    log_probabilities,
                    values,
                    result.rewards,
                    result.dones,
                    active_masks,
                )
                global_step += training.parallel_envs
                for info in result.infos:
                    if "episode_summary" not in info:
                        continue
                    episode_index += 1
                    summary = dict(info["episode_summary"])
                    recent_episodes.append(summary)
                    _write_episode(
                        csv_writer,
                        csv_stream,
                        summary,
                        global_step,
                        episode_index,
                    )
                    for key in CSV_FIELDS[2:]:
                        if key == "result":
                            continue
                        tensorboard.add_scalar(
                            f"episode/{key}", float(summary[key]), global_step
                        )
                    tensorboard.add_scalar(
                        "episode/win", float(summary["result"] == "win"), global_step
                    )
                    tensorboard.add_scalar(
                        "episode/draw", float(summary["result"] == "draw"), global_step
                    )
                    tensorboard.add_scalar(
                        "episode/loss", float(summary["result"] == "loss"), global_step
                    )
                observations = result.observations
                states = result.states
                active_masks = result.active_masks

            last_values = learner.values(states)
            buffer.compute_returns_and_advantages(
                last_values, config.mappo.gamma, config.mappo.gae_lambda
            )
            last_metrics = learner.update(buffer)
            for key, value in last_metrics.items():
                tensorboard.add_scalar(f"update/{key}", value, global_step)
            update_writer.writerow({
                "global_step": global_step, "episodes": episode_index,
                **{f"{label}_rate": sum(row["result"] == label for row in recent_episodes)
                   / max(len(recent_episodes), 1) for label in ("win", "draw", "loss")},
                **{f"mean_{key}": _mean(recent_episodes, key) for key in summary_keys},
                **last_metrics,
            })
            update_stream.flush()

            if global_step >= next_log:
                print(
                    _training_log(
                        global_step, episode_index, recent_episodes, last_metrics
                    ),
                    flush=True,
                )
                while next_log <= global_step:
                    next_log += training.log_interval_steps

            if global_step >= next_evaluation:
                evaluation = evaluate_policy(
                    config,
                    learner,
                    episodes=training.evaluation_episodes,
                    seed_start=config.seed + 1_000_000,
                )
                for key, value in evaluation.items():
                    tensorboard.add_scalar(f"evaluation/{key}", value, global_step)
                print(
                    f"[EVAL] step={global_step} W/D/L="
                    f"{evaluation['win_rate']:.1%}/{evaluation['draw_rate']:.1%}/"
                    f"{evaluation['loss_rate']:.1%} return={evaluation['mean_return']:.3f} "
                    f"alive_diff={evaluation['survival_difference']:.3f}",
                    flush=True,
                )
                if _is_better_evaluation(evaluation, best_evaluation):
                    best_evaluation = evaluation
                    learner.save(
                        checkpoint_dir / "best.pt",
                        global_step=global_step,
                        extra={
                            "episode_index": episode_index,
                            "best_evaluation": best_evaluation,
                        },
                    )
                while next_evaluation <= global_step:
                    next_evaluation += training.evaluation_interval_steps

            if global_step >= next_checkpoint:
                learner.save(
                    checkpoint_dir / f"step_{global_step:09d}.pt",
                    global_step=global_step,
                    extra={
                        "episode_index": episode_index,
                        "best_evaluation": best_evaluation,
                    },
                )
                while next_checkpoint <= global_step:
                    next_checkpoint += training.checkpoint_interval_steps

        final_path = checkpoint_dir / "final.pt"
        learner.save(
            final_path,
            global_step=global_step,
            extra={
                "episode_index": episode_index,
                "best_evaluation": best_evaluation,
            },
        )
        return final_path
    finally:
        csv_stream.close()
        update_stream.close()
        tensorboard.close()
        environments.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train continuous MAPPO in Fixed 5v5 2D UAV Combat Baseline V2.1"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", default=None)
    initialization.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.steps is not None:
        config.training.total_env_steps = args.steps
    if args.output is not None:
        config.training.output_dir = args.output
    if args.device is not None:
        config.training.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    checkpoint = run_training(config, resume=args.resume, init_checkpoint=args.init_checkpoint)
    print(f"[DONE] checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
