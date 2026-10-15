from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import ProjectConfig, load_config
from .env import UAVCombatEnv
from .mappo import MAPPO
from .opponents import RuleBasedOpponent


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _save_gif(frames: list[np.ndarray], path: Path, dt: float) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(20, int(dt * 1000)),
        loop=0,
        optimize=False,
    )


def evaluate_policy(
    config: ProjectConfig,
    policy: MAPPO,
    *,
    episodes: int,
    seed_start: int,
    gif_path: str | None = None,
) -> dict[str, float]:
    env = UAVCombatEnv(
        config.environment,
        config.reward,
        config.render,
        render_mode="rgb_array" if gif_path else None,
        project_root=Path.cwd(),
    )
    opponent = RuleBasedOpponent()
    rows: list[dict[str, float | str]] = []
    frames: list[np.ndarray] = []
    for episode in range(episodes):
        observations, info = env.reset(seed=seed_start + episode)
        opponent.reset()
        done = False
        if episode == 0 and gif_path:
            frames.append(env.render())
        while not done:
            red_actions = policy.policy_actions(
                observations[0][None],
                info["active_masks"][0][None],
                deterministic=True,
            )[0]
            blue_actions = opponent.act(env, 1)
            observations, _, done, _, info = env.step(
                np.stack([red_actions, blue_actions])
            )
            if episode == 0 and gif_path:
                frames.append(env.render())
        rows.append(info["episode_summary"])
    env.close()
    if gif_path:
        _save_gif(frames, Path(gif_path), config.environment.dt)

    result_values = [str(row["result"]) for row in rows]
    red_alive = np.asarray([row["red_alive"] for row in rows], dtype=np.float64)
    blue_alive = np.asarray([row["blue_alive"] for row in rows], dtype=np.float64)
    return {
        "episodes": float(episodes),
        "wins": float(result_values.count("win")),
        "draws": float(result_values.count("draw")),
        "losses": float(result_values.count("loss")),
        "win_rate": float(result_values.count("win") / episodes),
        "draw_rate": float(result_values.count("draw") / episodes),
        "loss_rate": float(result_values.count("loss") / episodes),
        "red_alive": float(red_alive.mean()),
        "blue_alive": float(blue_alive.mean()),
        "survival_difference": float((red_alive - blue_alive).mean()),
        "red_combat_kill": float(
            np.mean([row["red_combat_kill"] for row in rows])
        ),
        "blue_combat_kill": float(
            np.mean([row["blue_combat_kill"] for row in rows])
        ),
        "red_oob": float(np.mean([row["red_oob"] for row in rows])),
        "blue_oob": float(np.mean([row["blue_oob"] for row in rows])),
        "red_collision": float(np.mean([row["red_collision"] for row in rows])),
        "blue_collision": float(np.mean([row["blue_collision"] for row in rows])),
        "mean_episode_length": float(
            np.mean([row["episode_length"] for row in rows])
        ),
        "mean_return": float(np.mean([row["return"] for row in rows])),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed 5v5 deterministic red MAPPO against radar-constrained blue"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gif", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config.training.device)
    policy = MAPPO(
        UAVCombatEnv.OBS_DIM, UAVCombatEnv.STATE_DIM, UAVCombatEnv.N_AGENTS,
        config.mappo, device, seed=config.seed,
    )
    policy.load(args.checkpoint, load_optimizers=False)
    policy.actor.eval()
    policy.critic.eval()
    metrics = evaluate_policy(
        config,
        policy,
        episodes=args.episodes,
        seed_start=args.seed_start,
        gif_path=args.gif,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
