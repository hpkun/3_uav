from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import load_config
from .env import UAVCombatEnv
from .mappo import MAPPO
from .opponents import RuleBasedOpponent
from .train import resolve_device, set_seed


def save_gif(frames: list[np.ndarray], path: Path, decision_dt: float) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(20, int(decision_dt * 1000)),
        loop=0,
        optimize=False,
    )


def evaluate(
    config_path: str,
    checkpoint_path: str | None,
    *,
    episodes: int,
    opponent: str,
    gif_path: str | None,
    seed: int | None,
) -> dict[str, float | None]:
    config = load_config(config_path)
    if seed is not None:
        config.seed = seed
    set_seed(config.seed)
    device = resolve_device(config.training.device)
    env = UAVCombatEnv(
        config.environment,
        config.reward,
        config.render,
        render_mode="rgb_array" if gif_path else None,
        project_root=Path.cwd(),
    )
    policy = MAPPO(env.obs_dim, env.state_dim, env.n_agents, config.mappo, device)
    if checkpoint_path:
        policy.load(checkpoint_path, load_optimizer=False)
    policy.actor.eval()
    rule = RuleBasedOpponent()
    wins = draws = 0
    returns: list[float] = []
    lengths: list[int] = []
    surviving: list[float] = []
    red_kills: list[int] = []
    blue_kills: list[int] = []
    first_kill_times: list[float] = []
    frames: list[np.ndarray] = []

    with torch.no_grad():
        for episode in range(episodes):
            observation, info = env.reset(seed=config.seed + 10_000 + episode)
            red_hidden = policy.initial_actor_hidden()
            blue_hidden = policy.initial_actor_hidden()
            episode_return = 0.0
            terminated = truncated = False
            if episode == 0 and gif_path:
                frames.append(env.render())
            while not (terminated or truncated):
                red_action, red_hidden = policy.act_policy(
                    observation[0],
                    red_hidden,
                    env.alive[0].astype(np.float32),
                    deterministic=True,
                )
                if opponent == "rule":
                    blue_action = rule.act(env, 1)
                else:
                    blue_action, blue_hidden = policy.act_policy(
                        observation[1],
                        blue_hidden,
                        env.alive[1].astype(np.float32),
                        deterministic=True,
                    )
                actions = np.stack([red_action, blue_action])
                observation, reward, terminated, truncated, info = env.step(actions)
                episode_return += float(np.mean(reward[0]))
                if episode == 0 and gif_path:
                    frames.append(env.render())
            wins += int(info["winner"] == 0)
            draws += int(info["winner"] is None)
            returns.append(episode_return)
            lengths.append(info["step"])
            surviving.append(info["alive"][0] / env.n_agents)
            red_kills.append(info["kills"][0])
            blue_kills.append(info["kills"][1])
            if info["first_kill_time"] is not None:
                first_kill_times.append(float(info["first_kill_time"]))

    if gif_path:
        save_gif(frames, Path(gif_path), config.environment.decision_dt)
    env.close()
    return {
        "episodes": float(episodes),
        "red_win_rate": wins / episodes,
        "draw_rate": draws / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_survival_fraction": float(np.mean(surviving)),
        "mean_red_kills": float(np.mean(red_kills)),
        "mean_blue_kills": float(np.mean(blue_kills)),
        "mean_kill_difference": float(np.mean(np.asarray(red_kills) - np.asarray(blue_kills))),
        "episodes_with_kill_fraction": len(first_kill_times) / episodes,
        "mean_first_kill_time": (
            float(np.mean(first_kill_times)) if first_kill_times else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a MAPPO UAV policy")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--opponent", choices=("rule", "self"), default="rule")
    parser.add_argument("--gif", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate(
        args.config,
        args.checkpoint,
        episodes=args.episodes,
        opponent=args.opponent,
        gif_path=args.gif,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
