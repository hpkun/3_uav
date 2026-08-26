from __future__ import annotations

import argparse
import json

import numpy as np

from uav_mappo_2d.config import load_config
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.opponents import RuleBasedOpponent


def run_rule_vs_rule(config_path: str, episodes: int, seed_start: int) -> dict[str, int]:
    config = load_config(config_path)
    env = UAVCombatEnv(config.environment, config.reward, config.render)
    red_rule = RuleBasedOpponent()
    blue_rule = RuleBasedOpponent()
    counts = {"red_win": 0, "blue_win": 0, "draw": 0}
    for episode in range(episodes):
        env.reset(seed=seed_start + episode)
        red_rule.reset()
        blue_rule.reset()
        done = False
        while not done:
            _, _, done, _, info = env.step(
                np.stack([red_rule.act(env, 0), blue_rule.act(env, 1)])
            )
        key = "draw" if info["winner"] is None else (
            "red_win" if info["winner"] == 0 else "blue_win"
        )
        counts[key] += 1
    env.close()
    return counts


def run_random_vs_rule(
    config_path: str, episodes: int, seed_start: int
) -> dict[str, int]:
    config = load_config(config_path)
    env = UAVCombatEnv(config.environment, config.reward, config.render)
    blue_rule = RuleBasedOpponent()
    rng = np.random.default_rng(seed_start + 9_999_991)
    counts = {"red_win": 0, "blue_win": 0, "draw": 0}
    for episode in range(episodes):
        env.reset(seed=seed_start + episode)
        blue_rule.reset()
        done = False
        while not done:
            red_actions = rng.uniform(-1.0, 1.0, size=(3, 2)).astype(np.float32)
            _, _, done, _, info = env.step(
                np.stack([red_actions, blue_rule.act(env, 1)])
            )
        key = "draw" if info["winner"] is None else (
            "red_win" if info["winner"] == 0 else "blue_win"
        )
        counts[key] += 1
    env.close()
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed 3v3 environment sanity checks")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=200_000)
    parser.add_argument(
        "--mode", choices=("both", "rule-vs-rule", "random-vs-rule"), default="both"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = {}
    if args.mode in ("both", "rule-vs-rule"):
        results["rule_vs_rule"] = run_rule_vs_rule(
            args.config, args.episodes, args.seed_start
        )
    if args.mode in ("both", "random-vs-rule"):
        results["random_red_vs_rule_blue"] = run_random_vs_rule(
            args.config, args.episodes, args.seed_start
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
