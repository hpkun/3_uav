from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from uav_mappo_2d.config import load_config
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.opponents import RuleBasedOpponent


def run_scenario(config_path: str, strategy: int, episodes: int, seed_start: int) -> dict:
    config = load_config(config_path)
    env = UAVCombatEnv(config.environment, config.reward, config.render)
    rules = [RuleBasedOpponent(strategy), RuleBasedOpponent(strategy)]
    rows = []
    buffer_agents = 0
    buffer_actions_valid = 0
    for episode in range(episodes):
        env.reset(seed=seed_start + episode)
        for rule in rules:
            rule.reset()
        done = False
        while not done:
            actions = np.stack([rules[0].act(env, 0), rules[1].act(env, 1)])
            for team in range(2):
                masks = env.available_action_masks(team)
                buffer = env.alive[team] & ~masks.all(axis=-1)
                buffer_agents += int(buffer.sum())
                buffer_actions_valid += int(masks[np.arange(env.N_AGENTS), actions[team]][buffer].sum())
            _, _, done, _, info = env.step(actions)
        rows.append(info["episode_summary"])
    env.close()
    wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    losses = episodes - wins - draws
    keys = ["red_alive", "blue_alive", "red_combat_kill", "blue_combat_kill",
            "red_collision", "blue_collision", "red_outside_map", "blue_outside_map",
            "red_ever_outside", "blue_ever_outside", "episode_length", "return",
            *UAVCombatEnv.REWARD_COMPONENTS]
    return {
        "strategy": strategy, "episodes": episodes, "seed_start": seed_start,
        "red_wins": wins, "draws": draws, "blue_wins": losses,
        "red_win_rate": wins / episodes, "draw_rate": draws / episodes, "blue_win_rate": losses / episodes,
        **{f"mean_{key}": float(np.mean([row[key] for row in rows])) for key in keys},
        "buffer_agent_steps": buffer_agents, "valid_buffer_agent_steps": buffer_actions_valid,
        "symmetry_gap": abs(wins - losses) / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Xu Strategy1/2/3 symmetric rule sanity checks")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=200000)
    parser.add_argument("--mode", choices=("all", "strategy1", "strategy2", "strategy3"), default="all")
    parser.add_argument("--output", default=None, help="Optional JSON result file")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    strategies = [1, 2, 3] if args.mode == "all" else [int(args.mode[-1])]
    results = {f"strategy{s}_vs_strategy{s}": run_scenario(args.config, s, args.episodes, args.seed_start) for s in strategies}
    payload = json.dumps(results, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
