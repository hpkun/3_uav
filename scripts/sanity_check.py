from __future__ import annotations

import argparse
import json
import math

import numpy as np

from uav_mappo_2d.config import load_config
from uav_mappo_2d.env import UAVCombatEnv, wrap_angle
from uav_mappo_2d.opponents import RuleBasedOpponent


def _flying_outward(env: UAVCombatEnv, team: int, agent: int) -> bool:
    half = env.config.map_size / 2.0
    margin = 500.0
    x, y = env.positions[team, agent]
    vx = math.cos(env.headings[team, agent])
    vy = math.sin(env.headings[team, agent])
    return bool(
        (x >= half - margin and vx > 0.0)
        or (x <= -half + margin and vx < 0.0)
        or (y >= half - margin and vy > 0.0)
        or (y <= -half + margin and vy < 0.0)
    )


def _turn_toward(env: UAVCombatEnv, team: int, agent: int, point: np.ndarray) -> float:
    delta = point - env.positions[team, agent]
    desired_heading = math.atan2(float(delta[1]), float(delta[0]))
    heading_error = float(wrap_angle(desired_heading - env.headings[team, agent]))
    return float(
        np.clip(
            heading_error / (env.config.max_turn_rate * env.config.dt),
            -1.0,
            1.0,
        )
    )


def _passive_red_actions(env: UAVCombatEnv) -> np.ndarray:
    actions = np.zeros((3, 2), dtype=np.float32)
    for agent in np.flatnonzero(env.alive[0]):
        agent = int(agent)
        if _flying_outward(env, 0, agent):
            actions[agent, 1] = _turn_toward(env, 0, agent, np.zeros(2))
        actions[agent, 0] = np.clip(
            (200.0 - env.speeds[0, agent])
            / (env.config.max_acceleration * env.config.dt),
            -1.0,
            1.0,
        )
    return actions


def _pursuit_red_actions(env: UAVCombatEnv) -> np.ndarray:
    actions = np.zeros((3, 2), dtype=np.float32)
    enemies = [int(index) for index in np.flatnonzero(env.alive[1])]
    if not enemies:
        return actions
    for agent in np.flatnonzero(env.alive[0]):
        agent = int(agent)
        if _flying_outward(env, 0, agent):
            target_point = np.zeros(2)
        else:
            target = min(
                enemies,
                key=lambda index: (
                    float(
                        np.linalg.norm(
                            env.positions[1, index] - env.positions[0, agent]
                        )
                    ),
                    index,
                ),
            )
            target_point = env.positions[1, target]
        actions[agent, 1] = _turn_toward(env, 0, agent, target_point)
        actions[agent, 0] = np.clip(
            (env.config.speed_max - env.speeds[0, agent])
            / (env.config.max_acceleration * env.config.dt),
            -1.0,
            1.0,
        )
    return actions


def _summarize(rows: list[dict[str, float | str]], mode: str) -> dict[str, object]:
    episodes = len(rows)
    red_wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    blue_wins = sum(row["result"] == "loss" for row in rows)

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    result: dict[str, object] = {
        "episodes": episodes,
        "red_wins": red_wins,
        "draws": draws,
        "blue_wins": blue_wins,
        "red_win_rate": red_wins / episodes,
        "draw_rate": draws / episodes,
        "blue_win_rate": blue_wins / episodes,
        "mean_red_alive": mean("red_alive"),
        "mean_blue_alive": mean("blue_alive"),
        "mean_red_combat_kills": mean("red_combat_kill"),
        "mean_blue_combat_kills": mean("blue_combat_kill"),
        "mean_red_oob": mean("red_oob"),
        "mean_blue_oob": mean("blue_oob"),
        "mean_episode_length": mean("episode_length"),
        "mean_red_return": mean("return"),
    }
    if mode == "rule-vs-rule":
        result["diagnostics"] = {
            "symmetry_gap": abs(result["red_win_rate"] - result["blue_win_rate"]),
            "formula": "abs(red_win_rate - blue_win_rate)",
        }
    else:
        blue_kills = float(result["mean_blue_combat_kills"])
        red_oob = float(result["mean_red_oob"])
        result["diagnostics"] = {
            "blue_combat_share": blue_kills / max(blue_kills + red_oob, 1e-12),
            "formula": "mean_blue_combat_kills / (mean_blue_combat_kills + mean_red_oob)",
        }
    return result


def run_scenario(
    config_path: str, mode: str, episodes: int, seed_start: int
) -> dict[str, object]:
    config = load_config(config_path)
    env = UAVCombatEnv(config.environment, config.reward, config.render)
    red_rule = RuleBasedOpponent()
    blue_rule = RuleBasedOpponent()
    rng = np.random.default_rng(seed_start + 9_999_991)
    rows: list[dict[str, float | str]] = []
    for episode in range(episodes):
        env.reset(seed=seed_start + episode)
        red_rule.reset()
        blue_rule.reset()
        done = False
        while not done:
            if mode == "rule-vs-rule":
                red_actions = red_rule.act(env, 0)
            elif mode == "random-vs-rule":
                red_actions = rng.uniform(-1.0, 1.0, size=(3, 2)).astype(np.float32)
            elif mode == "passive-vs-rule":
                red_actions = _passive_red_actions(env)
            elif mode == "pursuit-vs-rule":
                red_actions = _pursuit_red_actions(env)
            else:
                raise ValueError(f"Unknown sanity mode: {mode}")
            _, _, done, _, info = env.step(
                np.stack([red_actions, blue_rule.act(env, 1)])
            )
        rows.append(info["episode_summary"])
    env.close()
    return _summarize(rows, mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed 3v3 environment sanity checks")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=200_000)
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "rule-vs-rule",
            "random-vs-rule",
            "passive-vs-rule",
            "pursuit-vs-rule",
        ),
        default="all",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    modes = (
        [
            "rule-vs-rule",
            "random-vs-rule",
            "passive-vs-rule",
            "pursuit-vs-rule",
        ]
        if args.mode == "all"
        else [args.mode]
    )
    results = {
        mode.replace("-", "_"): run_scenario(
            args.config, mode, args.episodes, args.seed_start
        )
        for mode in modes
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
