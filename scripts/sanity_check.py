from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from uav_mappo_2d.config import load_config
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.opponents import RuleBasedOpponent


MODES = ("rule-vs-rule", "random-vs-rule", "passive-vs-rule", "pursuit-vs-rule")


def pursuit_red_actions(env):
    """Nearest detected direct pursuit at 120m/s; boundary recovery only."""
    actions = np.zeros((env.N_AGENTS, env.ACTION_DIM), dtype=np.float32)
    enemies = np.flatnonzero(env.team_detected_enemy_mask(0))
    for agent in np.flatnonzero(env.alive[0]):
        point = RuleBasedOpponent.boundary_point(env, 0, agent)
        speed = 100. if point is not None else 120.
        if point is None and len(enemies):
            target = min(enemies, key=lambda j: (float(np.linalg.norm(env.positions[1,j] - env.positions[0,agent])), j))
            point = env.positions[1,target]
        heading = (env.headings[0,agent] if point is None
                   else RuleBasedOpponent._heading_to(env, 0, agent, point))
        actions[agent] = RuleBasedOpponent.control(env, 0, agent, heading, speed)
    return actions


def summarize(rows):
    episodes = len(rows)
    wins = sum(row['result'] == 'win' for row in rows)
    draws = sum(row['result'] == 'draw' for row in rows)
    losses = episodes - wins - draws
    keys = [*UAVCombatEnv.EPISODE_METRICS, 'episode_length', 'return', *UAVCombatEnv.REWARD_COMPONENTS]
    result = {
        'episodes': episodes, 'red_wins': wins, 'draws': draws, 'blue_wins': losses,
        'red_win_rate': wins / episodes, 'draw_rate': draws / episodes, 'blue_win_rate': losses / episodes,
        **{f'mean_{key}': float(np.mean([row[key] for row in rows])) for key in keys},
    }
    audit = {}
    for label, teams in [('red', ['red']), ('blue', ['blue']), ('combined', ['red', 'blue'])]:
        weapon = sum(sum(int(row[f'{t}_weapon_loss']) for t in teams) for row in rows)
        collision = sum(sum(int(row[f'{t}_collision']) for t in teams) for row in rows)
        oob = sum(sum(int(row[f'{t}_oob']) for t in teams) for row in rows)
        total = weapon + collision + oob
        audit[label] = {'death_total': total, 'weapon_losses': weapon, 'collision': collision, 'oob': oob,
                        'weapon_death_share': weapon / total if total else 0.,
                        'collision_share': collision / total if total else 0.,
                        'oob_share': oob / total if total else 0.}
    result['death_audit'] = audit
    return result


def run_scenario(config_path, mode, episodes, seed_start):
    cfg = load_config(config_path)
    env = UAVCombatEnv(cfg.environment, cfg.reward, cfg.render)
    red_rule, blue_rule = RuleBasedOpponent(2), RuleBasedOpponent(2)
    rng = np.random.default_rng(seed_start + 9999991)
    rows = []
    for episode in range(episodes):
        env.reset(seed=seed_start + episode)
        red_rule.reset()
        blue_rule.reset()
        done = False
        while not done:
            if mode == 'rule-vs-rule':
                red = red_rule.act(env, 0)
            elif mode == 'random-vs-rule':
                red = rng.uniform(-1, 1, (env.N_AGENTS, env.ACTION_DIM)).astype(np.float32)
            elif mode == 'passive-vs-rule':
                red = np.zeros((env.N_AGENTS, env.ACTION_DIM), dtype=np.float32)
            elif mode == 'pursuit-vs-rule':
                red = pursuit_red_actions(env)
            else:
                raise ValueError(f'Unknown sanity mode: {mode}')
            _, _, done, _, info = env.step(np.stack([red, blue_rule.act(env, 1)]))
        rows.append(info['episode_summary'])
    env.close()
    return {'mode': mode, 'seed_start': seed_start, **summarize(rows)}


def main():
    parser = argparse.ArgumentParser(description='Continuous 5v5 V2 behavioral sanity checks')
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--mode', choices=('all', *MODES), default='all')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=200000)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    modes = MODES if args.mode == 'all' else [args.mode]
    results = {m.replace('-', '_'): run_scenario(args.config, m, args.episodes, args.seed_start) for m in modes}
    payload = json.dumps(results, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + '\n', encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
