from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from uav_mappo_2d.config import load_config
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.opponents import RuleBasedOpponent


MODES = ("greedy-vs-greedy", "random-vs-greedy", "passive-vs-greedy", "direct-pursuit-vs-greedy")


def pursuit_red_actions(env):
    """Nearest detected direct pursuit, no escape or acceleration; nominal boundary return."""
    actions = np.zeros((env.N_AGENTS, env.ACTION_DIM), dtype=np.float32)
    detected = env.team_detected_enemy_mask(0)
    for agent in np.flatnonzero(env.alive[0]):
        heading = RuleBasedOpponent.boundary_heading(env, 0, agent)
        if heading is None:
            target = RuleBasedOpponent.nearest_target(env, 0, agent, detected)
            if target is None:
                continue
            delta = env.positions[1, target] - env.positions[0, agent]
            heading = np.arctan2(delta[1], delta[0])
        actions[agent] = RuleBasedOpponent.control(env, 0, agent, heading)
    return actions


def summarize(rows):
    """Report actual kills; attack windows are geometric pairs, never a hit-rate denominator."""
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
        oob = sum(sum(int(row[f'{t}_oob']) for t in teams) for row in rows)
        total = weapon + oob
        audit[label] = {'death_total': total, 'weapon_losses': weapon, 'oob': oob,
                        'weapon_death_share': weapon / total if total else 0.,
                        'oob_share': oob / total if total else 0.}
    result['death_audit'] = audit
    result['collision_episode_rate'] = {
        label: sum(any(row[f'{team}_collision_events'] > 0 for team in teams) for row in rows) / episodes
        for label, teams in [('red', ['red']), ('blue', ['blue']), ('combined', ['red', 'blue'])]
    }
    result['weapon_combat_episode_rate'] = sum(
        row['red_combat_kill'] + row['blue_combat_kill'] > 0 for row in rows
    ) / episodes
    result['timeout_episode_rate'] = sum(row['timed_out'] for row in rows) / episodes
    return result


def run_scenario(config_path, mode, episodes, seed_start):
    cfg = load_config(config_path)
    env = UAVCombatEnv(cfg.environment, cfg.reward, cfg.render)
    red_rule, blue_rule = RuleBasedOpponent(), RuleBasedOpponent()
    rng = np.random.default_rng(seed_start + 9999991)
    rows = []
    helper_max_difference = 0.
    helper_exact_mismatch_steps = 0
    for episode in range(episodes):
        env.reset(seed=seed_start + episode)
        done = False
        while not done:
            if mode == 'greedy-vs-greedy':
                red = red_rule.act(env, 0)
            elif mode == 'random-vs-greedy':
                red = rng.uniform(-1, 1, (env.N_AGENTS, env.ACTION_DIM)).astype(np.float32)
            elif mode == 'passive-vs-greedy':
                red = np.zeros((env.N_AGENTS, env.ACTION_DIM), dtype=np.float32)
            elif mode == 'direct-pursuit-vs-greedy':
                red = pursuit_red_actions(env)
            else:
                raise ValueError(f'Unknown sanity mode: {mode}')
            if mode in ('greedy-vs-greedy','direct-pursuit-vs-greedy'):
                helper,formal = pursuit_red_actions(env),red_rule.act(env,0)
                helper_max_difference = max(helper_max_difference,float(np.max(np.abs(helper-formal))))
                helper_exact_mismatch_steps += int(not np.array_equal(helper,formal))
            _, _, done, _, info = env.step(np.stack([red, blue_rule.act(env, 1)]))
        rows.append({**info['episode_summary'], 'timed_out':env.step_count>=env.config.episode_steps})
    env.close()
    return {'mode': mode, 'seed_start': seed_start, **summarize(rows),
            'helper_max_action_difference':helper_max_difference,
            'helper_exact_mismatch_steps':helper_exact_mismatch_steps}


def main():
    parser = argparse.ArgumentParser(description='V2.5 Greedy Shooter behavioral sanity checks')
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
