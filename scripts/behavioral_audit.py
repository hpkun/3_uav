"""Read-only diagnostics around actual V2 collision/weapon transitions; no tuning."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import math
from pathlib import Path
import numpy as np

from sanity_check import pursuit_red_actions
from uav_mappo_2d.config import load_config
from uav_mappo_2d.env import UAVCombatEnv, wrap_angle
from uav_mappo_2d.opponents import RuleBasedOpponent


def audit(mode, episodes, seed_start):
    cfg = load_config('configs/base.yaml')
    env = UAVCombatEnv(cfg.environment, cfg.reward, cfg.render)
    counts = Counter()
    collisions, weapons = [], []
    for episode in range(episodes):
        seed = seed_start + episode
        env.reset(seed=seed)
        controllers = [RuleBasedOpponent(2), RuleBasedOpponent(2)]
        history = deque(maxlen=5)
        done = False
        while not done:
            before = env.positions.copy()
            old_alive = env.alive.copy()
            headings = env.headings.copy()
            priorities = {}
            for team in range(2):
                detected = env.team_detected_enemy_mask(team)
                for agent in np.flatnonzero(env.alive[team]):
                    if team == 0 and mode == 'pursuit-vs-rule':
                        priority = 'direct-pursuit'
                    elif controllers[team].boundary_point(env, team, agent) is not None:
                        priority = 'boundary'
                    elif controllers[team]._avoidance(env, team, agent, detected) is not None:
                        priority = 'avoidance'
                    else:
                        priority = 'tail-or-patrol'
                    priorities[(team, int(agent))] = priority
            red = controllers[0].act(env, 0) if mode == 'rule-vs-rule' else pursuit_red_actions(env)
            actions = np.stack([red, controllers[1].act(env, 1)])
            frame = {'step_before': env.step_count, 'positions': before.tolist(),
                     'headings_degrees': np.degrees(headings).tolist(), 'actions': actions.tolist()}
            history.append(frame)
            kills_before = env.combat_kills.copy()
            _, _, done, _, _ = env.step(actions)
            inside = old_alive & np.all(np.abs(env.positions) <= cfg.environment.map_size / 2, axis=-1)
            indices = np.flatnonzero(inside.ravel())
            p0, p1 = before.reshape(-1, 2), env.positions.reshape(-1, 2)
            collision_victims = set()
            for n, first in enumerate(indices):
                for second in indices[n+1:]:
                    r0 = p0[first] - p0[second]
                    dr = (p1[first]-p0[first]) - (p1[second]-p0[second])
                    tau = float(np.clip(-np.dot(r0, dr) / np.dot(dr, dr), 0, 1)) if np.dot(dr, dr) > 1e-12 else 0.
                    closest = float(np.linalg.norm(r0 + tau * dr))
                    if closest > cfg.environment.collision_distance:
                        continue
                    collision_victims.update([int(first), int(second)])
                    t1, i1 = divmod(int(first), 5)
                    t2, i2 = divmod(int(second), 5)
                    category = 'same_team' if t1 == t2 else 'cross_team'
                    counts['collision_pairs'] += 1
                    counts[category] += 1
                    angle = abs(math.degrees(float(wrap_angle(headings[t1,i1] - headings[t2,i2]))))
                    counts['opposed_heading_gt135'] += int(angle > 135)
                    counts['parallel_heading_lt45'] += int(angle < 45)
                    start_distance = float(np.linalg.norm(r0))
                    counts['start_distance_lt300'] += int(start_distance < 300)
                    avoided = sum(priorities[(t,i)] == 'avoidance' for t,i in [(t1,i1),(t2,i2)])
                    counts['both_avoiding'] += int(avoided == 2)
                    counts['one_avoiding'] += int(avoided == 1)
                    counts['neither_avoiding'] += int(avoided == 0)
                    if len(collisions) < 5:
                        collisions.append({'seed': seed, 'step': env.step_count,
                            'pair': [[t1,i1],[t2,i2]], 'category': category,
                            'heading_difference_degrees': angle, 'distance_before': start_distance,
                            'distance_after': float(np.linalg.norm(p1[first]-p1[second])),
                            'swept_distance': closest, 'tau': tau,
                            'priority': [priorities[(t1,i1)], priorities[(t2,i2)]],
                            'preceding_frames': list(history)})
            if np.any(env.combat_kills > kills_before):
                counts['weapon_kills'] += int(np.sum(env.combat_kills-kills_before))
                valid = inside.ravel().copy()
                valid[list(collision_victims)] = False
                for first in np.flatnonzero(valid):
                    t, i = divmod(int(first), 5)
                    for second in np.flatnonzero(valid):
                        ot, j = divmod(int(second), 5)
                        if t == ot:
                            continue
                        d, alpha, beta = env._compute_pair_geometry(t, i, ot, j)
                        if d <= 1000 and abs(alpha) <= math.radians(30)+1e-12 and abs(beta) <= math.radians(45)+1e-12 and len(weapons) < 5:
                            weapons.append({'seed': seed, 'step': env.step_count,
                                'attacker': [t,i], 'target': [ot,j], 'distance': d,
                                'ata_degrees': math.degrees(alpha), 'aspect_degrees': math.degrees(beta),
                                'preceding_frames': list(history)})
    env.close()
    return {'mode': mode, 'episodes': episodes, 'seed_start': seed_start,
            'counts': dict(counts), 'collision_examples': collisions, 'weapon_examples': weapons}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed-start', type=int, default=200000)
    parser.add_argument('--output', default='runs/v2_validation/behavioral_audit.json')
    args = parser.parse_args()
    results = {mode: audit(mode, args.episodes, args.seed_start)
               for mode in ['rule-vs-rule', 'pursuit-vs-rule']}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({mode: row['counts'] for mode, row in results.items()}, indent=2))
