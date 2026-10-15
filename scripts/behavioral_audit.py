"""Observe actual V2.1 chase/escape transitions, weapons and OOB without tuning."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import math
from pathlib import Path
import numpy as np

from sanity_check import pursuit_red_actions
from uav_mappo_2d.config import load_config
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.opponents import RuleBasedOpponent


def audit(mode, episodes, seed_start):
    cfg = load_config('configs/base.yaml')
    env = UAVCombatEnv(cfg.environment, cfg.reward, cfg.render)
    counts = Counter()
    cycles, weapons, oob_examples = [], [], []
    for episode in range(episodes):
        seed = seed_start + episode
        env.reset(seed=seed)
        red, blue = RuleBasedOpponent(), RuleBasedOpponent()
        history = deque(maxlen=5)
        phase_history = [deque(maxlen=3) for _ in range(5)]
        done = False
        while not done:
            previous = env.positions.copy()
            old_alive = env.alive.copy()
            headings, speeds = env.headings.copy(), env.speeds.copy()
            detected = env.team_detected_enemy_mask(1)
            for agent in np.flatnonzero(env.alive[1]):
                target = blue.nearest_target(env,1,agent,detected)
                distance = (None if target is None else float(np.linalg.norm(
                    env.positions[0,target]-env.positions[1,agent])))
                if blue.boundary_heading(env,1,agent) is not None:
                    phase = 'boundary'
                elif target is None:
                    phase = 'straight'
                else:
                    threshold = cfg.environment.blue_escape_distance
                    threshold = cfg.environment.attack_range if threshold is None else threshold
                    phase = 'escape' if distance <= threshold else 'chase'
                counts['blue_'+phase+'_agent_steps'] += 1
                changes = phase_history[agent]
                if not changes or changes[-1]['phase'] != phase:
                    changes.append({'phase':phase,'step_before':env.step_count,
                                    'target':None if target is None else int(target),'distance':distance})
                    if [row['phase'] for row in changes] == ['chase','escape','chase']:
                        counts['blue_chase_escape_chase_cycles'] += 1
                        if len(cycles) < 5:
                            cycles.append({'seed':seed,'blue_agent':int(agent),'transitions':list(changes)})
            actions = np.stack([red.act(env,0) if mode == 'rule-vs-rule' else pursuit_red_actions(env),
                                blue.act(env,1)])
            history.append({'step_before':env.step_count,'positions':previous.tolist(),
                            'headings_degrees':np.degrees(headings).tolist(),
                            'speeds':speeds.tolist(),'actions':actions.tolist()})
            kill_before, oob_before = env.combat_kills.copy(), env.oob_losses.copy()
            _,_,done,_,info = env.step(actions)
            counts['weapon_kills'] += int((env.combat_kills-kill_before).sum())
            counts['oob_deaths'] += int((env.oob_losses-oob_before).sum())
            inside = old_alive & np.all(np.abs(env.positions) <= cfg.environment.map_size/2,axis=-1)
            for team,agent in zip(*np.where(old_alive & ~inside)):
                if len(oob_examples) < 5:
                    oob_examples.append({'seed':seed,'step':env.step_count,'team':int(team),
                        'agent':int(agent),'position_before':previous[team,agent].tolist(),
                        'position_after':env.positions[team,agent].tolist(),
                        'nominal_prediction':(previous[team,agent]+speeds[team,agent]*np.array([
                            math.cos(headings[team,agent]),math.sin(headings[team,agent])])*cfg.environment.dt).tolist(),
                        'action':actions[team,agent].tolist(),'preceding_frames':list(history)})
            if np.any(env.combat_kills > kill_before):
                for team in range(2):
                    for attacker in np.flatnonzero(inside[team]):
                        legal = []
                        for target in np.flatnonzero(inside[1-team]):
                            d,alpha,beta = env._compute_pair_geometry(team,attacker,1-team,target)
                            if d <= cfg.environment.attack_range and abs(alpha) <= math.radians(cfg.environment.attack_ata_degrees)+1e-12 and abs(beta) <= math.radians(cfg.environment.attack_aspect_degrees)+1e-12:
                                legal.append((d,int(target),alpha,beta))
                        if legal and len(weapons) < 5:
                            d,target,alpha,beta = min(legal)
                            weapons.append({'seed':seed,'step':env.step_count,'attacker':[team,int(attacker)],
                                'target':[1-team,target],'distance':d,'ata_degrees':math.degrees(alpha),
                                'aspect_degrees':math.degrees(beta),'preceding_frames':list(history)})
        counts['red_collision_events'] += int(env.collision_events[0])
        counts['blue_collision_events'] += int(env.collision_events[1])
    env.close()
    return {'mode':mode,'episodes':episodes,'seed_start':seed_start,'counts':dict(counts),
            'blue_cycle_examples':cycles,'weapon_examples':weapons,'oob_examples':oob_examples}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--seed-start',type=int,default=200000)
    parser.add_argument('--output',default='runs/v2_1_validation/behavioral_audit.json')
    args = parser.parse_args()
    results = {mode:audit(mode,args.episodes,args.seed_start)
               for mode in ['rule-vs-rule','pursuit-vs-rule']}
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({mode:row['counts'] for mode,row in results.items()},indent=2))
