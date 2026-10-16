"""Read-only rear-pursuit trajectory audit; diagnostics never alter policy or environment."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import numpy as np

from sanity_check import pursuit_red_actions
from uav_mappo_2d.config import load_config
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.opponents import RuleBasedOpponent


def legal(env, geometry):
    distance, alpha, beta = geometry
    return bool(distance <= env.config.attack_range
                and abs(alpha) <= math.radians(env.config.attack_ata_degrees)+1e-12
                and abs(beta) <= math.radians(env.config.attack_aspect_degrees)+1e-12)


def audit(mode, episodes, seed_start, trace_path):
    cfg = load_config('configs/base.yaml')
    env = UAVCombatEnv(cfg.environment, cfg.reward, cfg.render)
    counts = Counter()
    examples = {'weapon':[], 'selected_target_window':[], 'oob':[],
                'geometry_improvement':[], 'collision_pairs':[]}
    rows = []
    fields = ['seed','step_before','blue_id','phase','target_id','target_distance',
              'blue_heading_degrees','target_heading_degrees','rear_x','rear_y','rear_distance',
              'ata_before_degrees','aspect_before_degrees','window_before',
              'action_acceleration','action_turn','target_distance_after','ata_after_degrees',
              'aspect_after_degrees','window_after','fired_target','fired_target_distance',
              'fired_target_ata_degrees','fired_target_aspect_degrees','fired_victim_confirmed',
              'blue_alive_after','oob','weapon_loss']
    trace_path.parent.mkdir(parents=True,exist_ok=True)
    with trace_path.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for episode in range(episodes):
            seed = seed_start+episode
            env.reset(seed=seed)
            red,blue = RuleBasedOpponent(),RuleBasedOpponent()
            previous_target = np.full(5,-1)
            run = np.zeros(5,dtype=int)
            no_window_run = np.zeros(5,dtype=int)
            done = False
            while not done:
                old_alive = env.alive.copy()
                previous = env.positions.copy()
                headings,speeds = env.headings.copy(),env.speeds.copy()
                detected = env.team_detected_enemy_mask(1)
                decisions = []
                for agent in np.flatnonzero(env.alive[1]):
                    target = blue.nearest_target(env,1,agent,detected)
                    phase = ('boundary' if blue.boundary_heading(env,1,agent) is not None
                             else ('straight' if target is None else 'rear-pursuit'))
                    row = {key:None for key in fields}
                    row.update(seed=seed,step_before=env.step_count,blue_id=int(agent),
                               phase=phase,target_id=None if target is None else int(target),
                               blue_heading_degrees=math.degrees(headings[1,agent]))
                    if target is not None:
                        geometry = env._compute_pair_geometry(1,agent,0,target)
                        point = blue.rear_point(env,1,target)
                        row.update(target_distance=geometry[0],
                                   target_heading_degrees=math.degrees(env.headings[0,target]),
                                   rear_x=float(point[0]),rear_y=float(point[1]),
                                   rear_distance=float(np.linalg.norm(point-env.positions[1,agent])),
                                   ata_before_degrees=math.degrees(geometry[1]),
                                   aspect_before_degrees=math.degrees(geometry[2]),
                                   window_before=legal(env,geometry))
                    counts['blue_'+phase+'_agent_steps'] += 1
                    if phase == 'rear-pursuit':
                        counts['target_changes'] += int(previous_target[agent] >= 0 and previous_target[agent] != target)
                        run[agent] = run[agent]+1 if previous_target[agent] == target else 1
                        counts['longest_same_target_rear_run'] = max(counts['longest_same_target_rear_run'],int(run[agent]))
                        previous_target[agent] = target
                    else:
                        run[agent] = 0
                        previous_target[agent] = -1
                    decisions.append(row)
                actions = np.stack([red.act(env,0) if mode == 'rule-vs-rule' else pursuit_red_actions(env),
                                    blue.act(env,1)])
                collision_before = env.collision_events.copy()
                _,_,done,_,info = env.step(actions)
                # In this frozen environment collision does not change alive. This is
                # the actual post-OOB, pre-weapon snapshot, reconstructed without writes.
                inside = old_alive & np.all(np.abs(env.positions)<=env.config.map_size/2,axis=-1)
                involved = set()
                flat0,flat1 = previous.reshape(-1,2),env.positions.reshape(-1,2)
                entities = np.flatnonzero(inside.ravel())
                for index,first in enumerate(entities):
                    for second in entities[index+1:]:
                        delta0 = flat0[first]-flat0[second]
                        motion = (flat1[first]-flat0[first])-(flat1[second]-flat0[second])
                        denom = float(np.dot(motion,motion))
                        tau = float(np.clip(-np.dot(delta0,motion)/denom,0,1)) if denom>1e-12 else 0.
                        closest = float(np.linalg.norm(delta0+tau*motion))
                        if closest>env.config.collision_distance:
                            continue
                        involved.update([int(first),int(second)])
                        kind = 'same_team' if first//5 == second//5 else 'cross_team'
                        counts['collision_pairs_'+kind] += 1
                        if len(examples['collision_pairs'])<5:
                            examples['collision_pairs'].append({'seed':seed,'step':env.step_count,
                                'pair':[[int(first//5),int(first%5)],[int(second//5),int(second%5)]],
                                'kind':kind,'swept_distance':closest})
                diagnosed = np.bincount(np.array(sorted(involved),dtype=int)//5,minlength=2)
                assert np.array_equal(diagnosed,env.collision_events-collision_before)
                for row in decisions:
                    agent,target = row['blue_id'],row['target_id']
                    row.update(action_acceleration=float(actions[1,agent,0]),
                               action_turn=float(actions[1,agent,1]),
                               blue_alive_after=bool(env.alive[1,agent]),
                               oob=bool(not inside[1,agent]),
                               weapon_loss=bool(inside[1,agent] and not env.alive[1,agent]),
                               window_after=False,fired_victim_confirmed=False)
                    if target is not None:
                        geometry = env._compute_pair_geometry(1,agent,0,target)
                        row.update(target_distance_after=geometry[0],
                                   ata_after_degrees=math.degrees(geometry[1]),
                                   aspect_after_degrees=math.degrees(geometry[2]),
                                   window_after=bool(inside[1,agent] and inside[0,target] and legal(env,geometry)))
                        if row['phase'] == 'rear-pursuit' and inside[1,agent] and inside[0,target]:
                            counts['rear_geometry_transitions'] += 1
                            alpha_better = abs(row['ata_after_degrees']) < abs(row['ata_before_degrees'])-1e-9
                            beta_better = abs(row['aspect_after_degrees']) < abs(row['aspect_before_degrees'])-1e-9
                            counts['ata_improved_steps'] += int(alpha_better)
                            counts['aspect_improved_steps'] += int(beta_better)
                            counts['both_angles_improved_steps'] += int(alpha_better and beta_better)
                            counts['selected_target_window_steps'] += int(row['window_after'])
                            counts['late_rear_steps_without_window'] += int(row['step_before']>=150 and not row['window_after'])
                            if alpha_better and beta_better and len(examples['geometry_improvement'])<5:
                                examples['geometry_improvement'].append(row)
                            if row['window_after'] and len(examples['selected_target_window'])<5:
                                examples['selected_target_window'].append(row)
                    if inside[1,agent]:
                        candidates = [(env._compute_pair_geometry(1,agent,0,j)[0],int(j))
                                      for j in np.flatnonzero(inside[0])
                                      if legal(env,env._compute_pair_geometry(1,agent,0,j))]
                        if candidates:
                            _,fired = min(candidates)
                            fired_geometry = env._compute_pair_geometry(1,agent,0,fired)
                            row.update(fired_target=fired,fired_target_distance=fired_geometry[0],
                                       fired_target_ata_degrees=math.degrees(fired_geometry[1]),
                                       fired_target_aspect_degrees=math.degrees(fired_geometry[2]),
                                       fired_victim_confirmed=bool(not env.alive[0,fired]))
                            counts['blue_attacker_steps_with_any_window'] += 1
                            if len(examples['weapon'])<5:
                                examples['weapon'].append(dict(row))
                        no_window_run[agent] = no_window_run[agent]+1 if not candidates else 0
                        counts['longest_blue_alive_run_without_any_window'] = max(
                            counts['longest_blue_alive_run_without_any_window'],int(no_window_run[agent]))
                    if row['oob'] and len(examples['oob'])<5:
                        row_example = dict(row)
                        nominal = previous[1,agent]+speeds[1,agent]*env.config.dt*np.array([
                            math.cos(headings[1,agent]),math.sin(headings[1,agent])])
                        row_example.update(position_before=previous[1,agent].tolist(),
                                           nominal_prediction=nominal.tolist(),
                                           position_after=env.positions[1,agent].tolist())
                        examples['oob'].append(row_example)
                    writer.writerow(row)
                if done:
                    rows.append(info['episode_summary'])
    env.close()
    for key in ['blue_combat_kill','blue_weapon_loss','blue_oob','blue_collision_events',
                'blue_attack_window','red_combat_kill','red_oob','red_collision_events']:
        counts[key] = sum(int(row[key]) for row in rows)
    return {'mode':mode,'episodes':episodes,'seed_start':seed_start,'counts':dict(counts),
            'episodes_with_blue_kill':sum(row['blue_combat_kill']>0 for row in rows),
            'episodes_with_blue_window':sum(row['blue_attack_window']>0 for row in rows),
            'trace_csv':trace_path.as_posix(),'examples':examples,'episode_summaries':rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--seed-start',type=int,default=200000)
    parser.add_argument('--output',default='runs/v2_2_rear_pursuit/behavioral_audit.json')
    args = parser.parse_args()
    path = Path(args.output)
    results = {mode:audit(mode,args.episodes,args.seed_start,path.parent/(mode+'_blue_steps.csv'))
               for mode in ['rule-vs-rule','direct-pursuit-vs-rule']}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({mode:row['counts'] for mode,row in results.items()},indent=2))
