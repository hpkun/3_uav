"""Read-only V2.3 Greedy Shooter audit; no diagnostic state affects control."""
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
from uav_mappo_2d.env import UAVCombatEnv, wrap_angle
from uav_mappo_2d.opponents import RuleBasedOpponent


def legal(env, geometry):
    d,alpha,beta = geometry
    return bool(d <= env.config.attack_range
                and abs(alpha) <= math.radians(env.config.attack_ata_degrees)+1e-12
                and abs(beta) <= math.radians(env.config.attack_aspect_degrees)+1e-12)


def collision_participants(env, previous, inside, counts):
    involved = set()
    p0,p1 = previous.reshape(-1,2),env.positions.reshape(-1,2)
    entities = np.flatnonzero(inside.ravel())
    for n,first in enumerate(entities):
        for second in entities[n+1:]:
            delta = p0[first]-p0[second]
            motion = (p1[first]-p0[first])-(p1[second]-p0[second])
            denom = float(np.dot(motion,motion))
            tau = float(np.clip(-np.dot(delta,motion)/denom,0,1)) if denom>1e-12 else 0.
            if np.linalg.norm(delta+tau*motion) <= env.config.collision_distance:
                involved.update([int(first),int(second)])
                counts['collision_pairs_same_team' if first//5==second//5 else 'collision_pairs_cross_team'] += 1
    return involved


def audit(mode, episodes, seed_start, trace_path):
    cfg = load_config('configs/base.yaml')
    env = UAVCombatEnv(cfg.environment,cfg.reward,cfg.render)
    counts = Counter()
    examples = {'weapon':[], 'selected_window':[], 'oob':[]}
    summaries = []
    fields = ['seed','step_before','agent_id','phase','selected_target_id','target_distance',
              'own_x','own_y','target_x','target_y','own_heading_degrees',
              'desired_direct_heading_degrees','desired_heading_degrees','heading_error_degrees',
              'action_turn','action_acceleration','ata_before_degrees','aspect_before_degrees',
              'window_before','ata_after_degrees','aspect_after_degrees','window_after',
              'weapon_fired','weapon_victim','weapon_distance','weapon_ata_degrees',
              'weapon_aspect_degrees','victim_confirmed','collision_event','oob','weapon_loss',
              'target_changed','same_target_steps','alive_after']
    max_action_error,helper_error = 0.,0.
    trace_path.parent.mkdir(parents=True,exist_ok=True)
    with trace_path.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for episode in range(episodes):
            seed = seed_start+episode
            env.reset(seed=seed)
            red,blue = RuleBasedOpponent(),RuleBasedOpponent()
            last_target = np.full(5,-1)
            same_target_steps = np.zeros(5,dtype=int)
            done = False
            while not done:
                previous,old_alive = env.positions.copy(),env.alive.copy()
                detected = env.team_detected_enemy_mask(1)
                decisions = []
                for agent in np.flatnonzero(env.alive[1]):
                    target = blue.nearest_target(env,1,agent,detected)
                    # Independent distance/ID ordering, not a second call to selection.
                    candidates = [(float(np.linalg.norm(env.positions[0,j]-env.positions[1,agent])),int(j))
                                  for j in np.flatnonzero(detected)]
                    expected = min(candidates)[1] if candidates else None
                    assert target == expected
                    counts['nearest_target_checks'] += 1
                    boundary = blue.boundary_heading(env,1,agent)
                    phase = 'boundary' if boundary is not None else ('straight' if target is None else 'direct-pursuit')
                    row = {key:None for key in fields}
                    row.update(seed=seed,step_before=env.step_count,agent_id=int(agent),phase=phase,
                               selected_target_id=None if target is None else int(target),
                               own_x=float(env.positions[1,agent,0]),own_y=float(env.positions[1,agent,1]),
                               own_heading_degrees=math.degrees(env.headings[1,agent]),
                               window_before=False,window_after=False,weapon_fired=False,victim_confirmed=False)
                    desired = boundary
                    if target is not None:
                        delta = env.positions[0,target]-env.positions[1,agent]
                        direct = math.atan2(float(delta[1]),float(delta[0]))
                        geometry = env._compute_pair_geometry(1,agent,0,target)
                        row.update(target_x=float(env.positions[0,target,0]),target_y=float(env.positions[0,target,1]),
                                   target_distance=geometry[0],desired_direct_heading_degrees=math.degrees(direct),
                                   ata_before_degrees=math.degrees(geometry[1]),
                                   aspect_before_degrees=math.degrees(geometry[2]),window_before=legal(env,geometry))
                        if desired is None:
                            desired = direct
                        row['target_changed'] = bool(last_target[agent]>=0 and last_target[agent]!=target)
                        same_target_steps[agent] = same_target_steps[agent]+1 if last_target[agent]==target else 1
                        last_target[agent] = target
                    else:
                        last_target[agent],same_target_steps[agent] = -1,0
                        row['target_changed'] = False
                    row['same_target_steps'] = int(same_target_steps[agent])
                    counts['target_changes'] += int(row['target_changed'])
                    counts['longest_same_target_run'] = max(counts['longest_same_target_run'],row['same_target_steps'])
                    if desired is not None:
                        row['desired_heading_degrees'] = math.degrees(desired)
                        row['heading_error_degrees'] = math.degrees(float(wrap_angle(desired-env.headings[1,agent])))
                    counts['blue_'+phase+'_agent_steps'] += 1
                    decisions.append(row)
                helper,formal = pursuit_red_actions(env),red.act(env,0)
                helper_error = max(helper_error,float(np.max(np.abs(helper-formal))))
                actions = np.stack([formal if mode=='greedy-vs-greedy' else helper,blue.act(env,1)])
                for row in decisions:
                    agent = row['agent_id']
                    desired_turn = (0. if row['heading_error_degrees'] is None else np.clip(
                        row['heading_error_degrees']/(env.config.max_turn_rate_degrees*env.config.dt),-1,1))
                    error = abs(float(actions[1,agent,1])-desired_turn)
                    max_action_error = max(max_action_error,error)
                    assert error<1e-6 and actions[1,agent,0]==0
                    counts['direct_action_checks'] += 1
                collision_before = env.collision_events.copy()
                _,_,done,_,info = env.step(actions)
                inside = old_alive & np.all(np.abs(env.positions)<=env.config.map_size/2,axis=-1)
                involved = collision_participants(env,previous,inside,counts)
                diagnosed = np.bincount(np.array(sorted(involved),dtype=int)//5,minlength=2)
                assert np.array_equal(diagnosed,env.collision_events-collision_before)
                for row in decisions:
                    agent,target = row['agent_id'],row['selected_target_id']
                    row.update(action_turn=float(actions[1,agent,1]),action_acceleration=float(actions[1,agent,0]),
                               collision_event=bool(5+agent in involved),oob=bool(not inside[1,agent]),
                               weapon_loss=bool(inside[1,agent] and not env.alive[1,agent]),
                               alive_after=bool(env.alive[1,agent]))
                    if target is not None:
                        geometry = env._compute_pair_geometry(1,agent,0,target)
                        row.update(ata_after_degrees=math.degrees(geometry[1]),aspect_after_degrees=math.degrees(geometry[2]),
                                   window_after=bool(inside[1,agent] and inside[0,target] and legal(env,geometry)))
                        counts['selected_target_window_steps'] += int(row['window_after'])
                    if inside[1,agent]:
                        choices = [(env._compute_pair_geometry(1,agent,0,j)[0],int(j)) for j in np.flatnonzero(inside[0])
                                   if legal(env,env._compute_pair_geometry(1,agent,0,j))]
                        if choices:
                            _,victim = min(choices)
                            geometry = env._compute_pair_geometry(1,agent,0,victim)
                            row.update(weapon_fired=True,weapon_victim=victim,weapon_distance=geometry[0],
                                       weapon_ata_degrees=math.degrees(geometry[1]),
                                       weapon_aspect_degrees=math.degrees(geometry[2]),
                                       victim_confirmed=bool(not env.alive[0,victim]))
                            counts['blue_firing_agent_steps'] += 1
                    for label,condition in [('weapon',row['weapon_fired']),('selected_window',row['window_after']),('oob',row['oob'])]:
                        if condition and len(examples[label])<5:
                            examples[label].append(dict(row))
                    writer.writerow(row)
            summaries.append({**info['episode_summary'],'timed_out':env.step_count>=env.config.episode_steps})
    env.close()
    for key in ['blue_combat_kill','blue_weapon_loss','blue_oob','blue_collision_events','blue_attack_window',
                'red_combat_kill','red_oob','red_collision_events']:
        counts[key] = sum(int(r[key]) for r in summaries)
    return {'mode':mode,'episodes':episodes,'seed_start':seed_start,'counts':dict(counts),
            'max_action_check_error':max_action_error,'helper_max_action_difference':helper_error,
            'episodes_with_blue_kill':sum(r['blue_combat_kill']>0 for r in summaries),
            'episodes_with_blue_window':sum(r['blue_attack_window']>0 for r in summaries),
            'timeouts':sum(r['timed_out'] for r in summaries),
            'trace_csv':trace_path.as_posix(),'examples':examples,'episode_summaries':summaries}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--seed-start',type=int,default=200000)
    parser.add_argument('--output',default='runs/v2_3_greedy_shooter/behavioral_audit.json')
    args = parser.parse_args()
    path = Path(args.output)
    results = {mode:audit(mode,args.episodes,args.seed_start,path.parent/(mode+'_blue_steps.csv'))
               for mode in ['greedy-vs-greedy','direct-pursuit-vs-greedy']}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({mode:{'counts':r['counts'],'action_error':r['max_action_check_error'],
                            'helper_error':r['helper_max_action_difference']} for mode,r in results.items()},indent=2))
