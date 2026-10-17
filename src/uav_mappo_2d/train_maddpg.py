"""Independent off-policy MADDPG trainer; frozen V2.4 process environments."""
import argparse
from collections import deque
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .config import load_config
from .env import UAVCombatEnv
from .evaluate import resolve_device
from .evaluate_maddpg import evaluate
from .maddpg import MADDPG, JointReplayBuffer, OUNoise, ACTOR_DIAGNOSTICS, ACTOR_AGGREGATES
from .maddpg_config import load_maddpg_config
from .parallel_env import ParallelUAVEnvs


EPISODE_FIELDS = ['global_step','episode','return','result',*UAVCombatEnv.EPISODE_METRICS,
                  *UAVCombatEnv.REWARD_COMPONENTS,'episode_length']


def due_updates(global_step, next_update_step, interval, ready):
    """Crossed individual-transition thresholds; discard pre-warmup thresholds."""
    count = 0
    while global_step >= next_update_step:
        count += int(ready)
        next_update_step += interval
    return count, next_update_step


def better_evaluation(candidate, current):
    return current is None or (candidate['win_rate'],candidate['survival_difference']) > (
        current['win_rate'],current['survival_difference'])


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else 0.


def log_line(step, total, episodes, replay_size, updates, rows, metrics):
    behavior = ' '.join(f'{label}(R/B)={mean(rows,"red_"+key):.2f}/{mean(rows,"blue_"+key):.2f}'
                       for label,key in [('alive','alive'),('kills','combat_kill'),('losses','weapon_loss'),
                                         ('OOB','oob'),('collision','collision_events'),('windows','attack_window'),
                                         ('contact','radar_contact_steps')])
    algorithm = (f'actor={metrics["mean_actor_loss"]:.4g} critic={metrics["mean_critic_loss"]:.4g} '
                 f'Q/y={metrics["mean_q"]:.3f}/{metrics["mean_target_q"]:.3f} '
                 f'TD={metrics["mean_td_abs"]:.3f} grad(A/C)='
                 f'{metrics["mean_actor_grad_norm"]:.3f}/{metrics["mean_critic_grad_norm"]:.3f}') if metrics else 'warmup (loss=N/A)'
    if metrics:
        algorithm += (f' Replay-batch[reg={metrics["mean_actor_reg"]:.3g} |z|={metrics["mean_pre_tanh_abs"]:.3g} '
                      f'zmax={metrics["max_pre_tanh_abs"]:.3g} tanh\'={metrics["mean_tanh_derivative"]:.3g} '
                      f'sat(A/W)={metrics["mean_accel_saturation"]:.1%}/{metrics["mean_turn_saturation"]:.1%}]')
    if not rows:
        return (f'[MADDPG] step={step}/{total} ({step/max(total,1):.1%}) episodes={episodes} '
                f'replay={replay_size} updates={updates} episode_metrics=N/A (no completed episodes) {algorithm}')
    reward = '/'.join(f'{mean(rows,k):.2f}' for k in UAVCombatEnv.REWARD_COMPONENTS)
    rates = '/'.join(f'{sum(r["result"]==label for r in rows)/max(len(rows),1):.1%}' for label in ('win','draw','loss'))
    return (f'[MADDPG] step={step}/{total} ({step/max(total,1):.1%}) episodes={episodes} '
            f'replay={replay_size} updates={updates} return={mean(rows,"return"):.3f} {behavior} '
            f'length={mean(rows,"episode_length"):.1f} reward(kill/loss/collision/OOB/terminal)={reward} '
            f'{algorithm} W/D/L={rates}')


def run_training(config, algorithm):
    training = config.training
    total, n = training.total_env_steps, training.parallel_envs
    if total < 1 or n < 1 or total % n:
        raise ValueError('steps must be positive and divisible by parallel_envs')
    if min(training.log_interval_steps,training.evaluation_interval_steps,
           training.checkpoint_interval_steps,training.evaluation_episodes) < 1:
        raise ValueError('Logging/evaluation/checkpoint intervals and evaluation_episodes must be positive')
    warmup = algorithm.warmup(config.environment.episode_steps)
    output = Path(training.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new MADDPG output directory; no implicit resume/overwrite')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    learner = MADDPG(algorithm, resolve_device(training.device), seed=config.seed)
    replay = JointReplayBuffer(algorithm.buffer_capacity, seed=config.seed)
    noise = OUNoise(n, algorithm, seed=config.seed+1)
    snapshot = {k: asdict(getattr(config,k)) for k in ('environment','reward','render')}
    (output/'config_snapshot.json').write_text(json.dumps(
        {'project':asdict(config),'maddpg':asdict(algorithm),'learning_starts_resolved':warmup},indent=2)+'\n',encoding='utf-8')
    writer = SummaryWriter(output/'tensorboard')
    recent, step, episode_index, best, metrics = deque(maxlen=100), 0, 0, None, {}
    next_update = algorithm.update_interval_steps
    next_log, next_eval, next_checkpoint = training.log_interval_steps, training.evaluation_interval_steps, training.checkpoint_interval_steps
    reward_error, action_max = 0., 0.
    envs = ParallelUAVEnvs(n,config.environment,config.reward,config.render,Path.cwd(),base_seed=config.seed)
    print(f'[START MADDPG] steps={total} parallel={n} device={learner.device} '
          f'learning_starts={warmup} interval={algorithm.update_interval_steps} pids={envs.worker_pids}',flush=True)
    try:
        observations, states, active, _ = envs.reset()
        with (output/'episodes.csv').open('x',newline='',encoding='utf-8') as es, \
             (output/'updates.csv').open('x',newline='',encoding='utf-8') as us:
            ew = csv.DictWriter(es,fieldnames=EPISODE_FIELDS)
            ew.writeheader()
            metric_fields = ['updated_agents',*['mean_'+k for k in ('actor_loss','critic_loss','q','target_q',
                              'td_abs','actor_grad_norm','critic_grad_norm')],
                             *[f'agent{i}_{k}' for i in range(5) for k in ('actor_loss','critic_loss','updated')]]
            metric_fields += [*ACTOR_AGGREGATES, *[f'agent{i}_{k}' for i in range(5) for k in ACTOR_DIAGNOSTICS]]
            uw = csv.DictWriter(us,fieldnames=['global_step','update','replay_size','noise_sigma',*metric_fields])
            uw.writeheader()
            while step < total:
                actions = learner.actions(observations,active,noise)
                action_max = max(action_max,float(np.abs(actions).max()))
                if not np.isfinite(actions).all() or action_max > 1:
                    raise FloatingPointError('Invalid MADDPG actions')
                result = envs.step(actions)
                for e in range(n):
                    replay.add(observations=observations[e],states=states[e],actions=actions[e],rewards=result.rewards[e],
                               next_observations=result.observations[e],next_states=result.states[e],dones=result.dones[e],
                               active_masks=active[e],next_active_masks=result.active_masks[e])
                    info = result.infos[e]
                    reward_error = max(reward_error,abs(sum(info[k] for k in UAVCombatEnv.REWARD_COMPONENTS)-info['total_reward']))
                    assert np.float32(info['total_reward']) == result.rewards[e]
                    if 'episode_summary' in info:
                        summary = info['episode_summary']
                        reward_error = max(reward_error,abs(sum(summary[k] for k in UAVCombatEnv.REWARD_COMPONENTS)-summary['return']))
                        episode_index += 1
                        recent.append(summary)
                        ew.writerow(dict(global_step=step+n,episode=episode_index,**summary))
                        for key,value in summary.items():
                            if key != 'result':
                                writer.add_scalar('episode/'+key,float(value),step+n)
                        for label in ('win','draw','loss'):
                            writer.add_scalar('episode/'+label,float(summary['result']==label),step+n)
                noise.reset(np.flatnonzero(result.dones))
                observations, states, active = result.observations,result.states,result.active_masks
                step += n
                count,next_update = due_updates(step,next_update,algorithm.update_interval_steps,len(replay)>=warmup)
                for _ in range(count):
                    metrics = learner.update(replay.sample(algorithm.batch_size))
                    uw.writerow(dict(global_step=step,update=learner.update_count,replay_size=len(replay),
                                     noise_sigma=algorithm.ou_sigma,**metrics))
                    tags = {'mean_actor_loss':'actor_loss_mean','mean_critic_loss':'critic_loss_mean',
                            'mean_q':'q_mean','mean_target_q':'target_q_mean','mean_td_abs':'td_abs_mean',
                            'mean_actor_grad_norm':'actor_grad_norm','mean_critic_grad_norm':'critic_grad_norm',
                            'mean_actor_reg':'actor_reg_mean','mean_pre_tanh_abs':'pre_tanh_abs_mean',
                            'max_pre_tanh_abs':'pre_tanh_abs_max','mean_tanh_derivative':'tanh_derivative_mean',
                            'mean_accel_saturation':'accel_saturation_mean','mean_turn_saturation':'turn_saturation_mean'}
                    for key,value in metrics.items():
                        writer.add_scalar('maddpg/'+tags.get(key,key),value,step)
                    writer.add_scalar('maddpg/replay_size',len(replay),step)
                    writer.add_scalar('maddpg/noise_sigma',algorithm.ou_sigma,step)
                es.flush()
                us.flush()
                if step >= next_log or step == total:
                    print(log_line(step,total,episode_index,len(replay),learner.update_count,recent,metrics),flush=True)
                    while next_log <= step:
                        next_log += training.log_interval_steps
                if step >= next_eval or step == total:
                    evaluation = evaluate(config,learner,episodes=training.evaluation_episodes,seed_start=config.seed+1_000_000)
                    for key,value in evaluation.items():
                        writer.add_scalar('evaluation/'+key,value,step)
                    print(f'[EVAL MADDPG] step={step} return={evaluation["mean_return"]:.3f} '
                          f'kills(R/B)={evaluation["red_combat_kill"]:.2f}/{evaluation["blue_combat_kill"]:.2f} '
                          f'OOB(R/B)={evaluation["red_oob"]:.2f}/{evaluation["blue_oob"]:.2f} '
                          f'W/D/L={evaluation["win_rate"]:.1%}/{evaluation["draw_rate"]:.1%}/{evaluation["loss_rate"]:.1%}',flush=True)
                    if better_evaluation(evaluation,best):
                        best = evaluation
                        learner.save(output/'checkpoints/best.pt',global_step=step,environment_config=snapshot,best_evaluation=best)
                    while next_eval <= step:
                        next_eval += training.evaluation_interval_steps
                if step >= next_checkpoint:
                    learner.save(output/f'checkpoints/step_{step:09d}.pt',global_step=step,environment_config=snapshot,best_evaluation=best)
                    while next_checkpoint <= step:
                        next_checkpoint += training.checkpoint_interval_steps
            final = output/'checkpoints/final.pt'
            learner.save(final,global_step=step,environment_config=snapshot,best_evaluation=best)
            report = dict(global_step=step,updates=learner.update_count,replay_size=len(replay),episodes=episode_index,
                          worker_pids=envs.worker_pids,action_max_abs=action_max,max_reward_accounting_error=reward_error,
                          best_evaluation=best,last_evaluation=evaluation)
            assert reward_error < 1e-9
            (output/'training_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            return final
    finally:
        envs.close()
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/base.yaml')
    parser.add_argument('--algo-config',default='configs/maddpg.yaml')
    parser.add_argument('--steps',type=int,default=None)
    parser.add_argument('--output',default=None)
    parser.add_argument('--device',default=None)
    parser.add_argument('--seed',type=int,default=None)
    args = parser.parse_args()
    cfg, algorithm = load_config(args.config),load_maddpg_config(args.algo_config)
    if args.steps is not None:
        cfg.training.total_env_steps = args.steps
    cfg.training.output_dir = args.output or f'runs/v2_4_maddpg_seed{args.seed if args.seed is not None else cfg.seed}'
    if args.device is not None:
        cfg.training.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    print(f'[DONE MADDPG] checkpoint={run_training(cfg,algorithm)}')


if __name__ == '__main__':
    main()
