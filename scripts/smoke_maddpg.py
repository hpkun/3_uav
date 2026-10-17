"""Short MADDPG linkage verification, never a 2M experiment."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import csv
from unittest.mock import patch
import numpy as np
import torch

from uav_mappo_2d.config import load_config
from uav_mappo_2d.maddpg_config import load_maddpg_config
from uav_mappo_2d.maddpg import MADDPG
from uav_mappo_2d.train_maddpg import run_training
from uav_mappo_2d.evaluate_maddpg import evaluate


def parameters(group):
    return [torch.cat([p.detach().cpu().flatten() for p in m.parameters()]) for m in group]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/v2_4_maddpg_smoke_seed7')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--algo-config',default='configs/maddpg.yaml')
    args = parser.parse_args()
    cfg = load_config('configs/base.yaml')
    cfg.training = replace(cfg.training,total_env_steps=2560,output_dir=args.output,device=args.device,
                           log_interval_steps=512,checkpoint_interval_steps=1280,
                           evaluation_interval_steps=1280,evaluation_episodes=4)
    algo = replace(load_maddpg_config(args.algo_config),batch_size=32,
                   learning_starts=128,buffer_capacity=4096)
    initial = MADDPG(algo,seed=cfg.seed)
    groups = ('actors','critics','target_actors','target_critics')
    before = {key:parameters(getattr(initial,key)) for key in groups}
    formula_errors = []
    original_update = MADDPG.update_actor
    def checked_update(learner, i, batch):
        active = batch['active_masks'][:,i].bool()
        if active.any():
            with torch.no_grad():
                z = learner.actors[i].pre_tanh(batch['observations'][active,i])
                action = z.tanh()
                joint = batch['actions'][active].detach().clone()
                joint[:,i] = action
                pg = -learner.critics[i](batch['states'][active],joint).mean()
                expected_reg = algo.actor_regularization * (z if algo.actor_regularization_mode=='pre_tanh' else action).square().mean()
                expected_loss = float(pg + expected_reg)
        result = original_update(learner,i,batch)
        if result is not None:
            formula_errors.append(max(abs(result['actor_reg']-float(expected_reg)),
                                      abs(result['actor_loss']-expected_loss)))
        return result
    # Read-only pre-step check on every real Replay batch; no RNG consumption.
    with patch.object(MADDPG,'update_actor',checked_update):
        path = run_training(cfg,algo)
    assert formula_errors and max(formula_errors)==0.
    loaded = MADDPG(algo,device=args.device,seed=999)
    checkpoint = loaded.load(path)
    changes = {key:[float(torch.linalg.vector_norm(p-old)) for p,old in zip(parameters(getattr(loaded,key)),before[key])]
               for key in groups}
    assert all(v>0 for values in changes.values() for v in values)
    for key in groups:
        for network,state in zip(getattr(loaded,key),checkpoint[key]):
            assert all(torch.equal(v,state[k]) for k,v in network.state_dict().items())
    old_targets = {key:[p.detach().clone() for p in getattr(loaded,key).parameters()]
                   for key in ('target_actors','target_critics')}
    loaded.soft_update()  # verify formula on this diagnostic copy only; no file write
    error = 0.
    for online,target in [('actors','target_actors'),('critics','target_critics')]:
        for p,t,old in zip(getattr(loaded,online).parameters(),getattr(loaded,target).parameters(),old_targets[target]):
            expected = (1-algo.tau)*old + algo.tau*p.detach()
            error = max(error,float((t-expected).abs().max()))
    assert error < 1e-7
    result = evaluate(cfg,loaded,episodes=4,seed_start=3000000)
    with (Path(args.output)/'updates.csv').open() as stream:
        updates = list(csv.DictReader(stream))
    with (Path(args.output)/'episodes.csv').open() as stream:
        episodes = list(csv.DictReader(stream))
    assert len(updates)==24 and checkpoint['global_step']==2560
    assert all(np.isfinite(float(v)) for row in updates for v in row.values())
    assert checkpoint['maddpg_config']['actor_regularization_mode'] == algo.actor_regularization_mode
    report = dict(global_step=2560,updates=len(updates),episodes=len(episodes),
                  actor_regularization_mode=algo.actor_regularization_mode,
                  actor_formula_checks=len(formula_errors),actor_formula_max_error=max(formula_errors),
                  parameter_l2_changes=changes,soft_update_tau=algo.tau,soft_formula_max_error=error,
                  checkpoint_reload_exact=True,metrics_finite=True,
                  metrics_ranges={k:{'min':min(float(r[k]) for r in updates),'max':max(float(r[k]) for r in updates)}
                                  for k in updates[0] if k.startswith('mean_') or k=='max_pre_tanh_abs'},
                  deterministic_holdout_evaluation=result)
    (Path(args.output)/'smoke_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
