"""V2.5 acceptance: 100 Greedy episodes, seeded replay, event/reward accounting.

No policy, map, reward, turn-rate or spawn tuning. Optional smoke is linkage only.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from sanity_check import summarize
from replay_html import classify_events, collect_trace, write_html
from uav_mappo_2d.config import load_config
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.mappo import MAPPO
from uav_mappo_2d.opponents import RuleBasedOpponent


ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate(directory):
    cfg = load_config(ROOT / 'configs/base.yaml')
    envs = [UAVCombatEnv(cfg.environment, cfg.reward, cfg.render) for _ in range(2)]
    rule = RuleBasedOpponent()
    rows, both_rewards, lengths = [], [], []
    max_error, steps = 0., 0
    try:
        for seed in range(200000, 200100):
            for env in envs:
                env.reset(seed=seed)
            totals = np.zeros(2)
            while True:
                env, replay = envs
                old_alive, old_losses = env.alive.copy(), env.weapon_losses.copy()
                old_oob = env.oob_losses.copy()
                actions = np.stack([rule.act(env, t) for t in range(2)])
                obs, rewards, done, truncated, info = env.step(actions)
                other_obs, other_rewards, other_done, _, other_info = replay.step(actions)
                np.testing.assert_array_equal(obs, other_obs)
                np.testing.assert_array_equal(rewards, other_rewards)
                for attr in ('alive', 'positions', 'headings', 'speeds'):
                    np.testing.assert_array_equal(getattr(env, attr), getattr(replay, attr))
                np.testing.assert_array_equal(env.global_state(), replay.global_state())
                assert env.np_random.bit_generator.state == replay.np_random.bit_generator.state
                assert done == other_done and not truncated
                assert info['combat_kills'] == other_info['combat_kills']
                components = info['team_reward_components']
                error = np.max(np.abs(sum(np.array(components[k]) for k in env.REWARD_COMPONENTS) - info['team_rewards']))
                max_error = max(max_error, float(error))
                np.testing.assert_allclose(rewards, np.repeat(np.array(info['team_rewards'])[:, None], 5, axis=1), atol=1e-5)
                events = classify_events(old_alive, env.positions, env.alive,
                                         cfg.environment.map_size, cfg.environment.collision_distance)
                for t, letter in enumerate(('R', 'B')):
                    assert sum(v.startswith(letter) for v in events['weapon_victims']) == env.weapon_losses[t] - old_losses[t]
                    assert sum(v.startswith(letter) for v in events['oob_victims']) == env.oob_losses[t] - old_oob[t]
                    assert env.alive[t].sum() + env.weapon_losses[t] + env.oob_losses[t] == 5
                    assert env.weapon_losses[t] == env.combat_kills[1-t]
                totals += info['team_rewards']
                steps += 1
                if done:
                    break
            row = {**info['episode_summary'], 'seed': seed, 'timed_out': env.step_count >= cfg.environment.episode_steps}
            max_error = max(max_error, abs(row['return'] - sum(row[k] for k in env.REWARD_COMPONENTS)), abs(totals[0] - row['return']))
            rows.append(row)
            both_rewards.append(totals.tolist())
            lengths.append(env.step_count)
            if len(rows) % 10 == 0:
                print(f'V2.5 Greedy {len(rows)}/100 | seeded replay exact | accounting error={max_error:.3g}', flush=True)
    finally:
        for env in envs:
            env.close()
    assert max_error < 1e-9
    report = dict(version='V2.5', seed_start=200000, episodes=100, checked_steps=steps,
                  environment=asdict(cfg.environment), reward=asdict(cfg.reward),
                  summary=summarize(rows), mean_team_returns=np.mean(both_rewards, axis=0).tolist(),
                  length_distribution={k: float(v) for k, v in zip(('min', 'p25', 'median', 'p75', 'max'), np.percentile(lengths, [0, 25, 50, 75, 100]))},
                  early_termination_le_20=sum(n <= 20 for n in lengths),
                  no_contact_episodes=sum(r['red_radar_contact_steps'] == 0 for r in rows),
                  no_weapon_combat_episodes=sum(r['red_combat_kill'] + r['blue_combat_kill'] == 0 for r in rows),
                  max_reward_accounting_error=max_error, seeded_replay_exact=True,
                  window_semantics='Geometric pairs post-OOB/post-collision/pre-weapon; not attempts/hits/kills or a hit-rate denominator.',
                  episode_summaries=rows)
    save(directory / 'validation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'episode_summaries'}, indent=2), flush=True)
    return report


def smoke_mappo(directory):
    from uav_mappo_2d.train import run_training, set_seed
    from uav_mappo_2d.evaluate import evaluate_policy

    cfg = load_config(ROOT / 'configs/smoke.yaml')
    cfg.training.output_dir = str(directory)
    cfg.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert not directory.exists(), 'Use a new smoke directory'
    set_seed(cfg.seed)
    initial = MAPPO(68, 60, 5, cfg.mappo, cfg.training.device, seed=cfg.seed)
    before = {key: torch.cat([p.detach().cpu().flatten() for p in getattr(initial, key).parameters()])
              for key in ('actor', 'critic')}
    checkpoint = run_training(cfg)
    loaded = MAPPO(68, 60, 5, cfg.mappo, cfg.training.device, seed=999)
    payload = loaded.load(checkpoint)
    changes = {}
    for key in ('actor', 'critic'):
        after = torch.cat([p.detach().cpu().flatten() for p in getattr(loaded, key).parameters()])
        changes[key] = float(torch.linalg.vector_norm(after - before[key]))
        assert changes[key] > 0
        assert all(torch.equal(v, payload[key][k]) for k, v in getattr(loaded, key).state_dict().items())
    assert payload['global_step'] == 1280
    with (directory / 'updates.csv').open() as stream:
        updates = list(csv.DictReader(stream))
    with (directory / 'episodes.csv').open() as stream:
        episodes = list(csv.DictReader(stream))
    assert updates and episodes
    assert all(np.isfinite(float(v)) for row in (*updates, *episodes) for k, v in row.items() if k != 'result')
    errors = [abs(float(row['return']) - sum(float(row[k]) for k in UAVCombatEnv.REWARD_COMPONENTS)) for row in episodes]
    assert max(errors) < 1e-9
    # MAPPO checkpoints retain their existing weights/optimizer format; this
    # diagnostic sidecar records the exact environment used for this smoke.
    save(directory / 'smoke_config_snapshot.json', asdict(cfg))
    snapshot = json.loads((directory / 'smoke_config_snapshot.json').read_text(encoding='utf-8'))
    assert snapshot['environment'] == asdict(cfg.environment)
    evaluation = evaluate_policy(cfg, loaded, episodes=4, seed_start=3000000)
    trace = collect_trace(cfg, loaded, 3000000)
    trace['metadata'].update(checkpoint_filename=checkpoint.name, checkpoint_global_step=payload['global_step'])
    write_html(trace, directory / 'replay_seed3000000.html')
    result = dict(device=str(loaded.device), cuda_available=torch.cuda.is_available(), env_steps=1280,
                  updates=len(updates), episodes=len(episodes), parameter_l2_changes=changes,
                  all_metrics_finite=True, checkpoint_reload_exact=True, config_snapshot_exact=True,
                  max_reward_accounting_error=max(errors), evaluation=evaluation,
                  html_replay_frames=len(trace['frames']), last_update=updates[-1])
    save(directory / 'smoke_validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/v2_5_deng_combat')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-only', action='store_true', help='Run MAPPO linkage after separate rule acceptance')
    args = parser.parse_args()
    if not args.smoke_only:
        validate(args.output)
    if args.smoke or args.smoke_only:
        smoke_mappo(args.output / 'mappo_smoke')
