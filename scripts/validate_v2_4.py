"""V2.4 deterministic acceptance against saved V2.3 audit; no long training."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
from contextlib import redirect_stdout
import types

import numpy as np

from sanity_check import summarize
from uav_mappo_2d.config import load_config
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.opponents import RuleBasedOpponent


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = UAVCombatEnv.REWARD_COMPONENTS
V23_REFERENCE = 'c33ed9e9953087792893a6c9d2d1330abd694a77'


def historical_environment(audit):
    """Load the committed V2.3 env in an isolated module, never patch production.

    Require the exact source hash recorded by the historical audit. Fail closed
    if the recorded commit is unavailable; do not invent a historical model.
    """
    source = subprocess.check_output(['git', 'show', f'{V23_REFERENCE}:src/uav_mappo_2d/env.py'], cwd=ROOT)
    normalized = source.replace(b'\r\n', b'\n')
    variants = [normalized, normalized.replace(b'\n', b'\r\n')]
    expected = audit['protected_sha256_before']['src/uav_mappo_2d/env.py']
    assert expected in [hashlib.sha256(v).hexdigest() for v in variants], 'Reference is not the audited V2.3 source'
    module = types.ModuleType('uav_mappo_2d._historical_v23_validation')
    module.__package__ = 'uav_mappo_2d'
    module.__file__ = str(ROOT / 'src/uav_mappo_2d/env.py')
    exec(compile(normalized, '<audited V2.3 env from recorded commit>', 'exec'), module.__dict__)
    return module.UAVCombatEnv


def validate(output):
    raise RuntimeError('Historical V2.4 validator: run from the V2.4 checkout with its config. '
                       'For current production use scripts/validate_v2_5.py; V2.5 cannot match V2.3 trajectories.')
    # The original historical validator is retained below for reference.
    audit_path = ROOT / 'runs/v2_3_greedy_shooter/collision_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    cfg = load_config(ROOT / 'configs/base.yaml')
    assert cfg.environment.collision_distance == 50 and cfg.reward.collision_penalty == .2
    old_class = historical_environment(audit)
    old = old_class(cfg.environment, replace(cfg.reward, collision_penalty=5.), cfg.render)
    new = UAVCombatEnv(cfg.environment, cfg.reward, cfg.render)
    policies = [RuleBasedOpponent(), RuleBasedOpponent()]
    rows, episode_checks, team_totals, accounting = [], [], [], []
    steps = 0
    excluded = {'red_collision_events', 'blue_collision_events', 'collision_reward', 'return'}
    for n, saved in enumerate(audit['episodes']):
        seed = saved['seed']
        assert seed == 200000 + n
        old_obs, _ = old.reset(seed=seed)
        new_obs, _ = new.reset(seed=seed)
        np.testing.assert_array_equal(new_obs, old_obs)
        done = False
        actual = {k: np.zeros(2) for k in COMPONENTS}
        total = np.zeros(2)
        while not done:
            old_actions = np.stack([policies[t].act(old, t) for t in range(2)])
            new_actions = np.stack([policies[t].act(new, t) for t in range(2)])
            np.testing.assert_array_equal(new_actions, old_actions)
            old_obs, _, old_done, old_truncated, old_info = old.step(old_actions)
            new_obs, _, done, truncated, info = new.step(new_actions)
            assert done == old_done and truncated == old_truncated
            for key in ('positions', 'headings', 'speeds', 'alive', 'combat_kills', 'weapon_losses',
                        'oob_losses', 'attack_window_count', 'radar_contact_steps'):
                np.testing.assert_array_equal(getattr(new, key), getattr(old, key))
            np.testing.assert_array_equal(new_obs, old_obs)
            np.testing.assert_array_equal(new.global_state(), old.global_state())
            assert new.last_winner == old.last_winner
            assert new.np_random.bit_generator.state == old.np_random.bit_generator.state
            for k in COMPONENTS:
                actual[k] += info['team_reward_components'][k]
                if k != 'collision_reward':
                    assert info['team_reward_components'][k] == old_info['team_reward_components'][k]
            total += info['team_rewards']
            accounting.append(float(np.max(np.abs(sum(np.array(info['team_reward_components'][k])
                                                       for k in COMPONENTS) - info['team_rewards']))))
            steps += 1
        historical = {**old_info['episode_summary'], 'timed_out': old.step_count >= cfg.environment.episode_steps}
        assert historical == saved['summary'], f'V2.3 reference drift: seed={seed}'
        summary = {**info['episode_summary'], 'timed_out': new.step_count >= cfg.environment.episode_steps}
        for key, value in historical.items():
            if key not in excluded:
                assert summary[key] == value, (seed, key)
        assert new.collision_events.tolist() == saved['participant_events']['endpoint'], seed
        errors = []
        for t, team in enumerate(('Red', 'Blue')):
            expected = saved['rewards'][team]['Liu-style']
            for key in COMPONENTS:
                errors.append(abs(float(actual[key][t]) - expected[key]))
            errors.append(abs(float(total[t]) - expected['total']))
        errors.extend(abs(summary[k] - saved['rewards']['Red']['Liu-style'][k]) for k in COMPONENTS)
        errors.append(abs(summary['return'] - saved['rewards']['Red']['Liu-style']['total']))
        assert max(errors) < 1e-9, (seed, max(errors))
        accounting.extend(errors)
        rows.append(summary)
        team_totals.append({k: v.tolist() for k, v in actual.items()} | {'total': total.tolist()})
        episode_checks.append(dict(seed=seed, endpoint_events=new.collision_events.tolist(),
                                   max_absolute_counterfactual_error=max(errors), trajectory_exact=True,
                                   counterfactual_match=True, summary=summary, actual_team_rewards=team_totals[-1]))
        if (n + 1) % 10 == 0:
            print(f'V2.4 acceptance {n+1}/100 | trajectory mismatch=0 | '
                  f'counterfactual max error={max(accounting):.3g}', flush=True)
    old.close()
    new.close()
    basic = summarize(rows)
    old_basic = audit['basic_results']
    changed = {'mean_red_collision_events', 'mean_blue_collision_events', 'mean_collision_reward',
               'mean_return', 'collision_episode_rate'}
    for key, value in basic.items():
        if key not in changed:
            assert value == old_basic[key], key
    means = {k: np.mean([r[k] for r in team_totals], axis=0).tolist() for k in (*COMPONENTS, 'total')}
    result = dict(version='V2.4 endpoint collision', config='configs/base.yaml', seed_start=200000,
                  episodes=100, trajectory_checked_steps=steps, trajectory_mismatches=0,
                  exact_trajectory_matches=100, counterfactual_episode_matches=100,
                  max_absolute_reward_accounting_error=max(accounting),
                  historical_reference=V23_REFERENCE,
                  historical_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                  basic_results=basic, team_reward_means=means, episode_checks=episode_checks)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('exact_trajectory_matches', 'counterfactual_episode_matches',
          'max_absolute_reward_accounting_error', 'basic_results', 'team_reward_means')}, indent=2), flush=True)


def smoke(directory):
    import torch
    from uav_mappo_2d.mappo import MAPPO
    from uav_mappo_2d.train import run_training, set_seed

    cfg = load_config(ROOT / 'configs/smoke.yaml')
    cfg.training.output_dir = str(directory)
    assert cfg.training.total_env_steps == 1280
    assert not directory.exists(), 'Choose a new smoke output directory; do not append to old runs'
    set_seed(cfg.seed)
    initial = MAPPO(UAVCombatEnv.OBS_DIM, UAVCombatEnv.STATE_DIM, UAVCombatEnv.N_AGENTS,
                    cfg.mappo, 'cpu', seed=cfg.seed)
    initial_actor = torch.cat([p.detach().flatten() for p in initial.actor.parameters()])
    captured = io.StringIO()
    with redirect_stdout(captured):
        checkpoint = run_training(cfg)
    (directory / 'smoke.log').write_text(captured.getvalue(), encoding='utf-8')
    print(captured.getvalue(), end='', flush=True)
    reloaded = MAPPO(UAVCombatEnv.OBS_DIM, UAVCombatEnv.STATE_DIM, UAVCombatEnv.N_AGENTS, cfg.mappo, 'cpu', seed=cfg.seed)
    payload = reloaded.load(checkpoint)
    trained_actor = torch.cat([p.detach().flatten() for p in reloaded.actor.parameters()])
    change = float(torch.linalg.vector_norm(trained_actor - initial_actor))
    assert change > 0 and payload['global_step'] == 1280
    assert all(torch.equal(v, payload['actor'][k]) for k, v in reloaded.actor.state_dict().items())
    assert all(torch.equal(v, payload['critic'][k]) for k, v in reloaded.critic.state_dict().items())
    with (directory / 'episodes.csv').open() as stream:
        episodes = list(csv.DictReader(stream))
    with (directory / 'updates.csv').open() as stream:
        updates = list(csv.DictReader(stream))
    for row in (*episodes, *updates):
        for key, value in row.items():
            if key != 'result':
                assert np.isfinite(float(value)), (key, value)
    errors = [abs(sum(float(row[k]) for k in COMPONENTS) - float(row['return'])) for row in episodes]
    assert max(errors, default=0.) < 1e-9
    ppo_keys = ('actor_loss', 'critic_loss', 'entropy', 'approx_kl', 'clip_fraction',
                'actor_grad_norm', 'critic_grad_norm', 'explained_variance')
    result = dict(env_steps=1280, updates=len(updates), completed_episodes=len(episodes),
                  actor_parameter_l2_change=change, checkpoint_reload_exact=True, all_metrics_finite=True,
                  max_reward_accounting_error=max(errors, default=0.),
                  ppo={k: {'min': min(float(r[k]) for r in updates), 'max': max(float(r[k]) for r in updates),
                           'last': float(updates[-1][k])} for k in ppo_keys}, last_update=updates[-1])
    (directory.parent / 'smoke_validation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/v2_4_endpoint_collision/validation.json')
    parser.add_argument('--smoke', action='store_true', help='After acceptance only, run existing 1280-step smoke')
    args = parser.parse_args()
    validate(args.output)
    if args.smoke:
        smoke(args.output.parent / 'smoke')
