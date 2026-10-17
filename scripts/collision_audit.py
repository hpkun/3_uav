"""External read-only collision audit. No environment or policy mutation/patching.

Diagnostic sets and burst history below never enter the environment or actions.
Liu/Gao mappings are user-specified counterfactuals, not full paper reproductions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from sanity_check import summarize
from uav_mappo_2d.config import load_config
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.opponents import RuleBasedOpponent


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = UAVCombatEnv.REWARD_COMPONENTS
KINDS = ('swept', 'endpoint', 'swept_only', 'same_endpoint')


def protected_hashes():
    paths = sorted((ROOT / 'src/uav_mappo_2d').glob('*.py')) + sorted((ROOT / 'configs').glob('*.yaml'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def post_oob_mask(old_alive, positions, map_size):
    return old_alive & np.all(np.abs(positions) <= map_size / 2, axis=-1)


def pair_geometry(previous, current, valid, threshold=50.):
    """One row per current swept pair; squared comparison mirrors env exactly."""
    previous, current = previous.reshape(-1, 2), current.reshape(-1, 2)
    first, second = np.triu_indices(len(previous), 1)
    keep = valid.ravel()[first] & valid.ravel()[second]
    first, second = first[keep], second[keep]
    r0 = previous[first] - previous[second]
    displacement = current - previous
    dr = displacement[first] - displacement[second]
    denominator = np.sum(dr * dr, axis=1)
    tau = np.zeros(len(first))
    np.divide(-np.sum(r0 * dr, axis=1), denominator, out=tau, where=denominator > 1e-12)
    tau = np.clip(tau, 0, 1)
    swept2 = np.sum((r0 + tau[:, None] * dr)**2, axis=1)
    end2 = np.sum((current[first] - current[second])**2, axis=1)
    # Every endpoint collision is necessarily also a swept collision.
    assert np.all(~(end2 <= threshold**2) | (swept2 <= threshold**2))
    result = []
    for k in np.flatnonzero(swept2 <= threshold**2):
        i, j = int(first[k]), int(second[k])
        assert valid.ravel()[i] and valid.ravel()[j]
        relation = ('RR' if i < 5 else 'BB') if i // 5 == j // 5 else 'RB'
        result.append(dict(first=i, second=j, pair_type=relation,
                           endpoint=bool(end2[k] <= threshold**2),
                           distance_previous=float(np.linalg.norm(r0[k])),
                           distance_endpoint=float(np.sqrt(end2[k])),
                           distance_swept=float(np.sqrt(swept2[k])), tau=float(tau[k])))
    return result


def participant_sets(pairs):
    sets = {key: set() for key in (*KINDS, 'same', 'cross', 'both', 'all')}
    for pair in pairs:
        ids = {pair['first'], pair['second']}
        sets['swept'].update(ids)
        sets['endpoint' if pair['endpoint'] else 'swept_only'].update(ids)
        sets['same' if pair['pair_type'] != 'RB' else 'cross'].update(ids)
        if pair['endpoint'] and pair['pair_type'] != 'RB':
            sets['same_endpoint'].update(ids)
    sets['both'] = sets['same'] & sets['cross']
    sets['all'] = sets['same'] | sets['cross']
    assert sets['all'] == sets['swept']
    assert sets['endpoint'] | sets['swept_only'] == sets['swept']
    sets['endpoint_swept_only_overlap'] = sets['endpoint'] & sets['swept_only']
    return sets


def runs(bits):
    """Inclusive zero-based start/end of all contiguous active runs."""
    result, start = [], None
    for i, value in enumerate([*bits, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            result.append(dict(start_step=start + 1, end_step=i, length=i - start))
            start = None
    return result


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {key: 0. for key in ('mean', 'median', 'std', 'min', 'p25', 'p75', 'p90', 'p95', 'max')}
    return dict(mean=float(a.mean()), median=float(np.median(a)), std=float(a.std(ddof=0)),
                min=float(a.min()), max=float(a.max()),
                **{f'p{q}': float(np.percentile(a, q)) for q in (25, 75, 90, 95)})


def recompute(actual, swept, endpoint, same_endpoint):
    """Other actual trajectory components unchanged; no rewards sent to env."""
    result = {}
    for label, count, weight in [('Current', swept, -5.), ('Liu-style', endpoint, -.2),
                                 ('Gao-base', same_endpoint, -5.)]:
        components = dict(actual)
        components['collision_reward'] = weight * count
        components['total'] = sum(components.values())
        absolute = sum(abs(components[k]) for k in COMPONENTS)
        components['collision_absolute_share'] = abs(components['collision_reward']) / absolute if absolute else 0.
        result[label] = components
    return result


def digest_step(digest, actions, env, info):
    # Full ordered action/physical/reward/RNG trace checksum, baseline vs audit.
    for a in (actions, env.positions, env.headings, env.speeds, env.alive):
        digest.update(a.tobytes())
    digest.update(json.dumps(info, sort_keys=True, default=lambda a: a.tolist()).encode())
    digest.update(repr(env.np_random.bit_generator.state).encode())


def baseline(cfg):
    """Uninstrumented original rollout for exact per-episode replay comparison."""
    env = UAVCombatEnv(cfg.environment, cfg.reward, cfg.render)
    policies = [RuleBasedOpponent(), RuleBasedOpponent()]
    rows, checksums = [], []
    for seed in range(200000, 200100):
        env.reset(seed=seed)
        done, digest = False, hashlib.sha256()
        while not done:
            actions = np.stack([policies[t].act(env, t) for t in range(2)])
            _, _, done, _, info = env.step(actions)
            digest_step(digest, actions, env, info)
        rows.append({**info['episode_summary'], 'timed_out': env.step_count >= cfg.environment.episode_steps})
        checksums.append(digest.hexdigest())
    env.close()
    return rows, checksums


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def burst_summary(rows, episodes):
    lengths = [r['length'] for r in rows]
    result = {'burst_count': len(rows), **distribution(lengths),
              'bursts_per_episode': len(rows) / episodes,
              'participant_events_per_episode': sum(lengths) / episodes, 'bins': {}}
    for label, select in [('1', lambda n: n == 1), ('2-3', lambda n: 2 <= n <= 3),
                           ('>=4', lambda n: n >= 4)]:
        selected = [n for n in lengths if select(n)]
        result['bins'][label] = dict(burst_count=len(selected),
            burst_share=len(selected) / len(rows) if rows else 0., participant_events=sum(selected),
            event_share=sum(selected) / sum(lengths) if sum(lengths) else 0.)
    assert sum(result['bins'][k]['participant_events'] for k in result['bins']) == sum(lengths)
    return result


def run(output):
    raise RuntimeError('Historical V2.3 swept/-5 audit: run from its historical checkout/config. '
                       'For current V2.5 use scripts/validate_v2_5.py or behavioral_audit.py.')
    # Historical mathematics and original audit remain available for reference/tests.
    before = protected_hashes()
    cfg = load_config(ROOT / 'configs/base.yaml')
    assert cfg.environment.collision_distance == 50 and cfg.reward.collision_penalty == 5
    print('Baseline: original Greedy-vs-Greedy, 100 episodes, seeds 200000-200099', flush=True)
    references, checksums = baseline(cfg)
    existing = json.loads((ROOT / 'runs/v2_3_greedy_shooter/sanity.json').read_text())['greedy_vs_greedy']
    reproduced = summarize(references)
    for key, expected in reproduced.items():
        assert expected == existing[key], f'Original sanity mismatch: {key}'
    print('Baseline matches saved V2.3 sanity. Starting external audit of identical 100 seeds.', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = UAVCombatEnv(cfg.environment, cfg.reward, cfg.render)
    policies = [RuleBasedOpponent(), RuleBasedOpponent()]
    pair_rows, set_rows, burst_rows, team_bursts, episodes, reward_rows = [], [], [], [], [], []
    stats = Counter()
    accounting_errors = []
    for episode, seed in enumerate(range(200000, 200100)):
        env.reset(seed=seed)
        done, digest = False, hashlib.sha256()
        history, pair_counts = [], Counter()
        counts = {key: np.zeros(2, dtype=int) for key in (*KINDS, 'same', 'cross', 'both', 'all',
                                                        'endpoint_swept_only_overlap')}
        actual = {key: np.zeros(2) for key in COMPONENTS}
        actual_returns = np.zeros(2)
        first_pair_index = len(pair_rows)
        while not done:
            previous, old_alive = env.positions.copy(), env.alive.copy()
            old_collision, old_oob = env.collision_events.copy(), env.oob_losses.copy()
            actions = np.stack([policies[t].act(env, t) for t in range(2)])
            _, _, done, _, info = env.step(actions)
            digest_step(digest, actions, env, info)
            valid = post_oob_mask(old_alive, env.positions, cfg.environment.map_size)
            np.testing.assert_array_equal((old_alive & ~valid).sum(axis=1), env.oob_losses - old_oob)
            assert not np.any(env.alive & ~valid)
            stats['post_oob_pre_weapon_deaths_included'] += int(np.sum(valid & ~env.alive))
            stats['oob_excluded'] += int(np.sum(old_alive & ~valid))
            pairs = pair_geometry(previous, env.positions, valid, cfg.environment.collision_distance)
            sets = participant_sets(pairs)
            per_team = {key: np.array([sum(i // 5 == t for i in ids) for t in range(2)])
                        for key, ids in sets.items()}
            expected_increment = env.collision_events - old_collision
            assert np.array_equal(per_team['swept'], expected_increment), (seed, env.step_count, 'collision mismatch')
            stats['audited_steps'] += 1
            assert len(pairs) == sum(p['endpoint'] for p in pairs) + sum(not p['endpoint'] for p in pairs)
            assert len(pairs) == sum(sum(p['pair_type'] == rel for p in pairs) for rel in ('RR', 'BB', 'RB'))
            stats['pair_partition_checks'] += 1
            history.append([i in sets['swept'] for i in range(10)])
            for key in counts:
                counts[key] += per_team[key]
            np.testing.assert_array_equal(per_team['same'] + per_team['cross'] - per_team['both'], per_team['all'])
            for t in range(2):
                row = dict(seed=seed, step=env.step_count, team='Red' if t == 0 else 'Blue')
                for key in ('same', 'cross', 'both', 'all', 'endpoint', 'swept_only', 'endpoint_swept_only_overlap'):
                    row[key + '_set'] = json.dumps(sorted(i % 5 for i in sets[key] if i // 5 == t))
                set_rows.append(row)
            for p in pairs:
                row = dict(seed=seed, step=env.step_count, **p,
                           first_team='Red' if p['first'] < 5 else 'Blue', first_id=p['first'] % 5,
                           second_team='Red' if p['second'] < 5 else 'Blue', second_id=p['second'] % 5)
                pair_rows.append(row)
                pair_counts[p['pair_type'] + ('_endpoint' if p['endpoint'] else '_swept_only')] += 1
            for key in COMPONENTS:
                actual[key] += np.array(info['team_reward_components'][key])
            actual_returns += np.array(info['team_rewards'])
        row = {**info['episode_summary'], 'timed_out': env.step_count >= cfg.environment.episode_steps}
        assert row == references[episode], f'Episode summary differs: seed={seed}'
        assert digest.hexdigest() == checksums[episode], f'Trajectory/RNG differs: seed={seed}'
        stats['exact_trajectory_replay_episodes'] += 1
        for key in COMPONENTS:
            assert abs(actual[key][0] - row[key]) < 1e-9
        history = np.array(history, dtype=bool)
        ep_bursts, ep_team_bursts = [], []
        for agent in range(10):
            for r in runs(history[:, agent]):
                ep_bursts.append(dict(seed=seed, team='Red' if agent < 5 else 'Blue', agent_id=agent % 5, **r))
        for t in range(2):
            for r in runs(history[:, t * 5:(t + 1) * 5].any(axis=1)):
                ep_team_bursts.append(dict(seed=seed, team='Red' if t == 0 else 'Blue', **r))
        burst_rows.extend(ep_bursts)
        team_bursts.extend(ep_team_bursts)
        assert sum(r['length'] for r in ep_bursts) == int(counts['swept'].sum())
        ep = dict(seed=seed, episode_length=env.step_count, summary=row,
                  participant_events={k: v.tolist() for k, v in counts.items()},
                  pair_counts={rel: {kind: pair_counts[rel + '_' + kind] for kind in ('endpoint', 'swept_only')}
                               for rel in ('RR', 'BB', 'RB')},
                  burst_count=[sum(r['team'] == label for r in ep_bursts) for label in ('Red', 'Blue')],
                  max_burst_length=[max([r['length'] for r in ep_bursts if r['team'] == label], default=0)
                                    for label in ('Red', 'Blue')],
                  pair_row_range=[first_pair_index, len(pair_rows)], rewards={})
        for t, label in enumerate(('Red', 'Blue')):
            mapped = recompute({k: float(v[t]) for k, v in actual.items()},
                               int(counts['swept'][t]), int(counts['endpoint'][t]), int(counts['same_endpoint'][t]))
            collision_error = abs(mapped['Current']['collision_reward'] - actual['collision_reward'][t])
            total_error = abs(mapped['Current']['total'] - actual_returns[t])
            accounting_errors.extend([collision_error, total_error])
            assert collision_error < 1e-9 and total_error < 1e-9
            if t == 0:
                assert abs(mapped['Current']['total'] - row['return']) < 1e-9
            ep['rewards'][label] = mapped
            for scheme, components in mapped.items():
                reward_rows.append(dict(seed=seed, team=label, scheme=scheme, **components))
        episodes.append(ep)
        if (episode + 1) % 10 == 0:
            print(f'Audit {episode + 1}/100 | steps={stats["audited_steps"]} | '
                  f'pair-steps={len(pair_rows)} | mismatch=0 | reward_error={max(accounting_errors):g}', flush=True)
    env.close()
    assert before == protected_hashes(), 'Protected source/config changed'
    pair_table = {rel: {kind: sum(ep['pair_counts'][rel][kind] for ep in episodes)
                        for kind in ('endpoint', 'swept_only')} for rel in ('RR', 'BB', 'RB')}
    total_pairs = len(pair_rows)
    assert total_pairs == sum(sum(v.values()) for v in pair_table.values())
    participant_totals = {key: np.sum([ep['participant_events'][key] for ep in episodes], axis=0).tolist()
                          for key in counts}
    per_episode, bursts_out, teams_out, rewards_out = {}, {}, {}, {}
    for t, label in enumerate(('Red', 'Blue', 'Combined')):
        belongs = lambda r: t == 2 or r['team'] == label
        bursts_out[label] = burst_summary([r for r in burst_rows if belongs(r)], 100)
        teams_out[label] = burst_summary([r for r in team_bursts if belongs(r)], 100)
        def select(v):
            return sum(v) if t == 2 else v[t]
        metrics = {key: [select(ep['participant_events'][key]) for ep in episodes]
                   for key in KINDS}
        metrics['cross_team_endpoint_pairs'] = [ep['pair_counts']['RB']['endpoint'] for ep in episodes]
        metrics['burst_count'] = [select(ep['burst_count']) for ep in episodes]
        metrics['max_burst_length'] = [max(ep['max_burst_length']) if t == 2 else ep['max_burst_length'][t]
                                       for ep in episodes]
        per_episode[label] = {k: distribution(v) for k, v in metrics.items()}
        if t == 2:
            continue
        rewards_out[label] = {}
        for scheme in ('Current', 'Liu-style', 'Gao-base'):
            entries = [ep['rewards'][label][scheme] for ep in episodes]
            absolute = sum(sum(abs(e[k]) for k in COMPONENTS) for e in entries)
            rewards_out[label][scheme] = dict(
                components_mean={k: float(np.mean([e[k] for e in entries])) for k in (*COMPONENTS, 'total')},
                collision_reward_distribution=distribution([e['collision_reward'] for e in entries]),
                pooled_absolute_component_share=sum(abs(e['collision_reward']) for e in entries) / absolute if absolute else 0.,
                mean_episode_collision_absolute_share=float(np.mean([e['collision_absolute_share'] for e in entries])))
    cases = {}
    def pair_sum(ep, rels=('RR', 'BB', 'RB'), kind=None):
        return sum(sum(ep['pair_counts'][rel].values()) if kind is None else ep['pair_counts'][rel][kind] for rel in rels)
    for label, score in [
        ('A_most_participant_events', lambda ep: sum(ep['participant_events']['swept'])),
        ('B_highest_swept_only_pair_share', lambda ep: pair_sum(ep, kind='swept_only') / pair_sum(ep) if pair_sum(ep) else 0.),
        ('C_most_cross_team_pairs', lambda ep: pair_sum(ep, ('RB',))),
        ('D_most_same_team_pairs', lambda ep: pair_sum(ep, ('RR', 'BB'))),
        ('E_longest_uav_burst', lambda ep: max(ep['max_burst_length']))]:
        chosen = max(episodes, key=lambda ep: (score(ep), -ep['seed']))
        start, stop = chosen['pair_row_range']
        available = pair_rows[start:stop]
        if label.startswith('B'):
            available = sorted(available, key=lambda p: (p['endpoint'], p['step'], p['first'], p['second']))
        elif label.startswith('C') or label.startswith('D'):
            cross = label.startswith('C')
            available = sorted(available, key=lambda p: ((p['pair_type'] == 'RB') != cross, p['step']))
        elif label.startswith('E'):
            longest = max((r for r in burst_rows if r['seed'] == chosen['seed']), key=lambda r: r['length'])
            agent = longest['agent_id'] + (0 if longest['team'] == 'Red' else 5)
            available = [p for p in available if agent in (p['first'], p['second'])
                         and longest['start_step'] <= p['step'] <= longest['end_step']]
        cases[label] = dict(score=score(chosen), episode=chosen, representative_pairs=available[:5])
    payload = dict(definitions={
        'threshold_m': 50, 'std': 'population ddof=0', 'percentiles': 'numpy linear',
        'pair_unit': 'pair-step, not distinct physical pair or encounter',
        'participant_unit': 'unique UAV per step',
        'swept_only_participant': 'union of endpoints of swept-only pairs; overlaps endpoint participants',
        'exclusive_swept_only_participant': 'swept minus endpoint (derive by subtraction)',
        'burst': 'contiguous current swept participant events, no bridging zero/death/episode gaps',
        'team_burst': 'contiguous team any-collision steps; Combined pools Red/Blue runs, not OR of teams',
        'diagnostics': 'Liu endpoint/all teams/-0.2; Gao endpoint/same team/-5; both 50m, actual other rewards'},
        protected_sha256_before=before, protected_sha256_after=protected_hashes(),
        validation=dict(**stats, step_mismatches=0, pair_partition_mismatches=0,
                        max_absolute_reward_accounting_error=max(accounting_errors),
                        saved_sanity_summary_exact_match=True, exact_episode_summary_matches=100,
                        baseline_and_audit_unique_seeds=100),
        basic_results=reproduced, baseline_episode_summaries=references,
        baseline_trajectory_sha256=checksums, pair_table=pair_table, current_swept_pair_steps=total_pairs,
        participant_totals=participant_totals, burst=bursts_out, team_burst=teams_out,
        per_episode_distributions=per_episode, reward=rewards_out, cases=cases, episodes=episodes)
    output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    for filename, rows in [('collision_pairs.csv', pair_rows), ('collision_participant_sets.csv', set_rows),
                           ('collision_bursts.csv', burst_rows), ('collision_team_bursts.csv', team_bursts),
                           ('episode_reward_recompute.csv', reward_rows)]:
        write_csv(output.parent / filename, rows)
    print(json.dumps({k: payload[k] for k in ('validation', 'pair_table', 'participant_totals')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/v2_3_greedy_shooter/collision_audit.json')
    args = parser.parse_args()
    run(args.output)
