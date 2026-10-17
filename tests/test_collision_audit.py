"""V2.3 historical swept mathematics plus explicit V2.4 mapping checks.

Historical Current remains swept/-5; it is not the current production reward.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collision_audit import (pair_geometry, participant_sets, post_oob_mask,
                             recompute, runs, burst_summary)


def scene(entities):
    positions = np.full((2, 5, 2), 10000.)
    valid = np.zeros((2, 5), dtype=bool)
    for i, xy in entities.items():
        positions.reshape(-1, 2)[i] = xy
        valid.ravel()[i] = True
    return positions, valid


@pytest.mark.parametrize('ids,relation', [((0, 1), 'RR'), ((5, 6), 'BB'), ((0, 5), 'RB')])
def test_endpoint_relation_and_inclusive_threshold(ids, relation):
    positions, valid = scene({ids[0]: (0, 0), ids[1]: (50, 0)})
    pairs = pair_geometry(positions, positions, valid)
    assert len(pairs) == 1 and pairs[0]['endpoint'] and pairs[0]['pair_type'] == relation
    assert pairs[0]['tau'] == 0


def test_swept_only_crossing():
    previous, valid = scene({0: (-100, 0), 5: (100, 0)})
    current, _ = scene({0: (100, 0), 5: (-100, 0)})
    pair = pair_geometry(previous, current, valid)[0]
    assert pair['distance_swept'] == 0 and pair['distance_endpoint'] == 200
    assert pair['tau'] == .5 and not pair['endpoint']


def test_no_collision():
    positions, valid = scene({0: (0, 0), 5: (50.00001, 0)})
    assert pair_geometry(positions, positions, valid) == []


def test_participant_dedup_and_same_cross_overlap():
    positions, valid = scene({0: (0, 0), 1: (30, 0), 5: (60, 0)})
    sets = participant_sets(pair_geometry(positions, positions, valid))
    assert sets['same'] == {0, 1} and sets['cross'] == {1, 5}
    assert sets['both'] == {1} and sets['swept'] == {0, 1, 5}
    assert sets['same_endpoint'] == {0, 1}
    assert len(sets['same']) + len(sets['cross']) - len(sets['both']) == len(sets['all'])


def test_endpoint_and_swept_only_participants_can_overlap():
    sets = participant_sets([dict(first=0, second=1, pair_type='RR', endpoint=True),
                             dict(first=0, second=5, pair_type='RB', endpoint=False)])
    assert sets['endpoint_swept_only_overlap'] == {0}
    assert sets['swept_only'] - sets['endpoint'] == {5}


def test_burst_lengths_and_buckets():
    result = runs([False, True, True, True, False, True, False, True, True, True, True])
    assert [r['length'] for r in result] == [3, 1, 4]
    assert result[-1] == dict(start_step=8, end_step=11, length=4)
    summary = burst_summary(result, 1)
    assert summary['burst_count'] == 3
    assert summary['bins']['1']['participant_events'] == 1
    assert summary['bins']['2-3']['participant_events'] == 3
    assert summary['bins']['>=4']['participant_events'] == 4
    assert runs([]) == [] and runs([False]) == []


def test_oob_excluded_but_boundary_included():
    previous, old_alive = scene({0: (4990, 0), 5: (4990, 10), 6: (5000, 10)})
    current = previous.copy()
    current[0, 0] = (5001, 0)
    valid = post_oob_mask(old_alive, current, 10000)
    assert not valid[0, 0] and valid[1, 1]
    pairs = pair_geometry(previous, current, valid)
    assert len(pairs) == 1 and pairs[0]['pair_type'] == 'BB'


def components():
    return dict(kill_reward=20., weapon_loss_reward=-10., collision_reward=-50.,
                oob_reward=-20., terminal_reward=20.)


def test_liu_recompute_and_other_components_unchanged():
    actual = components()
    result = recompute(actual, 10, 4, 2)
    assert result['Current']['collision_reward'] == -50  # V2.3 historical meaning
    assert result['Liu-style']['collision_reward'] == -.8
    assert result['Liu-style']['total'] == pytest.approx(9.2)
    assert actual == components()
    for scheme in result.values():
        for key in actual:
            if key != 'collision_reward':
                assert scheme[key] == actual[key]


def test_gao_recompute():
    result = recompute(components(), 10, 4, 2)
    assert result['Gao-base']['collision_reward'] == -10
    assert result['Gao-base']['total'] == 0
    assert result['Current']['collision_absolute_share'] == 50 / 120


def test_zero_denominator_absolute_share():
    actual = {k: 0. for k in components()}
    assert all(r['collision_absolute_share'] == 0 for r in recompute(actual, 0, 0, 0).values())


def test_v24_production_matches_historical_liu_counterfactual(make_env, zero_actions):
    env = make_env((0, 0), (0, 1), (1, 0))
    env.positions[0, 0], env.positions[0, 1], env.positions[1, 0] = (0, 0), (30, 0), (4000, 4000)
    env.headings.fill(0)
    env.speeds.fill(100)
    previous, old_alive = env.positions.copy(), env.alive.copy()
    _, _, _, _, info = env.step(zero_actions)
    valid = post_oob_mask(old_alive, env.positions, env.config.map_size)
    sets = participant_sets(pair_geometry(previous, env.positions, valid))
    assert sets['swept'] == {0, 1}
    actual = {k: info[k] for k in components()}
    mapped = recompute(actual, 2, 2, 2)
    assert mapped['Current']['collision_reward'] == -10  # V2.3 historical mapping
    assert mapped['Liu-style']['collision_reward'] == info['collision_reward'] == -.4
    assert mapped['Liu-style']['total'] == info['total_reward']


def test_pre_weapon_mask_keeps_weapon_victims(make_env, zero_actions):
    env = make_env((0, 0), (1, 0),hit_probability=1.)
    env.positions[0, 0], env.positions[1, 0] = (0, 0), (20, 0)
    env.headings.fill(0)
    env.speeds.fill(100)
    previous, old_alive = env.positions.copy(), env.alive.copy()
    env.step(zero_actions)
    valid = post_oob_mask(old_alive, env.positions, env.config.map_size)
    assert valid[1, 0] and not env.alive[1, 0]
    assert participant_sets(pair_geometry(previous, env.positions, valid))['swept'] == {0, 5}


def test_geometry_does_not_mutate_inputs():
    positions, valid = scene({0: (0, 0), 5: (20, 0)})
    saved, saved_mask = positions.copy(), valid.copy()
    pair_geometry(positions, positions, valid)
    np.testing.assert_array_equal(saved, positions)
    np.testing.assert_array_equal(saved_mask, valid)
