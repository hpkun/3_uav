"""Synthetic unit fixtures only; no real trained checkpoint required."""
from dataclasses import replace
from html.parser import HTMLParser
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from replay_html import (build_parser, classify_events, collect_trace, main, render_html,
                         sha256, trace_json, trajectory_points, world_to_canvas, write_html)
from uav_mappo_2d.config import ProjectConfig, EnvironmentConfig, MAPPOConfig
from uav_mappo_2d.evaluate import evaluate_policy
from uav_mappo_2d.mappo import MAPPO


@pytest.fixture
def replay_fixture():
    cfg = ProjectConfig(environment=replace(EnvironmentConfig(), episode_steps=4),
                        mappo=replace(MAPPOConfig(), hidden_size=8))
    torch.manual_seed(24)
    policy = MAPPO(68, 60, 5, cfg.mappo, 'cpu', seed=24)
    policy.actor.eval()
    policy.critic.eval()
    return cfg, policy


@pytest.mark.parametrize('point,expected', [((-5000,5000),(58,58)),
    ((5000,-5000),(942,942)), ((0,0),(500,500))])
def test_world_to_canvas(point, expected):
    np.testing.assert_allclose(world_to_canvas(*point), expected)


def test_world_equal_scale_on_nonsquare_canvas():
    center = np.array(world_to_canvas(0, 0, 1200, 800))
    assert center.tolist() == [600, 400]
    dx = np.array(world_to_canvas(1000, 0, 1200, 800)) - center
    dy = np.array(world_to_canvas(0, 1000, 1200, 800)) - center
    assert dx[0] == pytest.approx(-dy[1])
    with pytest.raises(ValueError):
        world_to_canvas(0, 0, 100, 100)


def test_trace_frames_and_finite_pre_action_fields(replay_fixture):
    cfg, policy = replay_fixture
    trace = collect_trace(cfg, policy, 3000000)
    frames = trace['frames']
    assert len(frames) == trace['episode_summary']['episode_length'] + 1 == 5
    assert [f['step'] for f in frames] == list(range(5))
    assert [f['time'] for f in frames] == list(range(5))
    assert frames[0]['team_rewards'] == [0, 0] and frames[0]['events'] is None
    assert frames[-1]['critic_value'] is None and not frames[-1]['rl_available']
    assert frames[-1]['actions_red'] is None
    for f in frames[:-1]:
        assert np.isfinite(f['actions_red']).all() and np.isfinite(f['critic_value'])
        assert np.isfinite(np.asarray(f['actor_mu_red'])[f['actor_active_red']]).all()
        assert np.isfinite(f['actor_std']).all()
        np.testing.assert_allclose(f['actions_red'], np.tanh(f['actor_mu_red']), atol=1e-7)
        assert abs(sum(v[0] for v in f['reward_components'].values()) - f['team_rewards'][0]) < 1e-9
    assert frames[-1]['cumulative_red_reward'] == pytest.approx(trace['episode_summary']['return'])


def test_deterministic_repeated_trace_exact_and_parameters_read_only(replay_fixture):
    cfg, policy = replay_fixture
    original = {k: v.clone() for k, v in policy.actor.state_dict().items()}
    first = collect_trace(cfg, policy, 3000010)
    second = collect_trace(cfg, policy, 3000010)
    assert first == second
    assert all(torch.equal(v, policy.actor.state_dict()[k]) for k, v in original.items())


def test_stochastic_seeded_repeated_trace_exact(replay_fixture):
    cfg, policy = replay_fixture
    first = collect_trace(cfg, policy, 3000010, deterministic=False)
    assert first == collect_trace(cfg, policy, 3000010, deterministic=False)
    assert first['frames'][0]['actions_red'] != collect_trace(cfg, policy, 3000010)['frames'][0]['actions_red']


def test_same_semantics_as_formal_deterministic_evaluation(replay_fixture):
    cfg, policy = replay_fixture
    trace = collect_trace(cfg, policy, 3000000)
    official = evaluate_policy(cfg, policy, episodes=1, seed_start=3000000)
    summary = trace['episode_summary']
    assert official['mean_return'] == summary['return']
    assert official['mean_episode_length'] == summary['episode_length']
    assert official['red_alive'] == summary['red_alive']
    assert official['blue_alive'] == summary['blue_alive']


def event_scene():
    old = np.zeros((2,5), dtype=bool)
    old[0,:2], old[1,0] = True, True
    positions = np.zeros((2,5,2))
    positions[0,0], positions[0,1], positions[1,0] = (5001,0), (4990,0), (4999,0)
    new = old.copy()
    new[0,:2] = False
    return old, positions, new


def test_oob_and_weapon_deaths_correct_no_identity_guessing():
    old, positions, new = event_scene()
    events = classify_events(old, positions, new)
    assert events['oob_victims'] == ['R0'] and events['weapon_victims'] == ['R1']


def test_endpoint_collision_keeps_later_weapon_victim_excludes_oob():
    old, positions, new = event_scene()
    events = classify_events(old, positions, new)
    assert events['collision_pairs'] == [['R1', 'B0']]
    assert events['collision_participants'] == ['R1', 'B0']


def test_endpoint_inclusive_and_participant_dedup():
    alive = np.zeros((2,5), dtype=bool)
    alive[0,:2], alive[1,0] = True, True
    positions = np.zeros((2,5,2))
    positions[0,1], positions[1,0] = (50,0), (25,0)
    events = classify_events(alive, positions, alive)
    assert len(events['collision_pairs']) == 3
    assert events['collision_participants'] == ['R0', 'R1', 'B0']
    positions[1,0] = (200,0)
    assert classify_events(alive, positions, alive)['collision_participants'] == ['R0', 'R1']


def test_no_swept_only_collision_and_boundary_is_not_oob():
    alive = np.zeros((2,5), dtype=bool)
    alive[0,0], alive[1,0] = True, True
    positions = np.zeros((2,5,2))
    positions[0,0], positions[1,0] = (100,0), (-100,0)
    assert classify_events(alive, positions, alive)['collision_participants'] == []
    positions[0,0] = (5000,0)
    assert classify_events(alive, positions, alive)['oob_victims'] == []


def test_trajectories_stop_at_death_and_never_reveal_future():
    frames=[]
    for x, active in [(0,True),(1,True),(2,False),(999,False)]:
        frames.append(dict(positions=[[[x,0]]], alive=[[active]]))
    assert trajectory_points(frames,0,0) == [[0,0],[1,0],[2,0]]
    assert trajectory_points(frames,0,0,stop=1) == [[0,0],[1,0]]
    assert trajectory_points(frames[2:],0,0) == []


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags=[]
    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_json_roundtrip_and_script_terminators_escaped(replay_fixture):
    trace = collect_trace(*replay_fixture, 3000000)
    trace['metadata']['checkpoint_filename'] = '</script><img src="external">'
    encoded = trace_json(trace)
    assert '</script>' not in encoded
    assert json.loads(encoded) == trace
    doc = Document()
    doc.feed(render_html(trace))
    assert not any(tag == 'img' for tag, _ in doc.tags)
    assert sum(tag == 'script' for tag, _ in doc.tags) == 2


def test_html_self_contained_controls_and_labels(replay_fixture):
    trace = collect_trace(*replay_fixture, 3000000)
    document = render_html(trace)
    parsed = Document()
    parsed.feed(document)
    ids = {a['id'] for _,a in parsed.tags if 'id' in a}
    assert {'play','previous','next','timeline','speed','trails','selected-uav','radar',
            'attack','red-agent','event-log','value'} <= ids
    assert not any(tag in ('link','img','iframe') or 'src' in attrs or 'href' in attrs for tag,attrs in parsed.tags)
    assert all(s not in document for s in ('https://','http://','fetch(', 'XMLHttpRequest', '__TRACE_JSON__'))
    assert 'R0–R4' in document and 'B0–B4' in document
    assert "(i<5?'R':'B')+(i%5)" in document
    assert 'ctx.rotate(-f.headings[team][agent])' in document
    assert 'max-width:900px' in document and 'max-height:900px' not in document
    for speed in ('0.25','0.5','1','2','4','8'):
        assert f'value="{speed}"' in document
    assert 'ArrowLeft' in document and 'ArrowRight' in document and "event.code==='Space'" in document


def test_cli_defaults():
    args = build_parser().parse_args(['--checkpoint','best.pt'])
    assert not args.stochastic and args.seed == 3000000 and args.output is None


def test_output_single_file_and_no_overwrite(tmp_path,replay_fixture):
    trace = collect_trace(*replay_fixture,3000000)
    output = tmp_path/'replay.html'
    write_html(trace,output)
    assert list(tmp_path.iterdir()) == [output]
    with pytest.raises(FileExistsError):
        write_html(trace,output)
    with pytest.raises(ValueError):
        write_html(trace,tmp_path/'checkpoint.pt')


def test_cli_checkpoint_payload_metadata_and_file_unchanged(tmp_path,replay_fixture,monkeypatch):
    cfg,policy = replay_fixture
    checkpoint = tmp_path/'best_named999.pt'
    policy.save(checkpoint, global_step=123)
    original_hash = sha256(checkpoint)
    config = tmp_path/'config.yaml'
    config.write_text('environment:\n  episode_steps: 4\nmappo:\n  hidden_size: 8\n',encoding='utf-8')
    output = tmp_path/'recording.html'
    monkeypatch.setattr(sys,'argv',['replay_html.py','--config',str(config),'--checkpoint',str(checkpoint),
                                   '--seed','3000000','--device','cpu','--output',str(output)])
    main()
    document = output.read_text(encoding='utf-8')
    start = document.index('>',document.index('<script id="replay-data"'))+1
    data = json.loads(document[start:document.index('</script>',start)])
    assert data['metadata']['checkpoint_global_step'] == 123
    assert data['metadata']['checkpoint_filename'] == 'best_named999.pt'
    assert sha256(checkpoint) == original_hash
    assert not list(tmp_path.glob('*.json'))
