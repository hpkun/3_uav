"""Current audits and replay must never equate a geometric window with a kill."""
import csv
from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import behavioral_audit
from replay_html import classify_events
from uav_mappo_2d.config import ProjectConfig


def test_audit_geometry_ignores_beta(make_env):
    env = make_env()
    for beta in (-math.pi, -math.pi/2, 0, math.pi/2, math.pi):
        assert behavioral_audit.legal(env, (1000, math.pi/2, beta))
        assert not behavioral_audit.legal(env, (1000.001, 0, beta))
        assert not behavioral_audit.legal(env, (1000, math.pi/2 + 1e-6, beta))


def test_audit_miss_windows_are_attempts_not_weapon_deaths(tmp_path, monkeypatch):
    cfg = ProjectConfig()
    cfg.environment = replace(cfg.environment, hit_probability=0, episode_steps=30)
    monkeypatch.setattr(behavioral_audit, 'load_config', lambda _: cfg)
    path = tmp_path / 'steps.csv'
    report = behavioral_audit.audit('greedy-vs-greedy', 1, 200000, path)
    assert report['counts']['blue_attack_attempt_agent_steps'] > 0
    assert report['counts']['blue_attack_window'] > 0
    assert report['counts']['blue_combat_kill'] == 0
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    attempts = [row for row in rows if row['attack_attempt'] == 'True']
    assert attempts
    assert all(row['target_weapon_death_confirmed'] == 'False' for row in attempts)
    assert all(row['successful_attacker_id'] == '' for row in rows)


def test_replay_alive_transition_confirms_miss_without_weapon_victim():
    alive = np.zeros((2, 5), dtype=bool)
    alive[:, 0] = True
    positions = np.zeros((2, 5, 2))
    positions[1, 0] = (500, 0)
    assert classify_events(alive, positions, alive)['weapon_victims'] == []


def test_html_attack_sector_uses_config_and_explains_windows():
    template = (Path(__file__).resolve().parents[1] / 'scripts/replay_template.html').read_text(encoding='utf-8')
    assert 'meta.config.environment.attack_angle_degrees' in template
    assert 'heading-halfAngle,heading+halfAngle' in template
    assert '几何' in template and '70%' in template
