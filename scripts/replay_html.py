"""Record one V2.4 MAPPO-vs-Greedy episode as an offline, single-file HTML.

No renderer, optimizer, production changes, external resources or sidecar files.
frames[t] is s_t: incoming reward/events refer to t-1 -> t, while its RL fields
refer to the pre-action state s_t and action t -> t+1. Terminal RL fields are null.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import random

import numpy as np
import torch

from uav_mappo_2d.config import load_config
from uav_mappo_2d.env import UAVCombatEnv
from uav_mappo_2d.evaluate import resolve_device
from uav_mappo_2d.mappo import MAPPO
from uav_mappo_2d.opponents import RuleBasedOpponent


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name('replay_template.html')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def world_to_canvas(x, y, width=1000, height=1000, padding=58):
    """Fixed 10km square, centered equal scale; positive world y points upward."""
    side = min(width, height) - 2 * padding
    if side <= 0:
        raise ValueError('Canvas must be larger than twice its padding')
    return ((width - side) / 2 + (x + 5000) / 10000 * side,
            (height - side) / 2 + (5000 - y) / 10000 * side)


def classify_events(old_alive, positions, new_alive, map_size=10000., threshold=50.):
    """Reconstruct identities outside env, respecting OOB -> collision -> weapon."""
    old_alive, new_alive = np.asarray(old_alive, bool), np.asarray(new_alive, bool)
    positions = np.asarray(positions)
    inside = np.all(np.abs(positions) <= map_size / 2, axis=-1)
    valid = old_alive & inside  # post-OOB/pre-weapon, not new_alive
    if np.any(new_alive & ~valid):
        raise AssertionError('An OOB/dead UAV unexpectedly survived or resurrected')
    oob, weapon = old_alive & ~inside, valid & ~new_alive
    first, second = np.triu_indices(old_alive.size, 1)
    keep = valid.ravel()[first] & valid.ravel()[second]
    first, second = first[keep], second[keep]
    flat = positions.reshape(-1, 2)
    collided = np.sum((flat[first] - flat[second])**2, axis=1) <= threshold**2
    pairs = list(zip(first[collided].tolist(), second[collided].tolist()))
    participants = sorted({i for pair in pairs for i in pair})
    def label(index):
        return f'{"R" if index < 5 else "B"}{index % 5}'
    def ids(mask):
        return [label(int(i)) for i in np.flatnonzero(mask.ravel())]
    return dict(oob_victims=ids(oob), weapon_victims=ids(weapon),
                collision_participants=[label(i) for i in participants],
                collision_pairs=[[label(i) for i in pair] for pair in pairs])


def trajectory_points(frames, team, agent, stop=None):
    """No future samples; include the death endpoint once, then stop."""
    points = []
    for frame in frames[:None if stop is None else stop + 1]:
        if not points and not frame['alive'][team][agent]:
            break
        points.append(frame['positions'][team][agent])
        if not frame['alive'][team][agent]:
            break
    return points


def _frame(env, info, cumulative, events):
    return dict(step=env.step_count, time=env.step_count * env.config.dt,
                positions=env.positions.tolist(), headings=env.headings.tolist(),
                speeds=env.speeds.tolist(), alive=env.alive.tolist(),
                counts={k: info[k] for k in UAVCombatEnv.EPISODE_METRICS},
                team_rewards=list(info['team_rewards']),
                reward_components={k: list(info['team_reward_components'][k]) for k in UAVCombatEnv.REWARD_COMPONENTS},
                cumulative_red_reward=float(cumulative), events=events,
                actions_red=None, actor_mu_red=None, actor_std=None,
                actor_active_red=None, critic_value=None, rl_available=False)


@torch.no_grad()
def collect_trace(config, policy, seed, *, deterministic=True):
    """Use the exact masked policy_actions call used by formal evaluate.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env = UAVCombatEnv(config.environment, config.reward, config.render)
    blue = RuleBasedOpponent()
    observations, info = env.reset(seed=seed)
    frames = [_frame(env, info, 0., None)]
    cumulative, done = 0., False
    try:
        while not done:
            active = env.alive[0].copy()
            # Never replace formal deterministic evaluation with a new controller.
            actions = policy.policy_actions(observations[0][None], info['active_masks'][0][None],
                                             deterministic=deterministic)[0]
            distribution = policy.actor.distribution(torch.as_tensor(observations[0], device=policy.device))
            mu = distribution.mean.cpu().numpy()
            std = distribution.stddev[0].cpu().numpy()
            value = float(policy.values(info['global_state'][None])[0])
            assert np.isfinite(actions).all() and np.isfinite(mu).all() and np.isfinite(std).all() and np.isfinite(value)
            frames[-1].update(actions_red=actions.tolist(),
                              actor_mu_red=[mu[i].tolist() if active[i] else [None, None] for i in range(5)],
                              actor_std=std.tolist(), actor_active_red=active.tolist(),
                              critic_value=value, rl_available=True)
            old_alive = env.alive.copy()
            old_counts = frames[-1]['counts']
            observations, _, done, _, info = env.step(np.stack([actions, blue.act(env, 1)]))
            events = classify_events(old_alive, env.positions, env.alive,
                                     config.environment.map_size, config.environment.collision_distance)
            events['increments'] = {k: int(info[k] - old_counts[k]) for k in UAVCombatEnv.EPISODE_METRICS
                                    if k not in ('red_alive', 'blue_alive')}
            for prefix, letter, other in [('red', 'R', 'blue'), ('blue', 'B', 'red')]:
                inc = events['increments']
                assert sum(v.startswith(letter) for v in events['oob_victims']) == inc[prefix + '_oob']
                assert sum(v.startswith(letter) for v in events['weapon_victims']) == inc[prefix + '_weapon_loss']
                assert inc[prefix + '_weapon_loss'] == inc[other + '_combat_kill']
                assert sum(v.startswith(letter) for v in events['collision_participants']) == inc[prefix + '_collision_events']
            assert abs(sum(info[k] for k in UAVCombatEnv.REWARD_COMPONENTS) - info['total_reward']) < 1e-9
            cumulative += float(info['total_reward'])
            frames.append(_frame(env, info, cumulative, events))
        summary = dict(info['episode_summary'])
        assert len(frames) == summary['episode_length'] + 1
        assert abs(cumulative - summary['return']) < 1e-9
    finally:
        env.close()
    return dict(metadata=dict(environment_version='V2.4', seed=seed, deterministic=deterministic,
                              config=asdict(config),
                              actor_log_std_raw=policy.actor.log_std.detach().cpu().tolist(),
                              actor_log_std_effective=policy.actor.log_std.detach().clamp(-5., 1.).cpu().tolist(),
                              actor_std_semantics='global learnable std, clamped log_std then exp; not state-dependent',
                              frame_alignment='state s_t with incoming reward/events (t-1->t); RL is pre-action (t->t+1); terminal RL is null'),
                initial_state=frames[0], frames=frames, episode_summary=summary)


def trace_json(trace):
    # Escape script terminators in checkpoint names/metadata; no eval or network.
    return json.dumps(trace, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace(
        '<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def render_html(trace):
    title = f'V2.4 Replay · {trace["metadata"].get("checkpoint_filename", "checkpoint")} · seed {trace["metadata"]["seed"]}'
    template = TEMPLATE.read_text(encoding='utf-8')
    return template.replace('__PAGE_TITLE__', html.escape(title)).replace('__TRACE_JSON__', trace_json(trace))


def write_html(trace, output):
    output = Path(output)
    if output.suffix.lower() != '.html':
        raise ValueError('--output must end in .html')
    document = render_html(trace)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite existing results/checkpoints; reruns need a new filename.
    with output.open('x', encoding='utf-8') as stream:
        stream.write(document)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seed', type=int, default=3000000)
    parser.add_argument('--device', default=None, help='cuda/cpu/auto; default follows config')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--stochastic', action='store_true', help='Seeded sampling instead of deterministic tanh(mean)')
    return parser


def main():
    args = build_parser().parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    output = args.output or Path('runs/replays') / f'{checkpoint.stem}_seed{args.seed}.html'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing replay: {output}')
    if output.suffix.lower() != '.html':
        raise ValueError('--output must end in .html')
    config = load_config(args.config)
    if config.environment.map_size != 10000.:
        raise ValueError('This fixed V2.4 replay uses a 10000m square battlefield')
    source_paths = sorted((ROOT / 'src/uav_mappo_2d').glob('*.py')) + [Path(args.config), checkpoint]
    original_hashes = {str(p): sha256(p) for p in source_paths}
    device = resolve_device(args.device or config.training.device)
    policy = MAPPO(UAVCombatEnv.OBS_DIM, UAVCombatEnv.STATE_DIM, UAVCombatEnv.N_AGENTS,
                   config.mappo, device, seed=config.seed)
    payload = policy.load(checkpoint, load_optimizers=False)
    policy.actor.eval()
    policy.critic.eval()
    trace = collect_trace(config, policy, args.seed, deterministic=not args.stochastic)
    trace['metadata'].update(checkpoint=str(checkpoint), checkpoint_filename=checkpoint.name,
                             checkpoint_global_step=payload.get('global_step'), checkpoint_sha256=sha256(checkpoint),
                             config_file=str(Path(args.config).resolve()), device=str(policy.device))
    assert original_hashes == {str(p): sha256(p) for p in source_paths}, 'A read-only input changed'
    path = write_html(trace, output)
    s = trace['episode_summary']
    print(f'HTML: {path.resolve()} | bytes={path.stat().st_size}')
    print(f'checkpoint={checkpoint.name} global_step={payload.get("global_step", "unknown")} '
          f'seed={args.seed} mode={"stochastic" if args.stochastic else "deterministic"}')
    print(f'frames={len(trace["frames"])} length={s["episode_length"]} result={s["result"]} return={s["return"]:.6f} '
          f'kills(R/B)={s["red_combat_kill"]}/{s["blue_combat_kill"]} OOB(R/B)={s["red_oob"]}/{s["blue_oob"]}')
    print('Read-only source/config/checkpoint hashes verified; no training or browser launch.')


if __name__ == '__main__':
    main()
