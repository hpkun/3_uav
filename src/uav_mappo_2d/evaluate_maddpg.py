"""Deterministic MADDPG evaluation against the unchanged Greedy Blue."""
import argparse
import json
from pathlib import Path
import torch

from .config import load_config
from .evaluate import evaluate_policy, resolve_device
from .maddpg import MADDPG
from .maddpg_config import MADDPGConfig


def evaluate(config, policy, *, episodes, seed_start):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    # Reuse the existing deterministic environment rollout/metric aggregation,
    # not its MAPPO model construction. MADDPG implements only policy_actions.
    return evaluate_policy(config, policy, episodes=episodes, seed_start=seed_start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--seed-start', type=int, default=3000000)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg.training.device)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload.get('algorithm') != 'maddpg':
        parser.error('Checkpoint is not MADDPG')
    policy = MADDPG(MADDPGConfig(**payload['maddpg_config']), device, seed=cfg.seed)
    policy.load(args.checkpoint, load_optimizers=False)
    metrics = evaluate(cfg, policy, episodes=args.episodes, seed_start=args.seed_start)
    text = json.dumps(metrics, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
