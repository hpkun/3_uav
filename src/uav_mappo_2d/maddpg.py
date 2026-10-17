"""Vanilla five-agent MADDPG; no parameter sharing, PPO or TD3 components."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .maddpg_config import MADDPGConfig
from .maddpg_models import MADDPGActor, MADDPGCritic


ACTOR_DIAGNOSTICS = ('actor_reg', 'pre_tanh_abs_mean', 'pre_tanh_abs_max',
                     'tanh_derivative_mean', 'accel_saturation', 'turn_saturation')
ACTOR_AGGREGATES = dict(zip(
    ('mean_actor_reg', 'mean_pre_tanh_abs', 'max_pre_tanh_abs', 'mean_tanh_derivative',
     'mean_accel_saturation', 'mean_turn_saturation'), ACTOR_DIAGNOSTICS))


@torch.no_grad()
def actor_diagnostics(z, action, reg):
    """Read-only current Actor, active Replay rows, before optimizer.step (not rollout)."""
    z, action = z.detach(), action.detach()
    return dict(actor_reg=float(reg.detach()), pre_tanh_abs_mean=float(z.abs().mean()),
                pre_tanh_abs_max=float(z.abs().max()),
                tanh_derivative_mean=float((1-action.square()).mean()),
                accel_saturation=float((action[:,0].abs() >= .95).float().mean()),
                turn_saturation=float((action[:,1].abs() >= .95).float().mean()))


class JointReplayBuffer:
    SHAPES = {'observations': (5,68), 'states': (60,), 'actions': (5,2), 'rewards': (),
              'next_observations': (5,68), 'next_states': (60,), 'dones': (),
              'active_masks': (5,), 'next_active_masks': (5,)}

    def __init__(self, capacity=1_000_000, seed=0):
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity, self.size, self.position = int(capacity), 0, 0
        self.rng = np.random.default_rng(seed)
        self.data = {k: np.empty((capacity, *shape), dtype=np.float32) for k,shape in self.SHAPES.items()}

    def __len__(self):
        return self.size

    def add(self, **transition):
        if set(transition) != set(self.SHAPES):
            raise ValueError('A complete joint transition is required')
        for key, shape in self.SHAPES.items():
            value = np.asarray(transition[key], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid replay field: {key}')
            self.data[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size or batch_size < 1:
            raise ValueError('Not enough replay transitions')
        indices = self.rng.integers(self.size, size=batch_size)  # uniform with replacement, shared for all agents
        return {k: value[indices].copy() for k,value in self.data.items()}


class OUNoise:
    def __init__(self, num_envs, config, seed=0):
        self.config, self.rng = config, np.random.default_rng(seed)
        self.state = np.full((num_envs,5,2), config.ou_mu, dtype=np.float32)

    def sample(self):
        c = self.config
        self.state += c.ou_theta * (c.ou_mu - self.state) + c.ou_sigma * self.rng.standard_normal(self.state.shape)
        return self.state.copy()

    def reset(self, environments):
        self.state[environments] = self.config.ou_mu


def td_target(rewards, dones, next_active, next_q, gamma):
    return (rewards + gamma * (1 - dones) * next_active * next_q).detach()


def clip_parameter_gradients(parameters, limit):
    """Match official TF per-variable clip_by_norm; return total PRE-clip norm."""
    norms = [nn.utils.clip_grad_norm_([p], limit) for p in parameters if p.grad is not None]
    return float(torch.stack(norms).square().sum().sqrt()) if norms else 0.


class MADDPG:
    def __init__(self, config=None, device='cpu', seed=0):
        self.config, self.device = config or MADDPGConfig(), torch.device(device)
        torch.manual_seed(seed)
        self.actors = nn.ModuleList([MADDPGActor(hidden_size=self.config.hidden_size) for _ in range(5)]).to(self.device)
        self.critics = nn.ModuleList([MADDPGCritic(hidden_size=self.config.hidden_size) for _ in range(5)]).to(self.device)
        self.target_actors, self.target_critics = deepcopy(self.actors), deepcopy(self.critics)
        for target in (self.target_actors, self.target_critics):
            target.requires_grad_(False)
            target.eval()
        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=self.config.actor_learning_rate) for a in self.actors]
        self.critic_optimizers = [torch.optim.Adam(q.parameters(), lr=self.config.critic_learning_rate) for q in self.critics]
        self.update_count = 0

    @torch.no_grad()
    def policy_actions(self, observations, active_masks, *, deterministic=True):
        if not deterministic:
            raise ValueError('MADDPG is deterministic; training exploration uses actions(..., noise=...)')
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(active_masks, dtype=torch.float32, device=self.device)
        actions = torch.stack([a(obs[...,i,:]) for i,a in enumerate(self.actors)], dim=-2)
        return (actions * mask[...,None]).cpu().numpy()

    def actions(self, observations, active_masks, noise=None):
        actions = self.policy_actions(observations, active_masks)
        if noise is not None:
            actions = np.clip(actions + noise.sample(), -1., 1.)
        return (actions * np.asarray(active_masks)[...,None]).astype(np.float32)

    def tensor_batch(self, batch):
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in batch.items()}

    @torch.no_grad()
    def targets(self, batch):
        actions = torch.stack([a(batch['next_observations'][:,i]) for i,a in enumerate(self.target_actors)], dim=1)
        actions *= batch['next_active_masks'][...,None]
        return torch.stack([td_target(batch['rewards'], batch['dones'], batch['next_active_masks'][:,i],
                                      q(batch['next_states'], actions), self.config.gamma)
                            for i,q in enumerate(self.target_critics)], dim=1)

    def update_critic(self, i, batch, target):
        active = batch['active_masks'][:,i].bool()
        if not active.any():
            return None
        q = self.critics[i](batch['states'][active], batch['actions'][active].detach())
        y = target[active].detach()
        loss = (q - y).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite critic loss')
        opt = self.critic_optimizers[i]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = clip_parameter_gradients(self.critics[i].parameters(), self.config.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return dict(critic_loss=float(loss.detach()), q=float(q.detach().mean()),
                    target_q=float(y.mean()), td_abs=float((q.detach()-y).abs().mean()), critic_grad_norm=grad)

    def update_actor(self, i, batch):
        active = batch['active_masks'][:,i].bool()
        if not active.any():
            return None
        critic = self.critics[i]
        critic.requires_grad_(False)
        try:
            joint = batch['actions'][active].detach().clone()
            z = self.actors[i].pre_tanh(batch['observations'][active,i])
            action = torch.tanh(z)
            joint[:,i] = action  # only actor i has an action gradient
            pg_loss = -critic(batch['states'][active], joint).mean()
            # V1 tests the original distribution-parameter regularization intent
            # under our bounded tanh adaptation; it is not line-by-line equivalence.
            regularized = z if self.config.actor_regularization_mode == 'pre_tanh' else action
            reg = self.config.actor_regularization * regularized.square().mean()
            loss = pg_loss + reg
            diagnostics = actor_diagnostics(z, action, reg)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite actor loss')
            opt = self.actor_optimizers[i]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = clip_parameter_gradients(self.actors[i].parameters(), self.config.max_grad_norm)
            opt.step()
            return dict(actor_loss=float(loss.detach()), actor_grad_norm=grad,
                        policy_gradient_loss=float(pg_loss.detach()), **diagnostics)
        finally:
            critic.requires_grad_(True)

    @torch.no_grad()
    def soft_update(self):
        for online, target in ((self.actors,self.target_actors),(self.critics,self.target_critics)):
            for p,t in zip(online.parameters(),target.parameters()):
                t.mul_(1-self.config.tau).add_(p, alpha=self.config.tau)

    def update(self, batch):
        b = self.tensor_batch(batch)
        targets = self.targets(b)  # all targets computed before any optimization
        critics = [self.update_critic(i,b,targets[:,i]) for i in range(5)]
        actors = [self.update_actor(i,b) for i in range(5)]
        self.soft_update()
        self.update_count += 1
        metrics = {'updated_agents': sum(r is not None for r in actors)}
        for key in ('actor_loss','critic_loss','q','target_q','td_abs','actor_grad_norm','critic_grad_norm'):
            rows = actors if key in ('actor_loss','actor_grad_norm') else critics
            values = [r[key] for r in rows if r is not None]
            metrics['mean_' + key] = float(np.mean(values)) if values else 0.
        for i in range(5):
            for key,rows in [('actor_loss',actors),('critic_loss',critics)]:
                metrics[f'agent{i}_{key}'] = rows[i][key] if rows[i] is not None else 0.
            metrics[f'agent{i}_updated'] = int(actors[i] is not None)
        for aggregate, key in ACTOR_AGGREGATES.items():
            values = [row[key] for row in actors if row is not None]
            metrics[aggregate] = float((max if aggregate == 'max_pre_tanh_abs' else np.mean)(values)) if values else 0.
        for i, row in enumerate(actors):
            for key in ACTOR_DIAGNOSTICS:
                # Empty active sets retain the existing zero + updated=0 convention.
                metrics[f'agent{i}_{key}'] = row[key] if row is not None else 0.
        if not np.isfinite(list(metrics.values())).all():
            raise FloatingPointError(f'Non-finite MADDPG metrics: {metrics}')
        return metrics

    def save(self, path, *, global_step, environment_config, best_evaluation=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(algorithm='maddpg', global_step=int(global_step), update_count=self.update_count,
                        actors=[m.state_dict() for m in self.actors], critics=[m.state_dict() for m in self.critics],
                        target_actors=[m.state_dict() for m in self.target_actors],
                        target_critics=[m.state_dict() for m in self.target_critics],
                        actor_optimizers=[o.state_dict() for o in self.actor_optimizers],
                        critic_optimizers=[o.state_dict() for o in self.critic_optimizers],
                        maddpg_config=asdict(self.config), environment_config=environment_config,
                        best_evaluation=best_evaluation), path)

    def load(self, path, *, load_optimizers=True):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get('algorithm') != 'maddpg':
            raise ValueError('Not a MADDPG checkpoint (MAPPO checkpoints are incompatible)')
        payload['maddpg_config'] = dict(payload['maddpg_config'])
        payload['maddpg_config'].setdefault('actor_regularization_mode', 'bounded_action')
        if payload['maddpg_config'] != asdict(self.config):
            raise ValueError('MADDPG config must match the checkpoint')
        for key in ('actors','critics','target_actors','target_critics'):
            if len(payload[key]) != 5:
                raise ValueError('MADDPG checkpoint requires five independent networks per group')
            for m,state in zip(getattr(self,key),payload[key]):
                m.load_state_dict(state)
        if load_optimizers:
            for key in ('actor_optimizers','critic_optimizers'):
                for opt,state in zip(getattr(self,key),payload[key]):
                    opt.load_state_dict(state)
        self.update_count = int(payload.get('update_count',0))
        return payload
