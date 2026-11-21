"""Masked Double DQN updates, deterministic inference, and resume state."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lidar_dqn import schema
from lidar_dqn_network import QNetwork
from lidar_dqn_replay import ReplayBuffer

VERSION = 1


def double_dqn_target(online_values, target_values, mask, reward, terminated, gamma):
    """Online selects the safe action, target evaluates that same action."""
    action = online_values.masked_fill(~mask, -torch.inf).argmax(dim=1, keepdim=True)
    estimate = target_values.gather(1, action).squeeze(1)
    return reward + gamma * torch.where(terminated, torch.zeros_like(estimate), estimate)


def atomic_save(data, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, temp)
    temp.replace(path)


def read_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get('version') != VERSION or saved.get('schema') != schema():
        raise ValueError('Incompatible Double DQN checkpoint schema.')
    return saved


def load_inference(path):
    saved = read_checkpoint(path)
    network = QNetwork(saved['config']['hidden'])
    network.load_state_dict(saved['online'])
    return network.eval().requires_grad_(False), saved


class DQNAgent:
    def __init__(self, config):
        self.config = config.copy()
        self.device = torch.device(config['device'])
        torch.manual_seed(config['seed'])
        self.rng = np.random.default_rng(config['seed'])
        self.online = QNetwork(config['hidden']).to(self.device)
        self.target = QNetwork(config['hidden']).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config['learning_rate'])
        self.replay = ReplayBuffer(config['capacity'])
        self.updates = self.target_updates = 0

    @torch.no_grad()
    def values(self, state):
        return self.online(torch.as_tensor(state, device=self.device)).cpu().numpy()

    def choose(self, state, mask, epsilon, *, q=None):
        valid = np.flatnonzero(mask)
        if not len(valid):
            raise ValueError('Action mask has no fallback.')
        if len(valid) == 1:
            self.choice_mode = 'forced_or_single_valid'
            return int(valid[0])
        if epsilon and self.rng.random() < epsilon:
            self.choice_mode = 'exploration'
            return int(self.rng.choice(valid))
        self.choice_mode = 'exploitation'
        return int(valid[np.argmax((self.values(state) if q is None else q)[valid])])

    def update(self, details=None):
        cfg = self.config
        if self.replay.size < max(cfg['warmup'], cfg['batch_size']):
            return None
        data = self.replay.sample(cfg['batch_size'], self.rng, self.device)
        q = self.online(data['state'])
        chosen = q.gather(1, data['action'][:, None]).squeeze(1)
        with torch.no_grad():
            following = self.online(data['next_state'])
            target_values = self.target(data['next_state'])
            target = double_dqn_target(
                following, target_values,
                data['next_mask'], data['reward'], data['terminated'], cfg['gamma'])
        loss = nn.functional.smooth_l1_loss(chosen, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite DQN loss.')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.online.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.updates % cfg['target_interval'] == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target_updates += 1
        stats = float(loss.detach()), float(q.detach().mean()), float(q.detach().max())
        if details is not None:
            with torch.no_grad():
                nxt=int(following[0].masked_fill(~data['next_mask'][0],-torch.inf).argmax())
                error=target-chosen
                details.update(loss=stats[0],batch_mean_q=stats[1],batch_max_q=stats[2],
                    online_q_value=float(chosen[0]),target_q_value=None if bool(data['terminated'][0]) else float(target_values[0,nxt]),
                    selected_next_action=None if bool(data['terminated'][0]) else nxt,
                    td_target=float(target[0]),td_error=float(error[0]),
                    batch_mean_td_target=float(target.mean()),batch_mean_td_error=float(error.mean()),
                    gradient_norm=float(norm),batch_sample_index=0,
                    gradient_update_count=self.updates,target_update_count=self.target_updates,
                    learning_rate=self.optimizer.param_groups[0]['lr'],replay_buffer_size=self.replay.size)
        return stats

    def save(self, path, progress, *, full=False):
        data = dict(version=VERSION, schema=schema(), config=self.config,
                    online=self.online.state_dict(), target=self.target.state_dict(),
                    optimizer=self.optimizer.state_dict(), progress=progress,
                    updates=self.updates, target_updates=self.target_updates)
        if full:
            data.update(replay=self.replay.state_dict(), rng=self.rng.bit_generator.state,
                        torch_rng=torch.random.get_rng_state())
        atomic_save(data, path)

    def resume(self, path):
        saved = read_checkpoint(path, self.device)
        if saved['config'] != self.config or 'replay' not in saved:
            raise ValueError('Resume requires the full checkpoint and its original configuration.')
        self.online.load_state_dict(saved['online'])
        self.target.load_state_dict(saved['target'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.replay.load_state_dict(saved['replay'])
        self.rng.bit_generator.state = saved['rng']
        torch.random.set_rng_state(saved['torch_rng'].cpu())
        self.updates, self.target_updates = saved['updates'], saved['target_updates']
        return saved['progress']
