"""Continuous Q-learning with normalized advantage functions (Gu et al., 2016)."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class NormalizedAdvantage(nn.Module):
    """Q(s,a) = V(s) - 1/2 ||L(s)^T (a - mu(s))||²."""

    def __init__(self):
        super().__init__()
        self.mean = nn.Sequential(nn.Linear(26, 64), nn.ReLU(), nn.Linear(64, 64),
                                  nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
        self.value_and_precision = nn.Sequential(nn.Linear(26, 64), nn.ReLU(),
                                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

    def components(self, observations):
        value, first, below, second = self.value_and_precision(observations).unbind(dim=-1)
        diagonal = F.softplus(torch.stack((first, second), dim=-1)) + 1e-3
        return value, diagonal, below

    def forward(self, observations, actions, freeze_mean=False):
        mean = self.mean(observations)
        if freeze_mean:
            mean = mean.detach()
        dx, dy = (actions - mean).unbind(dim=-1)
        value, diagonal, below = self.components(observations)
        # L = [[d1, 0], [below, d2]] with strictly positive diagonal.
        advantage = -.5 * ((diagonal[..., 0] * dx + below * dy).square()
                           + (diagonal[..., 1] * dy).square())
        return value + advantage

    def value(self, observations):
        return self.value_and_precision(observations)[..., 0]

    def action(self, observation):
        with torch.no_grad():
            return self.mean(torch.as_tensor(observation, dtype=torch.float32)).numpy()

    def initialize_mean(self, layers):
        """Initialize only mu; value and curvature are learned from rewards."""
        linear_layers = [layer for layer in self.mean if isinstance(layer, nn.Linear)]
        if len(layers) != len(linear_layers):
            raise ValueError('Incompatible teacher network depth')
        with torch.no_grad():
            for layer, data in zip(linear_layers, layers):
                layer.weight.copy_(torch.as_tensor(data['weight']))
                layer.bias.copy_(torch.as_tensor(data['bias']))

    def export_layers(self):
        return [dict(weight=layer.weight.detach().tolist(), bias=layer.bias.detach().tolist())
                for layer in self.mean if isinstance(layer, nn.Linear)]


class Replay:
    def __init__(self, capacity):
        self.observations = np.empty((capacity, 26), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((capacity, 2), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminals = np.empty(capacity, dtype=np.float32)
        self.capacity = capacity
        self.count = 0

    def add(self, observation, action, reward, next_observation, terminated):
        index = self.count % self.capacity
        self.observations[index] = observation
        self.next_observations[index] = next_observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminals[index] = terminated
        self.count += 1

    def sample(self, rng, size):
        indices = rng.integers(0, min(self.count, self.capacity), size)
        arrays = (self.observations, self.actions, self.rewards, self.next_observations, self.terminals)
        return tuple(torch.from_numpy(array[indices]) for array in arrays)


class NAF:
    def __init__(self, model, teacher=None, gamma=.995, tau=.005, anchor_weight=2.):
        self.model = model
        self.target_value = deepcopy(model.value_and_precision).requires_grad_(False)
        self.teacher = teacher
        self.gamma = gamma
        self.tau = tau
        self.anchor_weight = anchor_weight
        self.optimizer = torch.optim.Adam([
            dict(params=model.mean.parameters(), lr=2e-5 if teacher else 1e-4),
            dict(params=model.value_and_precision.parameters(), lr=3e-4),
        ])

    def update(self, batch, freeze_mean=False):
        observations, actions, rewards, next_observations, terminals = batch
        with torch.no_grad():
            # Since mu belongs to [-1, 1]², max_a Q_target(s',a) = V_target(s').
            target = rewards + self.gamma * (1 - terminals) * self.target_value(next_observations)[:, 0]
        prediction = self.model(observations, actions, freeze_mean=freeze_mean)
        td_loss = F.smooth_l1_loss(prediction, target)
        loss = td_loss
        if self.teacher is not None and not freeze_mean:
            loss = loss + self.anchor_weight * F.mse_loss(self.model.mean(observations), self.teacher(observations))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        self.optimizer.step()
        with torch.no_grad():
            for target_param, param in zip(self.target_value.parameters(), self.model.value_and_precision.parameters()):
                target_param.lerp_(param, self.tau)
        return td_loss.detach().item()
