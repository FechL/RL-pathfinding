"""Bounded replay storage with explicit terminal, truncation, and action mask."""
import numpy as np
import torch

from lidar_dqn import FEATURES, NUM_ACTIONS


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = self.cursor = 0
        self.arrays = dict(
            state=np.empty((capacity, len(FEATURES)), dtype=np.float32),
            action=np.empty(capacity, dtype=np.int64),
            reward=np.empty(capacity, dtype=np.float32),
            next_state=np.empty((capacity, len(FEATURES)), dtype=np.float32),
            terminated=np.empty(capacity, dtype=bool),
            truncated=np.empty(capacity, dtype=bool),
            next_mask=np.empty((capacity, NUM_ACTIONS), dtype=bool),
        )

    def push(self, state, action, reward, next_state, terminated, truncated, next_mask):
        if not 0 <= action < NUM_ACTIONS or not np.isfinite(reward):
            raise ValueError('Invalid action or reward in replay.')
        if not np.isfinite(state).all() or not np.isfinite(next_state).all():
            raise ValueError('Nonfinite replay observation.')
        if not np.asarray(next_mask).any():
            raise ValueError('Missing next action fallback.')
        values = (state, action, reward, next_state, terminated, truncated, next_mask)
        for array, value in zip(self.arrays.values(), values):
            array[self.cursor] = value
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng, device):
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return {key: torch.as_tensor(values[indices], device=device)
                for key, values in self.arrays.items()}

    def state_dict(self):
        return dict(cursor=self.cursor, size=self.size,
                    arrays={key: torch.from_numpy(value[:self.size].copy())
                            for key, value in self.arrays.items()})

    def load_state_dict(self, data):
        size, cursor = data['size'], data['cursor']
        if not 0 <= size <= self.capacity or not 0 <= cursor < self.capacity:
            raise ValueError('Replay does not fit checkpoint capacity.')
        for key, array in self.arrays.items():
            array[:size] = data['arrays'][key].cpu().numpy()
        self.size, self.cursor = size, cursor
