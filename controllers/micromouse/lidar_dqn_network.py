"""PyTorch action-value network for discrete Micromouse motor actions."""
from torch import nn

from lidar_dqn import FEATURES, NUM_ACTIONS


class QNetwork(nn.Module):
    def __init__(self, hidden=(128, 128)):
        super().__init__()
        sizes = (len(FEATURES), *hidden, NUM_ACTIONS)
        layers = []
        for index, (inputs, outputs) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(inputs, outputs))
            if index < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, observation):
        return self.layers(observation)
