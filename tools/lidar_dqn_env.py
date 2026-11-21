"""Discrete gym adapter over the shared DFS and continuous motion mission."""
import math

import gymnasium as gym
import numpy as np

from lidar_dqn import FEATURES, NUM_ACTIONS, action_mask, encode_observation, get_action_mask
from lidar_drl_env import LidarMission
from lidar_qlearning import STOP_ACTION


class DQNMission(LidarMission):
    def __init__(self, navigation, timeout):
        super().__init__(navigation, timeout)
        bounds = navigation.bounds
        self.distance_scale = math.hypot(bounds[1] - bounds[0], bounds[3] - bounds[2])
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = gym.spaces.Box(-1., 1., (len(FEATURES),), dtype=np.float32)

    def _observation(self):
        env = self.motion
        return encode_observation(env.pose, self.target, env.lidar_sectors, env.wheels,
                                  env.config, self.distance_scale, recovering=env.collision_recovery)

    def action_mask(self, observation):
        if self.motion.collision_recovery:
            return action_mask((STOP_ACTION,))
        if self.target is None:
            if self.route_search.scan_bearing is None:
                return action_mask((STOP_ACTION,))
            return action_mask((self.route_search.scan_action(self.motion.lidar_sectors),))
        return get_action_mask(observation) & action_mask(self.allowed_actions())

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Invalid DQN action index.')
        return self.tabular_step(action)
