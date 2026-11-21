"""Continuous observations and action constraints shared by DQN runtimes."""
import math

import numpy as np

from lidar_qlearning import ACTION_NAMES, ACTION_VW, STOP_ACTION, safe_actions, waypoint_bearing

FEATURES = (
    'bearing_sin', 'bearing_cos', 'target_distance_over_arena_diagonal',
    'front', 'front_left', 'left', 'right', 'front_right',
    'linear_speed', 'yaw_rate', 'scanning', 'collision_recovery',
)
NUM_ACTIONS = len(ACTION_NAMES)
LIDAR_RANGE = 4.


def action_mask(allowed):
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[list(allowed)] = True
    if not mask.any():
        mask[STOP_ACTION] = True
    return mask


def get_action_mask(observation):
    """Physical mask from the five normalized forward-sector minima."""
    obs = np.asarray(observation)
    if obs.shape != (len(FEATURES),) or not np.isfinite(obs).all():
        raise ValueError('Expected one finite DQN observation.')
    if obs[-1]:
        return action_mask((STOP_ACTION,))
    return action_mask(safe_actions(obs[3:8] * LIDAR_RANGE))


def encode_observation(pose, target, sectors, wheels, config, distance_scale, *, recovering=False):
    pose = np.asarray(pose)
    sectors = np.asarray(sectors)
    wheels = np.asarray(wheels)
    if pose.shape != (3,) or sectors.shape != (5,) or wheels.shape != (2,):
        raise ValueError('Expected 3 pose, 5 LiDAR, and 2 wheel measurements.')
    if not all(np.isfinite(x).all() for x in (pose, sectors, wheels)):
        raise ValueError('Nonfinite DQN sensor input.')
    if not math.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError('Distance scale must be finite and positive.')
    if target is None:
        relative = (0., 0., 0.)
    else:
        bearing = waypoint_bearing(pose, target)
        relative = (math.sin(bearing), math.cos(bearing),
                    min(1., math.dist(pose[:2], target) / distance_scale))
    left, right = wheels / config.wheel_limit
    result = np.array([
        *relative, *np.clip(sectors / LIDAR_RANGE, 0., 1.),
        np.clip((left + right) / 2, -1., 1.), np.clip((right - left) / 2, -1., 1.),
        float(target is None), float(recovering),
    ], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite DQN observation.')
    return result


def schema():
    return dict(observation=list(FEATURES),
                actions=[dict(name=name, velocity=list(velocity))
                         for name, velocity in zip(ACTION_NAMES, ACTION_VW)])
