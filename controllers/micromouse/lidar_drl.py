"""Continuous SAC observation, motor limits and portable policy inference."""
import json
import math
from pathlib import Path

import numpy as np

from continuous_control import actor_from_layers
from lidar_qlearning import ACTION_VW, STOP_ACTION, safe_actions, waypoint_bearing

ALGORITHM = 'lidar_sac'
OBSERVATION_SIZE = 11


def observation(pose, target, sectors, wheels, wheel_limit):
    bearing = waypoint_bearing(pose, target) if target is not None else 0.
    distance = min(1., math.dist(pose[:2], target)) if target is not None else 0.
    return np.array([math.sin(bearing), math.cos(bearing), distance,
                     *np.clip(np.asarray(sectors) / .8, 0., 1.),
                     *np.clip(np.asarray(wheels) / wheel_limit, -1., 1.),
                     float(target is not None)], dtype=np.float32)


def drive_command(action, sectors, allowed):
    """Return the safety category and continuous (velocity, yaw rate)."""
    forward, yaw = np.clip(action, -1., 1.)
    linear, angular = .23 * (forward + 1.) / 2., 2.7 * yaw
    forward_actions = (0, 1, 2)
    pivot = 3 if angular >= 0. else 4
    arc = 1 if angular >= 0. else 2
    if not any(choice in allowed for choice in forward_actions):
        if pivot not in allowed:
            pivot = next((choice for choice in (3, 4) if choice in allowed), STOP_ACTION)
        return pivot, ACTION_VW[pivot]
    if arc not in allowed:
        angular = 0.
    if 0 not in safe_actions(sectors):
        linear = 0.
    return arc if angular else 0, (linear, angular)


def load_actor(path):
    data = json.loads(Path(path).read_text())
    if (data.get('algorithm'), data.get('format_version')) != (ALGORITHM, 1):
        raise ValueError('Expected a LiDAR SAC actor. Train with bash run.sh train --algorithm drl.')
    return actor_from_layers(data, OBSERVATION_SIZE), data
