"""Sensor-based tabular Q-learning for a hidden yellow maze goal."""
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from planner import ROBOT_RADIUS, SAFETY_MARGIN, segment_is_free

ACTION_DELTAS = ((1, 0), (0, 1), (-1, 0), (0, -1))
ACTION_NAMES = ('EAST', 'NORTH', 'WEST', 'SOUTH')
HEADING_NAMES = ACTION_NAMES
SENSOR_NAMES = ('front', 'left', 'back', 'right')
SENSOR_BINS = ('blocked', 'near', 'clear')
SENSOR_SCHEMA = 'gps-cell + heading + 4 relative IR bins + yellow color flag'


def state_key(state):
    return ','.join(str(int(value)) for value in state)


def parse_state_key(key):
    return tuple(int(value) for value in key.split(','))


def heading_bin(yaw):
    return int(round((yaw % (2 * math.pi)) / (math.pi / 2))) % 4


def distance_bin(distance):
    if distance < .10:
        return 0
    if distance < .25:
        return 1
    return 2


def ir_value_bin(value):
    """Map the Webots IR lookup-table output to blocked, near, or clear."""
    value = float(value)
    if value >= 1500:
        return 0
    if value >= 450:
        return 1
    return 2


def relative_sensor_bins(front_left, front_right, left, rear, right):
    front = min(ir_value_bin(front_left), ir_value_bin(front_right))
    return front, ir_value_bin(left), ir_value_bin(rear), ir_value_bin(right)


class SensorMaze:
    """Kinematic maze used only to train from local sensors and a color flag.

    The yellow-cell geometry belongs to the simulator's color-sensor model.
    It is not returned in an observation, reward shaping term, or Q-table key.
    The agent knows its own grid position from localization, but receives no
    target coordinates, map connectivity, path guidance, or exploring starts.
    """
    def __init__(self, start, walls, bounds, cell_size, yellow_cells, *,
                 sensor_range=.4, sensor_forward_offset=.055):
        self.start = tuple(start)
        self.walls = list(walls)
        self.bounds = tuple(bounds)
        self.cell_size = float(cell_size)
        self.width = round((bounds[1] - bounds[0]) / cell_size)
        self.height = round((bounds[3] - bounds[2]) / cell_size)
        self.yellow_cells = frozenset(tuple(map(int, cell)) for cell in yellow_cells)
        self.sensor_range = float(sensor_range)
        self.sensor_forward_offset = float(sensor_forward_offset)
        self.position = self.start
        self.heading = 0
        self.path = []
        self.max_state_count = self.width * self.height * 4 * (3 ** 4) * 2
        reach = self.cell_size + self.sensor_range + ROBOT_RADIUS
        self.local_walls = []
        for y in range(self.height):
            for x in range(self.width):
                center = self.center(x, y)
                self.local_walls.append([wall for wall in self.walls
                    if abs(wall.center[0]-center[0]) <= reach + wall.size[0]/2
                    and abs(wall.center[1]-center[1]) <= reach + wall.size[1]/2])
        self.signature = self._scene_signature()

    def _scene_signature(self):
        scene = dict(start=self.start, bounds=self.bounds, cell=self.cell_size,
                     walls=sorted((w.name, *w.center, *w.size) for w in self.walls),
                     sensor_schema=SENSOR_SCHEMA, yellow_sensor='front-facing floor color, yellow')
        def normalize(value):
            if isinstance(value, float):
                return round(value, 6)
            if isinstance(value, (tuple, list)):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            return value
        return hashlib.sha256(json.dumps(normalize(scene), sort_keys=True).encode()).hexdigest()

    def cell_xy(self, position):
        x = math.floor((position[0] - self.bounds[0]) / self.cell_size)
        y = math.floor((position[1] - self.bounds[2]) / self.cell_size)
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError('Position lies outside the maze')
        return x, y

    def cell_id(self, position):
        x, y = self.cell_xy(position)
        return y * self.width + x

    def center(self, x, y):
        return (self.bounds[0] + (x + .5) * self.cell_size,
                self.bounds[2] + (y + .5) * self.cell_size)

    def ray_distance(self, origin, angle):
        dx, dy = math.cos(angle), math.sin(angle)
        nearest = self.sensor_range
        xmin, xmax, ymin, ymax = self.bounds
        for distance, valid in (
            ((xmin - origin[0]) / dx if dx < -1e-12 else (xmax - origin[0]) / dx if dx > 1e-12 else math.inf,
             abs(dx) > 1e-12),
            ((ymin - origin[1]) / dy if dy < -1e-12 else (ymax - origin[1]) / dy if dy > 1e-12 else math.inf,
             abs(dy) > 1e-12),
        ):
            if valid and distance >= 0:
                nearest = min(nearest, distance)
        x, y = self.cell_xy(origin)
        for wall in self.local_walls[y*self.width+x]:
            lower, upper = 0., self.sensor_range
            intersects = True
            for axis, direction in ((0, dx), (1, dy)):
                wall_min = wall.center[axis] - wall.size[axis] / 2
                wall_max = wall.center[axis] + wall.size[axis] / 2
                if abs(direction) < 1e-12:
                    if not wall_min <= origin[axis] <= wall_max:
                        intersects = False
                        break
                    continue
                first = (wall_min - origin[axis]) / direction
                last = (wall_max - origin[axis]) / direction
                lower = max(lower, min(first, last))
                upper = min(upper, max(first, last))
                if lower > upper:
                    intersects = False
                    break
            if intersects and 0 <= lower <= self.sensor_range:
                nearest = min(nearest, lower)
        return max(0., nearest)

    def color_sensor(self, position=None, heading=None):
        """Simulate a short-range downward camera over the robot's nose."""
        position = self.position if position is None else position
        heading = self.heading if heading is None else heading
        dx, dy = ACTION_DELTAS[heading]
        sensed_point = (position[0] + self.sensor_forward_offset * dx,
                        position[1] + self.sensor_forward_offset * dy)
        try:
            return self.cell_xy(sensed_point) in self.yellow_cells
        except ValueError:
            return False

    def sensor_bins(self, position=None, heading=None):
        position = self.position if position is None else position
        heading = self.heading if heading is None else heading
        heading_angle = heading * math.pi / 2
        angles = (heading_angle, heading_angle + math.pi / 2,
                  heading_angle + math.pi, heading_angle - math.pi / 2)
        return tuple(distance_bin(self.ray_distance(position, angle)) for angle in angles)

    def state(self):
        return (self.cell_id(self.position), self.heading, *self.sensor_bins(),
                int(self.color_sensor()))

    def reset(self):
        self.position = self.start
        self.heading = 0
        self.path = [self.position]
        return self.state()

    def step(self, action):
        if not 0 <= action < len(ACTION_DELTAS):
            raise ValueError('Unknown tabular action')
        x, y = self.cell_xy(self.position)
        dx, dy = ACTION_DELTAS[action]
        next_xy = x + dx, y + dy
        self.heading = action
        if not (0 <= next_xy[0] < self.width and 0 <= next_xy[1] < self.height):
            return self.state(), -10., False, {'collision': True, 'yellow_detected': False}
        target = self.center(*next_xy)
        free = segment_is_free(self.position, target,
                               self.local_walls[y*self.width+x], self.bounds,
                               ROBOT_RADIUS + SAFETY_MARGIN)
        if free:
            self.position = target
            self.path.append(target)
        yellow = self.color_sensor()
        if yellow:
            return self.state(), 100., True, {'collision': False, 'yellow_detected': True}
        if not free:
            return self.state(), -10., False, {'collision': True, 'yellow_detected': False}
        return self.state(), -1., False, {'collision': False, 'yellow_detected': False}


def q_update_table(q_table, state, action, reward, next_state, terminated, alpha=.2, gamma=.95):
    state_id, next_state_id = state_key(state), state_key(next_state)
    q_table.setdefault(state_id, [0.] * len(ACTION_DELTAS))
    q_table.setdefault(next_state_id, [0.] * len(ACTION_DELTAS))
    old = q_table[state_id][action]
    target = reward if terminated else reward + gamma * max(q_table[next_state_id])
    q_table[state_id][action] = old + alpha * (target - old)
    return old, q_table[state_id][action], target


def greedy_action(values, *, rng=None, random_ties=False):
    best = max(values)
    options = [index for index, value in enumerate(values) if value == best]
    return rng.choice(options) if random_ties and rng is not None else options[0]


def train(env, episodes=4000, seed=42, max_steps=500, exploration_fraction=.8,
          alpha=.2, gamma=.95, epsilon_floor=.10, exploitation_epsilon=.02,
          viewer=None, on_episode=None):
    if episodes < 2 or max_steps < 1:
        raise ValueError('Use at least two episodes and a positive step limit')
    if not 0 < exploration_fraction < 1:
        raise ValueError('Exploration fraction must be between zero and one')
    rng = random.Random(seed)
    q_table, visits, history = {}, Counter(), []
    exploration_episodes = max(1, min(episodes - 1, round(episodes * exploration_fraction)))
    stopped = False
    for episode_index in range(episodes):
        exploring = episode_index < exploration_episodes
        phase = 'eksplorasi' if exploring else 'eksploitasi'
        if exploring:
            progress = episode_index / max(1, exploration_episodes - 1)
            epsilon = 1. - (1. - epsilon_floor) * progress
        else:
            epsilon = exploitation_epsilon
        state = env.reset()
        total_reward, collisions, q_updates, success = 0., 0, 0, False
        for step in range(1, max_steps + 1):
            state_id = state_key(state)
            values = q_table.setdefault(state_id, [0.] * len(ACTION_DELTAS))
            visits[state_id] += 1
            if rng.random() < epsilon:
                action = rng.randrange(len(ACTION_DELTAS))
            else:
                action = greedy_action(values, rng=rng, random_ties=exploring)
            next_state, reward, success, info = env.step(action)
            q_update_table(q_table, state, action, reward, next_state, success, alpha, gamma)
            q_updates += 1
            total_reward += reward
            collisions += int(info['collision'])
            state = next_state
            if viewer and not viewer.update(env, episode_index + 1, episodes, phase, epsilon,
                                             step, total_reward, q_table, visits, history,
                                             chosen_action=action):
                stopped = True
                break
            if success:
                break
        row = dict(episode=episode_index + 1, phase=phase, epsilon=epsilon,
                   reward=total_reward, steps=step, success=bool(success),
                   collisions=collisions, unique_states=len(q_table), q_updates=q_updates,
                   timeout=not success and step >= max_steps)
        history.append(row)
        if on_episode:
            on_episode(row, q_table, visits, history)
        if viewer:
            viewer.episode_finished(row, history, env, q_table, visits)
        if stopped:
            break
    return q_table, history, visits


def save_policy(path, env, q_table, metadata):
    data = dict(format_version=2, algorithm='sensor_tabular_q_learning',
                sensor_schema=SENSOR_SCHEMA, actions=ACTION_NAMES,
                q_table=q_table, state_count_bound=env.max_state_count,
                scene_signature=env.signature, metadata=metadata)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def load_policy(path, env):
    data = json.loads(Path(path).read_text())
    if data.get('format_version') != 2 or data.get('algorithm') != 'sensor_tabular_q_learning':
        raise ValueError('This is not a sensor-based tabular Q policy; train it with tools/train_q_learning.py')
    if data.get('scene_signature') != env.signature or data.get('sensor_schema') != SENSOR_SCHEMA:
        raise ValueError('Policy sensors or maze geometry do not match this run; retrain tabular Q-learning')
    q_table = data.get('q_table')
    if not isinstance(q_table, dict):
        raise ValueError('Malformed tabular Q table')
    for key, values in q_table.items():
        parse_state_key(key)
        if len(values) != len(ACTION_DELTAS) or any(not math.isfinite(float(v)) for v in values):
            raise ValueError(f'Malformed Q values for state {key}')
    return q_table, data.get('metadata', {})
