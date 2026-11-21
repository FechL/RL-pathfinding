"""Online depth-first exploration of a regular maze using local LiDAR."""
import math
import random

import numpy as np

from planner import ROBOT_RADIUS
from lidar_qlearning import STOP_ACTION, beam_angles, safe_actions

DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def world_scan(pose, local_hits):
    """Transform finite LiDAR hits from the robot frame to the GPS frame."""
    hits = np.asarray(local_hits, dtype=float).reshape(-1, 2)
    hits = hits[np.isfinite(hits).all(axis=1)]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return hits @ np.array([[c, s], [-s, c]]) + pose[:2]


def synthetic_hits(ranges):
    angles = beam_angles(len(ranges))
    hit = ranges < 4.
    return ranges[hit, None] * np.column_stack((np.cos(angles[hit]), np.sin(angles[hit])))


class LidarMazeSearch:
    def __init__(self, start, goal, bounds, cell_size, seed=0, replay=()):
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.origin = np.array([bounds[0], bounds[2]])
        self.size = cell_size
        self.shape = (round((bounds[1] - bounds[0]) / cell_size),
                      round((bounds[3] - bounds[2]) / cell_size))
        self.rng = random.Random(seed)
        self.stack = [self.cell(start)]
        self.visited = set(self.stack)
        self.edges = {}
        self.checked = {}
        self.order = {}
        self.scan_bearing = None
        self.route = []
        self.replay = iter(replay)
        self.replaying = bool(replay)
        self.final = False

    def cell(self, point):
        return tuple(np.floor((point - self.origin) / self.size).astype(int))

    def center(self, cell):
        return self.origin + (np.asarray(cell) + .5) * self.size

    @staticmethod
    def clear_corridor(start, end, hits):
        delta = end - start
        length = np.linalg.norm(delta)
        if length < 1e-9:
            return True
        along = delta / length
        across = np.array([-along[1], along[0]])
        offsets = hits - start
        longitudinal = offsets @ along
        lateral = np.abs(offsets @ across)
        return not np.any((longitudinal >= 0) & (longitudinal <= length + .04)
                          & (lateral <= ROBOT_RADIUS + .02))

    def scan_action(self, sectors):
        pivot = 3 if self.scan_bearing >= 0 else 4
        return pivot if pivot in safe_actions(sectors) else STOP_ACTION

    def next_target(self, pose, local_hits):
        """Observe a reached cell and choose one unvisited branch or backtrack."""
        self.scan_bearing = None
        if self.replaying:
            target = tuple(next(self.replay, self.goal))
            self.final = target == self.goal
            self.route.append(target)
            return target

        cell = self.stack[-1]
        center = self.center(cell)
        if not self.route and math.dist(pose[:2], center) > .055:
            self.route.append(tuple(center))
            return self.route[-1]

        hits = world_scan(pose, local_hits)

        def bearing(point):
            dx, dy = np.asarray(point) - pose[:2]
            return (math.atan2(dy, dx) - pose[2] + math.pi) % (2 * math.pi) - math.pi

        # Keep the entire corridor probe well inside the front scan, so an
        # unseen opening cannot be mistaken for an observed clear passage.
        visible_angle = math.radians(55)
        if math.dist(center, self.goal) <= .75 * self.size:
            goal_bearing = bearing(self.goal)
            if abs(goal_bearing) > visible_angle and math.dist(pose[:2], self.goal) > .09:
                self.scan_bearing = goal_bearing
                return None
            if self.clear_corridor(pose[:2], np.asarray(self.goal), hits):
                self.final = True
                self.route.append(self.goal)
                return self.goal

        if cell not in self.order:
            neighbors = []
            for dx, dy in DIRECTIONS:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if all(0 <= index < limit for index, limit in zip(neighbor, self.shape)):
                    neighbors.append(neighbor)
            self.order[cell] = sorted(
                neighbors,
                key=lambda neighbor: math.dist(self.center(neighbor), self.goal)
                + self.rng.uniform(0, 2.5 * self.size),
            )
            self.checked[cell] = set()
            self.edges[cell] = set()

        unknown = []
        for neighbor in self.order[cell]:
            if neighbor in self.checked[cell] or neighbor in self.visited:
                continue
            probe = center + .69 * (self.center(neighbor) - center)
            direction = bearing(probe)
            if abs(direction) > visible_angle:
                unknown.append(direction)
                continue
            self.checked[cell].add(neighbor)
            if self.clear_corridor(center, probe, hits):
                self.edges[cell].add(neighbor)

        neighbor = next((n for n in self.order[cell]
                         if n in self.edges[cell] and n not in self.visited), None)
        if neighbor is None and unknown:
            self.scan_bearing = unknown[0]
            return None
        if neighbor is not None:
            self.stack.append(neighbor)
            self.visited.add(neighbor)
        elif len(self.stack) > 1:
            self.stack.pop()
        else:
            return None
        target = tuple(self.center(self.stack[-1]))
        self.route.append(target)
        return target
