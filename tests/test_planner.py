import json
import math
from pathlib import Path
import random
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/micromouse"))
from planner import Obstacle, distance, path_length, plan_path, segment_is_free


def maze_points(layout):
    width, height = layout["grid_size"]
    cell = layout["cell_size"]

    def center(index):
        return (-width*cell/2+(index[0]+0.5)*cell,
                -height*cell/2+(index[1]+0.5)*cell)

    start = center(layout["start_cell"])
    cells = layout["goal_cells"]
    goal = (sum(center(cell)[0] for cell in cells)/len(cells),
            sum(center(cell)[1] for cell in cells)/len(cells))
    return start, goal


def maze_walls():
    text = (ROOT / "worlds/world.wbt").read_text()
    pattern = re.compile(
        r'MazeWall\s*\{\s*translation\s+([-\d.]+)\s+([-\d.]+)\s+[-\d.]+\s+'
        r'size\s+([-\d.]+)\s+([-\d.]+)\s+[-\d.]+\s+name\s+"([^"]+)"', re.S)
    return [Obstacle(name, "wall", (float(x), float(y)), (float(sx), float(sy)))
            for x, y, sx, sy, name in pattern.findall(text)]


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.layout = json.loads((ROOT / "worlds/maze_layout.json").read_text())
        width, height = self.layout["grid_size"]
        cell = self.layout["cell_size"]
        self.bounds = (-width*cell/2, width*cell/2,
                       -height*cell/2, height*cell/2)
        self.obstacles = maze_walls()
        self.start, self.goal = maze_points(self.layout)

    def assert_safe_route(self, start, goal):
        path, _ = plan_path(start, goal, self.obstacles, self.bounds)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], tuple(start))
        self.assertEqual(path[-1], tuple(goal))
        for a, b in zip(path, path[1:]):
            self.assertTrue(segment_is_free(a, b, self.obstacles, self.bounds))
        return path

    def test_start_to_center_goal_route_is_safe(self):
        path = self.assert_safe_route(self.start, self.goal)
        self.assertGreater(path_length(path), distance(self.start, self.goal))

    def test_reverse_and_alternative_maze_routes(self):
        width, height = self.layout["grid_size"]
        cell = self.layout["cell_size"]

        def center(i, j):
            return (-width*cell/2+(i+0.5)*cell,
                    -height*cell/2+(j+0.5)*cell)

        for start, goal in [(self.goal, self.start),
                            (center(0, height-1), center(width-1, 0)),
                            (center(width-1, height-1), self.goal)]:
            with self.subTest(start=start, goal=goal):
                self.assert_safe_route(start, goal)

    def test_small_start_goal_variations(self):
        generator = random.Random(20261025)
        for _ in range(10):
            start = [v+generator.uniform(-.04, .04) for v in self.start]
            goal = [v+generator.uniform(-.04, .04) for v in self.goal]
            self.assert_safe_route(start, goal)

    def test_no_route_terminates_with_finite_search(self):
        barrier = [Obstacle("wall", "wall", (0, 0), (.1, 7.0))]
        path, explored = plan_path((-1, 0), (1, 0), barrier, self.bounds)
        self.assertIsNone(path)
        self.assertLess(len(explored), 18000)

    def test_segment_cannot_tunnel_through_a_box(self):
        obstacle = [Obstacle("box", "wall", (0, 0), (.2, .2))]
        self.assertTrue(segment_is_free((-.4, 0), (-.4, 0), obstacle, self.bounds))
        self.assertTrue(segment_is_free((.4, 0), (.4, 0), obstacle, self.bounds))
        self.assertFalse(segment_is_free((-.4, 0), (.4, 0), obstacle, self.bounds))

    def test_cylinder_has_circular_footprint(self):
        obstacle = [Obstacle("drum", "drum", (0, 0), (.2, .2))]
        self.assertFalse(segment_is_free((-.4, 0), (.4, 0), obstacle, self.bounds))
        self.assertTrue(segment_is_free((.25, .25), (.25, .25), obstacle, self.bounds))

    def test_endpoints_outside_arena_or_inside_wall_fail(self):
        for start in [(-3.18, 0), self.obstacles[0].center]:
            self.assertIsNone(plan_path(start, self.goal, self.obstacles, self.bounds)[0])

    def test_exact_goal_and_short_motion(self):
        self.assertEqual(plan_path(self.start, self.start, self.obstacles, self.bounds)[0], [self.start])
        a, b = (-3.0, -3.0), (-2.9, -3.0)
        self.assertEqual(plan_path(a, b, [], self.bounds)[0][-1], b)


if __name__ == "__main__":
    unittest.main()
