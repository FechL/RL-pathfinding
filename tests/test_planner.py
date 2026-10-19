import json
import math
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/epuck"))
from planner import Obstacle, distance, path_length, plan_path, segment_is_free


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.layout = json.loads((ROOT / "worlds/navigation_lab.json").read_text())
        self.bounds = (-1.8, 1.8, -1.3, 1.3)
        self.obstacles = [Obstacle(o["id"], o["kind"], o["position"], o["size"][:2]) for o in self.layout["obstacles"]]

    def assert_safe_route(self, start, goal):
        path, _ = plan_path(start, goal, self.obstacles, self.bounds)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], tuple(start))
        self.assertEqual(path[-1], tuple(goal))
        for a, b in zip(path, path[1:]):
            self.assertTrue(segment_is_free(a, b, self.obstacles, self.bounds))
        self.assertLess(path_length(path)/distance(start, goal), 1.4)

    def test_default_route_is_short_and_safe(self):
        self.assert_safe_route(self.layout["start"], self.layout["goal"])

    def test_reverse_and_alternative_routes(self):
        for start, goal in [((1.45, .95), (-1.45, -.95)), ((-1.45, .95), (1.45, -.95)), ((-1.45, 0), (1.45, 0))]:
            with self.subTest(start=start, goal=goal):
                self.assert_safe_route(start, goal)

    def test_small_start_goal_variations(self):
        generator = random.Random(20261018)
        for _ in range(20):
            start = [v+generator.uniform(-.08, .08) for v in self.layout["start"]]
            goal = [v+generator.uniform(-.08, .08) for v in self.layout["goal"]]
            self.assert_safe_route(start, goal)

    def test_no_route_terminates_with_finite_search(self):
        barrier = [Obstacle("wall", "divider", (0, 0), (.1, 2.6))]
        path, explored = plan_path((-1, 0), (1, 0), barrier, self.bounds)
        self.assertIsNone(path)
        self.assertLess(len(explored), 4000)

    def test_segment_cannot_tunnel_through_a_box(self):
        obstacle = [Obstacle("box", "crate", (0, 0), (.2, .2))]
        self.assertTrue(segment_is_free((-.4, 0), (-.4, 0), obstacle, self.bounds))
        self.assertTrue(segment_is_free((.4, 0), (.4, 0), obstacle, self.bounds))
        self.assertFalse(segment_is_free((-.4, 0), (.4, 0), obstacle, self.bounds))

    def test_cylinder_has_circular_footprint(self):
        obstacle = [Obstacle("drum", "drum", (0, 0), (.2, .2))]
        self.assertFalse(segment_is_free((-.4, 0), (.4, 0), obstacle, self.bounds))
        self.assertTrue(segment_is_free((.15, .15), (.15, .15), obstacle, self.bounds))

    def test_endpoints_outside_arena_or_inside_obstacle_fail(self):
        for start in [(-1.79, 0), self.obstacles[0].center]:
            self.assertIsNone(plan_path(start, (1.45, .95), self.obstacles, self.bounds)[0])

    def test_exact_goal_and_empty_motion(self):
        point = (-1.45, -.95)
        self.assertEqual(plan_path(point, point, self.obstacles, self.bounds)[0], [point])
        self.assertEqual(plan_path((-1.4, -1), (-1.3, -1), [], self.bounds)[0][-1], (-1.3, -1))


if __name__ == "__main__":
    unittest.main()
