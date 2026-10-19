"""A* navigation demonstration using Webots Supervisor position updates."""
import json
import math
import os
from pathlib import Path
import sys
import time

from planner import (GRID_RESOLUTION, ROBOT_RADIUS, SAFETY_MARGIN, Obstacle,
                     distance, path_length, physical_clearance, plan_path,
                     segment_is_free)

MOVE_SPEED = 0.12  # m/s of simulated time; no wheel motor policy is used.
GOAL_TOLERANCE = 0.01
PROJECT = Path(__file__).resolve().parents[2]


def read_scene(supervisor):
    arena = supervisor.getFromDef("ARENA")
    if arena is None:
        raise RuntimeError("DEF ARENA was not found in the active world")
    width, height = arena.getField("floorSize").getSFVec2f()
    group = supervisor.getFromDef("OBSTACLES").getField("children")
    obstacles = []
    for i in range(group.getCount()):
        node = group.getMFNode(i)
        pos = node.getField("translation").getSFVec3f()
        obstacles.append(Obstacle(node.getField("name").getSFString(),
                                  node.getField("kind").getSFString(), tuple(pos[:2]),
                                  tuple(node.getField("size").getSFVec3f()[:2])))
    robot = supervisor.getFromDef("epuck")
    goal = supervisor.getFromDef("GOAL").getField("translation").getSFVec3f()
    return robot, tuple(goal[:2]), obstacles, (-width/2, width/2, -height/2, height/2)


class MapView:
    def __init__(self, start, goal, obstacles, bounds, path, explored, show):
        import cv2
        import numpy as np
        self.cv2, self.np, self.show = cv2, np, show
        self.bounds, self.start, self.goal = bounds, start, goal
        self.obstacles, self.path, self.explored = obstacles, path, explored
        self.scale, self.pad, self.header = 215, 45, 82
        self.width = round((bounds[1]-bounds[0])*self.scale)+2*self.pad
        self.height = round((bounds[3]-bounds[2])*self.scale)+self.header+2*self.pad
        self.title = "E-puck Navigation Lab - A*"
        self.background = self.draw_background()
        self.last_image = self.background.copy()
        if show:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, self.width, self.height)

    def pixel(self, point):
        return (round(self.pad+(point[0]-self.bounds[0])*self.scale),
                round(self.header+self.pad+(self.bounds[3]-point[1])*self.scale))

    def text(self, image, text, point, size=0.45, color=(65, 80, 91), weight=1):
        self.cv2.putText(image, text, point, self.cv2.FONT_HERSHEY_SIMPLEX,
                         size, color, weight, self.cv2.LINE_AA)

    def draw_background(self):
        cv2, np = self.cv2, self.np
        image = np.full((self.height, self.width, 3), (244, 242, 238), dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (self.width, self.header), (61, 43, 29), -1)
        self.text(image, "E-PUCK / NAVIGATION LAB", (25, 34), 0.75, (255, 255, 255), 2)
        self.text(image, f"A* | {len(self.obstacles)} static obstacles | 5 cm search samples | 3 cm safety margin", (25, 61), 0.43, (208, 220, 226))
        lo = self.pixel((self.bounds[0], self.bounds[3]))
        hi = self.pixel((self.bounds[1], self.bounds[2]))
        cv2.rectangle(image, lo, hi, (222, 213, 202), 2)
        for x in np.arange(self.bounds[0], self.bounds[1]+0.001, 0.2):
            cv2.line(image, self.pixel((x, self.bounds[2])), self.pixel((x, self.bounds[3])), (230, 225, 219), 1)
        for y in np.arange(self.bounds[2], self.bounds[3]+0.001, 0.2):
            cv2.line(image, self.pixel((self.bounds[0], y)), self.pixel((self.bounds[1], y)), (230, 225, 219), 1)
        for p in self.explored:
            cv2.circle(image, self.pixel(p), 1, (213, 201, 184), -1)
        for ob in self.obstacles:
            x, y = ob.center
            sx, sy = ob.size
            color = (67, 140, 186) if ob.kind == "crate" else (157, 133, 89)
            margin = round((ROBOT_RADIUS+SAFETY_MARGIN)*self.scale)
            if ob.circular:
                radius = round(sx/2*self.scale)
                cv2.circle(image, self.pixel(ob.center), radius+margin, (221, 214, 194), -1)
                cv2.circle(image, self.pixel(ob.center), radius, color, -1)
            else:
                a, b = self.pixel((x-sx/2, y+sy/2)), self.pixel((x+sx/2, y-sy/2))
                cv2.rectangle(image, (a[0]-margin, a[1]-margin), (b[0]+margin, b[1]+margin), (221, 214, 194), -1)
                cv2.rectangle(image, a, b, color, -1)
            px, py = self.pixel(ob.center)
            self.text(image, ob.name.split()[0], (px-10, py+4), 0.4, (255, 255, 255))
        if self.path:
            points = np.array([self.pixel(p) for p in self.path], dtype=np.int32)
            cv2.polylines(image, [points], False, (153, 127, 45), 3, cv2.LINE_AA)
            for p in self.path[1:-1]:
                cv2.circle(image, self.pixel(p), 4, (153, 127, 45), -1)
        for name, p, color in [("START", self.start, (190, 135, 40)), ("GOAL", self.goal, (108, 151, 34))]:
            center = self.pixel(p)
            cv2.circle(image, center, 14, color, 2)
            self.text(image, name, (center[0]-23, center[1]+30), 0.44, color)
        self.text(image, "XY coordinates in metres | Pale area: robot footprint + safety margin", (self.pad, self.height-14), 0.4)
        return image

    def update(self, position, trail):
        image = self.background.copy()
        if len(trail) > 1:
            self.cv2.polylines(image, [self.np.array([self.pixel(p) for p in trail], dtype=self.np.int32)], False, (220, 173, 64), 2)
        self.cv2.circle(image, self.pixel(position), max(5, round(ROBOT_RADIUS*self.scale)), (57, 75, 235), -1)
        self.last_image = image
        if self.show:
            self.cv2.imshow(self.title, image)
            self.cv2.waitKey(1)

    def save(self, path):
        self.cv2.imwrite(str(path), self.last_image)

    def close(self):
        if self.show:
            self.cv2.destroyAllWindows()


def main():
    if not os.environ.get("WEBOTS_HOME"):
        raise SystemExit("Run the controller through Webots: bash run.sh")
    try:
        from controller import Supervisor
    except ImportError as error:
        raise SystemExit("Use the built-in Webots controller API through bash run.sh. Do not install the controller package from pip.") from error
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    if supervisor.step(timestep) == -1:
        return
    robot, goal, obstacles, bounds = read_scene(supervisor)
    field, rotation = robot.getField("translation"), robot.getField("rotation")
    initial = field.getSFVec3f()
    start = tuple(initial[:2])
    show = os.environ.get("EPUCK_NO_GUI") != "1" and (sys.platform != "linux" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")))
    output = Path(os.environ.get("EPUCK_RESULTS_DIR", str(PROJECT / "results")))
    output.mkdir(parents=True, exist_ok=True)
    timer = time.perf_counter()
    path, explored = plan_path(start, goal, obstacles, bounds)
    planning_seconds = time.perf_counter()-timer
    print(f"Python controller: {sys.executable}")
    print(f"Arena: {bounds}; obstacles: {len(obstacles)}; start: {start}; goal: {goal}")
    view = MapView(start, goal, obstacles, bounds, path, explored, show)
    trail, current_index, steps, completed, violation = [start], 1, 0, False, None
    clearance = physical_clearance(start, obstacles, bounds)
    view.update(start, trail)
    capture = os.environ.get("EPUCK_CAPTURE_DIR")
    if capture:
        Path(capture).mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        if supervisor.step(timestep) == -1:
            view.close()
            return
    if capture:
        supervisor.exportImage(str(Path(capture) / "world_overview.png"), 100)
    began = supervisor.getTime()
    if path:
        print(f"Path found: {len(path)} waypoints, {path_length(path):.3f} m; expanded nodes: {len(explored)}")
        while supervisor.step(timestep) != -1:
            actual = tuple(field.getSFVec3f()[:2])
            trail.append(actual)
            clearance = min(clearance, physical_clearance(actual, obstacles, bounds))
            if current_index >= len(path):
                completed = distance(actual, goal) <= GOAL_TOLERANCE
                break
            target = path[current_index]
            dist = distance(actual, target)
            if dist < 0.001:
                current_index += 1
                continue
            step = min(MOVE_SPEED*timestep/1000, dist)
            point = (actual[0]+(target[0]-actual[0])*step/dist, actual[1]+(target[1]-actual[1])*step/dist)
            if not segment_is_free(actual, point, obstacles, bounds, ROBOT_RADIUS+SAFETY_MARGIN-0.001):
                violation = "Movement stopped because the free-space boundary changed"
                break
            field.setSFVec3f([point[0], point[1], initial[2]])
            rotation.setSFRotation([0, 0, 1, math.atan2(target[1]-actual[1], target[0]-actual[0])])
            robot.resetPhysics()
            steps += 1
            view.update(point, trail)
        else:
            view.close()
            return
    final = tuple(field.getSFVec3f()[:2])
    result = {
        "status": "success" if completed else "failed", "controller": "A* with Supervisor position updates",
        "python": sys.executable, "start": start, "goal": goal, "final_position": final,
        "arena_bounds": bounds, "obstacle_count": len(obstacles), "grid_resolution_m": GRID_RESOLUTION,
        "robot_radius_m": ROBOT_RADIUS, "safety_margin_m": SAFETY_MARGIN,
        "waypoints": path, "expanded_nodes": len(explored), "planning_seconds": planning_seconds,
        "planned_length_m": path_length(path) if path else None, "travelled_length_m": path_length(trail),
        "simulation_duration_s": supervisor.getTime()-began, "movement_steps": steps,
        "goal_error_m": distance(final, goal), "min_physical_clearance_m": clearance,
        "safety_violation": violation, "trajectory": trail,
        "obstacles": [{"name": ob.name, "kind": ob.kind, "center": ob.center, "size": ob.size} for ob in obstacles],
    }
    (output / "latest_run.json").write_text(json.dumps(result, indent=2)+"\n")
    view.update(final, trail)
    view.save(output / "map.png")
    print(f"SIMULATION_RESULT: {result['status']}; goal error={result['goal_error_m']:.5f} m; clearance={clearance:.4f} m")
    if capture:
        supervisor.exportImage(str(Path(capture) / "world_finished.png"), 100)
    if os.environ.get("EPUCK_TEST_EXIT") == "1":
        supervisor.simulationQuit(0 if completed else 1)
    else:
        while supervisor.step(timestep) != -1:
            if show:
                view.cv2.waitKey(1)
    view.close()


if __name__ == "__main__":
    main()
