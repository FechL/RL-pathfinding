"""Rule-based micromouse baseline: A* global route plus differential-drive control."""
import json
import math
import os
from pathlib import Path
import sys
import time

from planner import (GRID_RESOLUTION, ROBOT_RADIUS, SAFETY_MARGIN, Obstacle,
                     distance, path_length, physical_clearance, plan_path)

WHEEL_RADIUS = 0.027
AXLE_TRACK = 0.104
LINEAR_SPEED = 0.18
TURN_SPEED = 2.8
TARGET_TOLERANCE = 0.035
PROJECT = Path(__file__).resolve().parents[2]


def wrap_angle(angle):
    return (angle+math.pi) % (2*math.pi)-math.pi


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def read_scene(supervisor):
    arena = supervisor.getFromDef("ARENA")
    if arena is None:
        raise RuntimeError("DEF ARENA was not found in the active world")
    width, height = arena.getField("floorSize").getSFVec2f()
    cell_size = arena.getField("cellSize").getSFFloat()
    group = supervisor.getFromDef("WALLS").getField("children")
    walls = []
    for index in range(group.getCount()):
        node = group.getMFNode(index)
        pos = node.getField("translation").getSFVec3f()
        size = node.getField("size").getSFVec3f()
        name = node.getField("name").getSFString()
        walls.append(Obstacle(name, "wall", tuple(pos[:2]), tuple(size[:2])))
    robot = supervisor.getFromDef("MICROMOUSE")
    if robot is None:
        raise RuntimeError("DEF MICROMOUSE was not found in the active world")
    goal_node = supervisor.getFromDef("GOAL")
    if goal_node is None:
        raise RuntimeError("DEF GOAL was not found in the active world")
    goal = goal_node.getField("translation").getSFVec3f()
    bounds = (-width/2, width/2, -height/2, height/2)
    return robot, tuple(goal[:2]), walls, bounds, cell_size


class MapView:
    def __init__(self, start, goal, walls, bounds, path, explored, cell_size, show):
        import cv2
        import numpy as np
        self.cv2, self.np, self.show = cv2, np, show
        self.bounds, self.start, self.goal = bounds, start, goal
        self.walls, self.path, self.explored = walls, path, explored
        self.cell_size = cell_size
        self.scale, self.pad, self.header = 230, 42, 78
        self.width = round((bounds[1]-bounds[0])*self.scale)+2*self.pad
        self.height = round((bounds[3]-bounds[2])*self.scale)+self.header+2*self.pad
        self.title = "Micromouse M1 - A* Maze Solver"
        self.background = self.draw_background()
        self.last_image = self.background.copy()
        if show:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, self.width, self.height)

    def pixel(self, point):
        return (round(self.pad+(point[0]-self.bounds[0])*self.scale),
                round(self.header+self.pad+(self.bounds[3]-point[1])*self.scale))

    def text(self, image, text, point, size=0.45, color=(55, 73, 77), weight=1):
        self.cv2.putText(image, text, point, self.cv2.FONT_HERSHEY_SIMPLEX,
                         size, color, weight, self.cv2.LINE_AA)

    def draw_background(self):
        cv2, np = self.cv2, self.np
        image = np.full((self.height, self.width, 3), (235, 239, 233), dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (self.width, self.header), (42, 57, 62), -1)
        self.text(image, "MICROMOUSE M1 / MAZE SOLVER", (22, 33), 0.75, (255, 255, 255), 2)
        self.text(image, f"{len(self.walls)} walls | {self.cell_size*100:.0f} cm cells | A* route | {len(self.explored)} expanded cells", (23, 60), 0.43, (196, 215, 212))
        lo = self.pixel((self.bounds[0], self.bounds[3]))
        hi = self.pixel((self.bounds[1], self.bounds[2]))
        cv2.rectangle(image, lo, hi, (144, 158, 153), 2)
        for x in np.arange(self.bounds[0]+self.cell_size,
                           self.bounds[1], self.cell_size):
            cv2.line(image, self.pixel((x, self.bounds[2])),
                     self.pixel((x, self.bounds[3])), (210, 218, 211), 1)
        for y in np.arange(self.bounds[2]+self.cell_size,
                           self.bounds[3], self.cell_size):
            cv2.line(image, self.pixel((self.bounds[0], y)),
                     self.pixel((self.bounds[1], y)), (210, 218, 211), 1)
        for point in self.explored:
            cv2.circle(image, self.pixel(point), 2, (192, 202, 195), -1)
        for wall in self.walls:
            x, y = wall.center
            sx, sy = wall.size
            a, b = self.pixel((x-sx/2, y+sy/2)), self.pixel((x+sx/2, y-sy/2))
            cv2.rectangle(image, a, b, (82, 101, 105), -1)
            cv2.rectangle(image, a, b, (59, 76, 81), 1)
        if self.path:
            points = np.array([self.pixel(p) for p in self.path], dtype=np.int32)
            cv2.polylines(image, [points], False, (33, 136, 198), 3, cv2.LINE_AA)
        start = self.pixel(self.start)
        goal = self.pixel(self.goal)
        cv2.circle(image, start, 12, (45, 157, 111), 3)
        cv2.rectangle(image, (goal[0]-round(self.cell_size*self.scale),
                              goal[1]-round(self.cell_size*self.scale)),
                      (goal[0]+round(self.cell_size*self.scale),
                       goal[1]+round(self.cell_size*self.scale)), (77, 151, 208), 2)
        self.text(image, "START", (start[0]+12, start[1]+25), 0.38, (30, 125, 87))
        self.text(image, "GOAL", (goal[0]+18, goal[1]-18), 0.38, (56, 115, 170))
        self.text(image, "Blue: planned route | Green: start | Orange: 2 x 2 goal", (self.pad, self.height-14), 0.40)
        return image

    def update(self, position, heading, trail):
        image = self.background.copy()
        if len(trail) > 1:
            self.cv2.polylines(image, [self.np.array([self.pixel(p) for p in trail], dtype=self.np.int32)], False, (56, 184, 214), 2)
        center = self.pixel(position)
        self.cv2.circle(image, center, max(5, round(ROBOT_RADIUS*self.scale)), (48, 89, 226), -1)
        nose = self.pixel((position[0]+0.10*math.cos(heading), position[1]+0.10*math.sin(heading)))
        self.cv2.line(image, center, nose, (32, 54, 134), 3, self.cv2.LINE_AA)
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
        raise SystemExit("Use the built-in Webots controller API through bash run.sh.") from error

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    if supervisor.step(timestep) == -1:
        return
    robot, goal, walls, bounds, cell_size = read_scene(supervisor)
    start_position = robot.getField("translation").getSFVec3f()
    start = tuple(start_position[:2])
    gps = supervisor.getDevice("gps")
    imu = supervisor.getDevice("imu")
    gps.enable(timestep)
    imu.enable(timestep)
    left_motor = supervisor.getDevice("left wheel motor")
    right_motor = supervisor.getDevice("right wheel motor")
    for motor in (left_motor, right_motor):
        motor.setPosition(float("inf"))
        motor.setVelocity(0)
    distance_sensors = {}
    for name in ("front left sensor", "front right sensor", "left side sensor", "right side sensor"):
        sensor = supervisor.getDevice(name)
        sensor.enable(timestep)
        distance_sensors[name] = sensor

    show = os.environ.get("MICROMOUSE_NO_GUI") != "1" and (
        sys.platform != "linux" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")))
    output = Path(os.environ.get("MICROMOUSE_RESULTS_DIR", str(PROJECT / "results")))
    output.mkdir(parents=True, exist_ok=True)
    capture = os.environ.get("MICROMOUSE_CAPTURE_DIR")
    if capture:
        Path(capture).mkdir(parents=True, exist_ok=True)
    timer = time.perf_counter()
    path, explored = plan_path(start, goal, walls, bounds)
    planning_seconds = time.perf_counter()-timer
    print(f"Controller: {sys.executable}")
    print(f"Maze: {round((bounds[1]-bounds[0])/cell_size)} x {round((bounds[3]-bounds[2])/cell_size)} cells; walls: {len(walls)}; start: {start}; goal: {goal}")
    if path:
        print(f"A* route found: {len(path)} waypoints, {path_length(path):.2f} m; expanded cells: {len(explored)}")
    else:
        print("A* could not find a route to the goal")
    view = MapView(start, goal, walls, bounds, path, explored, cell_size, show)
    trail, current_index, steps = [start], 1, 0
    clearance = physical_clearance(start, walls, bounds)
    current = start
    heading = 0.0
    view.update(start, heading, trail)
    for _ in range(10):
        if supervisor.step(timestep) == -1:
            view.close()
            return
    if capture:
        supervisor.exportImage(str(Path(capture) / "maze_overview.png"), 100)

    began = supervisor.getTime()
    completed = False
    violation = None
    best_target_distance = math.inf
    last_progress_time = began
    while path and supervisor.step(timestep) != -1:
        current = tuple(gps.getValues()[:2])
        heading = imu.getRollPitchYaw()[2]
        current_clearance = physical_clearance(current, walls, bounds)
        clearance = min(clearance, current_clearance)
        if current_clearance < 0:
            violation = "The robot crossed a wall or arena boundary"
            break
        if not trail or distance(current, trail[-1]) >= 0.003:
            trail.append(current)
        if current_index >= len(path):
            completed = distance(current, goal) <= TARGET_TOLERANCE*1.5
            break

        target = path[current_index]
        dx, dy = target[0]-current[0], target[1]-current[1]
        target_distance = math.hypot(dx, dy)
        if target_distance <= TARGET_TOLERANCE:
            print(f"Reached waypoint {current_index+1}/{len(path)}")
            current_index += 1
            best_target_distance = math.inf
            last_progress_time = supervisor.getTime()
            continue
        if target_distance < best_target_distance-0.004:
            best_target_distance = target_distance
            last_progress_time = supervisor.getTime()
        elif supervisor.getTime()-last_progress_time > 15:
            violation = f"No progress toward waypoint {current_index+1} for 15 seconds"
            print(violation)
            break

        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_heading-heading)
        if abs(heading_error) > 0.16:
            left_speed = -math.copysign(TURN_SPEED, heading_error)
            right_speed = math.copysign(TURN_SPEED, heading_error)
        else:
            linear = min(LINEAR_SPEED, max(0.05, target_distance*1.2))
            angular = clamp(2.2*heading_error, -0.75, 0.75)
            left_speed = (linear-angular*AXLE_TRACK/2)/WHEEL_RADIUS
            right_speed = (linear+angular*AXLE_TRACK/2)/WHEEL_RADIUS
            limit = 17.0
            peak = max(abs(left_speed), abs(right_speed))
            if peak > limit:
                left_speed *= limit/peak
                right_speed *= limit/peak

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)
        steps += 1
        if steps % 4 == 0:
            view.update(current, heading, trail)
    else:
        if path:
            view.close()
            return

    left_motor.setVelocity(0)
    right_motor.setVelocity(0)
    final = tuple(gps.getValues()[:2])
    actual_node_position = robot.getField("translation").getSFVec3f()
    sensor_values = {name: sensor.getValue() for name, sensor in distance_sensors.items()}
    result = {
        "status": "success" if completed else "failed",
        "controller": "A* with differential-drive micromouse following",
        "python": sys.executable,
        "maze_size_cells": [round((bounds[1]-bounds[0])/cell_size), round((bounds[3]-bounds[2])/cell_size)],
        "cell_size_m": cell_size,
        "wall_count": len(walls),
        "start": start,
        "goal": goal,
        "arena_bounds": bounds,
        "final_position": final,
        "supervisor_position": actual_node_position,
        "grid_resolution_m": GRID_RESOLUTION,
        "robot_radius_m": ROBOT_RADIUS,
        "safety_margin_m": SAFETY_MARGIN,
        "waypoints": path,
        "expanded_nodes": len(explored),
        "planning_seconds": planning_seconds,
        "planned_length_m": path_length(path) if path else None,
        "travelled_length_m": path_length(trail),
        "simulation_duration_s": supervisor.getTime()-began,
        "movement_steps": steps,
        "goal_error_m": distance(final, goal),
        "min_physical_clearance_m": clearance,
        "safety_violation": violation,
        "sensor_readings": sensor_values,
        "trajectory": trail,
        "walls": [{"name": wall.name, "center": wall.center, "size": wall.size} for wall in walls],
    }
    (output / "latest_run.json").write_text(json.dumps(result, indent=2)+"\n")
    view.update(final, heading, trail)
    view.save(output / "map.png")
    print(f"MICROMOUSE_RESULT: {result['status']}; goal error={result['goal_error_m']:.3f} m; clearance={clearance:.3f} m; steps={steps}")
    if capture:
        supervisor.exportImage(str(Path(capture) / "maze_finished.png"), 100)
    if os.environ.get("MICROMOUSE_TEST_EXIT") == "1":
        supervisor.simulationQuit(0 if completed else 1)
    else:
        while supervisor.step(timestep) != -1:
            if show:
                view.cv2.waitKey(1)
    view.close()


if __name__ == "__main__":
    main()
