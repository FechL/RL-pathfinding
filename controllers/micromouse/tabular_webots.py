"""Execute a learned tabular policy using only GPS, IMU, IR and yellow color."""
import json
import math
import os
from pathlib import Path
import random
import time

from planner import Obstacle
from tabular_q_learning import (ACTION_DELTAS, ACTION_NAMES, SensorMaze,
                                greedy_action, load_policy,
                                relative_sensor_bins, state_key)

PROJECT = Path(__file__).resolve().parents[2]
WHEEL_RADIUS = 0.027
AXLE_TRACK = 0.104
MAX_WHEEL_SPEED = 15.0
TURN_SPEED = 2.4
LINEAR_SPEED = 0.14
ACTION_TIMEOUT_S = 8.0
GOAL_PIXEL_FRACTION = .25


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def camera_sees_yellow(camera):
    image = camera.getImage()
    if not image:
        return False
    width, height = camera.getWidth(), camera.getHeight()
    x0, x1 = width//3, max(width//3+1, (2*width)//3)
    y0, y1 = height//3, max(height//3+1, (2*height)//3)
    samples = []
    # Sampling a small patch via Webots' channel helpers avoids converting the
    # whole camera image to nested Python lists at each simulator time step.
    stride_x, stride_y = max(1, (x1-x0)//6), max(1, (y1-y0)//6)
    for y in range(y0, y1, stride_y):
        for x in range(x0, x1, stride_x):
            red = camera.imageGetRed(image, width, x, y)
            green = camera.imageGetGreen(image, width, x, y)
            blue = camera.imageGetBlue(image, width, x, y)
            samples.append(red > 155 and green > 135 and blue < 115
                           and red > blue*1.45 and green > blue*1.35)
    return bool(samples) and sum(samples)/len(samples) >= GOAL_PIXEL_FRACTION


def read_arena(supervisor):
    arena = supervisor.getFromDef('ARENA')
    robot = supervisor.getFromDef('MICROMOUSE')
    group = supervisor.getFromDef('WALLS')
    if arena is None or robot is None or group is None:
        raise RuntimeError('Active world needs ARENA, MICROMOUSE, and WALLS definitions')
    width, height = arena.getField('floorSize').getSFVec2f()
    cell_size = arena.getField('cellSize').getSFFloat()
    start = tuple(robot.getField('translation').getSFVec3f()[:2])
    walls = []
    children = group.getField('children')
    for index in range(children.getCount()):
        node = children.getMFNode(index)
        pos = node.getField('translation').getSFVec3f()
        size = node.getField('size').getSFVec3f()
        name = node.getField('name').getSFString()
        walls.append(Obstacle(name, 'wall', tuple(pos[:2]), tuple(size[:2])))
    bounds = (-width/2, width/2, -height/2, height/2)
    return arena, robot, SensorMaze(start, walls, bounds, cell_size, yellow_cells=())


def run(supervisor):
    timestep = int(supervisor.getBasicTimeStep())
    arena, robot, env = read_arena(supervisor)
    model_path = Path(os.environ.get('MICROMOUSE_POLICY', str(PROJECT/'models/tabular_q_learning.json')))
    if not model_path.is_absolute():
        model_path = PROJECT/model_path
    q_table, metadata = load_policy(model_path, env)
    gps, imu = supervisor.getDevice('gps'), supervisor.getDevice('imu')
    camera = supervisor.getDevice('yellow color sensor')
    sensors = {name: supervisor.getDevice(name) for name in (
        'front left sensor', 'front right sensor', 'left side sensor',
        'rear sensor', 'right side sensor')}
    motors = [supervisor.getDevice('left wheel motor'), supervisor.getDevice('right wheel motor')]
    for device in [gps, imu, camera, *sensors.values()]:
        device.enable(timestep)
    for motor in motors:
        motor.setPosition(float('inf'))
        motor.setVelocity(0.)
    for _ in range(3):
        if supervisor.step(timestep) == -1:
            return

    rng = random.Random(7)
    trajectory, decisions, collisions = [], [], 0
    final_state = None
    beginning = supervisor.getTime()
    max_steps = int(os.environ.get('MICROMOUSE_MAX_STEPS', '500'))

    def observe():
        position = tuple(gps.getValues()[:2])
        yaw = imu.getRollPitchYaw()[2]
        cell = env.cell_id(position)
        heading = round((yaw % (2*math.pi))/(math.pi/2)) % 4
        bins = relative_sensor_bins(
            sensors['front left sensor'].getValue(),
            sensors['front right sensor'].getValue(),
            sensors['left side sensor'].getValue(),
            sensors['rear sensor'].getValue(),
            sensors['right side sensor'].getValue())
        yellow = camera_sees_yellow(camera)
        return position, yaw, (cell, heading, *bins, int(yellow)), yellow

    def wheels(left, right):
        peak = max(abs(left), abs(right), 1e-9)
        if peak > MAX_WHEEL_SPEED:
            left, right = left*MAX_WHEEL_SPEED/peak, right*MAX_WHEEL_SPEED/peak
        motors[0].setVelocity(left)
        motors[1].setVelocity(right)

    def stop():
        wheels(0., 0.)

    def tick():
        return supervisor.step(timestep) != -1

    def turn_to(desired):
        started = supervisor.getTime()
        while supervisor.getTime()-started < 3.5:
            yaw = imu.getRollPitchYaw()[2]
            error = wrap_angle(desired-yaw)
            if abs(error) <= .06:
                stop()
                return True
            direction = math.copysign(1., error)
            wheels(-direction*TURN_SPEED, direction*TURN_SPEED)
            if not tick():
                return False
        stop()
        return False

    def move_one_cell(action):
        nonlocal collisions
        position, yaw, state, yellow = observe()
        final_state = state
        heading = action * math.pi/2
        if not turn_to(heading):
            collisions += 1
            return False, 'turn_timeout'
        x, y = env.cell_xy(position)
        dx, dy = ACTION_DELTAS[action]
        nx, ny = x+dx, y+dy
        if not (0 <= nx < env.width and 0 <= ny < env.height):
            collisions += 1
            return False, 'arena_boundary'
        target = env.center(nx, ny)
        began = supervisor.getTime()
        last_distance = math.inf
        last_progress = began
        while supervisor.getTime()-began < ACTION_TIMEOUT_S:
            position = tuple(gps.getValues()[:2])
            yaw = imu.getRollPitchYaw()[2]
            left = sensors['front left sensor'].getValue()
            right = sensors['front right sensor'].getValue()
            if camera_sees_yellow(camera):
                stop()
                return True, 'yellow_detected'
            distance = math.hypot(target[0]-position[0], target[1]-position[1])
            if distance <= .035:
                stop()
                return True, 'cell_reached'
            if distance < last_distance-.004:
                last_distance, last_progress = distance, supervisor.getTime()
            elif supervisor.getTime()-last_progress > 1.5:
                collisions += 1
                stop()
                return False, 'no_progress_collision'
            # A very close forward IR return is treated as a blocked action.
            if min(left, right) >= 3900:
                collisions += 1
                stop()
                return False, 'ir_collision'
            error = wrap_angle(heading - yaw)
            angular = clamp(2.0*error, -.55, .55)
            linear = min(LINEAR_SPEED, max(.045, distance*.8))
            left_speed = (linear-angular*AXLE_TRACK/2)/WHEEL_RADIUS
            right_speed = (linear+angular*AXLE_TRACK/2)/WHEEL_RADIUS
            wheels(left_speed, right_speed)
            if not tick():
                return False, 'simulator_closed'
        collisions += 1
        stop()
        return False, 'action_timeout'

    stop()
    success = False
    termination = 'timeout'
    for step in range(1, max_steps+1):
        if supervisor.step(timestep) == -1:
            termination = 'simulator_closed'
            break
        position, yaw, state, yellow = observe()
        trajectory.append([*position, yaw])
        if yellow:
            success, termination = True, 'yellow_detected'
            print(f'Goal sensor: yellow detected at step {step}; coordinates are not queried.', flush=True)
            break
        key = state_key(state)
        values = q_table.get(key, [0.] * len(ACTION_NAMES))
        action = greedy_action(values, rng=rng, random_ties=True)
        before = position
        moved, outcome = move_one_cell(action)
        after, after_yaw, next_state, goal_after = observe()
        final_state = next_state
        goal_after = goal_after or outcome == 'yellow_detected'
        reward = 100. if goal_after else (-10. if not moved else -1.)
        decisions.append(dict(step=step, state=state, action=ACTION_NAMES[action],
                              q_values=[round(float(q), 6) for q in values],
                              reward=reward, collision=not moved, outcome=outcome,
                              position_before=before, position_after=after,
                              yellow_detected=goal_after))
        if goal_after:
            success, termination = True, 'yellow_detected'
            print(f'Goal sensor: yellow detected after action {step}.', flush=True)
            break
        if step % 25 == 0:
            print(f"Sim step {step}/{max_steps}: cell={state[0]}, action={ACTION_NAMES[action]}, q={values}, collisions={collisions}", flush=True)
    stop()
    result = dict(status='success' if success else 'failed', algorithm='tabular_q_learning',
                  termination=termination, model=str(model_path), policy_metadata=metadata,
                  episode_steps=len(decisions), collisions=collisions,
                  elapsed_sim_seconds=supervisor.getTime()-beginning,
                  final_state=final_state,
                  yellow_detection_only=True, decisions=decisions, trajectory=trajectory)
    output = Path(os.environ.get('MICROMOUSE_RESULTS_DIR', str(PROJECT/'results')))
    output.mkdir(parents=True, exist_ok=True)
    (output/'latest_run.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f"MICROMOUSE_RESULT: {result['status']}; yellow_sensor={success}; steps={len(decisions)}; collisions={collisions}", flush=True)
    if os.environ.get('MICROMOUSE_TEST_EXIT') == '1':
        supervisor.simulationQuit(0 if success else 1)
    else:
        while supervisor.step(timestep) != -1:
            pass
