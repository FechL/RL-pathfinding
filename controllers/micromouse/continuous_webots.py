"""Execute continuous policy inference and motor updates at each physics tick."""
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from continuous_control import NavigationMap, read_policy
from planner import ROBOT_RADIUS, Obstacle, distance, path_length, physical_clearance
from motion_metrics import motion_metrics


def read_hidden_scene(supervisor):
    """Read only arena geometry and walls; deliberately never query DEF GOAL."""
    arena = supervisor.getFromDef('ARENA')
    robot = supervisor.getFromDef('MICROMOUSE')
    group = supervisor.getFromDef('WALLS')
    if arena is None or robot is None or group is None:
        raise RuntimeError('Hidden-goal simulation needs ARENA, MICROMOUSE and WALLS')
    width, height = arena.getField('floorSize').getSFVec2f()
    cell_size = arena.getField('cellSize').getSFFloat()
    children = group.getField('children')
    walls = []
    for index in range(children.getCount()):
        node = children.getMFNode(index)
        position = node.getField('translation').getSFVec3f()
        size = node.getField('size').getSFVec3f()
        name = node.getField('name').getSFString()
        walls.append(Obstacle(name, 'wall', tuple(position[:2]), tuple(size[:2])))
    bounds = (-width/2, width/2, -height/2, height/2)
    return robot, walls, bounds, cell_size


def infrared_range_m(sensor):
    """Invert the five-point IR lookup table into the same range as training."""
    raw = float(sensor.getValue())
    return float(np.interp(raw, [0, 450, 1500, 2800, 4095], [.4, .2, .1, .05, 0]))


def camera_sees_yellow(camera):
    image = camera.getImage()
    if image is None:
        return False
    width, height = camera.getWidth(), camera.getHeight()
    x0, x1 = width//3, 2*width//3
    y0, y1 = height//3, 2*height//3
    yellow = total = 0
    for y in range(y0, y1, 2):
        for x in range(x0, x1, 2):
            red = camera.imageGetRed(image, width, x, y)
            green = camera.imageGetGreen(image, width, x, y)
            blue = camera.imageGetBlue(image, width, x, y)
            yellow += red > 150 and green > 135 and blue < 120 and red > blue*1.45 and green > blue*1.35
            total += 1
    return total > 0 and yellow/total >= .3


def run(supervisor, algorithm):
    from micromouse import MapView, PROJECT, read_scene
    timestep=int(supervisor.getBasicTimeStep())
    dt=timestep/1000
    if timestep != 16:
        raise ValueError('The trained continuous controller requires a 16 ms Webots basicTimeStep')
    if supervisor.step(timestep)==-1:return
    hidden_goal = algorithm == 'hidden_goal'
    if hidden_goal:
        robot,walls,bounds,cell_size=read_hidden_scene(supervisor)
        goal=None
    else:
        robot,goal,walls,bounds,cell_size=read_scene(supervisor)
    start=tuple(robot.getField('translation').getSFVec3f()[:2])
    guided = algorithm not in ('rl_unguided','hidden_goal')
    nav=NavigationMap(start,goal,walls,bounds,cell_size,guided=guided,hidden_goal=hidden_goal)
    policy_name = {'naf':'naf_actor.json','sac':'continuous_actor.json','rl':'rl_actor.json',
                   'rl_unguided':'rl_unguided_actor.json',
                   'hidden_goal':'hidden_goal_actor.json'}[algorithm]
    label = ('SAC continuous sensor-goal' if hidden_goal else
             'RL dengan panduan' if algorithm == 'rl' else
             'RL tanpa panduan' if algorithm == 'rl_unguided' else algorithm.upper())
    policy_path = Path(os.environ.get('MICROMOUSE_POLICY', str(PROJECT/'models'/policy_name)))
    if (algorithm.startswith('rl') or hidden_goal) and not policy_path.is_file():
        raise SystemExit(f'No policy is installed. Train using bash run.sh train --algorithm {algorithm}.')
    actor,metadata=read_policy(policy_path,nav,algorithm)
    reset_pose = os.environ.get('MICROMOUSE_RL_RESET_POSE')
    if reset_pose is not None and algorithm == 'rl':
        x, y, yaw = map(float, json.loads(reset_pose))
        if not all(map(math.isfinite, (x, y, yaw))) or not nav.free((x, y), (x, y)):
            raise ValueError('The requested evaluation pose is unsafe')
        translation = robot.getField('translation')
        translation.setSFVec3f([x, y, translation.getSFVec3f()[2]])
        robot.getField('rotation').setSFRotation([0., 0., 1., yaw])
        robot.resetPhysics()
        start = (x, y)
    gps=supervisor.getDevice('gps');imu=supervisor.getDevice('imu')
    gps.enable(timestep);imu.enable(timestep)
    motors=[supervisor.getDevice(f'{side} wheel motor') for side in ('left','right')]
    encoders=[supervisor.getDevice(f'{side} wheel sensor') for side in ('left','right')]
    for motor,encoder in zip(motors,encoders):
        motor.setPosition(float('inf'));motor.setVelocity(0);encoder.enable(timestep)
    sensors={}
    sensor_names=('front left sensor','front right sensor','left side sensor','rear sensor','right side sensor')
    for name in sensor_names:
        sensor=supervisor.getDevice(name);sensor.enable(timestep);sensors[name]=sensor
    camera=None
    if hidden_goal:
        camera=supervisor.getDevice('yellow camera')
        camera.enable(timestep)
    for _ in range(10):
        if supervisor.step(timestep)==-1:return
    show=os.environ.get('MICROMOUSE_NO_GUI')!='1' and (sys.platform!='linux' or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))
    output=Path(os.environ.get('MICROMOUSE_RESULTS_DIR',str(PROJECT/'results')));output.mkdir(parents=True,exist_ok=True)
    capture=os.environ.get('MICROMOUSE_CAPTURE_DIR')
    if capture:
        Path(capture).mkdir(parents=True,exist_ok=True)
        supervisor.exportImage(str(Path(capture)/'maze_overview.png'),100)
    view=MapView(start,goal,walls,bounds,[],set(),cell_size,show,f'{label} continuous control')
    began=supervisor.getTime()
    pose=np.array([*gps.getValues()[:2],imu.getRollPitchYaw()[2]])
    trail=[pose[:2].tolist()];telemetry=[]
    commands=np.zeros(2);measured=np.zeros(2);previous_encoder=np.array([e.getValue() for e in encoders])
    steps=0;success=False;violation=None
    policy_ticks = 1 if nav.guided else round(nav.config.dt/dt)
    minimum_clearance = .005 if nav.guided else 0.
    clearance=physical_clearance(pose[:2],walls,bounds)
    best_progress=math.inf;last_progress=began
    inference_seconds=0.
    guidance = ('five IR sensors and yellow camera flag; no goal coordinates' if hidden_goal else
                'known-map local bearing' if nav.guided else 'goal and simulated local range observations')
    print(f'{label}: continuous forward velocity / yaw-rate policy; {guidance}',flush=True)
    while supervisor.step(timestep)!=-1:
        current_time=supervisor.getTime()
        pose=np.array([*gps.getValues()[:2],imu.getRollPitchYaw()[2]])
        positions=np.array([e.getValue() for e in encoders])
        measured=(positions-previous_encoder)/dt;previous_encoder=positions
        physical=physical_clearance(pose[:2],walls,bounds)
        clearance=min(clearance,physical)
        trail.append(pose[:2].tolist())
        if physical<minimum_clearance or not nav.free(trail[-2],trail[-1],ROBOT_RADIUS):
            violation='Physical wall clearance or swept trajectory check failed';break
        yellow = camera_sees_yellow(camera) if hidden_goal else False
        if hidden_goal and yellow:
            success=True;break
        if not hidden_goal and distance(pose[:2],goal)<=nav.config.goal_tolerance:
            success=True;break
        if current_time-began>=nav.config.max_seconds:
            violation='Episode time limit';break
        if nav.guided:
            _,progress=nav.guidance(pose[:2])
            if progress<best_progress-.02:
                best_progress=progress;last_progress=current_time
            elif current_time-last_progress>15:
                violation='No navigation progress for 15 seconds';break
        if steps%policy_ticks==0:
            timer=time.perf_counter()
            clipped_measured=np.clip(measured,-nav.config.wheel_limit,nav.config.wheel_limit)
            if hidden_goal:
                ir=np.array([infrared_range_m(sensors[name])/.4 for name in sensor_names],dtype=np.float32)
                observation=nav.observation(pose,clipped_measured,ir_values=ir,yellow_detected=yellow)
            else:
                observation=nav.observation(pose,clipped_measured)
            action=actor(observation)
            inference_seconds+=time.perf_counter()-timer
        commands=nav.config.wheels(action,commands,dt)
        for motor,command in zip(motors,commands):motor.setVelocity(float(command))
        telemetry_row=dict(time_s=current_time-began,pose=pose.tolist(),action=action.tolist(),
                           commanded_wheels=commands.tolist(),measured_wheels=measured.tolist(),
                           ir={name:s.getValue() for name,s in sensors.items()})
        if hidden_goal:
            telemetry_row['yellow_detected']=yellow
        telemetry.append(telemetry_row)
        steps+=1
        if show and steps%2==0:view.update(pose[:2],pose[2],trail)
    else:
        view.close();return
    for motor in motors:motor.setVelocity(0)
    result=dict(status='success' if success else 'failed',algorithm=algorithm,controller=f'{label} continuous forward/yaw control',
                policy_metadata={k:v for k,v in metadata.items() if k!='layers'},python=sys.executable,
                start=start,arena_bounds=bounds,cell_size_m=cell_size,
                final_position=pose[:2].tolist(),
                min_physical_clearance_m=clearance,safety_violation=violation,
                simulation_duration_s=supervisor.getTime()-began,movement_steps=steps,
                planned_length_m=None,travelled_length_m=path_length(trail),
                inference_seconds=inference_seconds,policy_interval_s=policy_ticks*dt,
                wheel_response_time_s=nav.config.wheel_response_time_s,trajectory=trail,telemetry=telemetry,
                motion=motion_metrics([r['time_s'] for r in telemetry],[r['commanded_wheels'] for r in telemetry]),
                walls=[dict(name=w.name,center=w.center,size=w.size) for w in walls])
    if hidden_goal:
        result['goal_detection']='yellow camera only'
        result['yellow_detected']=bool(success)
    else:
        result['goal']=goal
        result['goal_error_m']=distance(pose[:2],goal)
    (output/'latest_run.json').write_text(json.dumps(result,indent=2)+'\n')
    view.update(pose[:2],pose[2],trail);view.save(output/'map.png')
    if capture:supervisor.exportImage(str(Path(capture)/'maze_finished.png'),100)
    detail=(f"yellow={int(success)}" if hidden_goal else f"goal_error={result['goal_error_m']:.4f}m")
    print(f"MICROMOUSE_RESULT: {result['status']}; time={result['simulation_duration_s']:.3f}s; {detail}; clearance={clearance:.4f}m; reason={violation}",flush=True)
    if os.environ.get('MICROMOUSE_TEST_EXIT')=='1':
        view.close();supervisor.simulationQuit(0 if success else 1)
    else:
        while supervisor.step(timestep)!=-1:
            if show:view.cv2.waitKey(1)
        view.close()
