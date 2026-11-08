"""Execute continuous policy inference and motor updates at each physics tick."""
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from continuous_control import NavigationMap, read_policy
from planner import ROBOT_RADIUS, distance, path_length, physical_clearance
from motion_metrics import motion_metrics


def run(supervisor, algorithm):
    from micromouse import MapView, PROJECT, read_scene
    timestep=int(supervisor.getBasicTimeStep())
    dt=timestep/1000
    if timestep != 16:
        raise ValueError('The trained continuous controller requires a 16 ms Webots basicTimeStep')
    if supervisor.step(timestep)==-1:return
    robot,goal,walls,bounds,cell_size=read_scene(supervisor)
    start=tuple(robot.getField('translation').getSFVec3f()[:2])
    guided = algorithm != 'rl_unguided'
    nav=NavigationMap(start,goal,walls,bounds,cell_size,guided=guided)
    policy_name = {'naf':'naf_actor.json','sac':'continuous_actor.json','rl':'rl_actor.json',
                   'rl_unguided':'rl_unguided_actor.json'}[algorithm]
    label = 'RL dengan panduan' if algorithm == 'rl' else 'RL tanpa panduan' if algorithm == 'rl_unguided' else algorithm.upper()
    policy_path = Path(os.environ.get('MICROMOUSE_POLICY', str(PROJECT/'models'/policy_name)))
    if algorithm.startswith('rl') and not policy_path.is_file():
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
    for name in ('front left sensor','front right sensor','left side sensor','right side sensor'):
        sensor=supervisor.getDevice(name);sensor.enable(timestep);sensors[name]=sensor
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
    guidance = 'known-map local bearing' if nav.guided else 'goal and simulated local range observations'
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
        if distance(pose[:2],goal)<=nav.config.goal_tolerance:
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
            observation=nav.observation(pose,np.clip(measured,-nav.config.wheel_limit,nav.config.wheel_limit))
            action=actor(observation)
            inference_seconds+=time.perf_counter()-timer
        commands=nav.config.wheels(action,commands,dt)
        for motor,command in zip(motors,commands):motor.setVelocity(float(command))
        telemetry.append(dict(time_s=current_time-began,pose=pose.tolist(),action=action.tolist(),
                              commanded_wheels=commands.tolist(),measured_wheels=measured.tolist(),
                              ir={name:s.getValue() for name,s in sensors.items()}))
        steps+=1
        if show and steps%2==0:view.update(pose[:2],pose[2],trail)
    else:
        view.close();return
    for motor in motors:motor.setVelocity(0)
    result=dict(status='success' if success else 'failed',algorithm=algorithm,controller=f'{label} continuous forward/yaw control',
                policy_metadata={k:v for k,v in metadata.items() if k!='layers'},python=sys.executable,
                start=start,goal=goal,arena_bounds=bounds,cell_size_m=cell_size,
                final_position=pose[:2].tolist(),goal_error_m=distance(pose[:2],goal),
                min_physical_clearance_m=clearance,safety_violation=violation,
                simulation_duration_s=supervisor.getTime()-began,movement_steps=steps,
                planned_length_m=None,travelled_length_m=path_length(trail),
                inference_seconds=inference_seconds,policy_interval_s=policy_ticks*dt,
                wheel_response_time_s=nav.config.wheel_response_time_s,trajectory=trail,telemetry=telemetry,
                motion=motion_metrics([r['time_s'] for r in telemetry],[r['commanded_wheels'] for r in telemetry]),
                walls=[dict(name=w.name,center=w.center,size=w.size) for w in walls])
    (output/'latest_run.json').write_text(json.dumps(result,indent=2)+'\n')
    view.update(pose[:2],pose[2],trail);view.save(output/'map.png')
    if capture:supervisor.exportImage(str(Path(capture)/'maze_finished.png'),100)
    print(f"MICROMOUSE_RESULT: {result['status']}; time={result['simulation_duration_s']:.3f}s; goal_error={result['goal_error_m']:.4f}m; clearance={clearance:.4f}m; reason={violation}",flush=True)
    if os.environ.get('MICROMOUSE_TEST_EXIT')=='1':
        view.close();supervisor.simulationQuit(0 if success else 1)
    else:
        while supervisor.step(timestep)!=-1:
            if show:view.cv2.waitKey(1)
        view.close()
