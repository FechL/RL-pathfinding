"""Frozen Double DQN controller for Webots, using exported checkpoint weights."""
import json
import math
import os
from pathlib import Path

import numpy as np

from continuous_control import ControlConfig
from lidar_dqn import FEATURES, NUM_ACTIONS, action_mask, encode_observation, get_action_mask, schema
from lidar_q_env import COLLISION_RECOVERY_SECONDS, COLLISION_RECOVERY_SPEED, GOAL_TOLERANCE, WAYPOINT_TOLERANCE
from lidar_qlearning import ACTION_VW, WaypointRecovery, aggregate_point_cloud
from maze_search import LidarMazeSearch
from planner import path_length, physical_clearance
from q_learning import ContinuousMaze, scene_signature


def load_export(path):
    payload = json.loads(Path(path).read_text())
    if (payload.get('algorithm'),payload.get('version'),payload.get('schema')) != ('masked_double_dqn',1,schema()):
        raise ValueError('Incompatible DQN export.')
    layers = [(np.asarray(layer['weight'],dtype=np.float32),
               np.asarray(layer['bias'],dtype=np.float32)) for layer in payload['layers']]
    inputs = len(FEATURES)
    for weights,bias in layers:
        if (weights.ndim!=2 or weights.shape[1]!=inputs or bias.shape!=(weights.shape[0],)
                or not np.isfinite(weights).all() or not np.isfinite(bias).all()):
            raise ValueError('Invalid exported layer.')
        inputs = weights.shape[0]
    if not layers or inputs!=NUM_ACTIONS:
        raise ValueError('DQN export must produce six Q values.')

    def predict(obs):
        for index,(weights,bias) in enumerate(layers):
            obs = weights @ obs + bias
            if index+1<len(layers):
                obs = np.maximum(obs,0.)
        return obs

    return predict, payload


def run(supervisor):
    from micromouse import MapView, PROJECT, read_scene
    period = int(supervisor.getBasicTimeStep())
    dt = period / 1000
    if supervisor.step(period) == -1:
        return
    robot,goal,walls,bounds,cell_size = read_scene(supervisor)
    start = tuple(robot.getField('translation').getSFVec3f()[:2])
    signature = scene_signature(ContinuousMaze(start,goal,walls,bounds,cell_size))
    path = Path(os.environ.get('MICROMOUSE_POLICY', str(PROJECT/'output/drl/models/webots_model.json')))
    predict, payload = load_export(path)
    if payload['maze_signature'] != signature:
        raise ValueError('DQN export belongs to a different maze.')
    gps, imu, lidar = [supervisor.getDevice(name) for name in ('gps','imu','lidar')]
    encoders = [supervisor.getDevice(f'{side} wheel sensor') for side in ('left','right')]
    motors = [supervisor.getDevice(f'{side} wheel motor') for side in ('left','right')]
    for sensor in (gps,imu,lidar,*encoders):
        sensor.enable(period)
    lidar.enablePointCloud()
    if not math.isclose(lidar.getFov(),math.pi,rel_tol=1e-5):
        raise ValueError('DQN expects the front-facing LiDAR.')
    for motor in motors:
        motor.setPosition(math.inf)
        motor.setVelocity(0.)
    for _ in range(4):
        if supervisor.step(period)==-1:
            return
    seed = int(os.environ.get('MICROMOUSE_SEARCH_SEED','0'))
    timeout = float(os.environ.get('MICROMOUSE_TIMEOUT_S','300'))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('MICROMOUSE_TIMEOUT_S must be positive and finite.')
    search = LidarMazeSearch(start,goal,bounds,cell_size,seed=seed)
    distance_scale = math.hypot(bounds[1]-bounds[0],bounds[3]-bounds[2])
    show = os.environ.get('MICROMOUSE_NO_GUI')!='1' and bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    view = MapView(start,goal,walls,bounds,[],set(),cell_size,show,'LiDAR Double DQN') if show else None
    config = ControlConfig()
    position = np.array([encoder.getValue() for encoder in encoders])
    commands = np.zeros(2)
    velocity = (0.,0.)
    target = None
    recovery = WaypointRecovery()
    elapsed_start = supervisor.getTime()
    next_decision = 0.
    contact_until = 0.
    latched = False
    collisions = 0
    clearance = math.inf
    trajectory, decisions = [], []
    status = 'interrupted'
    while supervisor.step(period)!=-1:
        elapsed = supervisor.getTime()-elapsed_start
        pose = np.array([*gps.getValues()[:2],imu.getRollPitchYaw()[2]])
        trajectory.append(pose[:2].tolist())
        new_position = np.array([encoder.getValue() for encoder in encoders])
        wheels, position = (new_position-position)/dt, new_position
        contact = physical_clearance(pose[:2],walls,bounds)
        clearance = min(clearance,contact)
        if elapsed >= contact_until and contact >= .003:
            latched = False
        if elapsed >= contact_until and contact < -.002:
            collisions += int(not latched)
            latched = True
            contact_until = elapsed + COLLISION_RECOVERY_SECONDS
            recovery.active = True
        backing = elapsed < contact_until
        if not backing and search.final and math.dist(pose[:2],goal)<=GOAL_TOLERANCE:
            status = 'success'
            break
        if elapsed >= timeout:
            status = 'timeout'
            break
        if backing:
            rear = pose[:2]-.035*np.array([math.cos(pose[2]),math.sin(pose[2])])
            safe = physical_clearance(rear,walls,bounds) >= min(0.,contact)-.001
            velocity = (COLLISION_RECOVERY_SPEED if safe else 0.,0.)
            next_decision = 0.
        else:
            reached = target is not None and not search.final and math.dist(pose[:2],target)<=WAYPOINT_TOLERANCE
            if elapsed+1e-8 >= next_decision or reached:
                points = lidar.getPointCloud()
                sectors = aggregate_point_cloud(points)
                if target is None or reached:
                    target = search.next_target(pose,[(point.x,point.y) for point in points])
                    recovery = WaypointRecovery()
                obs = encode_observation(pose,target,sectors,wheels,config,distance_scale)
                if target is None:
                    if search.scan_bearing is None:
                        status = 'search_exhausted'
                        break
                    mask = action_mask((search.scan_action(sectors),))
                else:
                    mask = get_action_mask(obs) & action_mask(recovery.actions(pose,target,sectors,elapsed))
                q = predict(obs)
                if not np.isfinite(q).all():
                    raise FloatingPointError('Nonfinite Webots Q values.')
                action = int(np.where(mask,q,-np.inf).argmax())
                velocity = ACTION_VW[action]
                decisions.append(dict(time_s=elapsed,observation=obs.tolist(),mask=mask.tolist(),
                                      q=q.tolist(),action=action,target=target,pose=pose.tolist(),
                                      wheel_velocity_rad_s=wheels.tolist(),lidar_sector_min_m=sectors.tolist(),
                                      collision_count=collisions,scanning=target is None))
                next_decision = elapsed+.16
                if view:
                    view.update(pose[:2],pose[2],trajectory,lidar_points=points)
        commands = config.wheels_for_velocity(*velocity,commands,dt)
        for motor, command in zip(motors,commands):
            motor.setVelocity(float(command))
    for motor in motors:
        motor.setVelocity(0.)
    output = Path(os.environ.get('MICROMOUSE_RESULTS_DIR',str(PROJECT/'output/drl/data/webots')))
    output.mkdir(parents=True,exist_ok=True)
    result = dict(algorithm='masked_double_dqn',status=status,seed=seed,
                  checkpoint_sha256=payload['checkpoint_sha256'],start=start,goal=goal,
                  simulation_duration_s=supervisor.getTime()-elapsed_start,
                  collision_count=collisions, travelled_length_m=path_length(trajectory),
                  goal_error_m=math.dist(gps.getValues()[:2],goal),
                  min_clearance_m=clearance if trajectory else None,
                  trajectory=trajectory, decisions=decisions, targets=search.route)
    filename = output/'latest_run.json'
    filename.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if view:
        view.save(output/'map.png')
        view.close()
    print(f'DQN Webots: {status} | {result["simulation_duration_s"]:.2f}s | {filename}',flush=True)
    if os.environ.get('MICROMOUSE_TEST_EXIT')=='1':
        supervisor.simulationQuit(0)
