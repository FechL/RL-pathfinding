"""Run a learned sparse tabular Q policy with Webots' physical 360 lidar."""
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from continuous_control import ControlConfig
from lidar_qlearning import (ACTION_NAMES,ACTION_VW,TabularLidarQ,
                             aggregate_point_cloud,discretize_state,gps_cell,
                             safe_actions)
from planner import ROBOT_RADIUS,physical_clearance,path_length
from q_learning import ContinuousMaze,scene_signature


def find_policy(project):
    explicit=os.environ.get('MICROMOUSE_POLICY')
    if explicit:
        path=Path(explicit).expanduser()
        return path if path.is_absolute() else project/path
    runs=sorted((project/'reports/lidar_q_learning/runs').glob('*/q_table.json'),reverse=True)
    if runs:
        return runs[0]
    return project/'models/lidar_q_table.json'


def run(supervisor):
    from micromouse import MapView,PROJECT,read_scene
    timestep=int(supervisor.getBasicTimeStep())
    if timestep!=16:
        raise ValueError('Lidar Q controller expects a Webots basicTimeStep of 16 ms')
    if supervisor.step(timestep)==-1:return
    robot,goal,walls,bounds,cell_size=read_scene(supervisor)
    position=robot.getField('translation').getSFVec3f()
    start=tuple(position[:2])
    maze=ContinuousMaze(start,goal,walls,bounds,cell_size)
    signature=scene_signature(maze)
    policy_path=find_policy(PROJECT)
    if not policy_path.is_file():
        raise SystemExit('Q-table not found. Run: bash run.sh train --algorithm q_learning')
    agent,metadata=TabularLidarQ.load(policy_path)
    trained_signature=metadata.get('training',{}).get('maze_signature')
    if trained_signature and trained_signature!=signature:
        raise ValueError('The Q-table was trained for a different maze/start/goal. Train this world first.')

    gps=supervisor.getDevice('gps');gps.enable(timestep)
    imu=supervisor.getDevice('imu');imu.enable(timestep)
    lidar=supervisor.getDevice('lidar');lidar.enable(timestep);lidar.enablePointCloud()
    motors=[supervisor.getDevice('left wheel motor'),supervisor.getDevice('right wheel motor')]
    for motor in motors:
        motor.setPosition(float('inf'));motor.setVelocity(0.)
    for _ in range(4):
        if supervisor.step(timestep)==-1:return

    show=os.environ.get('MICROMOUSE_NO_GUI')!='1' and (sys.platform!='linux' or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))
    view=MapView(start,goal,walls,bounds,[],set(),cell_size,show,'Tabular Q-Learning + LiDAR')
    config=ControlConfig()
    decision_ticks=max(1,round(.16/(timestep/1000)))
    timeout_s=float(os.environ.get('MICROMOUSE_TIMEOUT_S','120'))
    max_ticks=math.ceil(timeout_s/(timestep/1000))
    commands=np.zeros(2,dtype=float)
    action=0
    previous_action=None
    steps=0
    success=False
    collision=False
    began=supervisor.getTime()
    trail=[]
    telemetry=[]
    clearance=float('inf')
    scan=lidar.getPointCloud()
    sectors=aggregate_point_cloud(scan)
    pose=np.array([*gps.getValues()[:2],imu.getRollPitchYaw()[2]])
    state=discretize_state(pose,goal,sectors,previous_action,
                           position_cell=gps_cell(pose[:2],bounds,cell_size))
    policy_ticks=0
    print(f'Tabular Q + LiDAR; table states={len(agent.q):,}; goal={tuple(goal)}; policy={policy_path}',flush=True)

    while supervisor.step(timestep)!=-1:
        pose=np.array([*gps.getValues()[:2],imu.getRollPitchYaw()[2]])
        point=pose[:2].copy();trail.append(point.tolist())
        physical=physical_clearance(point,walls,bounds)
        clearance=min(clearance,physical)
        if physical<-.002:
            collision=True
            break
        if np.linalg.norm(np.asarray(goal)-point)<=.09:
            success=True
            break
        if supervisor.getTime()-began>=timeout_s or steps>=max_ticks:
            break

        if policy_ticks%decision_ticks==0:
            scan=lidar.getPointCloud()
            sectors=aggregate_point_cloud(scan)
            state=discretize_state(pose,goal,sectors,previous_action,
                                   position_cell=gps_cell(pose[:2],bounds,cell_size))
            values=agent.values(state)
            allowed=safe_actions(sectors)
            action=max(allowed,key=lambda candidate:float(values[candidate]))
            previous_action=action
            policy_ticks=0
        commands=config.wheels_for_velocity(*ACTION_VW[action],commands,timestep/1000)
        for motor,speed in zip(motors,commands):motor.setVelocity(float(speed))
        telemetry.append(dict(time_s=supervisor.getTime()-began,pose=pose.tolist(),
                              distance_from_base_m=float(np.linalg.norm(point-np.asarray(start))),
                              distance_to_goal_m=float(np.linalg.norm(point-np.asarray(goal))),
                              q_state=list(state),q_values=agent.values(state).tolist(),
                              action=ACTION_NAMES[action],action_index=action,
                              lidar_sector_min_m=sectors.tolist(),
                              lidar_points=len(scan),commanded_wheels=commands.tolist(),
                              clearance_m=physical))
        steps+=1;policy_ticks+=1
        if show and steps%2==0:
            view.update(point,pose[2],trail,lidar_points=scan)

    for motor in motors:motor.setVelocity(0.)
    final=np.array(gps.getValues()[:2])
    final_heading=float(imu.getRollPitchYaw()[2])
    if not trail or np.linalg.norm(final-trail[-1])>1e-9:trail.append(final.tolist())
    result=dict(status='success' if success else 'collision' if collision else 'timeout',
                algorithm='tabular_q_lidar',policy=str(policy_path.resolve()),
                table_states=len(agent.q),goal=tuple(goal),start=start,final_position=final.tolist(),
                goal_error_m=float(np.linalg.norm(final-np.asarray(goal))),
                distance_from_base_m=float(np.linalg.norm(final-np.asarray(start))),
                simulation_duration_s=supervisor.getTime()-began,movement_steps=steps,
                travelled_length_m=path_length(trail),min_physical_clearance_m=clearance,
                trajectory=trail,telemetry=telemetry)
    output=Path(os.environ.get('MICROMOUSE_RESULTS_DIR',str(PROJECT/'results')))
    output.mkdir(parents=True,exist_ok=True)
    (output/'latest_run.json').write_text(json.dumps(result,indent=2)+'\n')
    if show:
        view.update(final,final_heading,trail,lidar_points=scan)
        view.save(output/'map.png')
        view.close()
    print('MICROMOUSE_RESULT: '+json.dumps(result,sort_keys=True),flush=True)
