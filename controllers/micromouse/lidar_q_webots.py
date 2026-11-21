"""Search the maze with DFS, then use tabular Q for each motion segment."""
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from continuous_control import ControlConfig
from lidar_qlearning import (ACTION_NAMES,ACTION_VW,TabularLidarQ,
                             WaypointRecovery,aggregate_point_cloud,discretize_state)
from maze_search import depth_first_path
from planner import physical_clearance,path_length
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
        raise ValueError('LiDAR Q controller expects a Webots basicTimeStep of 16 ms')
    if supervisor.step(timestep)==-1:return
    robot,goal,walls,bounds,cell_size=read_scene(supervisor)
    position=robot.getField('translation').getSFVec3f()
    start=tuple(position[:2])
    maze=ContinuousMaze(start,goal,walls,bounds,cell_size)
    signature=scene_signature(maze)
    path,search_cells=depth_first_path(start,goal,walls,bounds,cell_size)
    policy_path=find_policy(PROJECT)
    if not policy_path.is_file():
        raise SystemExit('Q movement table not found. Run: bash run.sh train --algorithm q_learning')
    try:
        agent,metadata=TabularLidarQ.load(policy_path)
    except ValueError as error:
        raise SystemExit(f'{error}. Train a movement-only Q-table before simulation.') from error
    training=metadata.get('training',{})
    trained_signature=training.get('maze_signature')
    if trained_signature and trained_signature!=signature:
        raise ValueError('The Q table was trained for a different maze. Train this world first.')
    if training.get('role') and 'continuous differential-drive motion' not in training['role']:
        raise ValueError('This Q table was not trained for robot motion; train the start-to-goal DFS/Q controller first.')

    gps=supervisor.getDevice('gps');gps.enable(timestep)
    imu=supervisor.getDevice('imu');imu.enable(timestep)
    lidar=supervisor.getDevice('lidar');lidar.enable(timestep);lidar.enablePointCloud()
    motors=[supervisor.getDevice('left wheel motor'),supervisor.getDevice('right wheel motor')]
    for motor in motors:
        motor.setPosition(float('inf'));motor.setVelocity(0.)
    for _ in range(4):
        if supervisor.step(timestep)==-1:return

    show=os.environ.get('MICROMOUSE_NO_GUI')!='1' and (sys.platform!='linux' or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))
    view=MapView(start,goal,walls,bounds,path,set(),cell_size,show,'DFS search + tabular Q motion')
    config=ControlConfig()
    decision_ticks=max(1,round(.16/(timestep/1000)))
    timeout_s=float(os.environ.get('MICROMOUSE_TIMEOUT_S','300'))
    max_ticks=math.ceil(timeout_s/(timestep/1000))
    commands=np.zeros(2,dtype=float)
    action=0
    recovery=WaypointRecovery()
    target_index=1 if len(path)>1 else 0
    steps=0;policy_ticks=0
    success=False;collision=False;timed_out=False
    began=supervisor.getTime()
    trail=[];telemetry=[];clearance=float('inf')
    scan=lidar.getPointCloud();sectors=aggregate_point_cloud(scan)
    print(f'DFS search visited {search_cells} cells and found {len(path)-1} cell transitions; Q table states={len(agent.q):,}; policy={policy_path}',flush=True)

    while supervisor.step(timestep)!=-1:
        pose=np.array([*gps.getValues()[:2],imu.getRollPitchYaw()[2]])
        point=pose[:2].copy();trail.append(point.tolist())
        physical=physical_clearance(point,walls,bounds)
        clearance=min(clearance,physical)
        goal_error=float(np.linalg.norm(np.asarray(goal)-point))
        if physical<-.002:
            collision=True
            break
        if goal_error<=.09:
            success=True
            break
        if supervisor.getTime()-began>=timeout_s or steps>=max_ticks:
            timed_out=True
            break

        target=np.asarray(path[min(target_index,len(path)-1)],dtype=float)
        target_error=float(np.linalg.norm(target-point))
        if target_error<=.055 and target_index<len(path)-1:
            target_index+=1
            recovery=WaypointRecovery()
            policy_ticks=0
            target=np.asarray(path[target_index],dtype=float)
            target_error=float(np.linalg.norm(target-point))

        if policy_ticks%decision_ticks==0:
            scan=lidar.getPointCloud()
            sectors=aggregate_point_cloud(scan)
            state=discretize_state(pose,target,sectors)
            values=agent.values(state)
            allowed=recovery.actions(pose,target,sectors,supervisor.getTime()-began)
            action=max(allowed,key=lambda candidate:float(values[candidate]))
            policy_ticks=0
        commands=config.wheels_for_velocity(*ACTION_VW[action],commands,timestep/1000)
        for motor,speed in zip(motors,commands):motor.setVelocity(float(speed))
        telemetry.append(dict(time_s=supervisor.getTime()-began,pose=pose.tolist(),
                              search_path_index=target_index,target=target.tolist(),
                              distance_to_waypoint_m=target_error,distance_to_goal_m=goal_error,
                              q_state=list(state),q_values=agent.values(state).tolist(),
                              action=ACTION_NAMES[action],action_index=action,
                              waypoint_recovery=recovery.active,
                              lidar_sector_min_m=sectors.tolist(),lidar_points=len(scan),
                              commanded_wheels=commands.tolist(),clearance_m=physical))
        steps+=1;policy_ticks+=1
        if show and steps%2==0:
            view.update(point,pose[2],trail,lidar_points=scan)

    for motor in motors:motor.setVelocity(0.)
    final=np.array(gps.getValues()[:2])
    final_heading=float(imu.getRollPitchYaw()[2])
    if not trail or np.linalg.norm(final-trail[-1])>1e-9:trail.append(final.tolist())
    result=dict(status='success' if success else 'collision' if collision else 'timeout',
                algorithm='dfs_search_with_tabular_q_motion',search_algorithm='depth_first_search',
                policy=str(policy_path.resolve()),table_states=len(agent.q),
                search_cells_visited=search_cells,search_path=path,goal=tuple(goal),start=start,
                final_position=final.tolist(),goal_error_m=float(np.linalg.norm(final-np.asarray(goal))),
                distance_from_base_m=float(np.linalg.norm(final-np.asarray(start))),
                simulation_duration_s=supervisor.getTime()-began,movement_steps=steps,
                travelled_length_m=path_length(trail),planned_path_length_m=path_length(path),
                min_physical_clearance_m=clearance,trajectory=trail,telemetry=telemetry,
                timeout=timed_out)
    output=Path(os.environ.get('MICROMOUSE_RESULTS_DIR',str(PROJECT/'results')))
    output.mkdir(parents=True,exist_ok=True)
    (output/'latest_run.json').write_text(json.dumps(result,indent=2)+'\n')
    if show:
        view.update(final,final_heading,trail,lidar_points=scan)
        view.save(output/'map.png')
        view.close()
    print('MICROMOUSE_RESULT: '+json.dumps(result,sort_keys=True),flush=True)
    if os.environ.get('MICROMOUSE_TEST_EXIT')=='1':
        supervisor.simulationQuit(0)
