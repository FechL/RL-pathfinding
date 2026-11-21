"""Explore the maze with LiDAR and drive with tabular Q-learning."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np

from continuous_control import ControlConfig
from lidar_drl import drive_command, load_actor, observation
from lidar_q_env import (COLLISION_PENALTY, COLLISION_RECOVERY_SECONDS,
                         COLLISION_RECOVERY_SPEED)
from lidar_qlearning import (ACTION_NAMES,ACTION_VW,TabularLidarQ,
                             WaypointRecovery,aggregate_point_cloud,discretize_state)
from maze_search import LidarMazeSearch
from planner import physical_clearance,path_length
from q_learning import ContinuousMaze,scene_signature


def find_policy(project, drl=False):
    explicit=os.environ.get('MICROMOUSE_POLICY')
    if explicit:
        path=Path(explicit).expanduser()
        return path if path.is_absolute() else project/path
    directory, filename = ('lidar_drl', 'actor.json') if drl else ('lidar_q_learning', 'q_table.json')
    runs=sorted((project/'reports'/directory/'runs').glob(f'*/{filename}'),reverse=True)
    if runs:
        return runs[0]
    return project/'models'/('lidar_sac_actor.json' if drl else 'lidar_q_table.json')


def run(supervisor, *, drl=False):
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
    policy_path=find_policy(PROJECT, drl)
    if not policy_path.is_file():
        algorithm = 'drl' if drl else 'q_learning'
        raise SystemExit(f'Policy not found. Run: bash run.sh train --algorithm {algorithm}')
    try:
        if drl:
            actor, metadata = load_actor(policy_path)
        else:
            agent, metadata = TabularLidarQ.load(policy_path)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    training=metadata.get('training',{})
    if training.get('role') and 'continuous differential-drive motion' not in training['role']:
        raise ValueError('This Q table was not trained for robot motion; train the start-to-goal DFS/Q controller first.')

    gps=supervisor.getDevice('gps');gps.enable(timestep)
    imu=supervisor.getDevice('imu');imu.enable(timestep)
    lidar=supervisor.getDevice('lidar');lidar.enable(timestep);lidar.enablePointCloud()
    if not math.isclose(lidar.getFov(), math.pi, rel_tol=1e-5):
        raise ValueError('The Q controller requires a front-facing 180-degree LiDAR')
    motors=[supervisor.getDevice('left wheel motor'),supervisor.getDevice('right wheel motor')]
    for motor in motors:
        motor.setPosition(float('inf'));motor.setVelocity(0.)
    for _ in range(4):
        if supervisor.step(timestep)==-1:return

    output = Path(os.environ.get('MICROMOUSE_RESULTS_DIR', str(PROJECT / 'results')))
    policy_hash = sha256(policy_path.read_bytes()).hexdigest()
    archive = output / 'lidar_routes' / f'{signature}-{policy_hash}'
    records = [json.loads(record.read_text()) for record in archive.glob('*.json')]
    best = min(records, key=lambda record: record['duration_s'], default=None)
    replay = os.environ.get('MICROMOUSE_BEST_ROUTE') == '1'
    if replay and best is None:
        raise SystemExit('Belum ada rute yang mencapai goal. Jalankan bash run.sh sim --fast terlebih dahulu.')

    attempt = uuid4()
    seed = int(os.environ.get('MICROMOUSE_SEARCH_SEED', attempt.int))
    search = LidarMazeSearch(start, goal, bounds, cell_size, seed,
                             replay=best['targets'] if replay else ())
    show=os.environ.get('MICROMOUSE_NO_GUI')!='1' and (sys.platform!='linux' or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))
    label = 'LiDAR + SAC' if drl else 'LiDAR + tabular Q'
    view=MapView(start,goal,walls,bounds,[],set(),cell_size,show,label)
    config=ControlConfig()
    decision_ticks=max(1,round(.16/(timestep/1000)))
    timeout_s=float(os.environ.get('MICROMOUSE_TIMEOUT_S','300'))
    max_ticks=math.ceil(timeout_s/(timestep/1000))
    commands=np.zeros(2,dtype=float)
    action=0
    policy_velocity = ACTION_VW[action]
    recovery=WaypointRecovery()
    target = None
    exhausted = False
    steps=0;policy_ticks=0
    success=False;timed_out=False
    collision_count = 0
    contact_latched = False
    recovery_until = 0.
    state = None
    began=supervisor.getTime()
    trail=[];telemetry=[];clearance=float('inf')
    scan=lidar.getPointCloud();sectors=aggregate_point_cloud(scan)
    print(f'{"Replay rute terbaik" if replay else "Eksplorasi LiDAR"} | seed={seed} | policy={policy_path}',
          flush=True)

    while supervisor.step(timestep)!=-1:
        pose=np.array([*gps.getValues()[:2],imu.getRollPitchYaw()[2]])
        point=pose[:2].copy();trail.append(point.tolist())
        physical=physical_clearance(point,walls,bounds)
        clearance=min(clearance,physical)
        goal_error=float(np.linalg.norm(np.asarray(goal)-point))
        elapsed = supervisor.getTime() - began
        collision_event = False
        recovering = elapsed < recovery_until
        if not recovering and physical >= .003:
            contact_latched = False
        if not recovering and recovery_until:
            recovery_until = 0.
            policy_ticks = decision_ticks
        if physical < -.002 and not recovering:
            collision_event = not contact_latched
            collision_count += int(collision_event)
            contact_latched = True
            recovery_until = elapsed + COLLISION_RECOVERY_SECONDS
            recovering = True
            recovery.active = True
        if not recovering and search.final and goal_error<=.09:
            success=True
            break
        if elapsed>=timeout_s or steps>=max_ticks:
            timed_out=True
            break

        needs_target = target is None or (
            not search.final and math.dist(point, target) <= .055
        )
        if recovering:
            if policy_ticks >= decision_ticks:
                scan = lidar.getPointCloud()
                sectors = aggregate_point_cloud(scan)
                policy_ticks = 0
            state = None
            # Check the retreat against the simulator's collision geometry.
            # The policy's inputs remain the forward LiDAR and GPS-derived state.
            rear = point - .035 * np.array([math.cos(pose[2]), math.sin(pose[2])])
            rear_clearance = physical_clearance(rear, walls, bounds)
            safe_retreat = rear_clearance >= min(0., physical) - .001
            velocity = (COLLISION_RECOVERY_SPEED if safe_retreat else 0., 0.)
        elif steps == 0 or policy_ticks >= decision_ticks or (needs_target and target is not None):
            scan = lidar.getPointCloud()
            if needs_target:
                target = search.next_target(pose, [(hit.x, hit.y) for hit in scan])
                if target is None and search.scan_bearing is None:
                    exhausted = True
                    break
                if target is not None:
                    recovery = WaypointRecovery()
                if show:
                    view.path = search.route
                    view.explored = [search.center(cell) for cell in search.edges]
                    view.background = view.draw_background()
            sectors=aggregate_point_cloud(scan)
            if target is None:
                action = search.scan_action(sectors)
                state = None
                policy_velocity = ACTION_VW[action]
            else:
                allowed=recovery.actions(pose,target,sectors,supervisor.getTime()-began)
                if drl:
                    state = observation(pose,target,sectors,commands,config.wheel_limit)
                    action, policy_velocity = drive_command(actor(state),sectors,allowed)
                else:
                    state=discretize_state(pose,target,sectors)
                    values=agent.values(state)
                    action=max(allowed,key=lambda candidate:float(values[candidate]))
                    policy_velocity = ACTION_VW[action]
            policy_ticks=0
        if not recovering:
            velocity = policy_velocity
        commands=config.wheels_for_velocity(*velocity,commands,timestep/1000)
        for motor,speed in zip(motors,commands):motor.setVelocity(float(speed))
        telemetry.append(dict(time_s=supervisor.getTime()-began,pose=pose.tolist(),
                              search_path_index=len(search.route),target=list(target) if target is not None else None,
                              distance_to_waypoint_m=math.dist(point,target) if target is not None else None,distance_to_goal_m=goal_error,
                              q_state=list(state) if state is not None and not drl else None,
                              observation=state.tolist() if state is not None and drl else None,
                              q_values=agent.values(state).tolist() if state is not None and not drl else None,
                              velocity_command=list(velocity),
                              scanning=target is None and not recovering,
                              action='Lepas dari tembok' if recovering else ACTION_NAMES[action],
                              action_index=None if recovering else action,
                              collision_event=collision_event, collision_recovery=recovering,
                              waypoint_recovery=target is not None and recovery.active,
                              lidar_sector_min_m=sectors.tolist(),lidar_points=len(scan),
                              commanded_wheels=commands.tolist(),clearance_m=physical))
        steps+=1;policy_ticks+=1
        if show and steps%2==0:
            view.update(point,pose[2],trail,lidar_points=scan)

    for motor in motors:motor.setVelocity(0.)
    final=np.array(gps.getValues()[:2])
    final_heading=float(imu.getRollPitchYaw()[2])
    if not trail or np.linalg.norm(final-trail[-1])>1e-9:trail.append(final.tolist())
    result=dict(status='success' if success else 'search_exhausted' if exhausted else 'timeout',
                algorithm='online_lidar_dfs_with_sac_motion' if drl else 'online_lidar_dfs_with_tabular_q_motion',
                search_algorithm='online_lidar_dfs',
                search_mode='best_route' if replay else 'exploration', search_seed=seed,
                policy=str(policy_path.resolve()),table_states=None if drl else len(agent.q),
                search_cells_visited=len(search.edges),search_path=search.route,goal=tuple(goal),start=start,
                final_position=final.tolist(),goal_error_m=float(np.linalg.norm(final-np.asarray(goal))),
                distance_from_base_m=float(np.linalg.norm(final-np.asarray(start))),
                simulation_duration_s=supervisor.getTime()-began,movement_steps=steps,
                travelled_length_m=path_length(trail),target_path_length_m=path_length([start,*search.route]),
                min_physical_clearance_m=clearance,trajectory=trail,telemetry=telemetry,
                collision_count=collision_count, collision=collision_count>0,
                collision_penalty_total=collision_count*COLLISION_PENALTY,
                timeout=timed_out)
    output.mkdir(parents=True,exist_ok=True)
    if success and search.route and search.final:
        record = dict(duration_s=result['simulation_duration_s'], targets=search.route,
                      seed=seed, mode=result['search_mode'])
        archive.mkdir(parents=True, exist_ok=True)
        temporary = archive / f'{attempt.hex}.tmp'
        temporary.write_text(json.dumps(record) + '\n')
        temporary.replace(archive / f'{attempt.hex}.json')
        records.append(record)
    result['successful_route_attempts'] = len(records)
    result['best_route_time_s'] = min((record['duration_s'] for record in records), default=None)
    (output/'latest_run.json').write_text(json.dumps(result,indent=2)+'\n')
    if show:
        view.update(final,final_heading,trail,lidar_points=scan)
        view.save(output/'map.png')
        view.close()
    best_time = result['best_route_time_s']
    best_label = f'{best_time:.2f} s' if best_time is not None else 'belum ada'
    print(f"{result['status']} | {result['simulation_duration_s']:.2f} s | "
          f"percobaan sukses: {len(records)} | waktu terbaik: {best_label}", flush=True)
    print('MICROMOUSE_RESULT: '+json.dumps(result,sort_keys=True),flush=True)
    if os.environ.get('MICROMOUSE_TEST_EXIT')=='1':
        supervisor.simulationQuit(0)
