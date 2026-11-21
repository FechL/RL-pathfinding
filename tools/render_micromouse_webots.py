"""Render a separate Webots video from captured GPS, IMU, and LiDAR logs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'controllers/micromouse'))
from lidar_qlearning import ACTION_NAMES, waypoint_bearing
from render_micromouse_recording import Dashboard, Encoder, MAP_COLUMNS, map_frames, probe
from train_q_learning import read_world

SECTORS=('front','front_left','left','right','front_right')
ANGLES=(0.,math.pi/4,math.pi/2,-math.pi/2,-math.pi/4)


def measured_rows(source,meta):
    samples=source.get('telemetry',source.get('decisions',[]))
    last=None
    events=0
    for index,item in enumerate(samples,1):
        if 'pose' not in item:
            raise ValueError('Telemetry has no measured XY and heading for Webots replay.')
        pose=item['pose']
        elapsed=item['time_s']
        target=item.get('target')
        bearing=waypoint_bearing(pose,target) if target is not None else None
        distance=math.dist(pose[:2],target) if target is not None else None
        sectors=item['lidar_sector_min_m']
        hit=bool(item.get('collision_event'))
        events=item.get('collision_count',events+int(hit))
        action=item.get('action_index',item.get('action'))
        row=dict(run_id=meta['run_id'],algorithm=meta['algorithm'],episode=1,
            episode_key='webots-000001',recording_phase='webots',training_phase='webots inference',
            step=index,simulation_time=elapsed,timestamp=None,global_row=index,
            robot_x=pose[0],robot_y=pose[1],heading=pose[2],post_x=pose[0],post_y=pose[1],
            post_heading=pose[2],target_x=target[0] if target is not None else None,
            target_y=target[1] if target is not None else None,target_bearing=bearing,target_distance=distance,
            action_id=action,action_name=item.get('action') if isinstance(item.get('action'),str) else
                ACTION_NAMES[action] if action is not None else 'Pemulihan',
            q_values=item.get('q_values',item.get('q')),action_mask=item.get('mask'),
            lidar_rays_m=list(zip(ANGLES,sectors)),collision_count=events,
            collision_event=hit,scanning_active=item.get('scanning',False),
            recovery_active=item.get('collision_recovery',False),
            epsilon=0.,reward_step=None,cumulative_reward=None,
            state_code=','.join(map(str,item['q_state'])) if item.get('q_state') else None,
            linear_velocity=None,angular_velocity=None)
        row.update({f'lidar_{name}_m':v for name,v in zip(SECTORS,sectors)})
        if 'wheel_velocity_rad_s' in item:
            left,right=item['wheel_velocity_rad_s']
            row.update(linear_velocity=.027*(left+right)/2,angular_velocity=.027*(right-left)/.104)
        elif last is not None and elapsed>last['simulation_time']:
            dt=elapsed-last['simulation_time']
            dx=pose[0]-last['robot_x']; dy=pose[1]-last['robot_y']
            yaw=(pose[2]-last['heading']+math.pi)%(2*math.pi)-math.pi
            row.update(linear_velocity=(dx*math.cos(pose[2])+dy*math.sin(pose[2]))/dt,
                       angular_velocity=yaw/dt)
        last=row
        yield row
    if last is None:
        raise ValueError('No real Webots telemetry rows.')
    x,y=source['trajectory'][-1]
    yield dict(last,step=last['step']+1,global_row=last['global_row']+1,
               simulation_time=source['simulation_duration_s'],robot_x=x,robot_y=y,
               heading=None,post_x=x,post_y=y,post_heading=None,
               linear_velocity=None,angular_velocity=None,
               target_x=None,target_y=None,target_bearing=None,target_distance=None,
               lidar_rays_m=[],**{f'lidar_{n}_m':None for n in SECTORS},
               action_id=None,action_name='Status akhir',action_mask=None,q_values=None,
               collision_count=source.get('collision_count',events),collision_event=False,
               state_code=None,episode_end=True,episode_success=source['status']=='success',
               timeout=source['status']=='timeout',search_exhausted=source['status']=='search_exhausted')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--stride',type=int,required=True)
    args=p.parse_args()
    if args.stride<1:
        p.error('Positive stride required.')
    source=json.loads(args.source.read_text())
    telemetry=args.output/'telemetry'
    video=args.output/'videos'
    filename=video/'webots_evaluation.mp4'
    if filename.exists():
        p.error('Webots recording already exists.')
    manifest=telemetry/'recording_manifest.json'
    meta=json.loads(manifest.read_text())
    if meta['status']!='encoded':
        p.error('Render kinematic videos before adding Webots.')
    maze=read_world()
    digest=hashlib.sha256(args.source.read_bytes()).hexdigest()
    web=dict(meta,environment='webots',run_id='webots-'+digest[:12],bounds=list(maze.bounds),
             start=source['start'],goal=source['goal'])
    dashboard=Dashboard(web,meta['width'],meta['height'])
    encoder=Encoder(filename,meta['width'],meta['height'],meta['fps'],meta['codec'])
    trail=[]; mapped=[]
    with (telemetry/'webots_step_telemetry.jsonl').open('w') as stream:
        for row in measured_rows(source,web):
            trail.append(dashboard.pixel((row['post_x'],row['post_y'])))
            stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
            if (row['step']-1)%args.stride==0 or row['collision_event'] or row.get('episode_end'):
                encoder.write(dashboard.render(row,trail,None))
                mapped.append(row)
    encoder.close()
    info=probe(filename)
    if int(info['nb_frames'])!=len(mapped):
        raise ValueError('Webots frame mapping mismatch.')
    with (telemetry/'webots_frame_mapping.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=MAP_COLUMNS,extrasaction='ignore'); writer.writeheader()
        map_frames(writer,filename.name,mapped)
    shutil.copyfile(args.source,telemetry/'webots_source.json')
    meta['webots']=dict(status='encoded',result=source['status'],simulation_time_s=source['simulation_duration_s'],
        source_sha256=digest,source='webots_source.json',stride=args.stride,
        filename=filename.name,step_rows=row['step'],first_step=1,last_step=row['step'],
        frame_mapping='webots_frame_mapping.csv',telemetry='webots_step_telemetry.jsonl',
        notes='Sensor spokes are the measured sector minima. Reward unavailable in Webots inference. '
              'Q-only older telemetry has no mask or wheel readings; velocity is estimated from GPS/IMU differences. '
              'The final frame uses the recorded terminal position and has no fabricated IMU/LiDAR sample.',
        **info)
    temp=manifest.with_suffix('.tmp')
    temp.write_text(json.dumps(meta,indent=2)+'\n')
    temp.replace(manifest)


if __name__=='__main__':
    main()
