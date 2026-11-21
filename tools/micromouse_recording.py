"""Lossless step capture. Videos are rendered offline from these recorded poses."""
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from lidar_qlearning import ACTION_NAMES, waypoint_bearing

SECTORS = ('front','front_left','left','right','front_right')
Q_FIELDS = ('q_forward','q_curve_left','q_curve_right','q_turn_left','q_turn_right','q_stop')
REWARD = dict(progress='progress_reward', alignment='heading_reward', time='time_penalty',
              gps_travel='distance_penalty', close_wall='wall_penalty', goal='goal_bonus',
              waypoint='waypoint_bonus', collision='collision_penalty', stalled='stagnation_penalty')
STATE = ('bearing_bin','distance_bin',*(f'lidar_{k}_bin' for k in SECTORS))
EVENTS = ('collision_event','collision_count','recovery_active','scanning_active',
          'waypoint_reached','goal_reached','timeout','search_exhausted','terminated',
          'truncated','episode_success','episode_end')
LEARNING = ('alpha','gamma','state_visits_before','state_visits_after','q_old','td_target',
            'td_error','q_new','q_update_applied','loss','online_q_value','target_q_value',
            'selected_next_action','learning_rate','gradient_norm','replay_buffer_size',
            'gradient_update_count','target_update_count','batch_mean_q','batch_max_q',
            'batch_mean_td_target','batch_mean_td_error','batch_sample_index')
KEYS = ('run_id','algorithm','episode','training_phase','recording_phase','episode_key','step',
        'simulation_time','timestamp','global_row','frame_index')
ROBOT = ('robot_x','robot_y','heading','linear_velocity','angular_velocity','target_x',
         'target_y','target_bearing','target_distance',*(f'lidar_{k}_m' for k in SECTORS))
FIELDS = (*KEYS,*ROBOT,'post_x','post_y','post_heading','lidar_rays_m','observation','state_code',
          *STATE,'action_id','action_name','selected_action','action_mask','q_values','selected_q_value',
          *Q_FIELDS,'epsilon','exploration_or_exploitation',*REWARD.values(),'reward_step',
          'cumulative_reward',*EVENTS,*LEARNING)
SUMMARY = ('run_id','algorithm','episode','phase','recording_phase','episode_key','total_steps',
           'total_reward','average_reward_per_step','duration_simulation_s','duration_wall_clock_s',
           'success','collision','collision_count','timeout','search_exhausted','epsilon_start',
           'epsilon_end','unique_states_visited','replay_buffer_size','gradient_updates',
           'best_reward_so_far','best_successful_time_so_far','distance_to_goal_m','first_row','last_row')


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def json_value(value):
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def csv_value(value):
    return json.dumps(value,separators=(',',':'),default=json_value,allow_nan=False) if isinstance(value,(list,tuple,dict)) else value


class Recording:
    def __init__(self, directory, algorithm, motion, *, feature_names=()):
        self.root = Path(directory)
        self.root.mkdir(parents=True,exist_ok=True)
        self.path = self.root/'telemetry'
        self.path.mkdir(exist_ok=False)
        self.algorithm = algorithm
        self.id = f'{algorithm}-{uuid4().hex[:12]}'
        self.feature_names = tuple(feature_names)
        self.count = 0
        self.episode = None
        self.best = {}
        self.files = []
        self.jsonl = self.open('step_telemetry.jsonl')
        self.csv = self.writer('step_telemetry.csv',(*FIELDS,*self.feature_names))
        self.summaries = self.writer('episode_summary.csv',SUMMARY)
        self.updates = self.writer('learning_updates.csv',(*KEYS,'state_code','action_id',*LEARNING))
        nav=motion.nav
        self.meta=dict(run_id=self.id,algorithm=algorithm,step_seconds=motion.decision_seconds,
                       bounds=list(nav.bounds),start=motion.base_start.tolist(),goal=motion.base_goal.tolist(),
                       cell_size=nav.cell_size,feature_names=self.feature_names,
                       walls=[dict(center=list(w.center),size=list(w.size)) for w in nav.walls],
                       started_at=stamp(),status='recording',webots='not_recorded_by_this_run',
                       synchronization='global_row is the stable key for per-video frame index CSV files',
                       raw_pose_timing='robot and lidar are pre-action; post pose, reward and events are after action')
        self.save_meta()

    def open(self,name):
        f=(self.path/name).open('w',newline='',buffering=1024*1024)
        self.files.append(f)
        return f

    def writer(self,name,fields):
        result = csv.DictWriter(self.open(name),fieldnames=fields,extrasaction='ignore')
        result.writeheader()
        return result

    def save_meta(self):
        file=self.path/'recording_manifest.json'
        temp=file.with_suffix('.tmp')
        temp.write_text(json.dumps(self.meta,indent=2)+'\n')
        temp.replace(file)

    def begin(self, episode, phase, epsilon, recording_phase='training'):
        self.episode = dict(run_id=self.id,algorithm=self.algorithm,episode=episode,phase=phase,
                            recording_phase=recording_phase,episode_key=f'{recording_phase}-{episode:06d}',
                            epsilon_start=epsilon,first_row=self.count+1)
        self.clock=time.perf_counter()
        self.states=set()
        self.last=None

    def decision(self, motion, target, action, allowed, epsilon, *, state=None,
                 observation=None, q=None, scanning=False, recovering=False, **extras):
        x,y,heading=map(float,motion.pose)
        cfg=motion.config
        left,right=motion.wheels
        mask=[bool(i in allowed) for i in range(len(ACTION_NAMES))]
        if target is None:
            tx=ty=bearing=distance=None
        else:
            tx,ty=map(float,target)
            bearing=waypoint_bearing(motion.pose,target)
            distance=math.dist(motion.pose[:2],target)
        # Rays are a documented subset of actual 128 readings. The complete
        # five sector minima used by the policy are recorded separately.
        rays=motion.lidar_scan
        angles=np.linspace(-math.pi/2,math.pi/2,len(rays))
        row=dict(run_id=self.id,algorithm=self.algorithm,episode=self.episode['episode'],
                 episode_key=self.episode['episode_key'],training_phase=self.episode['phase'],
                 recording_phase=self.episode['recording_phase'],step=motion.steps+1,
                 timestamp=stamp(),global_row=self.count+1,frame_index=None,
                 robot_x=x,robot_y=y,heading=heading,target_x=tx,target_y=ty,target_bearing=bearing,
                 target_distance=distance,linear_velocity=cfg.wheel_radius*(left+right)/2,
                 angular_velocity=cfg.wheel_radius*(right-left)/cfg.axle_track,
                 lidar_rays_m=[[float(angles[i]),float(rays[i])] for i in range(0,len(rays),8)],
                 action_id=action,selected_action=action,
                 action_name='Pemulihan kontak' if recovering else ACTION_NAMES[action],
                 action_mask=mask,epsilon=epsilon,q_values=list(q) if q is not None else None,
                 scanning_active=scanning,recovery_active=recovering,**extras)
        row.update({f'lidar_{key}_m':float(v) for key,v in zip(SECTORS,motion.lidar_sectors)})
        if state is not None:
            key=','.join(map(str,state))
            self.states.add(key)
            row.update(state_code=key,**dict(zip(STATE,map(int,state))))
        if observation is not None:
            row['observation']=list(map(float,observation))
            row.update(dict(zip(self.feature_names,row['observation'])))
        if q is not None:
            row.update(dict(zip(Q_FIELDS,map(float,q))))
            row['selected_q_value']=float(q[action])
        return row

    def step(self,row,motion,info,*,terminal,truncated,exhausted,episode_end=False,learning=None):
        row.update(post_x=float(motion.pose[0]),post_y=float(motion.pose[1]),post_heading=float(motion.pose[2]),
                   simulation_time=motion.steps*motion.decision_seconds,
                   reward_step=float(info['reward_step']),cumulative_reward=float(motion.episode_return),
                   collision_event=bool(info['collision']),collision_count=int(info['collision_count']),
                   waypoint_reached=bool(info['waypoint_reached']),goal_reached=bool(info['success']),
                   timeout=bool(truncated),search_exhausted=bool(exhausted),
                   terminated=bool(terminal or exhausted),truncated=bool(truncated),
                   episode_success=bool(info['success']),episode_end=episode_end)
        row.update({dst:info['reward_parts'].get(src) for src,dst in REWARD.items()})
        if learning:
            row.update(learning)
            self.updates.writerow({k:csv_value(v) for k,v in row.items()})
        self.jsonl.write(json.dumps(row,default=json_value,allow_nan=False,separators=(',',':'))+'\n')
        self.csv.writerow({k:csv_value(v) for k,v in row.items()})
        self.last=row
        self.count+=1
        if self.count%256==0:
            self.flush()

    def end(self,motion,info,**extras):
        s=self.episode
        last=self.last or {}
        group=s['recording_phase']
        reward=float(motion.episode_return)
        duration=motion.steps*motion.decision_seconds
        success=bool(info['success'])
        best_reward,best_time=self.best.get(group,(-math.inf,None))
        best_reward=max(best_reward,reward)
        if success:
            best_time=duration if best_time is None else min(best_time,duration)
        self.best[group]=(best_reward,best_time)
        self.summaries.writerow(dict(s,total_steps=motion.steps,total_reward=reward,
            average_reward_per_step=reward/max(1,motion.steps),duration_simulation_s=duration,
            duration_wall_clock_s=time.perf_counter()-self.clock,success=success,
            collision=int(info['collision_count'])>0,collision_count=info['collision_count'],
            timeout=bool(last.get('timeout')),search_exhausted=bool(info['search_exhausted']),
            epsilon_end=last.get('epsilon',s['epsilon_start']),
            unique_states_visited=len(self.states) if self.algorithm=='q_learning' else None,
            best_reward_so_far=best_reward,best_successful_time_so_far=best_time,last_row=self.count,
            distance_to_goal_m=math.dist(motion.pose[:2],motion.base_goal),**extras))
        self.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def close(self, status='captured'):
        if self.meta['status']=='recording':
            for f in self.files:
                f.close()
            self.meta.update(status=status,step_rows=self.count,finished_at=stamp())
            self.save_meta()
