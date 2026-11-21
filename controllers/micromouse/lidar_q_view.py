"""Live maze, lidar and sparse Q-table view for tabular training."""
import math
import time
from collections import deque

import numpy as np

from lidar_qlearning import (ACTION_NAMES,beam_angles,
                             SAFE_DISTANCE,STATE_SPACE_SIZE,safe_actions)


class LidarQView:
    def __init__(self, navigation, speed=8.):
        import cv2
        self.cv2=cv2
        self.nav=navigation
        self.speed=float(speed)
        self.title='Micromouse - Tabular Q-Learning + LiDAR'
        self.table_title='Q-table - state discret + aksi'
        self.size=720
        self.header=62
        self.panel=540
        self.width=self.size+self.panel
        self.height=self.size+self.header
        self.pad=26
        self.show_table=False
        self.table_window_open=False
        self.table_page=0
        self.table_keys=[]
        self.table_indices={}
        self.table_synced_size=0
        self.closed=False
        self.last_tick=None
        self.rewards=deque(maxlen=120)
        self.successes=deque(maxlen=120)
        self.collisions=deque(maxlen=120)
        self.trail=[]
        self.episode_key=None
        cv2.namedWindow(self.title,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title,self.width,self.height)
        cv2.setMouseCallback(self.title,self._mouse)
        self.button=(self.size+20,14,self.width-18,48)

    def pixel(self, point):
        xmin,xmax,ymin,ymax=self.nav.bounds
        return (round(self.pad+(point[0]-xmin)/(xmax-xmin)*(self.size-2*self.pad)),
                round(self.header+self.pad+(ymax-point[1])/(ymax-ymin)*(self.size-2*self.pad)))

    def _mouse(self,event,x,y,flags,param):
        if event==self.cv2.EVENT_LBUTTONDOWN:
            x0,y0,x1,y1=self.button
            if x0<=x<=x1 and y0<=y<=y1:
                self.show_table=not self.show_table
                if not self.show_table:
                    try:self.cv2.destroyWindow(self.table_title)
                    except self.cv2.error:pass
                    self.table_window_open=False
            elif self.show_table and y<84:
                if x>self.width-145:self.table_page+=1
                elif x>self.width-270:self.table_page=max(0,self.table_page-1)

    def _table_mouse(self,event,x,y,flags,param):
        if event==self.cv2.EVENT_LBUTTONDOWN and y<84:
            if x>self.width-145:self.table_page+=1
            elif x>self.width-270:self.table_page=max(0,self.table_page-1)

    def record_episode(self,reward,success,collision):
        self.rewards.append(float(reward))
        self.successes.append(bool(success))
        self.collisions.append(bool(collision))

    def _background(self):
        cv2=self.cv2
        frame=np.full((self.height,self.width,3),248,dtype=np.uint8)
        for wall in self.nav.walls:
            center=np.asarray(wall.center);half=np.asarray(wall.size)/2
            cv2.rectangle(frame,self.pixel(center+[-half[0],half[1]]),
                          self.pixel(center+[half[0],-half[1]]),(76,87,96),-1)
        half=self.nav.cell_size/2
        cv2.rectangle(frame,self.pixel(self.nav.goal+[-half,half]),
                      self.pixel(self.nav.goal+[half,-half]),(20,220,255),-1)
        cv2.circle(frame,self.pixel(self.nav.start),8,(63,151,82),2)
        cv2.line(frame,(self.size,0),(self.size,self.height),(210,214,218),1)
        return frame

    @staticmethod
    def _state_text(state):
        bearing,distance,*lidar=state
        return f"b{bearing:02d} d{distance} | {''.join(map(str,lidar))}"

    def _draw_table(self,agent,current_state,current_actions):
        cv2=self.cv2
        if len(agent.state_order)>self.table_synced_size:
            new_keys=agent.state_order[self.table_synced_size:]
            for key in new_keys:
                self.table_indices[key]=len(self.table_keys)
                self.table_keys.append(key)
            self.table_synced_size=len(agent.state_order)
        frame=np.full((self.height,self.width,3),249,dtype=np.uint8)
        cv2.putText(frame,'Q-table: target-relative + 5 sektor LiDAR',
                    (24,36),cv2.FONT_HERSHEY_SIMPLEX,.62,(36,48,60),2,cv2.LINE_AA)
        cv2.putText(frame,f'{len(agent.q):,}/{STATE_SPACE_SIZE:,} state tersimpan | sekarang: {self._state_text(current_state)}',
                    (24,64),cv2.FONT_HERSHEY_SIMPLEX,.39,(65,76,87),1,cv2.LINE_AA)
        cv2.rectangle(frame,(self.width-268,17),(self.width-150,49),(224,232,239),-1)
        cv2.rectangle(frame,(self.width-140,17),(self.width-20,49),(210,228,239),-1)
        cv2.putText(frame,'< sebelumnya',(self.width-258,39),cv2.FONT_HERSHEY_SIMPLEX,.34,(40,52,64),1,cv2.LINE_AA)
        cv2.putText(frame,'berikutnya >',(self.width-130,39),cv2.FONT_HERSHEY_SIMPLEX,.34,(40,52,64),1,cv2.LINE_AA)
        columns=[('State',265),('Maju',128),('Ar kiri',128),('Ar kanan',128),
                 ('Putar kiri',128),('Putar kanan',128),('Stop',128),('N',58)]
        x=20;y=108
        cv2.rectangle(frame,(x,y),(self.width-20,y+34),(44,67,86),-1)
        cx=x
        for label,width in columns:
            cv2.putText(frame,label,(cx+8,y+23),cv2.FONT_HERSHEY_SIMPLEX,.37,(250,250,250),1,cv2.LINE_AA)
            cx+=width
        current_key=agent.key(current_state)
        current_values=agent.values(current_state)
        current_visits=agent.visits.get(current_key,0)
        page_size=10
        current_index=self.table_indices.get(current_key)
        history_count=len(self.table_keys)-(current_index is not None)
        pages=max(1,math.ceil(history_count/page_size))
        self.table_page=min(self.table_page,pages-1)
        begin=self.table_page*page_size
        history_keys=[]
        logical_index=begin
        while logical_index<len(self.table_keys) and len(history_keys)<page_size:
            state_index=len(self.table_keys)-1-logical_index
            if state_index!=current_index:
                history_keys.append(self.table_keys[state_index])
            logical_index+=1
        rows=[(current_state,current_values,current_visits,True)]
        rows.extend((agent.unkey(key),agent.q[key],agent.visits.get(key,0),False)
                    for key in history_keys)
        row_y=y+35
        for index,(state,values,count,is_current) in enumerate(rows):
            yy=row_y+index*38
            background=(214,241,225) if is_current else (239,244,247)
            cv2.rectangle(frame,(x,yy),(self.width-20,yy+37),background,-1)
            cv2.line(frame,(x,yy+37),(self.width-20,yy+37),(220,225,229),1)
            cx=x
            state_label=self._state_text(state)
            bins=np.asarray((SAFE_DISTANCE*.5,.30,.50),dtype=float)[list(state[2:])]
            valid_actions=current_actions if is_current else safe_actions(bins)
            best_action=valid_actions[int(np.argmax(values[list(valid_actions)]))]
            for col,(label,width) in enumerate(columns):
                if col==0:text=('AKTIF | '+state_label) if is_current else state_label
                elif 1<=col<=len(ACTION_NAMES):
                    value=float(values[col-1]);text=f'{value:+.3f}'
                    color=(37,110,54) if value>0 else (47,55,64) if abs(value)<1e-6 else (50,65,190)
                    cv2.putText(frame,text,(cx+8,yy+24),cv2.FONT_HERSHEY_SIMPLEX,.37,color,1,cv2.LINE_AA)
                    if col-1==best_action:
                        cv2.rectangle(frame,(cx+2,yy+2),(cx+width-2,yy+35),(102,182,116),1)
                    cx+=width
                    continue
                else:text=str(count)
                cv2.putText(frame,text,(cx+8,yy+24),cv2.FONT_HERSHEY_SIMPLEX,.34,(42,56,68),1,cv2.LINE_AA)
                cx+=width
        cv2.putText(frame,f'Bingkai hijau = aksi Q terbaik yang aman | state aktif hijau | Halaman {self.table_page+1}/{pages} | T/Q tutup',
                    (22,self.height-20),cv2.FONT_HERSHEY_SIMPLEX,.37,(70,78,86),1,cv2.LINE_AA)
        return frame

    def _draw_main(self,env,agent,episode,epsilon,steps_total):
        cv2=self.cv2
        frame=self._background()
        label=(id(env),episode)
        if label!=self.episode_key:self.trail=[];self.episode_key=label
        # Display a thin subset of raw beams; the Q state uses all 128.
        angles = beam_angles(len(env.lidar_scan))
        for index in range(0,len(env.lidar_scan),4):
            distance=float(env.lidar_scan[index])
            angle=env.pose[2]+angles[index]
            origin=env.pose[:2]
            end=origin+distance*np.array([math.cos(angle),math.sin(angle)])
            color=(72,155,86) if distance>1. else (65,172,221) if distance>.35 else (65,73,213)
            cv2.line(frame,self.pixel(origin),self.pixel(end),color,1,cv2.LINE_AA)
        point=self.pixel(env.pose[:2]);self.trail.append(point)
        if len(self.trail)>1:cv2.polylines(frame,[np.asarray(self.trail,dtype=np.int32)],False,(185,119,54),2)
        cv2.circle(frame,point,9,(57,77,224),-1,cv2.LINE_AA)
        cv2.circle(frame,self.pixel(env.base_start),6,(60,160,90),-1,cv2.LINE_AA)
        target=self.pixel(env.goal)
        cv2.circle(frame,target,8,(205,105,30),2,cv2.LINE_AA)
        cv2.line(frame,point,target,(112,153,190),1,cv2.LINE_AA)
        nose=self.pixel(env.pose[:2]+.15*np.array([math.cos(env.pose[2]),math.sin(env.pose[2])]))
        cv2.line(frame,point,nose,(38,40,47),2,cv2.LINE_AA)
        cv2.putText(frame,'TABULAR Q MOTION CONTROL + LIDAR 180',(18,29),cv2.FONT_HERSHEY_SIMPLEX,.52,(35,48,60),2,cv2.LINE_AA)
        speed_test = env.training_phase == 'speed_test'
        phase_label = 'UJI KECEPATAN' if speed_test else 'EKSPLORASI'
        phase_color = (45, 135, 215) if speed_test else (145, 100, 40)
        cv2.rectangle(frame, (18, 38), (187, 59), phase_color, -1)
        cv2.putText(frame, phase_label, (26, 53), cv2.FONT_HERSHEY_SIMPLEX,
                    .47, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame,(self.button[0],self.button[1]),(self.button[2],self.button[3]),(213,229,241),-1)
        cv2.rectangle(frame,(self.button[0],self.button[1]),(self.button[2],self.button[3]),(112,143,165),1)
        cv2.putText(frame,'BUKA Q-TABLE  [T / Q]',(self.button[0]+18,37),cv2.FONT_HERSHEY_SIMPLEX,.43,(38,58,72),1,cv2.LINE_AA)
        x=self.size+20;y=85
        if env.recovering_from_collision:
            mode = 'lepas dari tembok'
        elif env.scanning:
            mode = 'pindai cabang'
        else:
            mode = 'pemulihan' if env.recovery_active else 'normal'
        route_label = f' | uji rute #{env.training_route_id}' if speed_test else ''
        best_time = f'{env.best_speed_time:.2f}s' if env.best_speed_time is not None else 'belum ada'
        info=[f'Episode: {episode} | step total: {steps_total}',
              f'Langkah episode: {env.steps}/{env.max_steps}',
              f'Epsilon: {epsilon:.3f} | state Q: {len(agent.q):,}/{STATE_SPACE_SIZE:,}',
              f'Rute ditemukan: {env.routes_found}{route_label} | terbaik: {best_time}',
              f'Reward akumulasi: {env.episode_return:+.2f}',
              f'Target lokal #{env.waypoint_index} | target {np.linalg.norm(env.goal-env.pose[:2]):.2f}m | base {np.linalg.norm(env.pose[:2]-env.base_start):.2f}m',
              f'Jarak jalan {env.travelled_distance:.2f}m | aksi: {ACTION_NAMES[env.previous_action] if env.previous_action is not None else "-"}',
              f'Mode: {mode} | Tabrakan episode: {env.collision_count}',
              'Sektor: D, DK, K, Kn, DKn (meter)',
              f'LiDAR: {" ".join(f"{v:.2f}" for v in env.lidar_sectors)}',
              f'Front clearance: {float(env.lidar_sectors[0]):.2f} m',
              f'LiDAR x{env.last_reward_parts.get("lidar_weight",1):.1f} | waypoint {env.last_reward_parts.get("progress",0):+.2f} | GPS {env.last_reward_parts.get("gps_travel",0):+.2f} | wall {env.last_reward_parts.get("close_wall",0):+.2f}',
              f'Tabrakan -100 | waypoint +4 | goal +120 | timeout {env.max_steps*env.decision_seconds:.0f}s',
              f'Sukses 100 ep: {100*sum(self.successes)/max(1,len(self.successes)):.1f}%',
              f'Tabrakan 100 ep: {100*sum(self.collisions)/max(1,len(self.collisions)):.1f}%']
        for i,text in enumerate(info):
            cv2.putText(frame,text,(x,y+i*23),cv2.FONT_HERSHEY_SIMPLEX,.34,(56,67,78),1,cv2.LINE_AA)
        self._chart(frame,(x,y+325,self.panel-40,150),'Reward 100 episode terakhir',list(self.rewards))
        self._chart(frame,(x,y+500,self.panel-40,150),'Q-value state sekarang',agent.values(env.last_state).tolist())
        return frame

    def _chart(self,frame,box,title,values):
        cv2=self.cv2;x,y,w,h=box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(221,226,230),1)
        cv2.putText(frame,title,(x+7,y+19),cv2.FONT_HERSHEY_SIMPLEX,.35,(49,60,71),1,cv2.LINE_AA)
        if len(values)<1:return
        values=np.asarray(values,dtype=float)
        low=min(float(values.min()),0.);high=max(float(values.max()),0.)
        margin=max((high-low)*.12,.05);low-=margin;high+=margin
        points=[]
        for i,value in enumerate(values):
            px=x+8+(w-16)*i/max(1,len(values)-1)
            py=y+h-10-(h-42)*(value-low)/(high-low)
            points.append((round(px),round(py)))
        if len(points)>1:cv2.polylines(frame,[np.asarray(points,dtype=np.int32)],False,(61,129,184),2,cv2.LINE_AA)

    def update(self,env,agent,episode,epsilon,steps_total):
        if self.closed:return False
        now=time.monotonic()
        if self.last_tick is not None:
            time.sleep(max(0.,env.decision_seconds/self.speed-(now-self.last_tick)))
        self.last_tick=time.monotonic()
        self.cv2.imshow(self.title,self._draw_main(env,agent,episode,epsilon,steps_total))
        if self.show_table:
            if not self.table_window_open:
                cv2=self.cv2;cv2.namedWindow(self.table_title,cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.table_title,self.width,self.height)
                cv2.setMouseCallback(self.table_title,self._table_mouse)
                self.table_window_open=True
            self.cv2.imshow(self.table_title,self._draw_table(agent,env.last_state,env.allowed_actions))
        key=self.cv2.waitKey(1)&255
        if key in (ord('t'),ord('q')):
            self.show_table=not self.show_table
            if not self.show_table:
                try:self.cv2.destroyWindow(self.table_title)
                except self.cv2.error:pass
                self.table_window_open=False
        elif key==ord(']'):
            self.table_page+=1
        elif key==ord('['):
            self.table_page=max(0,self.table_page-1)
        elif key in (27,ord('x')):
            self.close();return False
        return not self.closed

    def close(self):
        if not self.closed:
            try:self.cv2.destroyAllWindows()
            except self.cv2.error:pass
            self.closed=True
