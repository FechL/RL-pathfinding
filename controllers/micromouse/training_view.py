"""Local OpenCV preview of the actual kinematic training state."""
import math
import time
import numpy as np


class TrainingView:
    def __init__(self, navigation, speed=8.):
        import cv2
        self.cv2=cv2
        self.nav=navigation
        self.speed=speed
        self.title='Micromouse - Training langsung (simulasi 2D)'
        self.size=760;self.pad=28;self.header=110
        self.closed=False;self.stage='Training SAC'
        self.key=None;self.trail=[];self.last_tick=None;self.last_frame=0.
        cv2.namedWindow(self.title,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title,self.size,self.size+self.header)
        self.background=self.frame_background()

    def pixel(self,p):
        xmin,xmax,ymin,ymax=self.nav.bounds
        return (round(self.pad+(p[0]-xmin)/(xmax-xmin)*(self.size-2*self.pad)),
                round(self.header+self.pad+(ymax-p[1])/(ymax-ymin)*(self.size-2*self.pad)))

    def frame_background(self):
        cv2=self.cv2
        frame=np.full((self.size+self.header,self.size,3),246,dtype=np.uint8)
        for wall in self.nav.walls:
            p=np.asarray(wall.center);half=np.asarray(wall.size)/2
            cv2.rectangle(frame,self.pixel(p+[-half[0],half[1]]),self.pixel(p+[half[0],-half[1]]),(85,95,105),-1)
        cv2.circle(frame,self.pixel(self.nav.start),7,(60,150,65),2)
        cv2.circle(frame,self.pixel(self.nav.goal),9,(35,145,230),-1)
        return frame

    def frame(self,env):
        cv2=self.cv2
        frame=self.background.copy()
        label=(id(env),env.episode_index)
        if label!=self.key:self.trail=[];self.key=label
        point=self.pixel(env.pose[:2]);self.trail.append(point)
        if len(self.trail)>1:cv2.polylines(frame,[np.asarray(self.trail,dtype=np.int32)],False,(200,150,55),2)
        radius=max(4,round(.075/(self.nav.bounds[1]-self.nav.bounds[0])*(self.size-2*self.pad)))
        cv2.circle(frame,point,radius,(45,75,220),-1)
        nose=self.pixel(env.pose[:2]+.13*np.array([math.cos(env.pose[2]),math.sin(env.pose[2])]))
        cv2.line(frame,point,nose,(35,35,70),2)
        if env.last_success:
            status = 'GOAL'
        elif env.last_collision:
            status = 'TABRAKAN'
        else:
            status = 'TIMEOUT' if env.finished else 'BERJALAN'
        lines=[f'{self.stage} | Episode {env.episode_index} | {status}',
               f'Step {env.steps} | Reward {env.episode_return:.1f} | Simulasi {env.steps*env.config.dt:.1f}/{env.max_steps*env.config.dt:.1f} dtk',
               f'Kecepatan tampilan {self.speed:g}x | Q / Esc / tutup jendela: akhiri training']
        for i,line in enumerate(lines):cv2.putText(frame,line,(18,26+i*28),cv2.FONT_HERSHEY_SIMPLEX,.48,(40,45,50),1,cv2.LINE_AA)
        return frame

    def update(self,env,force=False,dt=None):
        if self.closed:return
        now=time.monotonic()
        if self.last_tick is not None and not force:
            interval = env.config.dt if dt is None else dt
            time.sleep(max(0.,interval/self.speed-(now-self.last_tick)))
        self.last_tick=time.monotonic()
        if not force and self.last_tick-self.last_frame<1/32:return
        self.last_frame=self.last_tick
        self.cv2.imshow(self.title,self.frame(env))
        key=self.cv2.waitKey(1)&255
        if key in (ord('q'),27) or self.cv2.getWindowProperty(self.title,self.cv2.WND_PROP_VISIBLE)<1:
            self.close()

    def close(self):
        if not self.closed:
            self.cv2.destroyWindow(self.title)
            self.closed=True
