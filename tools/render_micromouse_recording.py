"""Render offline MP4s whose frames point to the original training telemetry."""
import argparse
from collections import Counter
import csv
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


class Encoder:
    def __init__(self,file,width,height,fps,codec):
        self.file=file
        self.scratch=file.with_suffix('.tmp.mp4')
        self.count=0
        self.process=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo',
            '-pixel_format','bgr24','-video_size',f'{width}x{height}','-framerate',str(fps),'-i','-',
            '-an','-c:v',codec,*(['-preset','veryfast','-crf','25'] if codec=='libx264' else ['-q:v','5']),
            '-pix_fmt','yuv420p','-threads','1','-movflags','+faststart',str(self.scratch)],
            stdin=subprocess.PIPE)
    def write(self,frame):
        self.process.stdin.write(frame.tobytes())
        self.count+=1
    def close(self):
        self.process.stdin.close()
        if self.process.wait()!=0 or self.count==0:
            self.scratch.unlink(missing_ok=True)
            raise RuntimeError(f'Encoding failed or no frames: {self.file}')
        self.scratch.replace(self.file)


def f(value, digits=2):
    return '-' if value is None else f'{value:.{digits}f}'


class Dashboard:
    COLORS=((178,108,60),(61,123,174),(60,144,94),(139,90,145))
    def __init__(self,meta,width,height):
        self.meta=meta
        self.w,self.h=width,height
        self.base=np.full((height,width,3),248,np.uint8)
        self.map_size=min(width*.47,height*.73)
        self.panel=round(self.map_size+45)
        self.yplots=round(39+self.map_size+24)
        platform = 'WEBOTS' if meta.get('environment')=='webots' else 'KINEMATIK'
        self.text(self.base,platform+'   /   '+meta['algorithm'].upper().replace('_',' '),17,25,.64,(48,59,61),2)
        for wall in meta['walls']:
            x,y=wall['center']; sx,sy=wall['size']
            cv2.rectangle(self.base,self.pixel((x-sx/2,y-sy/2)),self.pixel((x+sx/2,y+sy/2)),(102,105,108),-1)
        for key,color in [('start',(60,153,59)),('goal',(30,194,214))]:
            pt=self.pixel(meta[key])
            cv2.circle(self.base,pt,6,color,-1,cv2.LINE_AA)
            self.text(self.base,key.upper(),pt[0]-36 if key=='goal' else pt[0]+7,pt[1]-7,.34)
        self.history=[]

    def pixel(self,pos):
        xmin,xmax,ymin,ymax=self.meta['bounds']
        return (round(23+(pos[0]-xmin)/(xmax-xmin)*self.map_size),
                round(39+(ymax-pos[1])/(ymax-ymin)*self.map_size))

    @staticmethod
    def text(image,message,x,y,scale=.43,color=(44,51,58),weight=1):
        cv2.putText(image,str(message),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,
                    scale,color,weight,cv2.LINE_AA)

    def render(self,row,trail,summary,*,overview=False):
        frame=self.base.copy()
        before=(row['robot_x'],row['robot_y'])
        origin=self.pixel(before)
        heading=row['heading']
        # Use measured pre-action pose for the recorded scan and post-action
        # pose for the robot after applying the annotated command.
        for angle,distance in row.get('lidar_rays_m',[]):
            theta=heading+angle
            endpoint=(before[0]+distance*math.cos(theta),before[1]+distance*math.sin(theta))
            cv2.line(frame,origin,self.pixel(endpoint),(190,180,65),1,cv2.LINE_AA)
        if len(trail)>1:
            cv2.polylines(frame,[np.array(trail,np.int32)],False,(175,125,48),2,cv2.LINE_AA)
        post=(row['post_x'],row['post_y'])
        center=self.pixel(post)
        h=row['post_heading']
        cv2.circle(frame,center,7,(54,83,213),-1,cv2.LINE_AA)
        if h is not None:
            cv2.arrowedLine(frame,center,self.pixel((post[0]+.20*math.cos(h),post[1]+.20*math.sin(h))),
                            (38,65,114),2,cv2.LINE_AA,tipLength=.25)
        if row.get('target_x') is not None:
            target=self.pixel((row['target_x'],row['target_y']))
            cv2.circle(frame,target,7,(61,160,228),2,cv2.LINE_AA)
            self.text(frame,'DFS',target[0]+8,target[1]+3,.33)
        x=self.panel
        title=f"{row['training_phase'].upper().replace('_',' ')}  |  EP {row['episode']:02d}  |  STEP {row['step']:04d}"
        self.text(frame,title,x,49,.53,(61,104,62),2)
        status='GOAL' if row.get('episode_success') else 'TIMEOUT' if row.get('timeout') else 'CABANG HABIS' if row.get('search_exhausted') else 'GAGAL' if row.get('episode_end') else 'BERJALAN'
        if overview:
            status='TIMELAPSE  /  '+status
        lines=[
            f"Simulasi {f(row['simulation_time'])} s  |  {status}",
            f"Pose sebelum aksi: X {f(before[0])}  Y {f(before[1])} m   heading {f(heading)} rad",
            f"v {f(row['linear_velocity'])} m/s   yaw {f(row['angular_velocity'])} rad/s",
            f"Target ({f(row.get('target_x'))}, {f(row.get('target_y'))}) m  |  bearing {f(row.get('target_bearing'))} rad",
            f"Jarak target {f(row.get('target_distance'))} m  |  aksi {row['action_id'] if row['action_id'] is not None else '-'}: {row['action_name']}",
            'LiDAR (m)    Depan | Depan-kiri | Kiri | Kanan | Depan-kanan',
            '  |  '.join(f(row.get('lidar_'+name+'_m')) for name in ('front','front_left','left','right','front_right')),
            f"Reward {f(row.get('reward_step'))} / langkah  |  akumulasi {f(row.get('cumulative_reward'))}",
            f"Epsilon {f(row.get('epsilon'),3)}  |  kontak {row.get('collision_count','-')}  |  ini {'YA' if row.get('collision_event') else '-'}",
            "Kontrol: "+('pemulihan kontak' if row.get('recovery_active') else 'pindai DFS' if row.get('scanning_active') else 'aksi agent'),
            'Aksi valid: '+ ' '.join(str(i) for i,b in enumerate(row.get('action_mask') or []) if b),
            'Q (ID 0-5): '+ '  '.join(f(v,2) for v in (row.get('q_values') or []))
        ]
        if row['algorithm']=='q_learning':
            lines += [f"State {row.get('state_code') or '-'}  |  visits {row.get('state_visits_after') if row.get('state_visits_after') is not None else '-'}",
                      f"Update Q: {f(row.get('q_old'))} -> {f(row.get('q_new'))}   TD error {f(row.get('td_error'))}"]
        else:
            lines += [f"Loss {f(row.get('loss'),4)} | Q batch {f(row.get('batch_mean_q'))} | replay {row.get('replay_buffer_size','-')}",
                      f"Update online {row.get('gradient_update_count','-')}  |  target {row.get('target_update_count','-')}"]
        for idx,line in enumerate(lines):
            self.text(frame,line,x,75+idx*23,.42)
        self.text(frame,f"Run {row['run_id']}   |   baris telemetry {row['global_row']}",x,408,.37,(107,108,108))
        self.plots(frame)
        return frame

    def plots(self,frame):
        rows=self.history
        metrics=[
            ('Reward + rerata 5','total_reward'),('Langkah / episode','total_steps'),
            ('Sukses 5 episode (%)','success'),('Epsilon','epsilon_end')]
        w=(self.w-40)//4
        for i,(title,key) in enumerate(metrics):
            x=14+i*w; y=self.yplots
            cv2.rectangle(frame,(x,y),(x+w-10,self.h-14),(219,223,224),1)
            self.text(frame,title,x+8,y+18,.39)
            if not rows:
                self.text(frame,'Menunggu episode selesai',x+8,y+42,.32)
                continue
            v=[float(r[key]) if key!='success' else float(r[key]=='True')*100 for r in rows]
            rolling=np.array([np.mean(v[max(0,j-4):j+1]) for j in range(len(v))])
            data=rolling if key=='success' else np.array(v)
            lo=min(np.min(data),np.min(rolling)) if i==0 else np.min(data)
            hi=max(np.max(data),np.max(rolling)) if i==0 else np.max(data)
            if hi-lo<1e-9:
                hi+=1
            def draw(values,color):
                pixels=[(x+9+round(j*(w-28)/max(1,len(v)-1)),self.h-22-round((b-lo)/(hi-lo)*(self.h-y-60))) for j,b in enumerate(values)]
                if len(pixels)>1:
                    cv2.polylines(frame,[np.array(pixels,np.int32)],False,color,1,cv2.LINE_AA)
                else:
                    cv2.circle(frame,pixels[0],2,color,-1)
            draw(data,self.COLORS[i])
            if i==0:
                draw(rolling,(80,87,205))
            self.text(frame,f'{f(float(data[-1]))}   n={len(rows)}',x+8,y+36,.33)


MAP_COLUMNS=('video','frame_index','run_id','recording_phase','episode','step','simulation_time','timestamp','global_row')


def map_frames(writer,video,rows):
    for index,row in enumerate(rows):
        writer.writerow(dict(row,video=video,frame_index=index))


def concat(video_dir, names, destination):
    with tempfile.NamedTemporaryFile(mode='w',suffix='.txt',dir=video_dir) as stream:
        for name in names:
            stream.write(f"file '{name}'\n")
        stream.flush()
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','concat','-safe','0','-i',stream.name,
                        '-c','copy','-movflags','+faststart',str(video_dir/destination)],check=True)


def probe(path):
    out=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=codec_name,nb_frames,width,height,avg_frame_rate,duration','-of','json',str(path)],text=True)
    stream=json.loads(out)['streams'][0]
    assert float(stream['duration'])>0, path
    return stream


def render_best_clip(root, meta):
    """Render every step of the selected attempt without rerunning it."""
    tele, video = root/'telemetry', root/'videos'
    best=meta['best_attempt']
    name=best['filename']
    dashboard=Dashboard(meta,meta['width'],meta['height'])
    with (tele/'episode_summary.csv').open() as stream:
        for r in csv.DictReader(stream):
            if r['episode_key']==best['episode_key']:
                break
            if r['recording_phase']=='training':
                dashboard.history.append(r)
    first, last = int(best['first_row']), int(best['last_row'])
    with (tele/'step_telemetry.jsonl').open() as stream:
        selected=list(map(json.loads,itertools.islice(stream,first-1,last)))
    if len(selected)!=int(best['total_steps']) or any(r['episode_key']!=best['episode_key'] for r in selected):
        raise ValueError('Selected episode no longer matches the recorded telemetry.')
    encoder=Encoder(video/name,meta['width'],meta['height'],meta['fps'],meta['codec'])
    trail=[dashboard.pixel((selected[0]['robot_x'],selected[0]['robot_y']))]
    for row in selected:
        trail.append(dashboard.pixel((row['post_x'],row['post_y'])))
        frame=dashboard.render(row,trail,best)
        if name=='best_available_attempt.mp4':
            dashboard.text(frame,'UPAYA TERBAIK - BELUM MENCAPAI GOAL',dashboard.panel,434,.41,(52,59,160),2)
        encoder.write(frame)
    encoder.close()
    info=probe(video/name)
    if int(info['nb_frames'])!=len(selected):
        raise ValueError('Detailed clip frame count differs from its episode.')
    path=tele/'frame_mapping.csv'
    temp=path.with_suffix('.tmp')
    with path.open() as stream,temp.open('w',newline='') as destination:
        writer=csv.DictWriter(destination,fieldnames=MAP_COLUMNS,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(r for r in csv.DictReader(stream) if r['video']!=name)
        map_frames(writer,name,selected)
    temp.replace(path)
    next(v for v in meta['videos'] if v['filename']==name).update(info,frame_stride=1)
    meta['best_attempt']['frame_stride']=1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fps',type=int,default=24)
    p.add_argument('--width',type=int,default=960)
    p.add_argument('--height',type=int,default=600)
    p.add_argument('--stride',type=int,default=24)
    p.add_argument('--overview-stride',type=int,default=144)
    p.add_argument('--best-only',action='store_true',help='Re-render the previous best clip with all its steps')
    args=p.parse_args()
    if args.width<900 or args.height<560 or args.width%2 or args.height%2 or min(args.fps,args.stride,args.overview_stride)<1:
        p.error('Expected even resolution >=900x560 and positive frame rate and strides')
    root=args.output
    tele=root/'telemetry'
    manifest=tele/'recording_manifest.json'
    meta=json.loads(manifest.read_text())
    if args.best_only:
        if meta['status']!='encoded':
            p.error('Render the complete recording before refreshing the best clip.')
        render_best_clip(root,meta)
        temp=manifest.with_suffix('.tmp')
        temp.write_text(json.dumps(meta,indent=2)+'\n')
        temp.replace(manifest)
        return
    if meta['status'] not in ('captured','partial','interrupted'):
        p.error('Capture must be closed before rendering')
    with (tele/'episode_summary.csv').open() as stream:
        episodes={row['episode_key']:row for row in csv.DictReader(stream)}
    video=root/'videos'
    if video.exists():
        p.error('Video directory already exists. Move it aside before rendering again.')
    (video/'episodes').mkdir(parents=True)
    encs=subprocess.check_output(['ffmpeg','-hide_banner','-encoders'],text=True,stderr=subprocess.STDOUT)
    codec='libx264' if 'libx264 ' in encs else 'mpeg4'
    full_frames,evaluation_frames,overview_frames=[],[],[]
    collections={'training':[],'evaluation':[],'route_speed_test':[]}
    files={}
    dashboard=Dashboard(meta,args.width,args.height)
    overview=Encoder(video/'training_overview.mp4',args.width,args.height,args.fps,codec)
    mapping=tele/'frame_mapping.csv'
    read_count=0
    source_counts=Counter()
    with mapping.open('w',newline='') as stream,(tele/'step_telemetry.jsonl').open() as raw:
        writer=csv.DictWriter(stream,fieldnames=MAP_COLUMNS,extrasaction='ignore')
        writer.writeheader()
        for key,group in itertools.groupby(map(json.loads,raw),lambda row:row['episode_key']):
            rows=list(group); read_count+=len(rows)
            row0=rows[0];phase=row0['recording_phase']; ep=row0['episode']
            source_counts[phase]+=len(rows)
            summary=episodes.get(key)
            filename=f'episodes/{"episode" if phase=="training" else phase}_{ep:06d}.mp4'
            encoder=Encoder(video/filename,args.width,args.height,args.fps,codec)
            trail=[dashboard.pixel((row0['robot_x'],row0['robot_y']))]
            saved=[]
            for i,row in enumerate(rows):
                trail.append(dashboard.pixel((row['post_x'],row['post_y'])))
                first_last=i in (0,len(rows)-1)
                selected=first_last or i%args.stride==0 or row.get('collision_event') or row.get('goal_reached')
                if selected:
                    encoder.write(dashboard.render(row,trail,summary))
                    saved.append(row)
                if phase=='training' and (first_last or i%args.overview_stride==0):
                    overview.write(dashboard.render(row,trail,summary,overview=True))
                    overview_frames.append(row)
            encoder.close()
            files[filename]=saved
            collections[phase].append(filename)
            if phase=='training':
                full_frames.extend(saved)
                if summary:
                    dashboard.history.append(summary)
            elif phase=='evaluation':
                evaluation_frames.extend(saved)
            print(f'{key}: {len(saved)} frames / {len(rows)} steps',flush=True)
        overview.close()
        files['training_overview.mp4']=overview_frames
        for phase,name,frames in (('training','training_full.mp4',full_frames),
                                  ('evaluation','final_evaluation.mp4',evaluation_frames)):
            if collections[phase]:
                concat(video,collections[phase],name)
                files[name]=frames
        if collections['route_speed_test']:
            concat(video,collections['route_speed_test'],'route_speed_test.mp4')
            files['route_speed_test.mp4']=[r for name in collections['route_speed_test'] for r in files[name]]
        candidates=[row for row in episodes.values() if int(row['total_steps'])>0]
        wins=[r for r in candidates if r['success']=='True']
        pick=min(wins,key=lambda r:float(r['duration_simulation_s'])) if wins else min(candidates,key=lambda r:float(r['distance_to_goal_m']))
        phase=pick['recording_phase']
        name=f"episodes/{'episode' if phase=='training' else phase}_{int(pick['episode']):06d}.mp4"
        target='best_successful_run.mp4' if wins else 'best_available_attempt.mp4'
        shutil.copyfile(video/name,video/target)
        files[target]=files[name]
        descriptions=[]
        for name,saved in files.items():
            map_frames(writer,name,saved)
            info=probe(video/name)
            if int(info['nb_frames'])!=len(saved):
                raise ValueError(f'Mismatched video frame count: {name}')
            refs={r['episode_key'] for r in saved}
            descriptions.append(dict(filename=name,algorithm=meta['algorithm'],episode_keys=sorted(refs),
                first_global_row=saved[0]['global_row'],last_global_row=saved[-1]['global_row'],
                first_step=saved[0]['step'],last_step=saved[-1]['step'],status='encoded',**info))
    if read_count!=meta.get('step_rows'):
        raise ValueError(f'Step count mismatch: read {read_count}')
    phase_coverage={phase:dict(step_rows=source_counts[phase],
                               video_frames=sum(len(files[name]) for name in names),
                               steps_without_video_frame=source_counts[phase]-sum(len(files[name]) for name in names))
                    for phase,names in collections.items() if names}
    meta.update(status='encoded',capture_status=meta['status'],fps=args.fps,width=args.width,height=args.height,
        codec=codec,frame_sampling=dict(training_stride=args.stride,overview_stride=args.overview_stride,
                                       retained='first, last, collision and goal in full/evaluation; first and last in overview'),
        frame_mapping='frame_mapping.csv',recorded_episodes=len(episodes),phase_coverage=phase_coverage,
        omitted_episodes=[key for key,row in episodes.items() if int(row['total_steps'])==0],
        best_attempt=dict(filename=target,selection='minimum successful time' if wins else 'closest final position to goal',**pick),
        videos=descriptions)
    render_best_clip(root,meta)
    scratch=manifest.with_suffix('.tmp')
    scratch.write_text(json.dumps(meta,indent=2)+'\n')
    scratch.replace(manifest)


if __name__=='__main__':
    main()
