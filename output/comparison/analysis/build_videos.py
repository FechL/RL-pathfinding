"""Create synchronized paired footage with a source-row map for every frame."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path('tools').resolve()))
from render_micromouse_recording import Dashboard, Encoder, probe

cv2.setNumThreads(1)
out = Path('output/comparison')
roots = [Path('output/q_learning'), Path('output/drl')]
meta = [json.loads((r/'telemetry/recording_manifest.json').read_text()) for r in roots]
source_tracks = []
source_paths = []
for root, m in zip(roots, meta):
    tracks, paths = defaultdict(list), defaultdict(list)
    for line in (root/'telemetry/step_telemetry.jsonl').open():
        row = json.loads(line)
        key = row['episode_key']
        is_best = key == m['best_attempt']['episode_key']
        paths[key].append((row['post_x'], row['post_y']))
        kept = {k:row.get(k) for k in ('run_id','algorithm','episode','episode_key','step','global_row',
            'simulation_time','training_phase','robot_x','robot_y','heading','post_x','post_y','post_heading',
            'linear_velocity','angular_velocity','target_x','target_y','target_bearing','target_distance',
            'lidar_front_m','lidar_front_left_m','lidar_left_m','lidar_right_m','lidar_front_right_m',
            'lidar_rays_m','action_id','action_name','action_mask','reward_step','cumulative_reward','epsilon',
            'collision_event','collision_count','episode_success','episode_end','timeout','search_exhausted',
            'recovery_active','scanning_active','q_values','state_code','state_visits_after',
            'q_old','q_new','td_error','loss','batch_mean_q','replay_buffer_size','gradient_update_count',
            'target_update_count','timestamp')}
        if is_best or row['step']==1 or (row['step']-1)%6==0 or row['episode_end']:
            tracks[key].append(kept)
    source_tracks.append(tracks)
    source_paths.append(paths)

W,H,FPS = 960,600,12
dashboards = [Dashboard(m,W,H) for m in meta]
best_keys = [m['best_attempt']['episode_key'] for m in meta]
metrics = {}
with (out/'data/episode_metrics.csv').open() as src:
    for r in csv.DictReader(src):
        metrics[(r['run_id'],r['episode_key'])]=r
video_specs = {
    'training_comparison': ('Training  |  kedua run dimulai ulang pada awal setiap episode', 48,
                            [(f'training-{n:06d}',)*2 for n in range(1,37)]),
    'evaluation_side_by_side': ('Evaluasi beku  |  start, heading dan seed DFS berpasangan', 18,
                                [(f'evaluation-{n:06d}',)*2 for n in range(1,13)]),
    'best_route_comparison': ('Rekor masing-masing  |  Q: replay saat training  |  DQN: replay beku', 1,[best_keys]),
    'telemetry_comparison': ('Inspeksi telemetry  |  seed DFS 1  |  evaluasi beku', 6,[('evaluation-000002',)*2]),
    'trajectory_comparison': ('Lintasan aktual  |  seed DFS 0 dan 7  |  evaluasi beku', 18,
                              [(f'evaluation-{n:06d}',)*2 for n in (1,8)])
}

def panel(idx, key, selected, mode):
    d = dashboards[idx]
    pixels = [d.pixel(p) for p in source_paths[idx][key][:selected['step']]]
    image = d.render(selected,pixels,None)
    if mode == 'trajectory_comparison':
        eventual = [d.pixel(p) for p in source_paths[idx][key]]
        if len(eventual)>1:
            cv2.polylines(image, [np.array(eventual,np.int32)], False,(169,169,169),1,cv2.LINE_AA)
            cv2.polylines(image, [np.array(pixels,np.int32)], False,(175,125,48),2,cv2.LINE_AA)
            cv2.circle(image,pixels[-1],7,(54,83,213),-1,cv2.LINE_AA)
        d.text(image,'Abu-abu: lintasan akhir  |  Biru: lintasan hingga frame',25,552,.52)
    elif mode == 'telemetry_comparison':
        d.text(image,'Aksi, mask dan Q memakai observasi sebelum aksi.',25,552,.52)
        d.text(image,'Pose lintasan memakai hasil sesudah aksi.',25,575,.52)
    if mode == 'best_route_comparison':
        info=metrics[(selected['run_id'],key)]
        length = float(info['sampled_path_m'])
        d.text(image,f"Rekor: {float(info['duration_s']):.2f} s   |   {info['steps']} langkah   |   {info['collision_count']} kontak", 25,552,.55)
        d.text(image,f"Panjang lintasan dari pose telemetry: {length:.2f} m",25,576,.55)
    return image

def best_source_frames(root, path, m):
    with (root/'telemetry/frame_mapping.csv').open() as src:
        mapping = [r for r in csv.DictReader(src) if r['video']=='best_successful_run.mp4']
    vid = cv2.VideoCapture(str(root/'videos/best_successful_run.mp4'))
    if not vid.isOpened():
        raise RuntimeError('Could not open validated source clip')
    return vid, mapping

man = {}
with (out/'data/video_frame_mapping.csv').open('w', newline='') as sink:
    writer=csv.writer(sink)
    writer.writerow(('video','frame_index','pair_index','shared_clock_s','side','run_id','episode_key',
                     'step','simulation_time','global_row','timestamp','held','source_file','source_frame'))
    for mode,(title,stride,pairs) in video_specs.items():
        filename = mode+'.mp4'
        fps = 24 if mode=='best_route_comparison' else FPS
        encoder=Encoder(out/'videos'/filename,W*2,H+40,fps,'libx264')
        footage = [best_source_frames(root, None, m) for root,m in zip(roots,meta)] if mode=='best_route_comparison' else None
        cached = [None,None]
        for index,keys in enumerate(pairs,1):
            ends = [tracks[key][-1]['step'] for tracks,key in zip(source_tracks,keys)]
            clocks = list(range(1,max(ends)+1,stride))
            if clocks[-1]!=max(ends):
                clocks.append(max(ends))
            by_step = [{r['step']:r for r in tracks[key]} for tracks,key in zip(source_tracks,keys)]
            for step in clocks:
                images=[]
                for idx,(key,end,data) in enumerate(zip(keys,ends,by_step)):
                    row=data[min(end,step)]
                    held=step>end
                    src_file, src_frame = 'telemetry/step_telemetry.jsonl', None
                    if footage:
                        capture, mapping = footage[idx]
                        token=mapping[row['step']-1]
                        assert int(token['global_row'])==row['global_row']
                        if not held:
                            ok, cached[idx] = capture.read()
                            assert ok
                        image=cached[idx].copy()
                        src_file,src_frame='videos/best_successful_run.mp4',token['frame_index']
                        info=metrics[(row['run_id'],key)]
                        cv2.rectangle(image,(0,510),(W,H),(248,248,248),-1)
                        dashboards[idx].text(image,f"Rekor {float(info['duration_s']):.2f} s  |  {end} langkah  |  {info['collision_count']} kontak  |  lintasan pose {float(info['sampled_path_m']):.2f} m",25,555,.57)
                    else:
                        image=panel(idx,key,row,mode)
                    if held:
                        dashboards[idx].text(image,'FRAME AKHIR DITAHAN',26,518,.48,(37,69,162),2)
                    writer.writerow((filename,encoder.count,index,step*.16,'left' if idx==0 else 'right',
                                     row['run_id'],key,row['step'],row['simulation_time'],row['global_row'],
                                     row['timestamp'],held,src_file,src_frame))
                    images.append(image)
                frame=np.full((H+40,W*2,3),246,np.uint8)
                frame[40:,:]=np.hstack(images)
                Dashboard.text(frame,title,21,25,.62,(32,54,74),2)
                Dashboard.text(frame,f"Pasangan {index}/{len(pairs)}   |   jam simulasi {step*.16:.2f} s", 1340,25,.56)
                encoder.write(frame)
        if footage:
            for capture,_ in footage:
                capture.release()
        encoder.close()
        man[filename] = dict(**probe(out/'videos'/filename), pairs=len(pairs), stride_steps=stride,
                             simulation_step_s=.16, playback_fps=fps, frame_count=encoder.count,
                             alignment='Identical simulated clock reset each pair; freeze finished side.',
                             provenance='Original validated MP4 frames' if footage else 'Actual recorded telemetry, re-rendered using project HUD')
(out/'data/video_manifest.json').write_text(json.dumps(man,indent=2)+'\n')
