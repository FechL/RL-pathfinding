"""Validate derived CSV/JSON, source integrity, and every comparative MP4."""
from collections import Counter, defaultdict
import csv
from hashlib import file_digest
from itertools import zip_longest
import json
import math
from pathlib import Path
import subprocess

out=Path('output/comparison')
data=out/'data'
def read(p): return json.loads(p.read_text())
def close(a,b): return math.isclose(float(a),float(b),abs_tol=1e-7)
inputs = read(data/'source_sha256.json')
for source,expected in inputs.items():
    with Path(source).open('rb') as handle:
        assert file_digest(handle,'sha256').hexdigest()==expected, source
details={}
by_episode={}
rows=Counter()
with (data/'common_telemetry.csv').open() as a, (data/'common_telemetry.jsonl').open() as b:
    for csvrow,line in zip_longest(csv.DictReader(a),b):
        assert csvrow is not None and line is not None
        r=json.loads(line)
        for name,value in r.items():
            as_csv=json.dumps(value,separators=(',', ':')) if name=='action_mask' else '' if value is None else str(value)
            assert csvrow[name]==as_csv, (name,r['global_row'])
        key=(r['run_id'],r['global_row'])
        assert key not in details
        details[key]=(r['episode_key'],r['step'],r['simulation_time'],r['timestamp'],r['episode_end'])
        rows[r['run_id']]+=1
        assert rows[r['run_id']]==r['global_row']
        e=by_episode.setdefault((r['run_id'],r['episode_key']),[0,0,0,False])
        e[0]+=1
        e[1]+=r['reward_step']
        e[2]+=int(r['collision_event'])
        e[3]=r['goal_reached']
        if r['recording_phase']!='training':
            assert r['epsilon']==0
with (data/'episode_metrics.csv').open() as src:
    for r in csv.DictReader(src):
        count,reward,contacts,goal=by_episode[(r['run_id'],r['episode_key'])]
        assert count==int(r['steps']) and close(reward,r['reward']) and contacts==int(r['collision_count'])
        assert str(goal)==r['success']
        assert close(r['duration_s'],int(r['steps'])*.16)
videos=read(data/'video_manifest.json')
paired=defaultdict(dict)
coverage=defaultdict(set)
with (data/'video_frame_mapping.csv').open() as src:
    for r in csv.DictReader(src):
        eid,step,timestamp_s,timestamp,terminal=details[(r['run_id'],int(r['global_row']))]
        assert (eid,step,timestamp)==(r['episode_key'],int(r['step']),r['timestamp'])
        assert close(timestamp_s,r['simulation_time'])
        assert r['side'] not in paired[(r['video'],int(r['frame_index']))]
        assert (r['held']=='True' and terminal and timestamp_s<float(r['shared_clock_s'])) or close(timestamp_s,r['shared_clock_s'])
        paired[(r['video'],int(r['frame_index']))][r['side']]=(r['pair_index'],r['shared_clock_s'])
        coverage[r['video']].add(r['pair_index'])
counts=Counter()
for (name,idx),sides in paired.items():
    assert idx==counts[name] and sides['left']==sides['right']
    counts[name]+=1
for name,v in videos.items():
    assert counts[name]==v['frame_count']==int(v['nb_frames'])
    assert len(coverage[name])==v['pairs']
    assert v['codec_name']=='h264' and float(v['duration'])>0
    subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(out/'videos'/name),'-f','null','-'],check=True)
q_dir=Path('output/q_learning/recording_run')
d_dir=Path('output/drl/recording_run')
sha_file=str(d_dir/'models/last_model.pt')
assert read(d_dir/'data/summary.json')['sha256']==inputs[sha_file]
actuals=list(csv.DictReader((data/'matched_evaluation.csv').open()))
assert len(actuals)==24
for seed, (q,d) in enumerate(zip(actuals[::2],actuals[1::2])):
    assert q['seed']==d['seed']==str(seed)
    assert close(q['heading_rad'],d['heading_rad'])
    assert q['updates']==d['updates']=='0'
assert sum(x['success']=='True' for x in actuals[::2])==9
assert sum(x['success']=='True' for x in actuals[1::2])==12
print(json.dumps(dict(source_hashes=len(inputs),telemetry_rows=rows,episodes=len(by_episode),
    frozen_pairs=len(actuals)//2,video_frames=counts,video_panels_mapped=sum(counts.values())*2,
    graphs=len(list((out/'graphs').glob('*.png'))),status='passed'),indent=2))
