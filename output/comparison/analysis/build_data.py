"""Rebuild comparative data from completed kinematic recording runs."""
import csv
from hashlib import file_digest
import itertools
import json
import math
from pathlib import Path
import platform
from statistics import mean, median, stdev
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path('controllers/micromouse').resolve()))
from lidar_dqn_agent import load_inference
from lidar_dqn_replay import ReplayBuffer
from lidar_qlearning import TabularLidarQ

TOP = Path('output/comparison')
DATA = TOP / 'data'
SOURCES = {a: Path('output') / a for a in ('q_learning', 'drl')}
SOURCE_DATA = {a: p / ('recording_run' if a == 'q_learning' else 'recording_run/data')
               for a, p in SOURCES.items()}
COMMON = '''run_id algorithm recording_phase training_phase episode episode_key global_row step simulation_time
timestamp robot_x robot_y heading post_x post_y post_heading target_x target_y target_bearing target_distance
linear_velocity angular_velocity lidar_front_m lidar_front_left_m lidar_left_m lidar_right_m lidar_front_right_m
action_id action_name action_mask reward_step cumulative_reward epsilon collision_event collision_count
recovery_active scanning_active waypoint_reached goal_reached episode_success timeout search_exhausted
episode_end terminated truncated q_forward q_curve_left q_curve_right q_turn_left q_turn_right q_stop'''.split()

def read(p):
    return json.loads(p.read_text())

def dump(p, data):
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))

def save_csv(p, records):
    records = list(records)
    with p.open('w', newline='') as f:
        fields = list(dict.fromkeys(k for r in records for k in r))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

def digest(p):
    with p.open('rb') as f:
        return file_digest(f, 'sha256').hexdigest()

def exact(a, b):
    return math.isclose(float(a), float(b), abs_tol=1e-7, rel_tol=1e-11)

metadata = {a: read(p / 'telemetry/recording_manifest.json') for a, p in SOURCES.items()}
config = {a: read(p / 'run.json') for a, p in SOURCE_DATA.items()}
for key in ('walls', 'start', 'goal', 'step_seconds', 'bounds', 'cell_size'):
    assert metadata['q_learning'][key] == metadata['drl'][key]
assert config['q_learning']['maze_signature'] == config['drl']['maze']['signature']

tracks, all_episodes = {}, {}
benchmark_input = {'q_learning': {}, 'drl': {}}
with (DATA / 'common_telemetry.csv').open('w', newline='') as csvfile, (DATA / 'common_telemetry.jsonl').open('w') as jsonfile:
    writer = csv.DictWriter(csvfile, fieldnames=COMMON)
    writer.writeheader()
    for alg, root in SOURCES.items():
        tracks[alg] = stats = {}
        with (root / 'telemetry/step_telemetry.jsonl').open() as stream:
            for line in stream:
                r = json.loads(line)
                item = {k: r.get(k) for k in COMMON}
                jsonfile.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')
                item['action_mask'] = json.dumps(item['action_mask'], separators=(',', ':'))
                writer.writerow(item)
                s = stats.setdefault(r['episode_key'], dict(
                    reward=0., path=0., collision_count=0, scan_steps=0, recovery_steps=0,
                    waypoints=0, steps=0, heading=r['heading'], last=None, zeros=0))
                s['steps'] += 1
                assert s['steps'] == r['step']
                assert r['action_mask'][r['action_id']]
                s['reward'] += r['reward_step']
                s['path'] += math.hypot(r['post_x']-r['robot_x'],r['post_y']-r['robot_y'])
                for target, field in [('collision_count','collision_event'),('scan_steps','scanning_active'),
                                      ('recovery_steps','recovery_active'),('waypoints','waypoint_reached')]:
                    s[target] += bool(r.get(field))
                s['zeros'] += int(r['action_id']==5 and not r['recovery_active'] and not r['scanning_active'])
                s['last'] = r
                if r['recording_phase'] == 'evaluation':
                    assert r['epsilon'] == 0 and not r.get('q_update_applied') and r.get('loss') is None
                    token = (r['episode'], r['step'])
                    encoded = r.get('state_code' if alg == 'q_learning' else 'observation')
                    if encoded:
                        benchmark_input[alg][token] = encoded
        assert sum(s['steps'] for s in stats.values()) == metadata[alg]['step_rows']
        updates = rows(root / 'telemetry/learning_updates.csv')
        by_ep = {}
        for episode, items in itertools.groupby(updates, lambda r: r['episode_key']):
            by_ep[episode] = list(items)
        ep_rows = rows(root / 'telemetry/episode_summary.csv')
        train = rows(SOURCE_DATA[alg] / 'training.csv')
        eval_data = read(SOURCE_DATA[alg] / 'evaluations.json')
        episode_tracks = []
        for ep in ep_rows:
            key = ep['episode_key']
            s = stats[key]
            last = s.pop('last')
            i = int(ep['episode'])-1
            phase = ep['recording_phase']
            raw = train[i] if phase == 'training' else eval_data[i] if phase == 'evaluation' else None
            for observed, logged in ((s['steps'],ep['total_steps']), (s['reward'],ep['total_reward']),
                                     (s['collision_count'],ep['collision_count']),
                                     (s['steps']*.16,ep['duration_simulation_s'])):
                assert exact(observed, logged)
            assert last['episode_end'] and last['goal_reached'] == (ep['success']=='True')
            success = ep['success']=='True'
            if raw is not None:
                assert exact(s['reward'],raw['reward']) and exact(s['steps'],raw['steps'])
                assert success == (str(raw['success'])=='True')
                assert exact(s['collision_count'],raw['collision_count'])
            length = None if raw is None else float(raw.get('travelled_distance_m', raw.get('distance_m')))
            updates_here = by_ep.get(key, [])
            detail = dict(algorithm=alg, run_id=ep['run_id'], episode=i+1, episode_key=key,
                          scope=phase, phase=ep['phase'], seed=None, heading_rad=s['heading'],
                          reward=s['reward'], steps=s['steps'], duration_s=float(ep['duration_simulation_s']),
                          wall_s=float(ep['duration_wall_clock_s']), success=success,
                          collision=s['collision_count']>0, collision_count=s['collision_count'],
                          timeout=ep['timeout']=='True', search_exhausted=ep['search_exhausted']=='True',
                          distance_m=length, sampled_path_m=s['path'], scan_steps=s['scan_steps'],
                          waypoint_events=s['waypoints'], recovery_steps=s['recovery_steps'],
                          recovery_s=s['recovery_steps']*.16, nonforced_stop_steps=s['zeros'],
                          epsilon_start=float(ep['epsilon_start']), epsilon_end=float(ep['epsilon_end']),
                          updates=len(updates_here), q_selected_mean=None, q_td_abs_mean=None,
                          q_states=None, replay_size=None, td_loss=None, dqn_mean_q=None, dqn_max_q=None,
                          target_updates=None, residual_goal_m=float(ep['distance_to_goal_m']))
            if length is not None:
                detail['avg_speed_m_s'] = length/detail['duration_s']
            if phase == 'training' and alg=='q_learning':
                detail['q_states'] = int(raw['q_states'])
                detail['q_selected_mean'] = mean(float(u['q_old']) for u in updates_here)
                detail['q_td_abs_mean'] = mean(abs(float(u['td_error'])) for u in updates_here)
            if phase == 'training' and alg=='drl':
                detail.update(replay_size=int(raw['replay_size']), td_loss=float(raw['td_loss']),
                              dqn_mean_q=float(raw['mean_q']), dqn_max_q=float(raw['max_q']),
                              target_updates=int(raw['target_updates']))
            if phase=='evaluation':
                detail['seed'] = int(raw['seed'])
                assert exact(detail['heading_rad'], ((detail['seed']%8)-3.5)*math.pi/4)
            episode_tracks.append(detail)
        assert len(train)==sum(e['scope']=='training' for e in episode_tracks)
        all_episodes[alg] = episode_tracks

# Use the same frozen artifacts and a deterministic subset of paired decision IDs.
qpath = SOURCE_DATA['q_learning'] / 'q_table.json'
model_path = SOURCES['drl'] / 'recording_run/models/last_model.pt'
q, _ = TabularLidarQ.load(qpath)
torch.set_num_threads(1)
network, checkpoint = load_inference(model_path)
picked = sorted(benchmark_input['q_learning'].keys() & benchmark_input['drl'].keys())[::7][:1024]
qinputs = [benchmark_input['q_learning'][k] for k in picked]
dinputs = [torch.tensor(benchmark_input['drl'][k]) for k in picked]
zero = np.zeros(6, dtype=np.float32)
def q_infer(key):
    return q.q.get(key, zero)
def d_infer(tensor):
    return network(tensor)
def time_calls(fn, inputs):
    for x in inputs:
        fn(x)
    t = time.perf_counter_ns()
    for x in inputs:
        fn(x)
    return (time.perf_counter_ns()-t)/len(inputs)/1000
timings = {'q_learning': [], 'drl': []}
calls = [(q_infer, qinputs, 'q_learning'), (d_infer, dinputs, 'drl')]
bench_cpu = next(int(num) for num in sorted(__import__('os').sched_getaffinity(0)))
os = __import__('os')
os.sched_setaffinity(0, {bench_cpu})
with torch.inference_mode():
    for i in range(21):
        for fn, xs, key in calls[::1 if i%2==0 else -1]:
            timings[key].append(time_calls(fn, xs))
cpu = subprocess.check_output(['lscpu'],text=True)
cpu = next(line.split(':',1)[1].strip() for line in cpu.splitlines() if line.startswith('Model name:'))
bench = dict(cpu=cpu, logical_cpu=bench_cpu, torch_threads=1, decisions_per_batch=len(picked), batches=21,
             python=sys.version, torch=torch.__version__, platform=platform.platform(),
             scope='Frozen six-Q-value scoring only. Inputs pre-encoded. No LiDAR, masking, argmax, or physics.',
             latency_us={a:dict(raw=t, median=median(t), mean=mean(t), sd=stdev(t),
                                minimum=min(t), maximum=max(t)) for a,t in timings.items()})
dump(DATA/'inference_benchmark.json',bench)

qdata = read(qpath)
visits = list(qdata['visits'].values())
buffer = ReplayBuffer(config['drl']['config']['capacity'])
replay_bytes = sum(x.nbytes for x in buffer.arrays.values())
params = sum(x.numel() for x in network.parameters())
weight_bytes = sum(x.numel()*x.element_size() for x in network.parameters())
model_dir = model_path.parent
memory = {
    'q_learning': dict(states=len(q.q), theoretical_states=qdata['state_space_upper_bound'],
                       coverage_percent=100*len(q.q)/qdata['state_space_upper_bound'],
                       visit_total=sum(visits), visit_mean=mean(visits), visit_median=median(visits),
                       visits_0=sum(v==0 for v in visits), visits_1_to_5=sum(1<=v<=5 for v in visits),
                       visits_6_to_20=sum(6<=v<=20 for v in visits),
                       visits_over_20=sum(v>20 for v in visits),
                       weight_bytes=sum(v.nbytes for v in q.q.values()),
                       file_bytes=qpath.stat().st_size),
    'drl': dict(parameters=params, trainable_architecture_parameters=params, observation=len(config['drl']['observation']),
                weight_bytes=weight_bytes, replay_allocated_bytes=replay_bytes,
                replay_bytes_per_item=replay_bytes//buffer.capacity, replay_capacity=buffer.capacity,
                file_bytes=model_path.stat().st_size,
                checkpoint_bytes={p.name:p.stat().st_size for p in model_dir.glob('*.pt')})}
def metrics(episodes):
    d = dict(episodes=len(episodes))
    ds = [e for e in episodes if e['success']]
    for f in ('success','collision','timeout','search_exhausted'):
        d[f] = sum(e[f] for e in episodes)
    d.update(failed=len(episodes)-len(ds), success_rate=100*len(ds)/len(episodes),
             success_after_collision=sum(e['collision'] for e in ds))
    for f in ('steps','collision_count','scan_steps','recovery_steps','waypoint_events','updates','wall_s'):
        d[f+'_total'] = sum(e[f] for e in episodes)
    for f in ('reward','steps','duration_s','distance_m','avg_speed_m_s'):
        vals = [e[f] for e in episodes if e.get(f) is not None]
        if vals:
            d[f] = dict(mean=mean(vals), median=median(vals), sd=stdev(vals) if len(vals)>1 else 0.,
                        min=min(vals), max=max(vals))
    d['best_success_s'] = min((e['duration_s'] for e in ds),default=None)
    d['mean_success_s'] = mean(e['duration_s'] for e in ds) if ds else None
    d['final_5_reward'] = mean(e['reward'] for e in episodes[-5:])
    return d

all_rows = list(itertools.chain.from_iterable(all_episodes.values()))
save_csv(DATA/'episode_metrics.csv',all_rows)
group_stats = {a:{phase:metrics([e for e in es if e['scope']==phase]) for phase in
                  sorted({e['scope'] for e in es})} for a,es in all_episodes.items()}
phase_stats = {a:{p:metrics([e for e in es if e['scope']=='training' and e['phase']==p]) for p in
                  sorted({e['phase'] for e in es if e['scope']=='training'})} for a,es in all_episodes.items()}
paired = []
for x,y in zip(*(list(e for e in es if e['scope']=='evaluation') for es in all_episodes.values()), strict=True):
    assert x['seed']==y['seed'] and exact(x['heading_rad'],y['heading_rad'])
    for e in (x,y):
        paired.append({**e,'pair_status':'matched_initial_conditions','full_protocol_status':'partially_matched'})
save_csv(DATA/'matched_evaluation.csv',paired)
dump(DATA/'matched_evaluation.json', dict(scope='Existing frozen evaluations. Initial conditions matched, complete protocol partially matched.',
                                         independent_training_seeds=1, unique_headings=8, paired_scenarios=len(paired)//2, results=paired))
summary = dict(sources={a:dict(data=str(SOURCE_DATA[a]), run_id=m['run_id'], steps=m['step_rows'],
                                best=m['best_attempt']) for a,m in metadata.items()},
               metrics=group_stats, training_phases=phase_stats, memory=memory,
               benchmark=bench, comparability='PARTIALLY MATCHED',
               train_initial_headings_equal=sum(exact(x['heading_rad'],y['heading_rad'])
                   for x,y in zip(all_episodes['q_learning'][:36],all_episodes['drl'][:36])),
               evaluation_pairs_both_success=sum(x['success'] and y['success']
                   for x,y in zip(paired[::2],paired[1::2])))
dump(DATA/'comparison_summary.json',summary)
save_csv(DATA/'comparison_metrics.csv',(
    dict(algorithm=a,scope=p,metric=k,statistic=h,value=n)
    for a, phases in group_stats.items() for p, ms in phases.items() for k,v in ms.items()
    for h,n in (v.items() if isinstance(v,dict) else [('value',v)])))
active = ['run.json','training.csv','evaluations.json','summary.json','routes.json']
integrity = {}
for a, p in SOURCES.items():
    selected = [SOURCE_DATA[a]/n for n in active]
    selected += [p/'telemetry'/n for n in ('recording_manifest.json','step_telemetry.csv','step_telemetry.jsonl',
                  'episode_summary.csv','learning_updates.csv','frame_mapping.csv')]
    selected += sorted((p/'videos').glob('*.mp4'))
    selected += [qpath] if a=='q_learning' else sorted(model_dir.glob('*.pt')) + [
                    SOURCE_DATA[a]/'evaluation_results.csv']
    integrity.update({str(f):digest(f) for f in selected})
dump(DATA/'source_sha256.json',integrity)
