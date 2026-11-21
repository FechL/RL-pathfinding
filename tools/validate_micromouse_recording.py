"""Cross-check episode completeness, rewards, masks, learning rows, and MP4 frames."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import subprocess

from render_micromouse_recording import probe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-episodes',type=int,required=True)
    parser.add_argument('--evaluation-episodes',type=int,required=True)
    args=parser.parse_args()
    tele=args.output/'telemetry'
    video=args.output/'videos'
    manifest=json.loads((tele/'recording_manifest.json').read_text())
    by_row={}
    episode_data={}
    gradients=Counter()
    step_groups=Counter()
    with (tele/'step_telemetry.jsonl').open() as raw,(tele/'step_telemetry.csv').open() as raw_csv:
        from itertools import zip_longest
        for rownum,(line,csvrow) in enumerate(zip_longest(raw,csv.DictReader(raw_csv)),1):
            assert line and csvrow
            r=json.loads(line)
            assert r['global_row']==rownum==int(csvrow['global_row'])
            ident=r['episode_key']
            step_groups[r['recording_phase']]+=1
            assert r['action_mask'][r['action_id']]
            assert r['action_id']==int(csvrow['action_id'])
            assert math.isclose(r['reward_step'],float(csvrow['reward_step']),abs_tol=1e-8)
            assert math.isclose(r['simulation_time'],r['step']*manifest['step_seconds'],abs_tol=1e-8)
            assert str(r['collision_event'])==csvrow['collision_event']
            assert str(r['goal_reached'])==csvrow['goal_reached']
            if r['recording_phase']!='training':
                assert r['epsilon']==0 and not r.get('q_update_applied') and r.get('loss') is None
            if r.get('q_update_applied'):
                assert not r['scanning_active'] and not r['recovery_active']
                assert math.isclose(r['td_target']-r['q_old'],r['td_error'],abs_tol=1e-8)
                gradients['q']+=1
            if r.get('loss') is not None:
                assert math.isfinite(r['loss'])
                gradients['dqn']+=1
            ep=episode_data.setdefault(ident,[0,0,0,r['global_row'],None])
            ep[0]+=1
            ep[1]+=r['reward_step']
            ep[2]+=int(r['collision_event'])
            ep[4]=r
            by_row[rownum]=(r['run_id'],r['episode'],r['step'],r['simulation_time'],r['timestamp'],r['recording_phase'])
    assert len(by_row)==manifest['step_rows']
    with (tele/'learning_updates.csv').open() as stream:
        updates=list(csv.DictReader(stream))
    assert len(updates)==sum(gradients.values())
    for row in updates:
        assert row['recording_phase']=='training'
        assert int(row['global_row']) in by_row
    with (tele/'episode_summary.csv').open() as stream:
        summaries=list(csv.DictReader(stream))
    assert len(summaries)==len(episode_data)
    for s in summaries:
        steps,reward,contacts,start,last=episode_data[s['episode_key']]
        assert last['episode_end']
        assert steps==int(s['total_steps'])==last['step']
        assert start==int(s['first_row'])
        assert last['global_row']==int(s['last_row'])
        assert contacts==int(s['collision_count'])
        assert str(last['goal_reached'])==s['success']
        assert math.isclose(reward,float(s['total_reward']),abs_tol=1e-7)
    phases=Counter(s['recording_phase'] for s in summaries)
    assert phases['training']==args.training_episodes
    assert phases['evaluation']==args.evaluation_episodes
    frames=Counter()
    covered={}
    with (tele/'frame_mapping.csv').open() as stream:
        for m in csv.DictReader(stream):
            token=(m['run_id'],int(m['episode']),int(m['step']),float(m['simulation_time']),
                   m['timestamp'],m['recording_phase'])
            assert token==by_row[int(m['global_row'])]
            name=m['video']; assert int(m['frame_index'])==frames[name]
            frames[name]+=1
            covered.setdefault(name,set()).add(int(m['episode']))
    assert len(covered['training_full.mp4'])==len(covered['training_overview.mp4'])==args.training_episodes
    assert len(covered['final_evaluation.mp4'])==args.evaluation_episodes
    for record in manifest['videos']:
        info=probe(video/record['filename'])
        assert int(info['nb_frames'])==frames[record['filename']]
    best=manifest['best_attempt']
    best_name=best['filename']
    wins=[float(s['duration_simulation_s']) for s in summaries if s['success']=='True']
    assert best_name==('best_successful_run.mp4' if wins else 'best_available_attempt.mp4')
    if wins:
        assert float(best['duration_simulation_s'])==min(wins)
    primary=['training_full.mp4','training_overview.mp4','final_evaluation.mp4',best_name]
    webots=manifest.get('webots')
    if isinstance(webots,dict) and webots.get('status')=='encoded':
        primary.append(webots['filename'])
        with (tele/'webots_step_telemetry.jsonl').open() as source:
            web={r['global_row']:r for r in map(json.loads,source)}
        with (tele/'webots_frame_mapping.csv').open() as stream:
            index=-1
            for index,row in enumerate(csv.DictReader(stream)):
                assert int(row['frame_index'])==index
                r=web[int(row['global_row'])]
                assert r['step']==int(row['step']) and r['simulation_time']==float(row['simulation_time'])
        assert index+1==int(probe(video/manifest['webots']['filename'])['nb_frames'])
    for name in primary:
        subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(video/name),
                        '-threads','1','-f','null','-'],check=True)
    summary=dict(result='passed',episodes=dict(phases),step_rows=len(by_row),step_phases=dict(step_groups),
                 learning_updates=dict(gradients),decoded_primary=primary,
                 videos_probed=len(manifest['videos'])+int('webots_evaluation.mp4' in primary),
                 best=best)
    (tele/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
