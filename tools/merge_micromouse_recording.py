"""Join a resumed capture at the last completed, checkpointed episode boundary."""
import argparse
import csv
import json
from pathlib import Path


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first',type=Path,required=True,help='Directory containing initial telemetry')
    p.add_argument('--continued',type=Path,required=True,help='Directory containing resumed telemetry')
    p.add_argument('--output',type=Path,required=True,help='Fresh directory for merged telemetry')
    args=p.parse_args()
    parts=[d/'telemetry' for d in (args.first,args.continued)]
    metadata=[json.loads((d/'recording_manifest.json').read_text()) for d in parts]
    for key in ('algorithm','bounds','start','goal','step_seconds','feature_names'):
        if metadata[0][key]!=metadata[1][key]:
            raise ValueError(f'Capture parts differ in {key}.')
    summaries=[read_rows(d/'episode_summary.csv') for d in parts]
    if (not summaries[0] or not summaries[1]
            or summaries[0][-1]['recording_phase']!='training'
            or summaries[1][0]['recording_phase']!='training'
            or int(summaries[0][-1]['episode'])+1!=int(summaries[1][0]['episode'])):
        raise ValueError('Capture segments must meet at a completed training episode.')
    rows_before_cut=int(summaries[0][-1]['last_row'])
    run=metadata[0]['run_id']
    out=args.output/'telemetry'
    out.mkdir(parents=True,exist_ok=False)
    offsets=[0,rows_before_cut]
    limits=[rows_before_cut,int(metadata[1]['step_rows'])]
    discarded=0
    with (out/'step_telemetry.jsonl').open('w') as destination:
        for i,d in enumerate(parts):
            with (d/'step_telemetry.jsonl').open() as source:
                for line in source:
                    row=json.loads(line)
                    if row['global_row']>limits[i]:
                        discarded+=1
                        continue
                    row.update(run_id=run,global_row=row['global_row']+offsets[i])
                    destination.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    for name in ('step_telemetry.csv','learning_updates.csv','episode_summary.csv'):
        best={}
        with (out/name).open('w',newline='') as destination:
            writer=None
            for i,d in enumerate(parts):
                with (d/name).open() as source:
                    reader=csv.DictReader(source)
                    if writer is None:
                        writer=csv.DictWriter(destination,fieldnames=reader.fieldnames)
                        writer.writeheader()
                    for row in reader:
                        row['run_id']=run
                        if name=='episode_summary.csv':
                            for column in ('first_row','last_row'):
                                row[column]=int(row[column])+offsets[i]
                            key=row['recording_phase']
                            reward,fast=best.get(key,(-float('inf'),None))
                            reward=max(reward,float(row['total_reward']))
                            if row['success']=='True':
                                t=float(row['duration_simulation_s'])
                                fast=t if fast is None else min(t,fast)
                            best[key]=reward,fast
                            row.update(best_reward_so_far=reward,best_successful_time_so_far=fast)
                        else:
                            n=int(row['global_row'])
                            if n>limits[i]:
                                continue
                            row['global_row']=n+offsets[i]
                        writer.writerow(row)
    count=rows_before_cut+limits[1]
    manifest=dict(metadata[0],status='captured',step_rows=count,
                  finished_at=metadata[1]['finished_at'],
                  segments=[dict(run_id=metadata[0]['run_id'],rows=rows_before_cut),
                            dict(run_id=metadata[1]['run_id'],rows=limits[1])],
                  resumed_after_episode=int(summaries[0][-1]['episode']),
                  discarded_interrupted_steps=discarded,
                  note='The interrupted partial episode is kept separately. It was restarted from its original episode-boundary checkpoint.')
    (out/'recording_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
