"""Train without Webots physics; evaluate learned options before wheel execution."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'controllers/micromouse'))
from planner import Obstacle,path_length,physical_clearance
from q_learning import ContinuousMaze,rollout,save_policy,train


def read_world():
    text=(ROOT/'worlds/world.wbt').read_text()
    layout=json.loads((ROOT/'worlds/maze_layout.json').read_text())
    walls=[]
    for block in re.findall(r'\bMazeWall\s*\{([^{}]+)\}',text):
        pos=list(map(float,re.search(r'translation\s+([^\n]+)',block)[1].split()))
        size=list(map(float,re.search(r'size\s+([^\n]+)',block)[1].split()))
        name=re.search(r'name\s+"([^"]+)"',block)[1]
        walls.append(Obstacle(name,'wall',tuple(pos[:2]),tuple(size[:2])))
    if not walls: raise ValueError('No maze walls in world')
    start=tuple(map(float,re.search(r'DEF MICROMOUSE Micromouse\s*\{\s*translation\s+(\S+)\s+(\S+)',text).groups()))
    goal=tuple(map(float,re.search(r'DEF GOAL Pose\s*\{\s*translation\s+(\S+)\s+(\S+)',text).groups()))
    w,h,cell=map(float,re.search(r'DEF ARENA NavigationArena\s*\{\s*floorSize\s+(\S+)\s+(\S+)\s+cellSize\s+(\S+)',text).groups())
    if layout['grid_size'] != [round(w/cell),round(h/cell)]:
        raise ValueError('Regenerate world after changing layout')
    return ContinuousMaze(start,goal,walls,(-w/2,w/2,-h/2,h/2),cell)


def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=6000)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--evaluation-runs',type=int,default=100)
    args=p.parse_args()
    if args.episodes < 1 or args.evaluation_runs < 1:
        p.error('episodes and evaluation-runs must be positive')
    env=read_world();began=time.perf_counter()
    q,history,examples=train(env,args.episodes,args.seed)
    seconds=time.perf_counter()-began
    root=ROOT/'reports/q_learning';root.mkdir(parents=True,exist_ok=True)
    with (root/'training.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    evaluations=[rollout(env,q,seed=10000+i,jitter=.015) for i in range(args.evaluation_runs)]
    greedy=rollout(env,q)
    baseline=rollout(env,[[0.]*4 for _ in range(env.state_count)])
    last=history[-100:]
    summary=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,episodes=args.episodes,
                 max_steps=600,alpha=.2,gamma=.99,epsilon_min=.03,epsilon_decay=.999,
                 continuous_landing_jitter_m=.015,training_seconds=seconds,
                 first_success_episode=next((r['episode'] for r in history if r['success']),None),
                 final_100_success_rate=sum(r['success'] for r in last)/len(last),
                 final_100_mean_reward=sum(r['reward'] for r in last)/len(last),
                 final_100_mean_steps=sum(r['steps'] for r in last)/len(last),
                 evaluation_runs=len(evaluations),evaluation_successes=sum(r['success'] for r in evaluations),
                 evaluation_collisions=sum(r['collisions'] for r in evaluations),
                 evaluation_mean_steps=sum(r['steps'] for r in evaluations)/len(evaluations),
                 greedy=greedy,untrained_greedy={k:v for k,v in baseline.items() if k!='path'},
                 greedy_length_m=path_length(greedy['path']),
                 greedy_min_waypoint_clearance_m=min(physical_clearance(p,env.walls,env.bounds) for p in greedy['path']),
                 update_examples=examples,
                 scope='Kinematic continuous XY option model with exact swept-disc geometry; not Webots physics training. Known map, fixed maze, exploring starts in first 80% episodes; last 20% mission-start only.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'evaluations.json').write_text(json.dumps(evaluations,indent=2)+'\n')
    save_policy(ROOT/'models/q_learning.json',env,q,{k:v for k,v in summary.items() if k not in ('greedy','update_examples')})
    print(json.dumps({k:v for k,v in summary.items() if k not in ('greedy','update_examples')},indent=2),flush=True)
    if not greedy['success'] or summary['evaluation_successes'] != len(evaluations) or summary['evaluation_collisions']:
        raise SystemExit('Policy validation failed; see summary.json')

if __name__=='__main__':main()
