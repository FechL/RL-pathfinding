"""Train sparse tabular Q-learning from planar lidar sectors and known goal."""
import argparse
import csv
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'controllers/micromouse'))
sys.path.insert(0,str(ROOT/'tools'))

from continuous_control import NavigationMap
from lidar_q_env import LidarQEnvironment
from lidar_qlearning import ACTION_NAMES,TabularLidarQ,safe_actions
from q_learning import scene_signature
from train_q_learning import read_world


def epsilon_for_episode(episode,total,exploration_fraction):
    boundary=max(1,int(total*exploration_fraction))
    if episode<boundary:
        return max(.45,1.-.55*episode/boundary)
    progress=(episode-boundary)/max(1,total-boundary)
    return max(.04,.45*(.04/.45)**progress)


def evaluate(env,agent,runs=12):
    results=[]
    for index in range(runs):
        state=env.reset()
        total=0.;collision=False;success=False
        for _ in range(env.max_steps):
            values=agent.values(state)
            action=max(safe_actions(env.lidar_sectors),key=lambda candidate:float(values[candidate]))
            next_state,reward,terminated,truncated,info=env.step(action)
            total+=reward;state=next_state
            if terminated:
                collision=info['collision'];success=info['success'];break
            if truncated:break
        results.append(dict(seed=index,reward=total,steps=env.steps,
                            success=success,collision=collision,
                            distance_to_goal_m=float(info.get('distance_to_goal_m',
                                math.hypot(env.nav.goal[0]-env.pose[0],env.nav.goal[1]-env.pose[1])))))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,default=1500)
    parser.add_argument('--max-steps',type=int,default=750)
    parser.add_argument('--timeout',type=float,default=120.)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--alpha',type=float,default=.16)
    parser.add_argument('--gamma',type=float,default=.995)
    parser.add_argument('--exploration-fraction',type=float,default=.60)
    parser.add_argument('--render',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--render-speed',type=float,default=8.)
    parser.add_argument('--evaluation-runs',type=int,default=12)
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.episodes<2 or args.max_steps<1 or args.timeout<=0 or args.evaluation_runs<1:
        parser.error('episodes >=2, max-steps, timeout and evaluation-runs must be positive')
    if not 0<args.exploration_fraction<1 or not 0<args.alpha<=1 or not 0<=args.gamma<=1:
        parser.error('exploration-fraction, alpha or gamma is outside its valid range')
    if not math.isfinite(args.render_speed) or args.render_speed<=0:
        parser.error('render-speed must be positive and finite')
    if args.render and sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('Tidak ada display grafis. Gunakan --no-render.')

    maze=read_world()
    nav=NavigationMap(maze.start,maze.goal,maze.walls,maze.bounds,maze.cell_size,guided=False)
    env=LidarQEnvironment(nav,timeout_seconds=args.timeout)
    env.max_steps=min(env.max_steps,args.max_steps)
    agent=TabularLidarQ(args.alpha,args.gamma,args.seed)
    created=datetime.now(timezone.utc)
    run_dir=args.run_dir or ROOT/'reports/lidar_q_learning/runs'/created.strftime('%Y%m%dT%H%M%S.%fZ')
    run_dir=run_dir if run_dir.is_absolute() else ROOT/run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error('Run directory must be empty so the previous Q-table and logs remain intact.')
    run_dir.mkdir(parents=True,exist_ok=True)
    table_path=run_dir/'q_table.json'
    history_path=run_dir/'training.csv'
    signature=scene_signature(maze)
    manifest=dict(created_at_utc=created.isoformat(),seed=args.seed,episodes=args.episodes,
                  max_steps=env.max_steps,step_duration_s=env.decision_seconds,
                  alpha=args.alpha,gamma=args.gamma,exploration_fraction=args.exploration_fraction,
                  lidar='Webots planar 360 degree LiDAR; 128 raw returns aggregated to 8 sector minima',
                  lidar_sector_order=['front','front-left','left','rear-left','rear','rear-right','right','front-right'],
                  state_schema='relative goal bearing 16 bins; goal distance 7 bins; eight lidar ranges each 3 bins; previous action',
                  action_names=ACTION_NAMES,goal_input='known goal coordinates and relative bearing/distance',
                  reward='8 x goal-distance progress - 0.025 per decision - near-wall penalty - low-progress spin penalty; +60 goal; -30 collision',
                  lidar_safety_mask='blocks only actions whose forward, diagonal, or rear sector is under 0.12 m; stop is a fallback only when no movement is safe',
                  collision_reward=-30.,goal_reward=60.,maze_signature=signature)
    (run_dir/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    view=None
    if args.render:
        from lidar_q_view import LidarQView
        view=LidarQView(nav,args.render_speed)

    began=time.perf_counter();success_count=collision_count=0;first_success=None;total_steps=0
    episodes_completed=0
    fieldnames=('episode','phase','epsilon','reward','steps','success','collision',
                'goal_distance_m','min_front_clearance_m','q_states','updates')
    try:
        with history_path.open('w',newline='') as csvfile:
            writer=csv.DictWriter(csvfile,fieldnames=fieldnames,lineterminator='\n')
            writer.writeheader()
            for episode in range(args.episodes):
                epsilon=epsilon_for_episode(episode,args.episodes,args.exploration_fraction)
                phase='exploration' if episode<args.episodes*args.exploration_fraction else 'exploitation'
                state=env.reset()
                updates=0
                terminal=truncated=False
                final_info={}
                for _ in range(env.max_steps):
                    action=agent.choose(state,epsilon,safe_actions(env.lidar_sectors))
                    next_state,reward,terminal,truncated,final_info=env.step(action)
                    agent.update(state,action,reward,next_state,terminal,
                                 safe_actions(env.lidar_sectors))
                    state=next_state;updates+=1;total_steps+=1
                    if view and not view.update(env,agent,episode+1,epsilon,
                                                total_steps):
                        raise KeyboardInterrupt
                    if terminal or truncated:break
                success=bool(final_info.get('success',False))
                collision=bool(final_info.get('collision',False))
                success_count+=success;collision_count+=collision
                if success and first_success is None:first_success=episode+1
                writer.writerow(dict(episode=episode+1,phase=phase,epsilon=epsilon,
                    reward=env.episode_return,steps=env.steps,success=success,collision=collision,
                    goal_distance_m=final_info.get('distance_to_goal_m',float('nan')),
                    min_front_clearance_m=final_info.get('front_clearance_m',float('nan')),
                    q_states=len(agent.q),updates=updates))
                csvfile.flush()
                episodes_completed=episode+1
                if view:view.record_episode(env.episode_return,success,collision)
                if (episode+1)%10==0 or episode+1==args.episodes:
                    agent.save(table_path,{**manifest,'episodes_completed':episodes_completed,
                        'elapsed_training_s':time.perf_counter()-began,
                        'success_rate_last_100':sum(view.successes)/max(1,len(view.successes)) if view else None})
                    print(f"episode {episode+1}/{args.episodes} | {phase} | eps={epsilon:.3f} | reward={env.episode_return:.2f} | success={success} collision={collision} | q_states={len(agent.q):,}",flush=True)
    except KeyboardInterrupt:
        print('Training dihentikan; menyimpan Q-table sementara.',flush=True)
    finally:
        agent.save(table_path,{**manifest,'episodes_completed':episodes_completed,
                               'elapsed_training_s':time.perf_counter()-began})
        if view:view.close()

    eval_results=evaluate(env,agent,args.evaluation_runs)
    summary=dict(**manifest,episodes_completed=episodes_completed,
                 first_success_episode=first_success,
                 training_successes=success_count,training_collisions=collision_count,
                 q_states=len(agent.q),training_seconds=time.perf_counter()-began,
                 evaluation_runs=len(eval_results),evaluation_successes=sum(x['success'] for x in eval_results),
                 evaluation_collisions=sum(x['collision'] for x in eval_results),
                 evaluation_mean_steps=sum(x['steps'] for x in eval_results)/len(eval_results),
                 policy_path=str(table_path.resolve()))
    (run_dir/'evaluations.json').write_text(json.dumps(eval_results,indent=2)+'\n')
    (run_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Run: {run_dir.resolve()}')
    print(f'Simulation: bash run.sh sim --algorithm q_learning --policy {table_path.relative_to(ROOT)}')


if __name__=='__main__':main()
