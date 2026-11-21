"""Train tabular Q motion control on full start-to-goal maze episodes."""
import argparse
import csv
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'controllers/micromouse'))
sys.path.insert(0,str(ROOT/'tools'))

from continuous_control import NavigationMap
from lidar_q_env import (COLLISION_PENALTY,FORWARD_LIDAR_WEIGHT,
                         GPS_TRAVEL_COST_PER_METER,GOAL_REWARD,
                         WAYPOINT_REWARD,
                         LidarQEnvironment)
from lidar_qlearning import (ACTION_NAMES,STATE_SPACE_SIZE,TabularLidarQ,
                             STOP_ACTION,WaypointRecovery)
from q_learning import scene_signature
from maze_search import LidarMazeSearch, synthetic_hits
from train_q_learning import read_world


def epsilon_for_episode(episode,total,exploration_fraction):
    boundary=max(1,int(total*exploration_fraction))
    if episode<boundary:
        return .30-.22*episode/boundary
    progress=(episode-boundary)/max(1,total-boundary)
    return max(.02,.08*(.02/.08)**progress)


def run_episode(env, agent, epsilon, heading, max_steps, *, seed, learn,
                view=None, episode=0, steps_offset=0):
    search = LidarMazeSearch(env.base_start, env.base_goal, env.nav.bounds,
                             env.nav.cell_size, seed=seed)
    env.reset(heading=heading)
    target = search.next_target(env.pose, synthetic_hits(env.lidar_scan))
    exhausted = target is None and search.scan_bearing is None
    state = env.last_state if target is None else env.set_waypoint(target)
    env.waypoint_index = len(search.route)
    total=0.;success=False;updates=0;info={}
    collision_count = 0
    min_front = float(min(env.lidar_sectors[[4,0,1]]))
    recovery=WaypointRecovery()
    allowed = recovery.actions(env.pose,env.goal,env.lidar_sectors,0.)
    for _ in range(0 if exhausted else max_steps):
        scanning = target is None
        recovering = env.collision_recovery
        if recovering:
            action = STOP_ACTION
        elif scanning:
            action = search.scan_action(env.lidar_sectors)
        elif learn:
            action=agent.choose(state,epsilon,allowed)
        else:
            values=agent.values(state)
            action=max(allowed,key=lambda a:float(values[a]))
        next_state,reward,terminated,truncated,info=env.step(action,final_goal=search.final,scanning=scanning)
        if info['collision']:
            collision_count += 1
            recovery.active = True
        min_front=min(min_front,float(info['front_clearance_m']))
        if ((scanning and not recovering and not info['collision']) or info['waypoint_reached']) and not (terminated or truncated):
            target = search.next_target(env.pose, synthetic_hits(env.lidar_scan))
            exhausted = target is None and search.scan_bearing is None
            if target is not None:
                env.waypoint_index = len(search.route)
                next_state = env.set_waypoint(target)
                recovery = WaypointRecovery()
        allowed=recovery.actions(env.pose,env.goal,env.lidar_sectors,
                                 env.steps*env.decision_seconds)
        env.scanning = scanning
        env.recovering_from_collision = recovering
        env.collision_count = collision_count
        env.recovery_active=recovery.active
        env.allowed_actions = (search.scan_action(env.lidar_sectors),) if target is None and not exhausted else allowed
        if learn and not scanning and not recovering:
            agent.update(state,action,reward,next_state,terminated or target is None,
                         allowed)
        state=next_state;total+=reward;updates+=1
        if view and not view.update(env,agent,episode,epsilon,steps_offset+updates):
            raise KeyboardInterrupt
        if terminated:
            success=bool(info['success'])
            break
        if truncated or exhausted:
            break
    result=dict(reward=total,steps=updates,success=success,collision=collision_count>0,
                collision_count=collision_count,
                start=env.base_start.tolist(),target=env.base_goal.tolist(),
                distance_to_goal_m=math.dist(env.pose[:2],env.base_goal),
                distance_from_base_m=float(info.get('distance_from_base_m',
                    math.hypot(env.pose[0]-env.base_start[0],
                               env.pose[1]-env.base_start[1]))),
                travelled_distance_m=env.travelled_distance,
                min_front_clearance_m=min_front,
                waypoints=len(search.route), search_exhausted=exhausted,
                cells_visited=len(search.edges), duration_s=updates*env.decision_seconds)
    return result,updates


def evaluate(env,agent,runs=40,max_steps=120):
    results=[]
    for index in range(runs):
        heading=((index%8)-3.5)*math.pi/4
        result,_=run_episode(env,agent,0.,heading,max_steps,seed=index,learn=False)
        results.append(dict(seed=index,**result))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,default=3000)
    parser.add_argument('--max-steps',type=int,
                        help='Optional decision cap; by default the timeout sets the episode length')
    parser.add_argument('--timeout',type=float,default=300.)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--alpha',type=float,default=.16)
    parser.add_argument('--gamma',type=float,default=.98)
    parser.add_argument('--exploration-fraction',type=float,default=.60)
    parser.add_argument('--render',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--render-speed',type=float,default=16.)
    parser.add_argument('--evaluation-runs',type=int,default=40)
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if (args.episodes<2 or (args.max_steps is not None and args.max_steps<1)
            or not math.isfinite(args.timeout) or args.timeout<=0 or args.evaluation_runs<1):
        parser.error('episodes >=2; max-steps, timeout and evaluation-runs must be positive')
    if not 0<args.exploration_fraction<1 or not 0<args.alpha<=1 or not 0<=args.gamma<=1:
        parser.error('exploration-fraction, alpha or gamma is outside its valid range')
    if not math.isfinite(args.render_speed) or args.render_speed<=0:
        parser.error('render-speed must be positive and finite')
    if args.render and sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('Tidak ada display grafis. Gunakan --no-render.')

    maze=read_world()
    nav=NavigationMap(maze.start,maze.goal,maze.walls,maze.bounds,maze.cell_size,guided=False)
    env=LidarQEnvironment(nav,timeout_seconds=args.timeout)
    if args.max_steps is not None:
        env.max_steps=min(env.max_steps,args.max_steps)
    agent=TabularLidarQ(args.alpha,args.gamma,args.seed)
    rng=random.Random(args.seed)
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
                  lidar='front-facing 180-degree LiDAR; 128 returns aggregated to five sectors',
                  state_schema='relative bearing to next DFS waypoint 8 bins, waypoint distance 4 bins, five forward LiDAR sectors 3 bins',
                  state_space_upper_bound=STATE_SPACE_SIZE,
                  action_names=ACTION_NAMES,
                  role='tabular Q-learning controls continuous differential-drive motion for an online LiDAR search from maze start to goal',
                  training_tasks='start-to-goal search with randomized initial heading; episodes end at goal, timeout, or exhausted search',
                  reward='4 x GPS next-waypoint progress + 0.20 x heading error reduction - time, travel, clearance and idle costs; +4 waypoint; +120 goal; -100 collision',
                  forward_lidar_reward_weight=FORWARD_LIDAR_WEIGHT,
                  gps_travel_cost_per_meter=GPS_TRAVEL_COST_PER_METER,
                  lidar_safety_mask='0.16 m directional clearance for translation; footprint clearance for pivots',
                  recovery='after 2 seconds without a new 2 cm waypoint progress, guides allowed Q actions until the next waypoint',
                  collision_recovery='contact incurs the penalty and triggers a brief supervised retreat within the same episode',
                  collision_reward=COLLISION_PENALTY,goal_reward=GOAL_REWARD,
                  waypoint_reward=WAYPOINT_REWARD,maze_signature=signature,
                  search_algorithm='online_lidar_dfs')
    (run_dir/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    view=None
    if args.render:
        from lidar_q_view import LidarQView
        view=LidarQView(nav,args.render_speed)

    began=time.perf_counter();success_count=collision_count=0;first_success=None;total_steps=0
    episodes_completed=0
    fieldnames=('episode','phase','epsilon','reward','steps','success','collision','collision_count',
                'goal_distance_m','route_waypoints','cells_visited','duration_s','search_exhausted',
                'distance_from_base_m','travelled_distance_m','min_front_clearance_m',
                'q_states','updates')
    try:
        with history_path.open('w',newline='') as csvfile:
            writer=csv.DictWriter(csvfile,fieldnames=fieldnames,lineterminator='\n')
            writer.writeheader()
            for episode in range(args.episodes):
                epsilon=epsilon_for_episode(episode,args.episodes,args.exploration_fraction)
                phase='exploration' if episode<args.episodes*args.exploration_fraction else 'exploitation'
                heading=rng.uniform(-math.pi,math.pi)
                final_info,updates=run_episode(env,agent,epsilon,heading,
                                               env.max_steps,seed=episode,learn=True,view=view,
                                               episode=episode+1,steps_offset=total_steps)
                total_steps+=updates
                success=final_info['success'];collision=final_info['collision']
                success_count+=success;collision_count+=collision
                if success and first_success is None:first_success=episode+1
                writer.writerow(dict(episode=episode+1,phase=phase,epsilon=epsilon,
                    reward=final_info['reward'],steps=final_info['steps'],
                    success=success,collision=collision,
                    collision_count=final_info['collision_count'],
                    goal_distance_m=final_info['distance_to_goal_m'],
                    route_waypoints=final_info['waypoints'],cells_visited=final_info['cells_visited'],
                    duration_s=final_info['duration_s'],search_exhausted=final_info['search_exhausted'],
                    distance_from_base_m=final_info['distance_from_base_m'],
                    travelled_distance_m=final_info['travelled_distance_m'],
                    min_front_clearance_m=final_info['min_front_clearance_m'],
                    q_states=len(agent.q),updates=updates))
                csvfile.flush();episodes_completed=episode+1
                if view:
                    view.record_episode(final_info['reward'],success,collision)
                if (episode+1)%10==0 or episode+1==args.episodes:
                    agent.save(table_path,{**manifest,'episodes_completed':episodes_completed,
                        'elapsed_training_s':time.perf_counter()-began})
                    print(f"episode {episode+1}/{args.episodes} | {phase} | eps={epsilon:.3f} | reward={final_info['reward']:.2f} | mission_success={success} collision={collision} | steps={updates} | q_states={len(agent.q):,}",flush=True)
    except KeyboardInterrupt:
        print('Training dihentikan; menyimpan Q-table sementara.',flush=True)
    finally:
        agent.save(table_path,{**manifest,'episodes_completed':episodes_completed,
                               'elapsed_training_s':time.perf_counter()-began})
        if view:view.close()

    eval_results=evaluate(env,agent,args.evaluation_runs,env.max_steps)
    summary=dict(**manifest,episodes_completed=episodes_completed,
                 first_success_episode=first_success,training_successes=success_count,
                 training_collisions=collision_count,q_states=len(agent.q),
                 training_seconds=time.perf_counter()-began,
                 evaluation_runs=len(eval_results),
                 evaluation_successes=sum(x['success'] for x in eval_results),
                 evaluation_collisions=sum(x['collision'] for x in eval_results),
                 evaluation_mean_steps=sum(x['steps'] for x in eval_results)/len(eval_results),
                 evaluation_mean_travelled_m=sum(x['travelled_distance_m'] for x in eval_results)/len(eval_results),
                 evaluation_mean_goal_distance_m=sum(x['distance_to_goal_m'] for x in eval_results)/len(eval_results),
                 policy_path=str(table_path.resolve()))
    (run_dir/'evaluations.json').write_text(json.dumps(eval_results,indent=2)+'\n')
    (run_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Run: {run_dir.resolve()}')
    print(f'Simulation: bash run.sh sim --algorithm q_learning --policy {table_path.resolve()}')


if __name__=='__main__':main()
