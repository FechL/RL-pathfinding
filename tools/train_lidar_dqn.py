"""Train Double DQN in the original continuous Micromouse environment."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

from evaluate_lidar_dqn import ROOT, evaluate, maze_mission, write_json
from lidar_dqn import schema
from lidar_dqn_agent import DQNAgent, VERSION, atomic_save, read_checkpoint
from train_lidar_q import remove_backtracking

BASE = '5c4cf33c2e045245be1cf49340a5a5e7eb27f5fa'
FIELDS = (
    'episode', 'seed', 'heading_rad', 'reward', 'steps', 'duration_s', 'distance_m',
    'success', 'collision', 'collision_count', 'timeout', 'search_exhausted',
    'epsilon', 'td_loss', 'mean_q', 'max_q', 'replay_size', 'gradient_updates',
    'target_updates', 'route_id', 'global_step', 'training_elapsed_s',
)


def epsilon_at(step, config, step_budget):
    fraction = min(1., step / max(1., config['epsilon_decay_fraction'] * step_budget))
    return config['epsilon_start'] + fraction * (config['epsilon_min'] - config['epsilon_start'])


def train_episode(mission, agent, episode, global_step, step_budget, recorder=None):
    config = agent.config
    seed = episode - 1
    heading = random.Random(config['seed'] + seed).uniform(-math.pi, math.pi)
    state, _ = mission.reset(seed=config['seed'] + seed, options=dict(heading=heading, search_seed=seed))
    mask = mission.action_mask(state)
    if recorder:
        recorder.begin(episode,'exploration',epsilon_at(global_step,config,step_budget))
    update_stats = []
    while True:
        epsilon = epsilon_at(global_step, config, step_budget)
        q = agent.values(state) if recorder else None
        action = agent.choose(state, mask, epsilon, q=q)
        if recorder:
            motion=mission.motion
            row=recorder.decision(motion,mission.target,action,np.flatnonzero(mask),epsilon,
                 observation=state,q=q,scanning=mission.target is None,recovering=motion.collision_recovery,
                 exploration_or_exploitation=agent.choice_mode,gamma=config['gamma'])
        next_state, reward, terminal, truncated, info = mission.step(action)
        next_mask = mission.action_mask(next_state)
        agent.replay.push(state, action, reward, next_state, terminal, truncated, next_mask)
        global_step += 1
        details = {} if recorder else None
        if global_step % config['update_every'] == 0:
            result = agent.update(details)
            if result is not None:
                update_stats.append(result)
        if recorder and mission.last_transition is not None:
            row.update(replay_buffer_size=agent.replay.size,gradient_update_count=agent.updates,
                       target_update_count=agent.target_updates,learning_rate=config['learning_rate'])
            recorder.step(row,motion,dict(mission.last_transition,collision_count=info['collision_count']),
                terminal=terminal,truncated=truncated,exhausted=info['search_exhausted'],
                episode_end=terminal or truncated,learning=details)
        state, mask = next_state, next_mask
        if terminal or truncated:
            break
    values = np.asarray(update_stats)
    if recorder:
        recorder.end(mission.motion,info,replay_buffer_size=agent.replay.size,gradient_updates=agent.updates)
    row = dict(episode=episode, heading_rad=heading, seed=seed, timeout=truncated,
               collision=info['collision_count'] > 0, epsilon=epsilon, global_step=global_step,
               replay_size=agent.replay.size, gradient_updates=agent.updates, target_updates=agent.target_updates,
               td_loss=float(values[:,0].mean()) if len(values) else None,
               mean_q=float(values[:,1].mean()) if len(values) else None,
               max_q=float(values[:,2].max()) if len(values) else None,
               reward=float(info['reward']),
               **{key:info[key] for key in ('steps','duration_s','distance_m','success','collision_count','search_exhausted')})
    return row, remove_backtracking(info['route'])


def validate_config(config):
    positive = ('episodes','evaluation_runs','capacity','batch_size','warmup','target_interval',
                'update_every','checkpoint_every','speed_trials')
    if any(type(config[key]) is not int or config[key] < 1 for key in positive):
        raise ValueError('Training counts must be positive integers.')
    if config['capacity'] < max(config['warmup'],config['batch_size']):
        raise ValueError('Replay capacity must cover a warm-up and batch.')
    if not 0 <= config['epsilon_min'] <= config['epsilon_start'] <= 1 or not 0 < config['epsilon_decay_fraction'] <= 1:
        raise ValueError('Invalid epsilon schedule.')
    if not 0 <= config['gamma'] <= 1 or any(not math.isfinite(config[k]) or config[k]<=0 for k in
                                            ('timeout','learning_rate','gradient_clip')):
        raise ValueError('Invalid optimization or timeout configuration.')
    if config['device'] != 'cpu' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable. Use device cpu.')


def run(config, output, *, resume=None, stop_after=None, record=None):
    validate_config(config)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data, models = output/'data', output/'models'
    if resume is None and any(folder.exists() for folder in (data, models)):
        raise ValueError('Use a fresh output directory or an explicit --resume checkpoint.')
    for folder in (data, models):
        folder.mkdir(parents=True, exist_ok=True)
    mission = maze_mission(config['timeout'])
    recorder = None
    if record:
        from lidar_dqn import FEATURES
        from micromouse_recording import Recording
        import atexit
        recorder=Recording(record,'double_dqn',mission.motion,feature_names=FEATURES)
        atexit.register(recorder.close,'interrupted')
    step_budget = config['episodes'] * mission.motion.max_steps
    agent = DQNAgent(config)
    manifest = dict(
        version=VERSION, algorithm='masked_double_dqn', architecture_base=BASE,
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, pytorch=str(torch.__version__), cuda_available=torch.cuda.is_available(),
        device=config['device'], machine=platform.machine(), cpu=platform.processor(),
        config=config, **schema(), parameters=sum(p.numel() for p in agent.online.parameters()),
        maze=dict(signature=mission.motion.nav.signature, start=mission.motion.base_start.tolist(),
                  goal=mission.motion.base_goal.tolist(), bounds=list(mission.motion.nav.bounds),
                  cell_size=mission.motion.nav.cell_size, distance_scale=mission.distance_scale,
                  step_s=mission.motion.decision_seconds, max_steps=mission.motion.max_steps),
        nominal_step_budget=step_budget, reward_source='LidarQEnvironment.step',
        eval_protocol='epsilon zero, seed i, heading ((i % 8) - 3.5) * pi/4',
        route_protocol='freeze final model; replay discovered candidates with headings 0, -0.10, +0.10 radians',
        scoring_best_checkpoint='successful training episode, then fewest contacts, then shortest duration; final checkpoint is evaluated')
    progress = dict(episode=0, global_step=0, epsilon=config['epsilon_start'], routes=[],
                    history=[], best_score=None, elapsed_s=0.)
    if resume is not None:
        progress = agent.resume(resume)
        if progress.get('maze_signature') != manifest['maze']['signature']:
            raise ValueError('Resume checkpoint uses a different maze.')
        # Fresh-output resume keeps the earlier best checkpoint from the parent run.
        prior_best = resume.parent/'best_model.pt'
        if prior_best.is_file() and (models/'best_model.pt').resolve() != prior_best.resolve():
            atomic_save(read_checkpoint(prior_best),models/'best_model.pt')
    progress['maze_signature'] = manifest['maze']['signature']
    if not (data/'run.json').exists():
        write_json(data/'run.json',manifest)
    clock = time.perf_counter()
    prior_elapsed = progress['elapsed_s']
    end = config['episodes'] if stop_after is None else min(stop_after, config['episodes'])
    with (data/'training.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(progress['history'])
        for episode in range(progress['episode']+1,end+1):
            row, route = train_episode(mission, agent, episode, progress['global_step'], step_budget,recorder)
            route_id = None
            if route:
                targets = [[float(value) for value in point] for point in route]
                found = next((r for r in progress['routes'] if r['targets']==targets),None)
                if found is None:
                    found = dict(route_id=len(progress['routes'])+1, targets=targets, discovered_episode=episode,
                                 exploration_duration_s=row['duration_s'])
                    progress['routes'].append(found)
                route_id = found['route_id']
            row['route_id'] = route_id
            row['training_elapsed_s'] = prior_elapsed+time.perf_counter()-clock
            progress.update(episode=episode,global_step=row['global_step'],epsilon=row['epsilon'],
                            elapsed_s=row['training_elapsed_s'])
            progress['history'].append(row)
            writer.writerow(row); stream.flush()
            score = [int(not row['success']),row['collision_count'],
                     row['duration_s'] if row['success'] else -row['reward']]
            if progress['best_score'] is None or score < progress['best_score']:
                progress['best_score'] = score
                agent.save(models/'best_model.pt',progress)
            # A resume checkpoint is written at episode boundaries. This includes
            # the full replay buffer, RNG, and history, so continuation is exact.
            agent.save(models/'checkpoint_latest.pt',progress,full=True)
            if episode % config['checkpoint_every']==0 or episode==end:
                agent.save(models/'last_model.pt',progress)
            print(f"Episode {episode}/{config['episodes']} | goal={row['success']} | "
                  f"tabrakan={row['collision_count']} | langkah={row['global_step']} | "
                  f"epsilon={row['epsilon']:.3f} | update={agent.updates}",flush=True)
    if progress['episode'] == 0:
        raise ValueError('No episode boundary to save.')
    agent.save(models/'last_model.pt',progress)
    atomic_save(dict(version=VERSION,schema=schema(),target=agent.target.state_dict(),
                     updates=agent.updates,target_updates=agent.target_updates),models/'target_network.pt')
    stats = dict(episodes=progress['episode'],steps=progress['global_step'],routes_found=len(progress['routes']),
                 training_successes=sum(r['success'] for r in progress['history']),
                 training_seconds=progress['elapsed_s'], gradient_updates=agent.updates,target_updates=agent.target_updates)
    report = dict(training=stats)
    if progress['episode']==config['episodes']:
        report.update(evaluate(models/'last_model.pt',data,progress['routes'],config,recorder))
    if recorder:
        recorder.close('captured' if progress['episode']==config['episodes'] and not resume else 'partial')
    write_json(data/'summary.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/lidar_dqn.json')
    parser.add_argument('--output',type=Path,default=ROOT/'output/drl')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-after-episodes',type=int)
    parser.add_argument('--record',type=Path,help='Record every step into a fresh telemetry directory')
    for name,kind in (('episodes',int),('timeout',float),('evaluation-runs',int),('warmup',int),
                      ('seed',int),('update-every',int),('speed-trials',int)):
        parser.add_argument('--'+name,type=kind)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    for key in config:
        value=getattr(args,key,None)
        if value is not None:
            config[key]=value
    print(json.dumps(run(config,args.output,resume=args.resume,stop_after=args.stop_after_episodes,
                         record=args.record),indent=2))


if __name__ == '__main__':
    main()
