"""Frozen Double DQN rollouts and route timing on the original Micromouse maze."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'controllers/micromouse'))

from continuous_control import NavigationMap
from lidar_dqn_agent import load_inference
from lidar_dqn_env import DQNMission
from train_q_learning import read_world
from train_lidar_q import remove_backtracking


def maze_mission(timeout):
    maze = read_world()
    nav = NavigationMap(maze.start, maze.goal, maze.walls, maze.bounds, maze.cell_size, guided=False)
    return DQNMission(nav, timeout)


@torch.inference_mode()
def rollout(mission, model, *, heading, seed, route=(), trace=None,recorder=None,
            episode=1,phase='evaluation'):
    state, _ = mission.reset(seed=seed, options=dict(heading=heading, search_seed=seed, route=route))
    if recorder:
        recorder.begin(episode,phase,0.,phase)
    count = 0
    while True:
        mask = mission.action_mask(state)
        q = model(torch.from_numpy(state)).numpy()
        if not np.isfinite(q).all():
            raise FloatingPointError('Nonfinite Q during evaluation.')
        decision = int(np.where(mask, q, -np.inf).argmax())
        if recorder:
            motion=mission.motion
            row=recorder.decision(motion,mission.target,decision,np.flatnonzero(mask),0.,
                observation=state,q=q,scanning=mission.target is None,recovering=motion.collision_recovery,
                exploration_or_exploitation='exploitation' if mask.sum()>1 else 'forced_or_single_valid')
        if trace is not None and mission.target is not None and not mission.motion.collision_recovery:
            trace.append(dict(seed=seed, step=count, observation=state.tolist(),
                              valid=mask.tolist(), q=q.tolist(), action=decision))
            if len(trace) >= 120:
                trace = None
        state, _, terminal, truncated, info = mission.step(decision)
        if recorder and mission.last_transition is not None:
            recorder.step(row,motion,dict(mission.last_transition,collision_count=info['collision_count']),
                terminal=terminal,truncated=truncated,exhausted=info['search_exhausted'],
                episode_end=terminal or truncated)
        count += 1
        if terminal or truncated:
            if recorder:
                recorder.end(motion,info)
            return dict(seed=seed, heading_rad=heading, timeout=truncated, **info)


def metrics(rows):
    wins = [r['duration_s'] for r in rows if r['success']]
    return dict(runs=len(rows), successes=len(wins), failures=len(rows)-len(wins),
                success_rate=len(wins)/len(rows),
                collision_episodes=sum(r['collision_count'] > 0 for r in rows),
                collision_events=sum(r['collision_count'] for r in rows),
                timeouts=sum(r['timeout'] for r in rows), search_exhausted=sum(r['search_exhausted'] for r in rows),
                average_reward=statistics.mean(r['reward'] for r in rows),
                median_reward=statistics.median(r['reward'] for r in rows),
                average_steps=statistics.mean(r['steps'] for r in rows),
                median_steps=statistics.median(r['steps'] for r in rows),
                average_duration_s=statistics.mean(r['duration_s'] for r in rows),
                best_success_s=min(wins, default=None),
                mean_distance_m=statistics.mean(r['distance_m'] for r in rows))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def evaluate(checkpoint, data, candidates, config, recorder=None):
    model, saved = load_inference(checkpoint)
    sha = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    mission = maze_mission(config['timeout'])
    samples = []
    rows = []
    for index in range(config['evaluation_runs']):
        heading = ((index % 8) - 3.5) * math.pi / 4
        r = rollout(mission, model, seed=index, heading=heading,
                    trace=samples if index == 0 else None,recorder=recorder,episode=index+1)
        rows.append(r)
        print(f"Evaluasi {index+1}: goal={r['success']} | waktu={r['duration_s']:.2f}s", flush=True)
    write_json(data / 'evaluations.json', rows)
    columns = [key for key in rows[0] if key not in ('route','replay')]
    with (data / 'evaluation_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    if samples:
        with (data / 'action_values.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['seed','step','observation','mask',*[f'q{i}' for i in range(6)],'action'])
            for row in samples:
                writer.writerow([row['seed'],row['step'],json.dumps(row['observation']),
                                 json.dumps(row['valid']),*row['q'],row['action']])
    if not candidates:
        # A successful frozen exploration can discover usable routes even
        # when exploratory training did not complete a mission.
        found = {}
        for row in rows:
            route = remove_backtracking(row['route'])
            if route and route not in found:
                found[route] = dict(route_id=len(found)+1,targets=route,
                                    discovered_seed=row['seed'],source='frozen_evaluation')
        candidates = list(found.values())
    trials = []
    for candidate in candidates:
        for trial in range(config['speed_trials']):
            # Conditions match between routes. The three starts share the same
            # position with small, explicitly recorded heading variation.
            heading = (0., -.10, .10)[trial % 3]
            result = rollout(mission, model, seed=trial, heading=heading, route=candidate['targets'],
                             recorder=recorder,episode=len(trials)+1,phase='route_speed_test')
            trials.append(dict(route_id=candidate['route_id'], trial=trial+1, **result))
        print(f"Uji rute {candidate['route_id']} selesai.", flush=True)
    wins = [r for r in trials if r['success']]
    best = min(wins, key=lambda r:r['duration_s'], default=None)
    write_json(data / 'routes.json', dict(candidates=candidates, trials=trials, best=best))
    return dict(checkpoint=str(checkpoint), sha256=sha, episode=saved['progress']['episode'],
                epsilon=0, online_frozen=True, evaluation=metrics(rows), route_tests=len(trials),
                route_successes=len(wins), fastest_route=best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--routes', type=Path, help='Optional routes.json with recorded DFS candidates')
    parser.add_argument('--runs', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    _, saved = load_inference(args.checkpoint)
    config = saved['config'].copy()
    if args.runs is not None:
        if args.runs < 1:
            parser.error('--runs must be positive')
        config['evaluation_runs'] = args.runs
    candidates = json.loads(args.routes.read_text())['candidates'] if args.routes else saved['progress']['routes']
    args.output.mkdir(parents=True, exist_ok=False)
    info = evaluate(args.checkpoint, args.output, candidates, config)
    write_json(args.output / 'summary.json', info)


if __name__ == '__main__':
    main()
