"""Train a hidden-goal, sensor-based tabular Q-learning micromouse."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'controllers/micromouse'))
sys.path.insert(0, str(ROOT / 'tools'))
from planner import Obstacle
from tabular_q_learning import (ACTION_NAMES, HEADING_NAMES, SENSOR_BINS,
                                SENSOR_NAMES, SensorMaze, save_policy, state_key,
                                train)
from q_learning_monitor import QLearningMonitor


def read_world():
    """Return the sensor simulator; the goal is only wired to its color sensor."""
    text = (ROOT / 'worlds/world.wbt').read_text()
    layout = json.loads((ROOT / 'worlds/maze_layout.json').read_text())
    walls = []
    for block in re.findall(r'\bMazeWall\s*\{([^{}]+)\}', text):
        pos = list(map(float, re.search(r'translation\s+([^\n]+)', block)[1].split()))
        size = list(map(float, re.search(r'size\s+([^\n]+)', block)[1].split()))
        name = re.search(r'name\s+"([^"]+)"', block)[1]
        walls.append(Obstacle(name, 'wall', tuple(pos[:2]), tuple(size[:2])))
    if not walls:
        raise ValueError('No maze walls found in worlds/world.wbt')
    arena = re.search(r'DEF ARENA NavigationArena\s*\{\s*floorSize\s+(\S+)\s+(\S+)\s+cellSize\s+(\S+)', text)
    start_match = re.search(r'DEF MICROMOUSE Micromouse\s*\{\s*translation\s+(\S+)\s+(\S+)', text)
    if not arena or not start_match:
        raise ValueError('The Webots arena or robot start pose was not found')
    width_m, height_m, cell_size = map(float, arena.groups())
    start = tuple(map(float, start_match.groups()))
    bounds = (-width_m/2, width_m/2, -height_m/2, height_m/2)
    expected_grid = [round(width_m/cell_size), round(height_m/cell_size)]
    if layout['grid_size'] != expected_grid:
        raise ValueError('Regenerate worlds/world.wbt after changing maze_layout.json')
    return SensorMaze(start, walls, bounds, cell_size, layout['goal_cells'])


def write_state_space(path, env, q_table, visits):
    fields = ['state_key', 'cell_id', 'cell_x', 'cell_y', 'heading',
              'ir_front', 'ir_left', 'ir_back', 'ir_right', 'yellow_detected',
              'visits', *[f'q_{name.lower()}' for name in ACTION_NAMES]]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for key, values in sorted(q_table.items(), key=lambda item: tuple(map(int, item[0].split(',')))):
            state = tuple(map(int, key.split(',')))
            cell, heading, *features = state
            x, y = cell % env.width, cell // env.width
            writer.writerow(dict(state_key=key, cell_id=cell, cell_x=x, cell_y=y,
                                 heading=HEADING_NAMES[heading],
                                 **{f'ir_{name}': SENSOR_BINS[features[index]]
                                    for index, name in enumerate(SENSOR_NAMES)},
                                 yellow_detected=bool(features[4]), visits=visits.get(key, 0),
                                 **{f'q_{name.lower()}': f'{value:.8f}'
                                    for name, value in zip(ACTION_NAMES, values)}))


def write_episode_history(path, history):
    fields = list(history[0]) if history else ['episode', 'phase', 'epsilon', 'reward',
                                               'steps', 'success', 'collisions',
                                               'unique_states', 'q_updates', 'timeout']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=6000)
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Time limit per episode; wall collisions do not end an episode')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=.2)
    parser.add_argument('--gamma', type=float, default=.95)
    parser.add_argument('--exploration-fraction', type=float, default=.8)
    parser.add_argument('--epsilon-floor', type=float, default=.10)
    parser.add_argument('--exploitation-epsilon', type=float, default=.02)
    parser.add_argument('--render-speed', type=int, default=64,
                        help='Refresh the live view every N decisions; larger values run faster')
    render = parser.add_mutually_exclusive_group()
    render.add_argument('--render', dest='render', action='store_true')
    render.add_argument('--no-render', dest='render', action='store_false')
    parser.set_defaults(render=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--model', type=Path, default=ROOT/'models/tabular_q_learning.json')
    args = parser.parse_args()
    if args.episodes < 2 or args.max_steps < 1 or args.render_speed < 1:
        parser.error('episodes must be at least 2; max-steps and render-speed must be positive')
    if not 0 < args.exploration_fraction < 1:
        parser.error('exploration-fraction must be between zero and one')
    if not 0 <= args.epsilon_floor <= 1 or not 0 <= args.exploitation_epsilon <= 1:
        parser.error('epsilon values must be between zero and one')
    if not 0 < args.alpha <= 1 or not 0 <= args.gamma <= 1:
        parser.error('alpha must be in (0,1] and gamma in [0,1]')

    env = read_world()
    run_dir = args.output_dir or ROOT/'reports/tabular_q_learning/runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir.mkdir(parents=True, exist_ok=True)
    show = args.render and (sys.platform != 'linux' or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))
    monitor = QLearningMonitor(env, args.render_speed, show=show) if args.render else None
    began = time.perf_counter()

    def progress(row, _q_table, _visits, _history):
        if not show and row['episode'] % 100 == 0:
            print(f"Episode {row['episode']}/{args.episodes}: phase={row['phase']}, reward={row['reward']:.1f}, steps={row['steps']}, success={row['success']}, collisions={row['collisions']}, states={row['unique_states']}", flush=True)

    try:
        q_table, history, visits = train(
            env, episodes=args.episodes, seed=args.seed, max_steps=args.max_steps,
            exploration_fraction=args.exploration_fraction, alpha=args.alpha,
            gamma=args.gamma, epsilon_floor=args.epsilon_floor,
            exploitation_epsilon=args.exploitation_epsilon,
            viewer=monitor, on_episode=progress)
    finally:
        if monitor:
            monitor.close()
    seconds = time.perf_counter() - began
    stamp = datetime.now(timezone.utc).isoformat()
    exploration_episodes = max(1, min(args.episodes-1, round(args.episodes*args.exploration_fraction)))
    write_episode_history(run_dir/'training.csv', history)
    write_state_space(run_dir/'state_space.csv', env, q_table, visits)
    metadata = dict(created_at_utc=stamp, seed=args.seed, episodes_completed=len(history),
                    requested_episodes=args.episodes, max_steps=args.max_steps,
                    alpha=args.alpha, gamma=args.gamma,
                    exploration_fraction=args.exploration_fraction,
                    exploration_episodes=exploration_episodes,
                    exploitation_episodes=args.episodes-exploration_episodes,
                    epsilon_floor=args.epsilon_floor,
                    exploitation_epsilon=args.exploitation_epsilon,
                    state_space_schema='cell_id,heading,IR(front,left,back,right),yellow_detected',
                    state_space_bound=env.max_state_count, visited_state_count=len(q_table),
                    training_seconds=round(seconds, 3),
                    success_count=sum(row['success'] for row in history),
                    first_success_episode=next((row['episode'] for row in history if row['success']), None),
                    final_100_success_rate=round(sum(row['success'] for row in history[-100:]) / max(1, len(history[-100:])), 4),
                    final_100_mean_reward=round(sum(row['reward'] for row in history[-100:]) / max(1, len(history[-100:])), 4),
                    final_100_mean_steps=round(sum(row['steps'] for row in history[-100:]) / max(1, len(history[-100:])), 4),
                    total_collisions=sum(row['collisions'] for row in history),
                    total_q_updates=sum(row['q_updates'] for row in history),
                    goal_knowledge='yellow color sensor only; no goal coordinate or distance reward',
                    reward_rule='-1 per safe move, -10 on collision (episode continues), +100 on yellow detection',
                    report_dir=str(run_dir.relative_to(ROOT)), model=str(args.model.relative_to(ROOT) if args.model.is_relative_to(ROOT) else args.model))
    save_policy(args.model, env, q_table, metadata)
    (run_dir/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if monitor:
        monitor.save(run_dir/'monitor.png')
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Training data: {run_dir}", flush=True)
    print(f"Q-table: {args.model}", flush=True)


if __name__ == '__main__':
    main()
