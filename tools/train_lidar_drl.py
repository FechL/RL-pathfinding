"""Train a continuous SAC controller on the same LiDAR mission as tabular Q."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'controllers/micromouse'))

from continuous_control import NavigationMap
from lidar_drl import ALGORITHM
from lidar_drl_env import LidarMission
from train_lidar_q import remove_backtracking
from train_q_learning import read_world


def export_actor(model, path):
    layers = [module for module in model.actor.latent_pi if isinstance(module, torch.nn.Linear)]
    layers.append(model.actor.mu)
    artifact = dict(format_version=1, algorithm=ALGORITHM, layers=[
        dict(weight=layer.weight.detach().cpu().tolist(), bias=layer.bias.detach().cpu().tolist())
        for layer in layers])
    path.write_text(json.dumps(artifact) + '\n')


def plots(rows, losses, directory):
    if not rows:
        return
    from matplotlib.figure import Figure
    figure = Figure(figsize=(10, 8), layout='constrained')
    axes = figure.subplots(2, 2).flat
    fields = [('reward', 'Reward per episode'), ('steps', 'Langkah per episode'),
              ('collision_count', 'Jumlah tabrakan'), ('duration_s', 'Durasi simulasi (s)')]
    switch = next((row['episode'] for row in rows if row['phase'] == 'speed_test'), None)
    for axis, (field, label) in zip(axes, fields):
        axis.plot([row['episode'] for row in rows], [row[field] for row in rows])
        if switch is not None:
            axis.axvline(switch, color='orange', linestyle='--', label='Uji kecepatan')
            axis.legend()
        axis.set(xlabel='Episode', ylabel=label)
    figure.savefig(directory / 'training.png', dpi=150)
    if losses:
        figure = Figure(figsize=(9, 4), layout='constrained')
        actor, critic = figure.subplots(1, 2)
        for axis, key, title in ((actor, 'actor_loss', 'Actor loss'),
                                 (critic, 'critic_loss', 'Critic loss')):
            axis.plot([row['step'] for row in losses], [row[key] for row in losses])
            axis.set(xlabel='Langkah training', title=title)
        figure.savefig(directory / 'losses.png', dpi=150)


class TrainingMonitor(BaseCallback):
    def __init__(self, mission, writer, stream, explore_steps, view):
        super().__init__()
        self.mission = mission
        self.writer = writer
        self.stream = stream
        self.explore_steps = explore_steps
        self.view = view
        self.rows = []
        self.losses = []
        self.routes = {}
        self.timed_routes = []
        self.trials = 0
        self.next_route = 0
        self.best = None
        self.started = 0.

    def _on_training_start(self):
        self.started = time.perf_counter()

    def choose_route(self):
        if not self.timed_routes:
            return ()
        route = self.timed_routes[self.next_route % len(self.timed_routes)]
        self.next_route += 1
        return route

    def _on_step(self):
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            route = remove_backtracking(info['route'])
            replay = info['replay']
            phase = 'speed_test' if replay else 'exploration'
            if replay:
                self.trials += 1
                if info['success'] and (self.best is None or info['duration_s'] < self.best['duration_s']):
                    self.best = dict(route_id=self.routes[replay],
                                     duration_s=info['duration_s'], episode=len(self.rows) + 1)
            elif route and route not in self.routes:
                self.routes[route] = len(self.routes) + 1
            route_id = self.routes.get(replay or route)
            row = dict(episode=len(self.rows) + 1, phase=phase, route_id=route_id,
                       total_steps=self.num_timesteps,
                       elapsed_training_s=time.perf_counter() - self.started,
                       **{key: info[key] for key in ('success','steps','reward','collision_count','duration_s','distance_m')})
            self.rows.append(row)
            self.writer.writerow(row)
            self.stream.flush()
            print(f"{phase} | episode {row['episode']} | sukses={row['success']} | "
                  f"reward={row['reward']:.2f} | waktu={row['duration_s']:.2f}s", flush=True)
            if self.view:
                self.view.record_episode(row['reward'], row['success'], row['collision_count'] > 0, row['steps'])

        if not self.timed_routes and self.routes and self.num_timesteps >= self.explore_steps:
            self.timed_routes = list(self.routes)
            print('UJI KECEPATAN dijadwalkan mulai episode berikutnya', flush=True)
        metrics = self.model.logger.name_to_value
        if self.num_timesteps % 100 == 0 and 'train/critic_loss' in metrics:
            self.losses.append(dict(step=self.num_timesteps,
                                    actor_loss=float(metrics['train/actor_loss']),
                                    critic_loss=float(metrics['train/critic_loss']),
                                    entropy_coefficient=float(metrics['train/ent_coef'])))
        if self.view:
            self.view.stage = 'UJI KECEPATAN / SAC' if self.mission.route_search.replaying else 'EKSPLORASI / SAC'
            for key in ('train/actor_loss', 'train/critic_loss', 'train/ent_coef'):
                if key in metrics:
                    self.view.record_training_metric(key, metrics[key])
            self.view.set_progress(self.num_timesteps, len(self.rows))
            self.view.update(self.mission.motion, dt=self.mission.motion.decision_seconds)
        return not self.view or not self.view.closed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--update-every', type=int, choices=(1, 2, 4, 8), default=2,
                        help='Satu update jaringan per N langkah simulasi. N lebih besar mengurangi biaya komputasi.')
    parser.add_argument('--timeout', type=float, default=300.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exploration-fraction', type=float, default=.6)
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--render-speed', type=float, default=16.)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if (args.steps < 1 or not 0 < args.exploration_fraction < 1 or
            not math.isfinite(args.timeout) or args.timeout <= 0 or
            not math.isfinite(args.render_speed) or args.render_speed <= 0):
        parser.error('Invalid training steps, timeout, exploration fraction, or render speed')
    if args.render and sys.platform.startswith('linux') and not (
            os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('Gunakan --no-render jika display tidak tersedia.')
    maze = read_world()
    nav = NavigationMap(maze.start, maze.goal, maze.walls, maze.bounds, maze.cell_size, guided=False)
    mission = LidarMission(nav, args.timeout)
    directory = args.run_dir or ROOT / 'reports/lidar_drl/runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        parser.error('Run directory must be empty.')
    view = None
    if args.render:
        from training_view import TrainingView
        view = TrainingView(nav, args.render_speed)
    torch.set_num_threads(1)
    model = SAC('MlpPolicy', mission, seed=args.seed, gamma=.98, learning_starts=min(256, args.steps // 4),
                train_freq=(args.update_every, 'step'), gradient_steps=1,
                policy_kwargs=dict(net_arch=[64, 64]), device='cpu', verbose=0)
    fields = ('episode','phase','route_id','total_steps','elapsed_training_s','success','steps',
              'reward','collision_count','duration_s','distance_m')
    with (directory / 'training.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        monitor = TrainingMonitor(mission, writer, stream, int(args.steps * args.exploration_fraction), view)
        mission.route_selector = monitor.choose_route
        try:
            model.learn(total_timesteps=args.steps, callback=monitor)
        except KeyboardInterrupt:
            print('Training dihentikan, menyimpan actor SAC.', flush=True)
        finally:
            training_seconds = time.perf_counter() - monitor.started if monitor.started else 0.
            if view:
                view.close()
    export_actor(model, directory / 'actor.json')
    model.save(directory / 'sac')
    with (directory / 'losses.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('step','actor_loss','critic_loss','entropy_coefficient'))
        writer.writeheader()
        writer.writerows(monitor.losses)
    plots(monitor.rows, monitor.losses, directory)
    first_success = next((row for row in monitor.rows if row['success']), None)
    summary = dict(algorithm=ALGORITHM, seed=args.seed, steps=model.num_timesteps, timeout_s=args.timeout,
                   training_seconds=training_seconds, render=args.render,
                   update_every_steps=args.update_every,
                   first_success_step=first_success['total_steps'] if first_success else None,
                   first_success_training_s=first_success['elapsed_training_s'] if first_success else None,
                   episodes=len(monitor.rows), routes_found=len(monitor.routes), best_speed_trial=monitor.best,
                   exploration_fraction=args.exploration_fraction,
                   routes=[dict(route_id=route_id, targets=targets) for targets, route_id in monitor.routes.items()])
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Run DRL: {directory.resolve()}')


if __name__ == '__main__':
    main()
