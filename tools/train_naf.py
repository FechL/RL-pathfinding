"""Train a continuous Q-function and export its greedy mean to Webots."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from naf import NAF, NormalizedAdvantage, Replay
from train_continuous import ROOT, evaluate, navigation
from continuous_control import read_policy
from continuous_env import ContinuousDriveEnv
from plot_training import generate


def atomic_write(path, contents):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(contents)
    temporary.replace(path)


def json_write(path, data):
    atomic_write(path, (json.dumps(data, indent=2) + '\n').encode())


def episode_row(env, wall_time, info):
    return dict(r=env.episode_return, l=env.steps, t=wall_time, **info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--teacher', type=Path, default=ROOT/'models/continuous_actor.json')
    parser.add_argument('--warm-start', action=argparse.BooleanOptionalAction, default=True,
                        help='Initialize the greedy mean from the Tugas 7 SAC policy')
    parser.add_argument('--plots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--activate', action='store_true', help='Install best successful trained policy for Webots')
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--render-speed', type=float, default=8.)
    parser.add_argument('--eval-every', type=int, default=5000)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1:
        parser.error('--steps and --eval-every must be positive')
    if not math.isfinite(args.render_speed) or args.render_speed <= 0:
        parser.error('--render-speed must be positive and finite')
    if args.render and sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('Tidak ada layar grafis. Gunakan --no-render untuk training tanpa jendela.')

    report = args.run_dir or ROOT/'reports/naf/runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    if report.exists() and any(report.iterdir()):
        parser.error('Choose an empty --run-dir to preserve existing results.')
    report.mkdir(parents=True, exist_ok=True)
    models = report/'models'
    models.mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    nav = navigation()
    env = ContinuousDriveEnv(nav, training=True, max_steps=1500)
    model = NormalizedAdvantage()
    teacher = None
    teacher_sha = None
    if args.warm_start:
        _, metadata = read_policy(args.teacher, nav)
        model.initialize_mean(metadata['layers'])
        teacher = deepcopy(model.mean).requires_grad_(False)
        teacher_sha = metadata['artifact_sha256']
    learner = NAF(model, teacher)
    replay = Replay(100000)
    warmup = min(1500, args.steps // 5)
    results = []
    best = None
    mean_updates = 0
    began = time.perf_counter()
    manifest = dict(algorithm='NAF', seed=args.seed, requested_new_steps=args.steps,
                    control_dt_s=nav.config.dt, wheel_response_time_s=nav.config.wheel_response_time_s,
                    gamma=learner.gamma, target_tau=learner.tau,
                    batch_size=128, replay_capacity=replay.capacity, update_every_steps=2,
                    value_only_warmup_steps=warmup if teacher is not None else 0,
                    exploration_std_start=.12 if teacher is not None else .35,
                    exploration_std_final=.06 if teacher is not None else .175,
                    mean_learning_rate=learner.optimizer.param_groups[0]['lr'],
                    value_precision_learning_rate=learner.optimizer.param_groups[1]['lr'],
                    teacher_sha256=teacher_sha, teacher_policy=args.teacher.name if teacher is not None else None,
                    teacher_anchor_weight=learner.anchor_weight if teacher is not None else 0.)
    json_write(report/'run.json', manifest)
    print(f'Training NAF: {report.resolve()}', flush=True)

    def assess(step):
        nonlocal best
        data = nav.metadata()
        data.update(algorithm='naf', layers=model.export_layers(),
                    training=dict(**manifest, total_steps=step, elapsed_seconds=time.perf_counter()-began))
        candidate = models/f'naf_{step}.json'
        json_write(candidate, data)
        # Float32 PyTorch and NumPy export implement the same ReLU/tanh policy.
        records = evaluate(model.action, nav)
        name = 'before_naf_updates' if step == 0 else 'naf_checkpoint'
        successes = sum(row['is_success'] for row in records)
        mean_time = float(np.mean([row['time_s'] for row in records]))
        result = dict(stage=name, steps=step, successes=successes, runs=len(records), mean_time_s=mean_time,
                      collisions=sum(row['collision'] for row in records), policy=candidate.name)
        results.append(result)
        json_write(report/'checkpoints.json', results)
        print(f'NAF {step}: {successes}/{len(records)} success; mean {mean_time:.3f}s', flush=True)
        if mean_updates and successes == len(records) and (best is None or mean_time < best['mean_time_s']):
            best = result
            atomic_write(models/'naf_actor.json', candidate.read_bytes())
            json_write(report/'evaluation.json', records)
            torch.save(model.state_dict(), models/'naf_best.pt')

    observation, _ = env.reset(seed=args.seed)
    viewer = None
    if args.render:
        from training_view import TrainingView
        viewer = TrainingView(nav, args.render_speed)
        viewer.stage = 'Training NAF'
        viewer.update(env, force=True)
        env.step_callback = viewer.update
    step = 0
    fields = ('r', 'l', 't', 'is_success', 'collision', 'time_s', 'goal_error_m')
    try:
        with (report/'training.monitor.csv').open('w') as log, (report/'td_loss.csv').open('w') as loss_log:
            monitor = csv.DictWriter(log, fieldnames=fields)
            monitor.writeheader()
            losses = csv.writer(loss_log)
            losses.writerow(['steps', 'mean_td_loss', 'updates'])
            assess(0)
            if args.plots:
                generate(report)
            td_losses = []
            env.curriculum = 1. if teacher is not None else .25
            for step in range(1, args.steps+1):
                progress = step / args.steps
                initial_sigma = manifest['exploration_std_start']
                sigma = initial_sigma * (1. - .5 * progress)
                if teacher is None and step <= 1000:
                    action = rng.uniform(-1., 1., size=2).astype(np.float32)
                else:
                    action = np.clip(model.action(observation) + rng.normal(0., sigma, size=2), -1., 1.)
                next_observation, reward, terminated, truncated, info = env.step(action)
                replay.add(observation, action, reward, next_observation, terminated)
                observation = next_observation
                if terminated or truncated:
                    monitor.writerow(episode_row(env, time.perf_counter()-began, info))
                    log.flush()
                    if teacher is None:
                        env.curriculum = min(1., .25 + 1.25*progress)
                    # Keep variable continuous starts, with frequent complete-mission starts.
                    env.mission_probability = .25
                    observation, _ = env.reset()
                if step >= 128 and step % 2 == 0:
                    freeze_mean = teacher is not None and step <= warmup
                    td_losses.append(learner.update(replay.sample(rng, 128), freeze_mean=freeze_mean))
                    mean_updates += not freeze_mean
                if td_losses and (step % 500 == 0 or step == args.steps or (viewer and viewer.closed)):
                    losses.writerow([step, np.mean(td_losses), len(td_losses)])
                    loss_log.flush()
                    td_losses.clear()
                if step % args.eval_every == 0 or step == args.steps:
                    assess(step)
                    if args.plots:
                        generate(report)
                if viewer is not None and viewer.closed:
                    break
            if step != results[-1]['steps']:
                assess(step)
                if args.plots:
                    generate(report)
    finally:
        env.close()
        if viewer is not None:
            viewer.close()

    json_write(report/'summary.json', dict(total_steps=step, seconds=time.perf_counter()-began,
                                          selected=best, checkpoint_evaluations=results,
                                          scope='Fixed-map kinematic checkpoint selection. Webots physics is evaluated separately.'))
    if not mean_updates:
        raise SystemExit('Training stopped before updating the greedy action network. Checkpoints and logs retained.')
    if best is None:
        raise SystemExit('No trained checkpoint reached the goal on all three seeds. Checkpoints and logs retained.')
    if args.activate:
        atomic_write(ROOT/'models/naf_actor.json', (models/'naf_actor.json').read_bytes())
    print(f'Best trained NAF: {models / "naf_actor.json"}', flush=True)


if __name__ == '__main__':
    main()
