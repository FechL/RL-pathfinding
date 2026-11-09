"""Plot a saved tabular Q-learning run and its visited state space."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def newest_run():
    runs = ROOT/'reports/tabular_q_learning/runs'
    choices = sorted((path for path in runs.iterdir() if path.is_dir()), reverse=True) if runs.exists() else []
    for path in choices:
        if (path/'training.csv').exists() and (path/'state_space.csv').exists():
            return path
    raise FileNotFoundError('No tabular Q run found. Run `bash run.sh train` first.')


def rolling(values, window=100):
    result = []
    for index in range(len(values)):
        chunk = values[max(0, index-window+1):index+1]
        result.append(sum(chunk)/len(chunk))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, help='Run folder; defaults to the most recent saved run')
    args = parser.parse_args()
    folder = args.report_dir or newest_run()
    if not folder.is_absolute():
        folder = ROOT/folder
    with (folder/'training.csv').open(newline='') as stream:
        history = list(csv.DictReader(stream))
    with (folder/'state_space.csv').open(newline='') as stream:
        states = list(csv.DictReader(stream))
    if not history:
        raise ValueError(f'No episode rows in {folder}/training.csv')
    output = folder/'figures'
    output.mkdir(parents=True, exist_ok=True)
    episode = np.array([int(row['episode']) for row in history])
    reward = np.array([float(row['reward']) for row in history])
    steps = np.array([int(row['steps']) for row in history])
    success = np.array([int(row['success'].lower() == 'true') for row in history])
    collisions = np.array([int(row['collisions']) for row in history])
    epsilon = np.array([float(row['epsilon']) for row in history])
    phase = [row['phase'] for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(episode, reward, color='#b5c5c3', linewidth=.6, label='per-episode reward')
    axes[0, 0].plot(episode, rolling(reward.tolist()), color='#dc7b22', linewidth=2, label='rolling mean (100)')
    axes[0, 0].set(title='Reward per episode', xlabel='Episode', ylabel='Reward')
    axes[0, 0].legend()
    axes[0, 1].plot(episode, steps, color='#92b5d2', linewidth=.7, label='steps')
    axes[0, 1].plot(episode, rolling(steps.tolist()), color='#286b9e', linewidth=2, label='rolling mean (100)')
    axes[0, 1].set(title='Steps until yellow detection or timeout', xlabel='Episode', ylabel='Steps')
    axes[0, 1].legend()
    collision_rate = collisions / np.maximum(steps, 1)
    axes[1, 0].plot(episode, rolling(success.tolist()), color='#24865b', linewidth=2, label='success rate (100)')
    axes[1, 0].plot(episode, rolling(collision_rate.tolist()), color='#b84a40', linewidth=1.5, label='collision/action rate (100)')
    axes[1, 0].set(title='Learning success and collisions', xlabel='Episode', ylabel='Rolling rate')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend()
    axes[1, 1].plot(episode, epsilon, color='#7652a4', linewidth=1.7)
    if 'eksploitasi' in phase:
        first = phase.index('eksploitasi')
        axes[1, 1].axvline(episode[first], color='#d78223', linestyle='--', label='start exploitation')
        axes[1, 1].legend()
    axes[1, 1].set(title='Epsilon: exploration → exploitation', xlabel='Episode', ylabel='ε', ylim=(0, 1.05))
    fig.suptitle('Tabular Q-learning training monitor', fontsize=16)
    fig.savefig(output/'training_metrics.png', dpi=160)
    plt.close(fig)

    summary_path = folder/'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    width, height = 16, 16
    visit_map = np.zeros((height, width), dtype=float)
    value_sum = np.zeros((height, width), dtype=float)
    value_count = np.zeros((height, width), dtype=float)
    action_counts = np.zeros(4, dtype=float)
    for row in states:
        x, y = int(row['cell_x']), int(row['cell_y'])
        visit_map[y, x] += int(row['visits'])
        q_values = [float(row[f'q_{name}']) for name in ('east', 'north', 'west', 'south')]
        value_sum[y, x] += max(q_values)
        value_count[y, x] += 1
        if int(row['visits']) > 0:
            action_counts[int(np.argmax(q_values))] += int(row['visits'])
    value_map = np.divide(value_sum, value_count, out=np.zeros_like(value_sum), where=value_count > 0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    visit_img = axes[0].imshow(visit_map, origin='lower', cmap='Blues')
    axes[0].set(title='State visits by cell', xlabel='Cell x', ylabel='Cell y')
    fig.colorbar(visit_img, ax=axes[0], label='decision count')
    value_img = axes[1].imshow(value_map, origin='lower', cmap='viridis')
    axes[1].set(title='Mean max Q(s,a) by cell', xlabel='Cell x', ylabel='Cell y')
    fig.colorbar(value_img, ax=axes[1], label='Q value')
    axes[2].bar(('East', 'North', 'West', 'South'), action_counts,
                color=('#487ca8', '#68a56d', '#d17a45', '#9569aa'))
    axes[2].set(title='Greedy action by visited state', ylabel='state visits')
    fig.suptitle('Learned tabular state space', fontsize=16)
    fig.savefig(output/'state_space.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(run=str(folder), visited_states=len(states),
                          state_space_bound=summary.get('state_space_bound'),
                          success_count=summary.get('success_count'),
                          training_graph=str(output/'training_metrics.png'),
                          state_space_graph=str(output/'state_space.png')), indent=2))


if __name__ == '__main__':
    main()
