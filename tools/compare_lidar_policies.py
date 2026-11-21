"""Compare frozen tabular Q and SAC on identical routes in the kinematic simulator."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'controllers/micromouse'))

from continuous_control import NavigationMap
from lidar_drl import load_actor
from lidar_drl_env import LidarMission
from lidar_qlearning import TabularLidarQ
from train_q_learning import read_world


def evaluate(mission, controller, actor, route, route_id, heading, seed):
    state, _ = mission.reset(seed=seed, options={'route': route, 'heading': heading})
    motion = mission.motion
    while True:
        if controller == 'tabular_q':
            choice = 0
            if mission.target is not None and not motion.collision_recovery:
                values = actor.values(motion.last_state)
                choice = max(mission.allowed_actions(), key=lambda action: float(values[action]))
            _, _, terminal, truncated, info = mission.tabular_step(choice)
        else:
            state, _, terminal, truncated, info = mission.step(actor(state))
        if terminal or truncated:
            return dict(algorithm=controller, route_id=route_id, heading_rad=heading,
                        timeout=truncated, **{key: info[key] for key in (
                            'success', 'collision_count', 'duration_s', 'distance_m', 'steps', 'reward')})


def draw(rows, output):
    from matplotlib.figure import Figure
    figure = Figure(figsize=(9, 4), layout='constrained')
    results, outcomes = figure.subplots(1, 2)
    for name, color in [('tabular_q', 'tab:blue'), ('sac', 'tab:orange')]:
        records = [row for row in rows if row['algorithm'] == name]
        successful = [row for row in records if row['success']]
        outcomes.bar(name, 100 * len(successful) / len(records), color=color)
        if successful:
            results.scatter([str(row['route_id']) for row in successful],
                            [row['duration_s'] for row in successful], label=name, color=color)
    if results.has_data():
        results.legend()
    else:
        results.text(.5, .5, 'Belum ada perjalanan sukses', ha='center', transform=results.transAxes)
    results.set(xlabel='Rute', ylabel='Waktu tempuh sukses (s)')
    outcomes.set(ylabel='Mencapai goal (%)', ylim=(0, 100))
    figure.savefig(output / 'comparison.png', dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q-policy', type=Path, required=True)
    parser.add_argument('--drl-policy', type=Path, required=True)
    parser.add_argument('--routes', type=Path, required=True,
                        help='Training routes.json from a tabular run that reached the goal')
    parser.add_argument('--limit-routes', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=300.)
    parser.add_argument('--output', type=Path, default=Path('results/lidar_comparison'))
    args = parser.parse_args()
    if args.limit_routes < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Expected positive route count and timeout.')
    routes = json.loads(args.routes.read_text())['routes']
    if not routes:
        parser.error('No routes have reached the goal in this training run.')
    routes = sorted(routes, key=lambda route: (
        route.get('best_speed_time_s') is None,
        route.get('best_speed_time_s') or route.get('exploration_time_s', math.inf)))
    maze = read_world()
    nav = NavigationMap(maze.start, maze.goal, maze.walls, maze.bounds, maze.cell_size, guided=False)
    q, _ = TabularLidarQ.load(args.q_policy)
    sac, _ = load_actor(args.drl_policy)
    mission = LidarMission(nav, args.timeout)
    rows = []
    for candidate in routes[:args.limit_routes]:
        route = candidate['targets']
        if not route or math.dist(route[-1], maze.goal) > 1e-6:
            parser.error(f"Route {candidate['route_id']} does not end at the current maze goal.")
        for seed, heading in enumerate((0., -.15, .15)):
            for name, policy in (('tabular_q', q), ('sac', sac)):
                rows.append(evaluate(mission, name, policy, route, candidate['route_id'], heading, seed))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'trials.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name in ('tabular_q', 'sac'):
        records = [row for row in rows if row['algorithm'] == name]
        times = [row['duration_s'] for row in records if row['success']]
        summary[name] = dict(runs=len(records), successes=len(times),
                             median_success_time_s=statistics.median(times) if times else None,
                             collisions=sum(row['collision_count'] for row in records),
                             timeouts=sum(row['timeout'] for row in records))
    report = dict(simulator='kinematic', maze_signature=nav.signature, timeout_s=args.timeout,
                  q_policy=str(args.q_policy.resolve()), drl_policy=str(args.drl_policy.resolve()),
                  routes_file=str(args.routes.resolve()), metrics=summary)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    draw(rows, args.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
