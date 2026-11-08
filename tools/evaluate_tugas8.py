"""Reproduce the Webots corner and kinematic route comparisons for Tugas 8."""
import argparse
import heapq
from itertools import groupby
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from train_continuous import ROOT, evaluate, navigation
from continuous_control import ControlConfig, read_policy
from planner import Obstacle, physical_clearance


OUTPUT = ROOT / "reports/tugas8"
FIGURE_DPI = 180
COLORS = ("#005A8D", "#A34313", "#626B39")
STYLES = ("-", "--", ":")
LABELS = {"naf": "NAF", "q_learning": "Q-learning tabular", "sac": "SAC"}
# First baseline corner with two straight links on both approaches.
CORNER_ENTRY = (0, -2.4, -3.0, .2)
CORNER_EXIT = (1, -2.8, -2.2, .2)
# Common gates upstream and downstream of the two alternative corridors.
ROUTE_ENTRY = (1, .8, -1.8, .2)
ROUTE_EXIT = (1, 2.0, .2, .2)
EXCLUDED_EDGE = (163, 164)


def crossing(times, positions, gate, not_before=0.):
    """Find the first positive-direction crossing of a finite gate."""
    axis, value, center, half_width = gate
    for i in range(len(times) - 1):
        a, b = positions[i:i+2]
        if a[axis] < value <= b[axis]:
            fraction = (value - a[axis]) / (b[axis] - a[axis])
            location = a + fraction * (b - a)
            time = times[i] + fraction * (times[i+1] - times[i])
            if abs(location[1-axis] - center) <= half_width and time >= not_before:
                return float(time), location
    raise ValueError(f"Trajectory did not cross gate {gate}")


def gate_line(ax, gate, label=None):
    axis, value, center, half_width = gate
    points = ([value, value], [center-half_width, center+half_width])
    x, y = points if axis == 0 else points[::-1]
    ax.plot(x, y, color="#444B52", ls="-.", lw=1.1, label=label)


def map_axes(ax, walls, bounds):
    for wall in walls:
        x, y = wall.center
        w, h = wall.size
        ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, color="#CBD2DA", linewidth=0))
    ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="X (m)", ylabel="Y (m)", aspect="equal")


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E8EE", linewidth=.7)
    ax.set_axisbelow(True)


def save(fig, output, name):
    path = output / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write(output, name, record):
    (output / name).write_text(json.dumps(record, indent=2) + "\n")


def webots_comparison(paths, output):
    records = [json.loads(path.read_text()) for path in paths]
    world = records[0]
    bounds = world["arena_bounds"]
    wall_data = world["walls"]
    walls = [Obstacle(w["name"], "wall", w["center"], w["size"]) for w in wall_data]
    config = ControlConfig()
    map_fig, maze = plt.subplots(figsize=(6.8, 6.3), layout="constrained")
    map_axes(maze, walls, bounds)
    corner_fig = plt.figure(figsize=(10., 4.9), layout="constrained")
    grid = corner_fig.add_gridspec(2, 2, width_ratios=(1.05, 1.45))
    corner = corner_fig.add_subplot(grid[:, 0])
    v_plot = corner_fig.add_subplot(grid[0, 1])
    w_plot = corner_fig.add_subplot(grid[1, 1], sharex=v_plot)
    map_axes(corner, walls, (-2.53, -1.98, -3.22, -2.69))
    gate_line(corner, CORNER_ENTRY, "Gerbang")
    gate_line(corner, CORNER_EXIT)
    metrics = []
    for data, path, color, style in zip(records, paths, COLORS, STYLES):
        if data["status"] != "success" or data["arena_bounds"] != bounds or data["walls"] != wall_data:
            raise ValueError("Comparison requires successful runs in the same Webots world")
        label = LABELS[data["algorithm"]]
        samples = data["telemetry"]
        times = np.array([s["time_s"] for s in samples])
        poses = np.array([s["pose"][:2] for s in samples])
        wheels = np.array([s["commanded_wheels"] for s in samples])
        if not np.all(np.diff(times) > 0):
            raise ValueError("Telemetry time must be strictly increasing")
        entry_time, entry = crossing(times, poses, CORNER_ENTRY)
        exit_time, exit_pos = crossing(times, poses, CORNER_EXIT, entry_time)
        inside = (times > entry_time) & (times < exit_time)
        segment = np.vstack((entry, poses[inside], exit_pos))
        elapsed = exit_time - entry_time
        selected = np.r_[np.searchsorted(times, entry_time, side="right")-1, np.flatnonzero(inside)]
        v = wheels[selected].mean(axis=1) * config.wheel_radius
        w = (wheels[selected, 1] - wheels[selected, 0]) * config.wheel_radius / config.axle_track
        t = np.r_[0., times[inside] - entry_time, elapsed]
        turning = np.abs(w) > .4
        durations = np.diff(t)
        turn_time = durations[turning].sum()
        metrics.append(dict(
            algorithm=data["algorithm"], source=str(path), full_duration_s=data["simulation_duration_s"],
            full_length_m=data["travelled_length_m"], full_clearance_m=data["min_physical_clearance_m"],
            goal_error_m=data["goal_error_m"], full_motion=data["motion"],
            entry_time_s=entry_time, exit_time_s=exit_time, corner_duration_s=elapsed,
            corner_length_m=float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum()),
            corner_clearance_m=min(physical_clearance(p, walls, bounds) for p in segment),
            moving_turn_fraction=float(durations[turning & (v > .05)].sum()/turn_time) if turn_time else None,
            peak_forward_m_s=float(v.max()), peak_abs_yaw_rad_s=float(np.abs(w).max()),
        ))
        kwargs = dict(color=color, ls=style, lw=1.65, label=label)
        maze.plot(*np.asarray(data["trajectory"]).T, **kwargs)
        corner.plot(*segment.T, **kwargs)
        v_plot.step(t, np.r_[v, v[-1]], where="post", **kwargs)
        w_plot.step(t, np.r_[w, w[-1]], where="post", **kwargs)
    maze.scatter(*world["start"], s=28, marker="o", color="#242E36", label="Start")
    maze.scatter(*world["goal"], s=35, marker="*", color="#242E36", label="Goal")
    maze.set_title("Lintasan pada maze Webots 16 × 16")
    maze.legend(loc="upper center", bbox_to_anchor=(.5, -.11), ncol=3, frameon=False)
    corner.set_title("Lintasan di belokan")
    v_plot.set(title="Perintah gerak di antara gerbang", ylabel="v (m/s)")
    v_plot.tick_params(labelbottom=False)
    w_plot.set(xlabel="Waktu sejak gerbang masuk (s)", ylabel="ω (rad/s)")
    w_plot.set_xlim(left=0.)
    for ax in (v_plot, w_plot):
        style_axes(ax)
    corner_fig.legend(*corner.get_legend_handles_labels(), loc="outside lower center",
                      ncol=len(records)+1, frameon=False)
    save(map_fig, output, "webots_trajectory.png")
    save(corner_fig, output, "webots_corner.png")
    write(output, "webots_comparison.json", dict(
        scope="First run per controller, one fixed 90-degree bend. This is a controller benchmark.",
        gates=dict(entry=CORNER_ENTRY, exit=CORNER_EXIT, fields=["axis", "value_m", "center_m", "half_width_m"]),
        timing="Interpolate positive crossings using Webots GPS positions. Time and clearance use the same gate-to-gate interval.",
        motion="Computed from commanded wheel speeds held until next timestamp. Moving turn means v > 0.05 m/s while |yaw rate| > 0.4 rad/s.",
        runs=metrics,
    ))


def smooth_comparison(before, after, output):
    """Compare body yaw from Webots IMU over the same corner and full run."""
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    records = []
    for path, label, color, style in zip((before, after), ("Sebelum", "Sesudah"), COLORS, STYLES):
        data = json.loads(path.read_text())
        samples = data["telemetry"]
        times = np.array([row["time_s"] for row in samples])
        poses = np.array([row["pose"] for row in samples])
        dt = np.diff(times)
        if data["status"] != "success" or len(times) < 4 or not np.allclose(dt, dt[0], atol=1e-8):
            raise ValueError("Expected successful runs with uniformly sampled Webots IMU data")
        entry, _ = crossing(times, poses[:, :2], CORNER_ENTRY)
        end, _ = crossing(times, poses[:, :2], CORNER_EXIT, entry)
        yaw = np.unwrap(poses[:, 2])
        corner = (times >= entry) & (times <= end)
        rate_times = (times[1:] + times[:-1]) / 2
        segment = (rate_times >= entry) & (rate_times <= end)
        ax.plot(rate_times[segment]-entry, (np.diff(yaw)/dt)[segment],
                label=label, color=color, ls=style, lw=1.7)
        if np.count_nonzero(corner) < 4:
            raise ValueError("The corner needs at least four IMU samples")

        def jerk_rms(headings):
            return float(np.sqrt(np.mean((np.diff(headings, n=3)/dt[0]**3)**2)))
        records.append(dict(label=label, source=str(path),
                            policy_sha256=data["policy_metadata"]["artifact_sha256"],
                            duration_s=data["simulation_duration_s"],
                            min_clearance_m=data["min_physical_clearance_m"],
                            corner_duration_s=end-entry, yaw_jerk_rms_rad_s3=jerk_rms(yaw),
                            corner_yaw_jerk_rms_rad_s3=jerk_rms(yaw[corner])))
    if records[0]["policy_sha256"] != records[1]["policy_sha256"]:
        raise ValueError("Expected the same policy before and after the motor changes")
    ax.set(title="Laju belok dari IMU di belokan Webots yang sama",
           xlabel="Waktu sejak gerbang masuk (s)", ylabel="Laju belok (rad/s)")
    style_axes(ax)
    ax.legend(frameon=False)
    save(fig, output, "smooth_turn.png")
    write(output, "smooth_comparison.json", dict(
        method="Uniform 16 ms IMU yaw samples, unwrapped. RMS jerk uses the third finite difference of yaw.",
        runs=records))


def exclude_guide_edge(nav, edge):
    """Recompute the guidance field after withholding one link from Dijkstra."""
    a, b = edge
    nav.neighbors[a].remove(b)
    nav.neighbors[b].remove(a)
    # The original no-successor cells are the direct-to-goal roots.
    queue = [(float(nav.costs[i]), int(i)) for i in np.flatnonzero(nav.next == -1)]
    heapq.heapify(queue)
    nav.costs[nav.next != -1] = np.inf
    nav.next.fill(-1)
    while queue:
        cost, i = heapq.heappop(queue)
        if cost > nav.costs[i] + 1e-9:
            continue
        for j in nav.neighbors[i]:
            new_cost = cost + nav.cell_size
            if new_cost < nav.costs[j] - 1e-9:
                nav.costs[j], nav.next[j] = new_cost, i
                heapq.heappush(queue, (new_cost, j))
    if not np.all(np.isfinite(nav.costs)):
        raise ValueError("Guide edge exclusion disconnected the maze")


def route_comparison(policy, output):
    nav = navigation()
    algorithm = json.loads(policy.read_text())["algorithm"]
    if algorithm not in ("naf", "sac"):
        raise ValueError("The route comparison requires a continuous NAF or SAC policy")
    actor, metadata = read_policy(policy, nav, algorithm)
    seeds = (1000, 1001, 1002)
    fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
    map_axes(ax, nav.walls, (-2.06, .44, .48, 2.45))
    gate_line(ax, ROUTE_ENTRY, "Gerbang")
    gate_line(ax, ROUTE_EXIT)
    alternatives = []
    for name, edge, color, style, expected_cell, other_cell in zip(
        ("Panduan baku", "Panduan alternatif"), (None, EXCLUDED_EDGE), COLORS, STYLES, (164, 179), (179, 164)
    ):
        if edge:
            exclude_guide_edge(nav, edge)
        route = [163]
        while route[-1] != 200 and nav.next[route[-1]] != -1:
            route.append(int(nav.next[route[-1]]))
        if route[-1] != 200:
            raise ValueError("Expected a route rejoining at cell 200")
        summaries = []
        for record in evaluate(actor, nav, seeds=seeds):
            positions = np.asarray(record.pop("trajectory"))
            record.pop("wheel_commands")
            times = np.arange(len(positions))*nav.config.dt
            visited = [cell for cell, _ in groupby(nav.maze.state(p) for p in positions)]
            followed = expected_cell in visited and other_cell not in visited
            elapsed = None
            segment_clearance = None
            if followed:
                entry_time, entry = crossing(times, positions, ROUTE_ENTRY)
                try:
                    exit_time, exit_pos = crossing(times, positions, ROUTE_EXIT, entry_time)
                except ValueError:
                    exit_time = None  # A collision or timeout can occur before the exit gate.
                if exit_time is not None:
                    elapsed = exit_time - entry_time
                    inside = (times > entry_time) & (times < exit_time)
                    segment = np.vstack((entry, positions[inside], exit_pos))
                    segment_clearance = min(physical_clearance(p, nav.walls, nav.bounds) for p in segment)
                    if record["seed"] == seeds[0]:
                        ax.plot(*segment.T, color=color, ls=style, lw=2, label=name)
            summaries.append(dict(**record, followed_expected_branch=followed, visited_cells=visited,
                                  corridor_duration_s=elapsed, corridor_clearance_m=segment_clearance))
        completions = [s for s in summaries if s["is_success"] and s["corridor_duration_s"] is not None]
        alternatives.append(dict(
            label=name, excluded_edge=edge, guide_cells_between_junctions=route,
            nominal_length_m=(len(route)-1)*nav.cell_size, runs=summaries, completed_runs=len(completions),
            mean_full_duration_s=statistics.mean(s["time_s"] for s in completions) if completions else None,
            mean_corridor_duration_s=statistics.mean(s["corridor_duration_s"] for s in completions) if completions else None,
        ))
    ax.set_title(f"Dua koridor dengan policy {LABELS[metadata['algorithm']]} yang sama")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, frameon=False)
    save(fig, output, f"route_comparison_{metadata['algorithm']}.png")
    paired_savings = []
    for a, b in zip(*(option["runs"] for option in alternatives)):
        if all(row["is_success"] and row["corridor_duration_s"] is not None for row in (a, b)):
            paired_savings.append(dict(seed=a["seed"], full_seconds=a["time_s"]-b["time_s"],
                                       corridor_seconds=a["corridor_duration_s"]-b["corridor_duration_s"]))
    write(output, f"route_comparison_{metadata['algorithm']}.json", dict(
        policy=str(policy), algorithm=metadata["algorithm"], artifact_sha256=metadata["artifact_sha256"],
        scope="Paired kinematic experiment on the same maze and actor; the Dijkstra guide selects the corridor.",
        method="Only the guide excludes edge 163–164 in the alternate scenario. Wall geometry and reset seeds stay identical.",
        gates=dict(entry=ROUTE_ENTRY, exit=ROUTE_EXIT, fields=["axis", "value_m", "center_m", "half_width_m"]),
        plot_seed=seeds[0], alternatives=alternatives, paired_savings=paired_savings,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    modes = parser.add_subparsers(dest="mode", required=True)
    webots = modes.add_parser("webots", help="Compare NAF, tabular Q-learning and optionally SAC")
    webots.add_argument("telemetry", nargs="+", type=Path)
    routes = modes.add_parser("routes", help="Measure two maze corridors with one continuous policy")
    routes.add_argument("policy", type=Path)
    smooth = modes.add_parser("smooth", help="Compare IMU yaw before and after smoothing")
    smooth.add_argument("before", type=Path)
    smooth.add_argument("after", type=Path)
    args = parser.parse_args()
    if args.mode == "webots":
        if not 2 <= len(args.telemetry) <= 3:
            parser.error("Provide 2 or 3 Webots telemetry files")
        webots_comparison(args.telemetry, args.output)
    elif args.mode == "routes":
        route_comparison(args.policy, args.output)
    else:
        smooth_comparison(args.before, args.after, args.output)


if __name__ == "__main__":
    main()
