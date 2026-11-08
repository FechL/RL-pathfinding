"""Run the actual Webots controller on isolated project copies and save evidence."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/micromouse"))
from planner import Obstacle, ROBOT_RADIUS, segment_is_free


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--algorithm", choices=("naf", "sac", "rl", "rl_unguided", "q_learning", "astar"), default="sac")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--keep-telemetry", action="store_true", help="Save full telemetry from the first validated run.")
    parser.add_argument("--reset-pose", type=float, nargs=3, metavar=("X", "Y", "YAW"),
                        help="Evaluate an RL policy from a specified pose instead of the mission start.")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    if args.reset_pose is not None and args.algorithm != "rl_unguided":
        parser.error("--reset-pose is only available for the unguided RL controller")
    if not shutil.which("webots"):
        parser.error("Webots was not found in PATH")
    report = args.report_dir or ROOT / "reports" / f"{args.algorithm}_webots"
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    records = []
    for number in range(1, args.runs+1):
        with tempfile.TemporaryDirectory(prefix="micromouse-maze-run-") as tmp:
            project = Path(tmp)
            for folder in ("controllers", "worlds", "protos", "models"):
                shutil.copytree(ROOT / folder, project / folder, ignore=shutil.ignore_patterns("*.zip", "*.pkl", "*.pt"))
            env = os.environ.copy()
            env.update(MICROMOUSE_ALGORITHM=args.algorithm, MICROMOUSE_TEST_EXIT="1", MICROMOUSE_NO_GUI="1", MICROMOUSE_RESULTS_DIR=str(project / "results"))
            env.pop("MICROMOUSE_RL_RESET_POSE", None)
            if args.reset_pose is not None:
                env["MICROMOUSE_RL_RESET_POSE"] = json.dumps(args.reset_pose)
            if number == 1:
                env["MICROMOUSE_CAPTURE_DIR"] = str(project / "captures")
            else:
                env.pop("MICROMOUSE_CAPTURE_DIR", None)
            command = [shutil.which("webots"), "--batch", "--mode=fast", "--minimize", "--stdout", "--stderr"]
            if number > 1:
                command.append("--no-rendering")
            command.append(str(project / "worlds/world.wbt"))
            process = subprocess.run(command, env=env, text=True, capture_output=True, timeout=600)
            log = process.stdout+process.stderr
            (report / f"webots_run_{number}.log").write_text(log)
            raw_result = project / "results/latest_run.json"
            if number == 1 and args.keep_telemetry and raw_result.is_file():
                shutil.copy2(raw_result, report / "telemetry.json")
            assert process.returncode == 0, log
            data = json.loads(raw_result.read_text())
            assert data["status"] == "success", data
            assert data["algorithm"] == args.algorithm, data
            assert data["python"] == str(Path.home()/"miniconda3/bin/python"), data["python"]
            assert data["goal_error_m"] <= .06
            assert data["min_physical_clearance_m"] >= .005
            obstacles = [Obstacle(w["name"], "wall", w["center"], w["size"]) for w in data["walls"]]
            for a, b in zip(data["trajectory"], data["trajectory"][1:]):
                assert segment_is_free(a, b, obstacles, data["arena_bounds"], ROBOT_RADIUS), (a,b)
            data.pop("trajectory")
            data.pop("telemetry", None)
            records.append(data)
            if number == 1:
                shutil.copy2(project / "results/map.png", figures / "navigation_map.png")
                for image in (project / "captures").glob("*.png"):
                    shutil.copy2(image, figures / image.name)
            print(f"Run {number}: success; time={data['simulation_duration_s']:.3f} s; length={data['travelled_length_m']:.3f} m; error={data['goal_error_m']:.6f} m; clearance={data['min_physical_clearance_m']:.4f} m", flush=True)
    start_scope = "the specified reset pose" if args.reset_pose is not None else "the mission start"
    result = {"validated_at_utc": datetime.now(timezone.utc).isoformat(), "webots": "R2025a", "runs": records,
              "reset_pose": args.reset_pose,
              "scope": f"{args.runs} deterministic physics resets from {start_scope} on the fixed 16 x 16 maze; selected algorithm drives simulated wheel motors. NAF and SAC use continuous velocity actions and kinematic training. Legacy tabular Q-learning uses cell options. Repeated fixed-maze runs measure repeatability."}
    (report / "validation_results.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
