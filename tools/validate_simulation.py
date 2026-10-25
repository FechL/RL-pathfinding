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
    args = parser.parse_args()
    report = ROOT / "reports"
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    records = []
    for number in range(1, args.runs+1):
        with tempfile.TemporaryDirectory(prefix="micromouse-maze-run-") as tmp:
            project = Path(tmp)
            for folder in ("controllers", "worlds", "protos"):
                shutil.copytree(ROOT / folder, project / folder)
            env = os.environ.copy()
            env.update(MICROMOUSE_TEST_EXIT="1", MICROMOUSE_NO_GUI="1", MICROMOUSE_RESULTS_DIR=str(project / "results"))
            if number == 1:
                env["MICROMOUSE_CAPTURE_DIR"] = str(project / "captures")
            else:
                env.pop("MICROMOUSE_CAPTURE_DIR", None)
            command = [shutil.which("webots"), "--batch", "--mode=fast", "--minimize", "--stdout", "--stderr"]
            if number > 1:
                command.append("--no-rendering")
            command.append(str(project / "worlds/world.wbt"))
            process = subprocess.run(command, env=env, text=True, capture_output=True, timeout=300)
            log = process.stdout+process.stderr
            (report / f"webots_run_{number}.log").write_text(log)
            assert process.returncode == 0, log
            data = json.loads((project / "results/latest_run.json").read_text())
            assert data["status"] == "success", data
            assert data["python"] == str(Path.home()/"miniconda3/bin/python"), data["python"]
            assert data["goal_error_m"] <= .06
            assert data["min_physical_clearance_m"] >= .005
            obstacles = [Obstacle(w["name"], "wall", w["center"], w["size"]) for w in data["walls"]]
            for a, b in zip(data["trajectory"], data["trajectory"][1:]):
                assert segment_is_free(a, b, obstacles, data["arena_bounds"], ROBOT_RADIUS), (a,b)
            data.pop("trajectory")
            records.append(data)
            if number == 1:
                shutil.copy2(project / "results/map.png", figures / "navigation_map.png")
                for image in (project / "captures").glob("*.png"):
                    shutil.copy2(image, figures / image.name)
            print(f"Run {number}: success; length={data['planned_length_m']:.3f} m; error={data['goal_error_m']:.6f} m; clearance={data['min_physical_clearance_m']:.4f} m", flush=True)
    result = {"validated_at_utc": datetime.now(timezone.utc).isoformat(), "webots": "R2025a", "runs": records,
              "scope": "Three deterministic resets of the 16 x 16 maze solve; A* uses the known wall map and a differential-drive robot, no RL training."}
    (report / "validation_results.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
