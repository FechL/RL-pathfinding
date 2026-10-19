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
sys.path.insert(0, str(ROOT / "controllers/epuck"))
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
        with tempfile.TemporaryDirectory(prefix="epuck-lab-test-") as tmp:
            project = Path(tmp)
            for folder in ("controllers", "worlds", "protos"):
                shutil.copytree(ROOT / folder, project / folder)
            # Disable camera overlays inherited from a previous desktop session.
            config = project / "worlds/.world.wbproj"
            if config.exists():
                text = config.read_text()
                config.write_text("\n".join(line for line in text.splitlines() if not line.startswith("renderingDevicePerspectives:"))+"\nrenderingDevicePerspectives: e-puck:camera;0;1;0;0\n")
            env = os.environ.copy()
            env.update(EPUCK_TEST_EXIT="1", EPUCK_NO_GUI="1", EPUCK_RESULTS_DIR=str(project / "results"))
            if number == 1:
                env["EPUCK_CAPTURE_DIR"] = str(project / "captures")
            else:
                env.pop("EPUCK_CAPTURE_DIR", None)
            command = [shutil.which("webots"), "--batch", "--mode=fast", "--minimize", "--stdout", "--stderr"]
            if number > 1:
                command.append("--no-rendering")
            command.append(str(project / "worlds/world.wbt"))
            process = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
            log = process.stdout+process.stderr
            (report / f"webots_run_{number}.log").write_text(log)
            assert process.returncode == 0, log
            data = json.loads((project / "results/latest_run.json").read_text())
            assert data["status"] == "success", data
            assert data["python"] == str(Path.home()/"miniconda3/bin/python"), data["python"]
            assert data["goal_error_m"] <= .01
            assert data["min_physical_clearance_m"] >= .029
            obstacles = [Obstacle(o["name"], o["kind"], o["center"], o["size"]) for o in data["obstacles"]]
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
              "scope": "Three deterministic resets of the same static layout; no RL training and no wheel-dynamics validation."}
    (report / "validation_results.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
