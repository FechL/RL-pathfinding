# Micromouse Maze Arena

A Webots R2025a micromouse simulation with a small differential-drive robot in a deterministic 16 × 16 maze. Each cell is 40 cm, so the arena is 6.4 × 6.4 m. The maze has dead ends, extra loop openings, and a 2 × 2 goal zone at the far corner opposite the start.

The current baseline uses A* for route planning and wheel motors for movement. The robot has four infrared wall sensors, GPS, an IMU, and wheel encoders. The sensors and motor actions provide a starting point for a later RL controller; no RL policy or training is included now. A* reads the known maze geometry through Webots Supervisor, so the current baseline is not sensor-only.

## Run

Install Webots R2025a and Miniconda, then from the project folder run:

```bash
conda activate base
python -m pip install -r requirements.txt
bash run.sh
```

Webots opens `worlds/world.wbt`. The OpenCV window displays the maze, explored cells, planned route, and mouse position. The robot moves with its differential-drive wheel motors. Use Reset or Reload in Webots to restart.

To hide the OpenCV map window while keeping the Webots GUI open:

```bash
MICROMOUSE_NO_GUI=1 bash run.sh
```

The Python path defaults to `~/miniconda3/bin/python`. If yours is different, update `base_python` in `run.sh` and `COMMAND` in `controllers/micromouse/runtime.ini`. Webots provides the `controller` Python module; do not install it from PyPI.

## Maze layout

`worlds/maze_layout.json` controls the grid size, cell width, maze seed, loop openings, start cell, and center goal cells. Regenerate the maze world and floor texture with:

```bash
python -m pip install Pillow
python tools/build_world.py
```

The same seed produces the same maze. Change `seed` to create another maze with the same dimensions.

## Future RL interface

The robot exposes four infrared distance readings, yaw from the IMU, position from GPS, and left/right wheel motors. A future agent can use those readings as observations and wheel speeds as continuous actions, while the current A* controller remains a deterministic baseline.

## Checks

```bash
python -m unittest discover -s tests -v
python tools/validate_simulation.py
```

The validator runs three deterministic maze solves and saves results to `reports/validation_results.json`.
