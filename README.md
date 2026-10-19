# E-puck Navigation Lab

A Webots simulation where an E-puck robot follows an A* path around obstacles to reach a goal. The project uses Python and Webots R2025a. It does not train a reinforcement learning agent.

## Start

On Linux, install Webots R2025a and Miniconda, then run these commands from the project folder:

```bash
conda activate base
python -m pip install -r requirements.txt
bash run.sh
```

The launcher opens `worlds/world.wbt`. An OpenCV window shows the obstacles, planned path, and robot position. Use Reset or Reload in Webots to restart.

To run without the OpenCV map window:

```bash
EPUCK_NO_GUI=1 bash run.sh
```

The Python path defaults to `~/miniconda3/bin/python`. If yours is different, update `base_python` in `run.sh` and `COMMAND` in `controllers/epuck/runtime.ini`. Webots provides the `controller` Python module; do not install it from PyPI.

To rebuild the world and floor texture after changing the layout, install Pillow and run:

```bash
python -m pip install Pillow
python tools/build_world.py
```

Keep box obstacles aligned with the arena axes; the planner does not support rotated box shapes.

## Checks

```bash
python -m unittest discover -s tests -v
python tools/validate_simulation.py
```

The validator runs three Webots simulations with the same layout and checks that the robot reaches the goal safely. Results are saved to `reports/validation_results.json`.

See [research notes](reports/research_notes.md) for design references and limitations.
