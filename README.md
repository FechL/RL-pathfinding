# Micromouse Q-Learning in a Continuous Environment

Webots R2025a micromouse navigation in a fixed 16 × 16 maze (6.4 × 6.4 m). The default controller follows motion options selected by a trained tabular Q-learning policy. An A* baseline remains available explicitly.

The physical robot has continuous XY position and yaw. Q-learning discretizes XY into 256 cell bins and learns four high-level motion options (east, north, west, south). A GPS/IMU feedback servo executes the options with differential-drive wheel velocities. This is high-level navigation RL, not direct wheel-velocity learning. The complete static wall map is known; IR is logged rather than used for routing.

## Run the learned policy

```bash
conda activate base
python -m pip install -r requirements.txt
bash run.sh
```

The included `models/q_learning.json` was trained for 6,000 episodes with seed 42. No training is required to run it. Set `MICROMOUSE_NO_GUI=1` to hide only the OpenCV window. Webots itself requires a usable graphical session. Python defaults to `~/miniconda3/bin/python`; change `run.sh` and `controllers/micromouse/runtime.ini` if needed. Webots supplies the `controller` module; do not install it from PyPI.

To run the earlier baseline:

```bash
MICROMOUSE_ALGORITHM=astar bash run.sh
```

## Train and evaluate

```bash
python tools/train_q_learning.py --episodes 6000 --seed 42 --evaluation-runs 100
python -m unittest discover -s tests -v
python tools/validate_simulation.py --runs 3 --algorithm q_learning
```

Training runs in a continuous XY kinematic option model using the actual world wall geometry and exact swept-disc collision checks. Reset and landing positions have uniform noise of ±0.015 m. No A*, shortest-path demonstrations, action masks, or route-label supervision are used. With 80% probability, an episode in the first 80% of training starts at a random nonterminal cell; all remaining episodes start at the real mission start. This exploring-starts curriculum helps sparse goal rewards propagate. Evaluation always uses the mission start, a separate random seed stream and epsilon 0. Time-limit truncation keeps the Bellman bootstrap; goal termination removes it.

Rewards: +100 on entering the goal zone, -1 per valid move, -10 for a blocked option (pose unchanged). Alpha 0.2, gamma 0.99, epsilon max(0.03, 0.999^episode), 600 decisions per episode. The Q-table has 1,024 values. A scene fingerprint rejects stale models after geometry changes. Once the learned policy enters the 2 × 2 goal room, the existing servo docks at its centre. There is no A* fallback if Q-learning fails.

Training produces `reports/q_learning/training.csv`, `summary.json`, and `evaluations.json`. Physics validation independently runs actual wheel motors in Webots and writes `reports/q_learning_webots/validation_results.json`, three logs, and screenshots. The validator checks final goal error, physical clearance, and every recorded trajectory segment. `results/latest_run.json` records an interactive run.

## Recorded results

For seed 42: 100/100 noisy kinematic evaluations succeed without blocked options; the deterministic greedy policy reaches the goal zone in 41 options (16.4 m). Seeds 43 and 44 independently learn the same 41-option result on this fixed maze. These seeds do not represent different maze layouts.

Three deterministic Webots resets reach the goal centre within 0.034 m, with minimum disc clearance above 0.087 m. See the JSON logs for actual measurements. Repeated deterministic physics runs establish repeatability on this map, not statistical generalization to unknown worlds or hardware.

## Maze generation

`worlds/maze_layout.json` controls size, seed, loops, start, and goal. To change it:

```bash
python -m pip install Pillow
python tools/build_world.py
python tools/train_q_learning.py
```

Reload the generated world. Training reads wall geometry from `worlds/world.wbt`, so save/regenerate it before retraining. The earlier files directly under `reports/validation_results.json` and `reports/webots_run_*.log` belong to a previous E-puck baseline and are not current Q-learning evidence.
