# Micromouse Continuous SAC with a Hidden Goal

The default learner uses **continuous Soft Actor-Critic (SAC)** in the fixed
16 × 16 Webots maze. The actor outputs real-valued forward speed and yaw rate;
it receives five continuous IR ranges, wheel speeds, continuous GPS position,
heading, a continuous signal for prolonged lack of motion, and a one-bit
yellow-camera reading. The policy and reward do not use
the goal coordinates or goal distance. It succeeds when the camera detects
the yellow floor tile. A collision ends the episode with a penalty; a separate
180-second timeout prevents an episode from running forever. After 1.5 seconds
without translation, the agent receives an anti-spin penalty and continuous
stagnation signal. Nearby walls reduce that penalty so the robot can pivot in
a dead end.

```bash
bash run.sh train                         # training animation + live metrics
bash run.sh train --algorithm rl          # same continuous hidden-goal SAC
bash run.sh train --render-speed 16       # faster visible preview
bash run.sh train --no-render             # fastest training, logs still saved
bash run.sh train --steps 100000 --exploration-fraction 0.75
bash run.sh sim                           # Webots robot run; hidden-goal actor
bash run.sh sim --fast                    # Webots simulation speed mode
bash run.sh graphs                        # graphs for the newest training run
```

Training has two SAC phases. The first uses higher entropy for **exploration**;
the second lowers target entropy for **exploitation**. Actions remain continuous
in both phases. The live local window shows the robot moving from the mission
start, its trajectory, continuous state values, all five IR readings, the
yellow sensor bit, stagnation signal, reward, steps, success/collision rates, critic Q estimate,
actor/critic loss, and entropy. Q here is SAC's critic estimate, not a Q-table.
There is no web service: `run.sh` starts the local simulator and monitor.

Each run is kept under `reports/continuous_hidden_goal/runs/`. It saves the
continuous state/action samples in `state_space.csv`, episode reward/steps and
outcomes in `training.monitor.csv`, SAC metrics in `training_metrics.csv`, and
plots including `figures/state_space.png`, `figures/episode_steps.png`, reward,
success/collision, Q, losses, and entropy. State-space plots also show the
anti-spin signal. The policy candidate is saved inside that run's
`models/` folder. To try a candidate in Webots without replacing the default:

```bash
bash run.sh sim --policy reports/continuous_hidden_goal/runs/<run>/models/hidden_goal_actor.json --fast
```

`--activate` installs a policy as the default only after all fixed-seed
kinematic evaluations reach the goal. Training and evaluation use a kinematic
surrogate; validate successful policies in Webots physics separately. The
training preview speed (`--render-speed`) changes wall-clock playback only;
`--no-render` removes display delays. Webots simulation can use `--fast`.

## Legacy guided continuous SAC

The older known-goal SAC and guided fine-tuning modes remain available by
selecting `sac` or `guided_rl` explicitly. They use a goal vector and shaped
distance reward, unlike the default hidden-goal task.

```bash
bash run.sh train --algorithm sac --warm-start
bash run.sh train --algorithm guided_rl
bash run.sh sim --algorithm sac
```

## Tugas 8 — Continuous Deep Q-Learning

Select **Normalized Advantage Functions (NAF)** for the Tugas 8 controller.
NAF uses the same 26 real-valued observations and two real-valued forward
velocity and yaw-rate actions as SAC. The legacy Q-table still represents
the discrete baseline. See [Gu et al. (2016)](https://proceedings.mlr.press/v48/gu16.html).

```bash
bash run.sh sim --algorithm naf
bash run.sh train --algorithm naf --render-speed 1
bash run.sh train --algorithm naf --steps 30000 --eval-every 2500 --seed 42 --no-render
bash run.sh graphs --report-dir reports/naf
```

The NAF learner represents `Q(s,a) = V(s) − ½ ‖L(s)ᵀ(a − μ(s))‖²`.
Its greedy action is `μ(s)` in the continuous action box, and its Bellman
target bootstraps with the target network's `V(s′)`. A goal or collision
ends the bootstrap. A time limit leaves the bootstrap active. Gaussian
exploration and replay supply continuous actions for the updates.

The recorded run initializes `μ` from the exported SAC controller and uses a
small teacher anchor during training. The value and precision parameters start
randomly. Use `--no-warm-start` to omit teacher initialization and anchoring.
The published comparison starts from the warm-start checkpoint before any
NAF updates. The best mean in the fixed kinematic evaluation fell from
52.651 s to 52.587 s, a difference of one control interval. Three seeded
evaluations at each of the 13 checkpoints reached the goal.

The training window shows the actual 2D kinematic rollout and reward. Each
run saves PNG plots, a training monitor and evaluation checkpoints under
`reports/naf/runs/`. With `--activate`, the best trained checkpoint that
reaches the goal in every evaluation replaces `models/naf_actor.json`.
Without that flag, use an absolute `MICROMOUSE_POLICY` to test a specific
export without changing the deployed policy.

```bash
xvfb-run -a ~/miniconda3/bin/python tools/validate_simulation.py \
  --algorithm naf --runs 3 --keep-telemetry --report-dir reports/tugas8/naf_webots
```

Recorded training evidence is in `reports/naf/`. Webots physics results
and the comparison of the same turn across controllers are in `reports/tugas8/`.
In the original report, NAF completed all three Webots resets in 53.008 s,
matching the warm-start policy's time. The matched-speed tabular controller took 98.000 s
and followed a different route in part of the maze. On the shared corner,
NAF took 1.317 s with 6.67 cm clearance and the tabular controller took
2.444 s with 7.89 cm clearance. NAF kept moving forward throughout the
samples classified as turning between the gates.

The smoother controller uses the same NAF weights and completed all three
Webots resets in 52.352 s, with 6.15 cm minimum clearance. In the same corner,
RMS yaw jerk from the IMU fell from 272.3 to 162.0 rad/s³. The updated
telemetry and before/after plot are in `reports/tugas8/smooth_naf_webots/`.

Regenerate the corner and route comparisons from their inputs:

```bash
.venv/bin/python tools/evaluate_tugas8.py webots \
  reports/tugas8/{naf_webots,tabular_webots,sac_webots}/telemetry.json
.venv/bin/python tools/evaluate_tugas8.py routes models/naf_actor.json
```

In the paired kinematic route test, the same trained NAF policy completed the
mission in a mean of 52.203 s with the alternate guide and 52.587 s with the
default guide. The comparison changes one graph edge in the Dijkstra guide
while retaining the same wall geometry and reset seeds. The guide selects the
corridor and NAF controls the continuous motion within it.

## What the agent learns

Both continuous policies make a fresh decision every 16 ms in Webots.
Training still requests actions every 64 ms and applies four 16 ms motor substeps.
The two real-valued outputs
request forward velocity (0–0.38 m/s) and yaw rate (−4–4 rad/s). Both can be
nonzero simultaneously. Shared motor conversion enforces a 17 rad/s wheel
speed limit and 45 rad/s² wheel acceleration limit. An 80 ms exponential
wheel response softens abrupt changes in the policy's requests. There
is no compulsory “stop, rotate 90°, move one cell” action sequence.

The 26 continuous observations contain a local navigation bearing, wheel
velocities, 16 geometric range rays, XY position, sine/cosine of heading, and
the goal vector in robot coordinates. The complete map is known. Range rays
are computed from that map and the current GPS/IMU pose; the four physical
IR sensors are recorded separately, not substituted for those rays.

A Dijkstra distance field over known maze connectivity supplies local guidance
and a dense progress reward. This guide biases exploration toward short
routes; it does not prescribe wheel commands or a timed trajectory. **This
is map-assisted continuous control, not autonomous mapping or proof of the
globally fastest route.** SAC can adjust its trajectory, speed and turns,
but establishing the fastest alternative corridor requires explicit route
comparisons and more training. Changing the map requires retraining.

The training reward is `12 × geodesic progress − elapsed seconds − wheel slew
penalty`, with +30 for reaching within 5 cm of the goal centre and −15 for
collision. Faster successful traversal is favoured; collisions terminate the
episode. A time limit is truncation, so SAC retains its value bootstrap.
Random continuous starting positions and headings provide a curriculum.

## Run

Install Webots R2025a, Miniconda and the lightweight runtime requirements:

```bash
~/miniconda3/bin/python -m pip install -r requirements.txt
bash run.sh
```

Webots uses `~/miniconda3/bin/python` via
`controllers/micromouse/runtime.ini`. Running the exported actor needs only
NumPy and OpenCV, not PyTorch. The model is `models/continuous_actor.json`;
map, controller configuration and observation version are checked at load.
No fallback controller is silently selected if the model is incompatible.

Set `MICROMOUSE_NO_GUI=1` to hide the OpenCV map. Webots still needs a display.
The map shows the actual driven trajectory; no full route is precomputed
from the continuous actor. `results/latest_run.json` records pose, actions,
measured and commanded wheel speeds, IR readings, elapsed time, trajectory,
clearance, and motion metrics. A stalled or unsafe episode stops and reports
failure.

Explicit older baselines:

```bash
MICROMOUSE_ALGORITHM=q_learning bash run.sh
MICROMOUSE_ALGORITHM=astar bash run.sh
```

## Train or continue training

Use an isolated environment for the training dependencies. For a CPU machine:

```bash
~/miniconda3/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-training.txt
bash run.sh train --steps 300000 --seed 42
```

Training uses Stable Baselines3 SAC with two critics and a squashed Gaussian
actor (64/64 hidden units). It learns in a continuous differential-drive
kinematic surrogate, with exact arc integration and swept collision checks
at 16 ms substeps. It does **not** train in Webots rigid-body physics. Real
motor slip and contact dynamics must be checked separately in Webots.

`--warm-start` explicitly uses 30,000 continuous feedback driving examples
to teach basic steering, then warms both critics for 4,000 updates. SAC
continues with its Q/entropy objective plus a behaviour-cloning regularizer
(weight 10) to retain safe basic driving while adjusting speed and turns.
The feedback reference is used only during training. The deployed actor is
a neural network, with no reference-action mixing or safety action mask.
Use `bash run.sh train --no-warm-start` to train pure SAC from scratch; it can take substantially more
experience. The included run is not evidence of learning without examples.

Each training invocation creates its own timestamped folder under
`reports/continuous/runs/`. The terminal prints its exact path. Use
`--run-dir reports/continuous/runs/my_run` to name a new run explicitly;
existing training logs cannot be overwritten. Runs and their full models
remain local and are excluded from Git.

Checkpoints are saved every 10,000 steps in that run's `models/` folder.
Every 20,000 new steps, three mission-start evaluations rank fully successful
candidates by mean time. The selected portable actor is saved as
`models/continuous_actor.json` inside the run, alongside the full SAC
checkpoints and final replay buffer. `--activate` explicitly copies a fully
successful best actor to the project's default Webots model. Without that
flag, training preserves the existing Webots actor. To try a new run without
changing the default, set `MICROMOUSE_POLICY` to its actor's absolute path
when running `bash run.sh`.

Continue the previously included full checkpoint with:

```bash
bash run.sh train --steps 200000 --resume models/sac_continuous.zip
```

For later runs, use the full checkpoint path inside that run's `models/`
folder. Training episode data and seeded evaluation trajectories are saved
inside each run. Historical evidence for the included actor remains under
`reports/continuous/`. Evaluation uses the actual mission start, deterministic
actor output, and separate seeds with start-position perturbations. A saved
checkpoint is evidence of training, not automatically evidence of success.

## See the training graphs

Regenerate the included training figures without retraining:

```bash
bash run.sh graphs
```

The newest hidden-goal run is selected by default. Its `figures/` folder has
`training_progress.png`, `episode_steps.png`, `episode_outcomes.png`,
`q_learning.png`, and `state_space.png`. These show kinematic training and
evaluation; validate policies in Webots physics separately. Historical runs
under `reports/continuous/` used known-goal guidance and have different logs.

All training runs from the same shell entry point: `bash run.sh train`.
It opens a local 2D maze window showing the robot's training movements, trail,
episode, reward, continuous state and sensor values, and collision/yellow status.
The phase label changes from exploration to exploitation. The default preview
speed is 8x simulation time for SAC and real time for `bash run.sh train --algorithm naf`.
Use `bash run.sh train --render-speed 1` for real-time SAC preview,
or `bash run.sh train --no-render` for fast training without a window.
Press Q/Esc or close the window to stop; during SAC, partial training is
saved as a run candidate and evaluated.
This is a kinematic training preview, not Webots 3D physics training.

Training prints progress to the terminal and saves local PNG graphs; no web
service or browser is needed. `bash run.sh` runs the trained robot in Webots,
and `bash run.sh graphs` regenerates the saved training figures.

PNGs refresh after each evaluation and at the end. To refresh them every five
seconds for a specific run:

```bash
bash run.sh graphs --report-dir reports/continuous/runs/my_run --watch
```

The faint lines show individual episodes; thick lines show a rolling mean over
up to 20 episodes. Hidden-goal training always starts at the mission start; a
short episode may be a collision. Check success rate and state-space plots as
well as reward and episode length. The recorded results later in this README
describe earlier known-goal SAC/NAF runs, not the new hidden-goal task.

## Verify and compare

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/train_continuous.py --evaluate-only
~/miniconda3/bin/python tools/validate_simulation.py --runs 3 --algorithm sac
~/miniconda3/bin/python tools/validate_simulation.py --runs 1 --algorithm q_learning --report-dir reports/continuous_baseline_webots
```

For a baseline with the same 0.38 m/s forward and 4 rad/s yaw speed budgets:

```bash
MICROMOUSE_BASELINE_SPEED=0.38 MICROMOUSE_BASELINE_TURN_SPEED=7.7037037037 xvfb-run -a ~/miniconda3/bin/python tools/validate_simulation.py --runs 1 --algorithm q_learning --report-dir reports/continuous_matched_baseline_webots
```

On a headless Linux machine, prefix the Webots validation command with
`xvfb-run -a`. Physics validation runs isolated project copies, checks goal
error, physical wall clearance and every recorded trajectory segment, and
saves logs/screenshots under `reports/sac_webots/`.

Compare successful runs using `simulation_duration_s`, `travelled_length_m`,
`goal_error_m`, `min_physical_clearance_m`, `moving_turn_fraction`, stopped
fraction and peak wheel acceleration. Turning fraction counts yaw rates
above 0.4 rad/s and forward speed above 0.05 m/s; it does not by itself prove
optimal cornering. Three deterministic resets show repeatability, not
statistical generalization to other mazes or hardware.

## Recorded continuous-control results

The included actor was selected after reward-driven SAC fine-tuning, using
30,000 continuous driving examples, critic warmup and a BC regularizer. The
selected checkpoint has 140,000 accumulated environment steps; training
continued to 220,000 and retained the faster fully successful checkpoint.
Its seeded kinematic mean time is 52.651 s, compared with 53.184 s directly
after the demonstration warm start. See `reports/continuous/checkpoints.json`.

Three deterministic Webots runs with the exact exported actor all succeed
in **53.008 s**, with at least **0.061 m physical clearance** and goal error
at most **0.0483 m**. The previous default takes **137.888 s**. With matched
forward/yaw speed budgets the old controller takes **98.000 s**, making this
continuous configuration about **45.9% faster** on this fixed maze. The old
baseline has no acceleration governor and uses a tighter final waypoint
tolerance, so this is a controller-configuration benchmark, not a controlled
claim that one RL algorithm is universally superior.

Commanded wheel acceleration is bounded at 45 rad/s². During samples with
yaw rates above 0.4 rad/s, forward commands exceed 0.05 m/s in 99.7% of samples;
the matched legacy baseline has zero moving-turn samples. Motion summaries
use commanded wheel speeds consistently across controllers; full runtime
telemetry also records measured encoder speeds. Results and the actor's
SHA-256 fingerprint are recorded in `reports/continuous/comparison.json`.

All 26 unit/integration tests pass, including arc integration, swept
collision checks, Gymnasium semantics, exported actor equivalence and a
complete mission with the included actor. The existing Word report is
historical and has not been rewritten to represent these SAC results.

## Maze and legacy experiments

`worlds/maze_layout.json` controls geometry and seed. Regenerate and retrain:

```bash
~/miniconda3/bin/python tools/build_world.py
.venv/bin/python tools/train_continuous.py --steps 300000 --warm-start
```

The old option-based learner is retained in `q_learning.py`, with its trainer
at `tools/train_q_learning.py`, model at `models/q_learning.json`, and reports
in `reports/q_learning/` and `reports/q_learning_webots/`. It uses 256 discrete
cell states and four cardinal actions; these are not the new SAC controller.
Earlier reports and the Word document describe those historical experiments,
not the new controller's performance.
