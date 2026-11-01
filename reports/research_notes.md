# Continuous environment Q-learning experiment

Updated 1 November 2026. The default route now comes from tabular Q-learning. The A* planner is retained only for explicit baseline selection and geometry helper reuse. Learning and rollout do not invoke `plan_path`.

Physical state is continuous XY/yaw; the policy bins XY at 0.40 m and chooses four cell-sized motion options. The low-level GPS/IMU wheel controller handles yaw. Training is performed in a kinematic XY option environment with continuous reset/landing noise, known world geometry and exact swept-disc collision checks. No wheel dynamics are learned during this stage. The first 4,800 episodes use exploring starts with probability 0.8; the last 1,200 always use the mission start.

Current evidence is under `reports/q_learning` and `reports/q_learning_webots`. The former records training and noisy kinematic evaluation; the latter records actual differential-drive physics runs. Seeds 42, 43, and 44 are training seeds on the same maze, not separate layouts. The three Webots runs are deterministic resets, not independent randomized trials.

Earlier E-puck validation under the top-level reports folder and older videos are historical baseline material and must not be presented as learned-policy evidence. See the new report for limitations and reproduction commands.
