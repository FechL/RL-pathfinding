# Micromouse maze design notes

## Current simulation

The arena is a 16 × 16 grid with 40 cm cells, for a total floor area of 6.4 × 6.4 m. A seeded depth-first maze generator creates dead ends; ten additional passages add alternate routes. The goal occupies four cells at the corner opposite the start. With the current seed, the generated world contains 213 internal wall segments.

A* searches an eight-connected 5 cm lattice. Every edge and smoothed path segment is checked against the rectangular wall footprints and the mouse radius. The baseline controller drives two wheel motors and uses GPS plus an IMU to follow the route. Four infrared distance sensors and wheel encoders are exposed on the robot for future sensor-based control.

The current planner reads the complete maze map through Webots Supervisor. The infrared readings are not used to select the route, and there is no RL policy or training loop. A future RL controller can instead use the distance sensors, IMU, and odometry as observations and wheel velocities as actions.

## Evidence and limits

`reports/validation_results.json` and `reports/webots_run_*.log` are retained from the earlier E-puck prototype; rerun `tools/validate_simulation.py` before using them as evidence for the micromouse. The simulation uses static walls and a simplified micromouse model. Successful simulation runs do not establish transfer to competition hardware.
