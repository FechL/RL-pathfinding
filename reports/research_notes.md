# Navigation arena design notes

Reviewed on October 18, 2026. These references informed the arena design. The implemented controller uses A*.

1. **Farama Foundation — Point Maze.** The [official documentation](https://robotics.farama.org/envs/maze/point_maze/) describes goal navigation with continuous observations and customizable maps, including open arenas and more complex mazes. This project uses a continuous floor, open routes, and obstacles that require small detours. Arena dimensions, safety margins, and obstacle counts are project choices.
2. **Safety-Gymnasium — Goal.** The [official documentation](https://safety-gymnasium.readthedocs.io/en/latest/environments/safe_navigation/goal.html) describes Goal0 through Goal2 with different object configurations. This inspired controlled obstacle variety and measuring both goal completion and clearance. This project's obstacles do not implement the benchmark's hazard costs.
3. **Florensa et al. 2017 — Reverse Curriculum Generation for Reinforcement Learning.** The [paper](https://arxiv.org/abs/1707.05300) describes learning from starting states near the goal before progressing to harder states. A possible future extension is to gradually increase the distance between start and goal. Curriculum learning and RL training are not implemented.
4. **Cyberbotics — Supervisor API R2025a.** The [official documentation](https://cyberbotics.com/doc/reference/supervisor?version=R2025a) describes node access, field updates, image export, and simulation control. This project uses these features to move the robot directly and automate checks. The results validate path geometry.

## Implementation choices

The 3.6 × 2.6 m arena contains eight fixed obstacles in four categories. Their footprints cover about 6.78% of the floor. The smallest gap between obstacles is about 0.280 m, and the smallest gap between an obstacle and a wall is about 0.350 m. The planner uses a robot radius of 0.037 m and an extra safety margin of 0.030 m. The layout avoids dead ends, moving obstacles, and random placements that could make the goal unreachable.

A* searches eight directions on a 0.05 m grid. The planner then simplifies waypoints while checking each full path segment. Positions and movement use continuous coordinates. This does not represent a continuous RL action space or guarantee the globally shortest continuous path.

## Evidence and limitations

`validation_results.json` contains measurements from repeated runs of the same layout. `webots_run_*.log` stores simulator output, and `figures/` stores screenshots and the map. Unit tests check alternative routes, small changes to start and goal positions, obstacle crossings, circular drum footprints, arena boundaries, and cases with no route. Passing these checks does not prove transfer to a physical robot or policy generalization.
