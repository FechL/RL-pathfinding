"""Gymnasium adapter around the Micromouse LiDAR search and motion model."""
import gymnasium as gym
import numpy as np

from lidar_drl import OBSERVATION_SIZE, drive_command, observation
from lidar_q_env import LidarQEnvironment
from lidar_qlearning import STOP_ACTION, WaypointRecovery
from maze_search import LidarMazeSearch, synthetic_hits


class LidarMission(gym.Env):
    def __init__(self, navigation, timeout):
        self.motion = LidarQEnvironment(navigation, timeout_seconds=timeout)
        self.action_space = gym.spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1., 1., (OBSERVATION_SIZE,), dtype=np.float32)
        self.episode_index = 0
        self.route_selector = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mission = options or {}
        self.replay = mission.get('route', ())
        if self.route_selector is not None:
            self.replay = self.route_selector()
        heading = mission.get('heading', 0. if self.replay else None)
        if heading is None:
            heading = self.np_random.uniform(-np.pi, np.pi)
        env = self.motion
        env.reset(heading=heading)
        self.episode_index += 1
        env.episode_index = self.episode_index
        self.collision_count = 0
        self.last_transition = None
        search_seed = mission.get('search_seed')
        if search_seed is None:
            search_seed = int(self.np_random.integers(2**32))
        self.route_search = LidarMazeSearch(
            env.base_start, env.base_goal, env.nav.bounds, env.nav.cell_size,
            seed=search_seed, replay=self.replay)
        self.target = None
        self._select_target()
        return self._observation(), {}

    def _select_target(self):
        env = self.motion
        self.target = self.route_search.next_target(env.pose, synthetic_hits(env.lidar_scan))
        self.recovery = WaypointRecovery()
        if self.target is not None:
            env.set_waypoint(self.target)

    def _observation(self):
        env = self.motion
        return observation(env.pose, self.target, env.lidar_sectors, env.wheels,
                           env.config.wheel_limit)

    def allowed_actions(self):
        env = self.motion
        return self.recovery.actions(env.pose, self.target, env.lidar_sectors,
                                     env.steps * env.decision_seconds)

    def step(self, action):
        return self._step(action, tabular=False)

    def tabular_step(self, action):
        return self._step(action, tabular=True)

    def _step(self, action, *, tabular):
        env = self.motion
        recovering = env.collision_recovery
        scanning = self.target is None
        exhausted = scanning and self.route_search.scan_bearing is None
        if exhausted:
            return self._observation(), 0., True, False, self._info(False, True)
        velocity = None
        if recovering:
            category = STOP_ACTION
        elif scanning:
            category = self.route_search.scan_action(env.lidar_sectors)
        elif tabular:
            category = action
        else:
            category, velocity = drive_command(action, env.lidar_sectors, self.allowed_actions())
        _, reward, terminal, truncated, info = env.step(
            category, final_goal=self.route_search.final, scanning=scanning, velocity=velocity)
        self.last_transition = dict(info,reward_step=reward)
        if info['collision']:
            self.collision_count += 1
            self.recovery.active = True
        if not (terminal or truncated or recovering or info['collision']) and (
                scanning or info['waypoint_reached']):
            self._select_target()
        exhausted = self.target is None and self.route_search.scan_bearing is None
        return self._observation(), reward, terminal or exhausted, truncated, self._info(
            info['success'], exhausted)

    def _info(self, success, exhausted):
        env = self.motion
        return dict(success=success, collision_count=self.collision_count,
                    search_exhausted=exhausted, steps=env.steps, reward=env.episode_return,
                    duration_s=env.steps * env.decision_seconds,
                    distance_m=env.travelled_distance,
                    replay=self.replay,
                    route=self.route_search.route.copy() if success else [])
