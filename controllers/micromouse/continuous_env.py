"""Gymnasium environment for direct continuous differential-drive learning."""
import math
import gymnasium as gym
import numpy as np
from planner import ROBOT_RADIUS, SAFETY_MARGIN


class ContinuousDriveEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, navigation, training=True, max_steps=None):
        super().__init__()
        self.nav = navigation
        self.config = navigation.config
        self.collision_radius = ROBOT_RADIUS + (SAFETY_MARGIN if navigation.guided else 0.)
        self.training = training
        self.max_steps = max_steps or round(self.config.max_seconds/self.config.dt)
        self.action_space = gym.spaces.Box(-1.,1.,shape=(2,),dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.,1.,shape=(26,),dtype=np.float32)
        self.curriculum = 1.
        self.mission_probability = 0.
        self.pose = np.zeros(3)
        self.wheels = np.zeros(2)
        self.steps = 0
        self.finished = False
        self.episode_index = 0
        self.episode_return = 0.
        self.last_success = False
        self.last_collision = False
        self.step_callback = None
        self.teacher_actor = None
        self.teacher_anchor_weight = 0.

    def sample_near_goal(self, radius, *, visible=False):
        for _ in range(1000):
            angle = self.np_random.uniform(-math.pi, math.pi)
            distance = self.np_random.uniform(.25, radius)
            position = self.nav.goal + distance*np.array([math.cos(angle), math.sin(angle)])
            target = self.nav.goal if visible else position
            if self.nav.free(position, target):
                return position
        raise RuntimeError('No safe curriculum start sampled near the goal')

    def goal_distance(self):
        if self.nav.guided:
            return self.nav.guidance(self.pose[:2])[1]
        return float(np.linalg.norm(self.pose[:2]-self.nav.goal))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        position = self.nav.start.copy()
        heading = float(options.get('heading',0.))
        if not self.nav.guided:
            if options.get('local'):
                position = self.sample_near_goal(.65, visible=True)
                heading = self.np_random.uniform(-math.pi, math.pi)
            elif self.training and 'position' not in options and self.np_random.random() >= self.mission_probability:
                near = self.np_random.random() < .65
                position = self.sample_near_goal(.8 if near else .8+3.2*self.curriculum, visible=near)
                heading = self.np_random.uniform(-math.pi, math.pi)
        elif self.training and 'position' not in options and self.np_random.random() >= self.mission_probability:
            candidates = self.nav.reachable[(self.nav.costs[self.nav.reachable]>.15) & (self.nav.costs[self.nav.reachable] < max(1.2,self.curriculum*float(self.nav.costs.max())))]
            state = int(self.np_random.choice(candidates))
            position = self.nav.centers[state].copy()
            target,_ = self.nav.guidance(position)
            heading = math.atan2(target[1]-position[1],target[0]-position[0])+self.np_random.uniform(-math.pi,math.pi)
        if 'position' in options:
            position = np.array(options['position'],dtype=float)
        for _ in range(30):
            jitter = self.np_random.uniform(-.025,.025,2) if (self.training or options.get('jitter')) else np.zeros(2)
            if self.nav.free(position+jitter,position+jitter):
                position += jitter; break
        else: raise ValueError('Unsafe reset')
        self.pose = np.array([*position,heading])
        self.wheels = np.zeros(2)
        self.steps = 0
        self.finished = False
        self.episode_index += 1
        self.episode_return = 0.
        self.last_success = self.last_collision = False
        self.potential = self.goal_distance()
        return self.nav.observation(self.pose,self.wheels),{}

    def step(self, action):
        if self.finished: raise RuntimeError('Reset after termination or truncation')
        action = np.clip(np.asarray(action,dtype=float),-1.,1.)
        teacher_action = None
        if self.teacher_actor is not None and self.teacher_anchor_weight:
            observation = self.nav.observation(self.pose,self.wheels)
            teacher_action = self.teacher_actor(observation)
        previous = self.wheels.copy()
        # Ramp at the same 16 ms motor-update rate used in Webots.
        motor_dt = self.config.dt/4
        collision = False
        for substep in range(4):
            self.wheels = self.config.wheels(action,self.wheels,motor_dt)
            candidate = self.config.integrate(self.pose,self.wheels,motor_dt)
            if not self.nav.free(self.pose[:2],candidate[:2],self.collision_radius):
                collision = True; self.wheels[:] = 0; break
            self.pose = candidate
            if self.step_callback is not None and substep < 3:
                self.step_callback(self, dt=motor_dt)
        self.steps += 1
        potential = self.goal_distance()
        success = np.linalg.norm(self.pose[:2]-self.nav.goal) <= self.config.goal_tolerance
        # Progress shapes exploration; elapsed time makes slower traversal costly.
        reward = 12.*(self.potential-potential)-self.config.dt
        reward -= .003*float(np.square((self.wheels-previous)/self.config.wheel_limit).sum())
        if teacher_action is not None:
            reward -= self.teacher_anchor_weight*float(np.square(action-teacher_action).sum())
        if collision: reward -= 15.
        if success: reward += 30.
        self.episode_return += reward
        self.last_success, self.last_collision = bool(success), collision
        self.potential = potential
        terminated = bool(collision or success)
        truncated = bool(self.steps>=self.max_steps and not terminated)
        self.finished = terminated or truncated
        if self.step_callback is not None:
            self.step_callback(self, force=self.finished, dt=motor_dt)
        return self.nav.observation(self.pose,self.wheels),float(reward),terminated,truncated,dict(
            is_success=bool(success),collision=collision,time_s=self.steps*self.config.dt,
            goal_error_m=float(np.linalg.norm(self.pose[:2]-self.nav.goal)))
