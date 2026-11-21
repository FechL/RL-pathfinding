"""Continuous differential-drive surrogate controlled by discrete Q actions."""
import math

import numpy as np

from lidar_qlearning import ACTION_NAMES, ACTION_VW, aggregate_lidar, discretize_state
from planner import ROBOT_RADIUS, SAFETY_MARGIN

FORWARD_LIDAR_WEIGHT=1.5
GPS_TRAVEL_COST_PER_METER=.5


class LidarQEnvironment:
    def __init__(self, navigation, *, decision_seconds=.16, timeout_seconds=120.):
        self.nav=navigation
        self.config=navigation.config
        self.decision_seconds=float(decision_seconds)
        self.substep=self.config.dt/4
        self.substeps=max(1,round(self.decision_seconds/self.substep))
        self.max_steps=max(1,math.ceil(timeout_seconds/self.decision_seconds))
        self.pose=np.zeros(3)
        self.wheels=np.zeros(2)
        self.lidar_scan=np.full(128,4.,dtype=np.float32)
        self.lidar_sectors=np.full(8,4.,dtype=np.float32)
        self.steps=0
        self.episode_return=0.
        self.travelled_distance=0.
        self.previous_action=None
        self.finished=False
        self.last_collision=False
        self.last_success=False
        self.last_reward_parts={}
        self.last_state=None
        self.last_position=self.nav.start.copy()

    def reset(self, heading=0.):
        self.pose=np.array([*self.nav.start,float(heading)],dtype=float)
        if not self.nav.free(self.pose[:2],self.pose[:2]):
            raise ValueError('Start pose intersects a wall')
        self.wheels=np.zeros(2)
        self.lidar_scan=self.nav.lidar_scan(self.pose)
        self.lidar_sectors=aggregate_lidar(self.lidar_scan)
        self.steps=0
        self.episode_return=0.
        self.travelled_distance=0.
        self.previous_action=None
        self.finished=False
        self.last_collision=False
        self.last_success=False
        self.last_reward_parts={}
        self.last_position=self.pose[:2].copy()
        self.last_state=discretize_state(self.pose,self.nav.goal,self.lidar_sectors)
        return self.last_state

    def step(self, action):
        if self.finished:
            raise RuntimeError('Reset before beginning another episode')
        if not 0<=int(action)<len(ACTION_NAMES):
            raise ValueError('Unknown tabular Q action')
        action=int(action)
        old_state=self.last_state
        old_heading=float(self.pose[2])
        old_position=self.pose[:2].copy()
        old_distance=float(np.linalg.norm(self.nav.goal-self.pose[:2]))
        linear,angular=ACTION_VW[action]
        collision=False
        for _ in range(self.substeps):
            self.wheels=self.config.wheels_for_velocity(linear,angular,self.wheels,self.substep)
            candidate=self.config.integrate(self.pose,self.wheels,self.substep)
            if not self.nav.free(self.pose[:2],candidate[:2],ROBOT_RADIUS+SAFETY_MARGIN):
                collision=True
                self.wheels[:]=0.
                break
            self.pose=candidate
        self.steps+=1
        self.lidar_scan=self.nav.lidar_scan(self.pose)
        self.lidar_sectors=aggregate_lidar(self.lidar_scan)
        travelled_delta=float(np.linalg.norm(self.pose[:2]-old_position))
        self.travelled_distance+=travelled_delta
        distance_from_base=float(np.linalg.norm(self.pose[:2]-self.nav.start))
        new_distance=float(np.linalg.norm(self.nav.goal-self.pose[:2]))
        success=new_distance<=.09
        progress_reward=8.*(old_distance-new_distance)
        time_cost=-.025
        gps_travel_cost=-GPS_TRAVEL_COST_PER_METER*travelled_delta
        frontal_clearance=float(np.min(self.lidar_sectors[[7,0,1]]))
        if action in (1,3):
            maneuver_clearance=float(self.lidar_sectors[1])
        elif action in (2,4):
            maneuver_clearance=float(self.lidar_sectors[7])
        elif action==5:
            maneuver_clearance=float(self.lidar_sectors[4])
        elif action==6:
            maneuver_clearance=float(np.min(self.lidar_sectors[[0,1,4,7]]))
        else:
            maneuver_clearance=float(self.lidar_sectors[0])
        lidar_weight=FORWARD_LIDAR_WEIGHT if action in (0,1,2,3,4) else 1.
        close_wall_penalty=-.45*lidar_weight*float(
            np.clip((.24-maneuver_clearance)/.24,0.,1.))
        yaw_change=(self.pose[2]-old_heading+math.pi)%(2*math.pi)-math.pi
        spinning_penalty=-.18 if abs(yaw_change)>.30 and progress_reward<.015 else 0.
        collision_penalty=-30. if collision else 0.
        goal_reward=60. if success else 0.
        stalled_penalty=-.10 if np.linalg.norm(self.pose[:2]-self.last_position)<.002 else 0.
        self.last_position=self.pose[:2].copy()
        reward=(progress_reward+time_cost+gps_travel_cost+close_wall_penalty+collision_penalty+
                goal_reward+stalled_penalty+spinning_penalty)
        terminal=bool(collision or success)
        truncated=bool(self.steps>=self.max_steps and not terminal)
        next_state=discretize_state(self.pose,self.nav.goal,self.lidar_sectors,action)
        self.last_state=next_state
        self.previous_action=action
        self.episode_return+=reward
        self.finished=terminal or truncated
        self.last_collision=collision
        self.last_success=success
        self.last_reward_parts=dict(progress=progress_reward,time=time_cost,
                                    gps_travel=gps_travel_cost,
                                    close_wall=close_wall_penalty,collision=collision_penalty,
                                    goal=goal_reward,stalled=stalled_penalty,
                                    spinning=spinning_penalty,lidar_weight=lidar_weight,
                                    maneuver_clearance=maneuver_clearance)
        info=dict(collision=collision,success=success,distance_to_goal_m=new_distance,
                  distance_from_base_m=distance_from_base,
                  travelled_distance_m=self.travelled_distance,
                  front_clearance_m=frontal_clearance,action=ACTION_NAMES[action],
                  maneuver_clearance_m=maneuver_clearance,lidar_weight=lidar_weight,
                  reward_parts=self.last_reward_parts.copy())
        return next_state,float(reward),terminal,truncated,info
