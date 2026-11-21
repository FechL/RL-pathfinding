"""Continuous differential-drive surrogate for full-route Q-learning."""
import math

import numpy as np

from lidar_qlearning import (ACTION_NAMES, ACTION_VW, aggregate_lidar,
                             discretize_state, waypoint_bearing)
from planner import ROBOT_RADIUS, SAFETY_MARGIN

FORWARD_LIDAR_WEIGHT=1.5
GPS_TRAVEL_COST_PER_METER=.5
COLLISION_PENALTY=-100.
GOAL_REWARD=120.
WAYPOINT_REWARD=4.
WAYPOINT_TOLERANCE=.055
GOAL_TOLERANCE=.09


class LidarQEnvironment:
    def __init__(self, navigation, *, decision_seconds=.16, timeout_seconds=300.):
        self.nav=navigation
        self.config=navigation.config
        # Keep the mission base/goal fixed while the controller retargets
        # successive DFS waypoints within one start-to-goal episode.
        self.base_start=np.asarray(navigation.start,dtype=float).copy()
        self.base_goal=np.asarray(navigation.goal,dtype=float).copy()
        self.start=self.base_start.copy()
        self.goal=self.base_goal.copy()
        self.decision_seconds=float(decision_seconds)
        self.substep=self.config.dt/4
        self.substeps=max(1,round(self.decision_seconds/self.substep))
        self.max_steps=max(1,math.ceil(timeout_seconds/self.decision_seconds))
        self.pose=np.zeros(3)
        self.wheels=np.zeros(2)
        self.lidar_scan=np.full(128,4.,dtype=np.float32)
        self.lidar_sectors=np.full(5,4.,dtype=np.float32)
        self.steps=0
        self.episode_return=0.
        self.travelled_distance=0.
        self.previous_action=None
        self.finished=False
        self.last_collision=False
        self.last_success=False
        self.last_reward_parts={}
        self.last_state=None

    def reset(self, start=None, goal=None, heading=0.):
        self.start=self.base_start.copy() if start is None else np.asarray(start,dtype=float)
        self.goal=self.base_goal.copy() if goal is None else np.asarray(goal,dtype=float)
        self.pose=np.array([*self.start,float(heading)],dtype=float)
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
        self.last_state=discretize_state(self.pose,self.goal,self.lidar_sectors)
        return self.last_state

    def set_waypoint(self, goal):
        """Retarget the next route waypoint without resetting robot motion."""
        if self.finished:
            raise RuntimeError('Cannot change waypoints after an episode ends')
        self.goal=np.asarray(goal,dtype=float)
        self.last_state=discretize_state(self.pose,self.goal,self.lidar_sectors)
        return self.last_state

    def step(self, action, *, final_goal=True, scanning=False):
        if self.finished:
            raise RuntimeError('Reset before beginning another episode')
        if not 0<=int(action)<len(ACTION_NAMES):
            raise ValueError('Unknown tabular Q action')
        action=int(action)
        old_bearing=abs(waypoint_bearing(self.pose,self.goal))
        old_position=self.pose[:2].copy()
        old_distance=float(np.linalg.norm(self.goal-self.pose[:2]))
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
        distance_from_base=float(np.linalg.norm(self.pose[:2]-self.base_start))
        new_distance=float(np.linalg.norm(self.goal-self.pose[:2]))
        waypoint_reached=not scanning and not collision and new_distance<=(GOAL_TOLERANCE if final_goal else WAYPOINT_TOLERANCE)
        success=bool(waypoint_reached and final_goal)
        waypoint_reward=WAYPOINT_REWARD if waypoint_reached and not final_goal else 0.
        progress_reward=0. if scanning else 4.*(old_distance-new_distance)
        time_cost=-.025
        gps_travel_cost=-GPS_TRAVEL_COST_PER_METER*travelled_delta
        frontal_clearance=float(np.min(self.lidar_sectors[[4,0,1]]))
        if action==1:
            maneuver_clearance=float(np.min(self.lidar_sectors[[0,1]]))
        elif action==2:
            maneuver_clearance=float(np.min(self.lidar_sectors[[0,4]]))
        elif action in (3,4,5):
            maneuver_clearance=float(self.lidar_sectors.min())
        else:
            maneuver_clearance=float(self.lidar_sectors[0])
        lidar_weight=FORWARD_LIDAR_WEIGHT if linear>0 else 1.
        close_wall_penalty=-.45*lidar_weight*float(
            np.clip((.24-maneuver_clearance)/.24,0.,1.))
        alignment_gain=old_bearing-abs(waypoint_bearing(self.pose,self.goal))
        alignment_reward=0. if scanning else .20*alignment_gain
        collision_penalty=COLLISION_PENALTY if collision else 0.
        goal_reward=GOAL_REWARD if success else 0.
        stalled_penalty=-.10 if not scanning and travelled_delta<.002 and alignment_gain<.05 else 0.
        reward=(progress_reward+time_cost+gps_travel_cost+close_wall_penalty+
                collision_penalty+goal_reward+stalled_penalty+alignment_reward)
        reward+=waypoint_reward
        terminal=bool(collision or success)
        truncated=bool(self.steps>=self.max_steps and not terminal)
        next_state=discretize_state(self.pose,self.goal,self.lidar_sectors)
        self.last_state=next_state
        self.previous_action=action
        self.episode_return+=reward
        self.finished=terminal or truncated
        self.last_collision=collision
        self.last_success=success
        self.last_reward_parts=dict(progress=progress_reward,time=time_cost,
                                    gps_travel=gps_travel_cost,
                                    close_wall=close_wall_penalty,collision=collision_penalty,
                                    goal=goal_reward,waypoint=waypoint_reward,
                                    stalled=stalled_penalty,
                                    alignment=alignment_reward,lidar_weight=lidar_weight,
                                    maneuver_clearance=maneuver_clearance)
        info=dict(collision=collision,success=success,waypoint_reached=waypoint_reached,
                  distance_to_goal_m=new_distance,
                  distance_from_base_m=distance_from_base,
                  travelled_distance_m=self.travelled_distance,
                  front_clearance_m=frontal_clearance,action=ACTION_NAMES[action],
                  maneuver_clearance_m=maneuver_clearance,lidar_weight=lidar_weight,
                  reward_parts=self.last_reward_parts.copy())
        return next_state,float(reward),terminal,truncated,info
