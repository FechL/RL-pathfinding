"""Shared continuous motor dynamics, map guidance and SAC observations.

The graph is a known-map navigation aid, not the policy's action space.
The actor outputs two real numbers on every control tick; it never outputs a
cell, waypoint index or a discrete turn. Training uses a kinematic surrogate.
"""
from dataclasses import asdict, dataclass
import heapq
import hashlib
import json
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
from planner import ROBOT_RADIUS, SAFETY_MARGIN, segment_is_free
from q_learning import ContinuousMaze, scene_signature


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class ControlConfig:
    # Actuator response in both simulators. This is independent of the
    # observation and action contract stored with older policy exports.
    wheel_response_time_s: ClassVar[float] = .08
    dt: float = .064
    wheel_radius: float = .027
    axle_track: float = .104
    wheel_limit: float = 17.
    wheel_acceleration: float = 45.
    max_linear: float = .38
    max_angular: float = 4.0
    goal_tolerance: float = .05
    max_seconds: float = 180.

    def wheels(self, action, previous, dt=None):
        """Bounded real-valued v/w commands and common wheel slew limits."""
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError('Expected two finite continuous actions')
        action = np.clip(action, -1, 1)
        v = (action[0] + 1) * .5 * self.max_linear
        w = action[1] * self.max_angular
        desired = np.array([(v-w*self.axle_track/2)/self.wheel_radius,
                            (v+w*self.axle_track/2)/self.wheel_radius])
        desired *= min(1., self.wheel_limit/max(np.max(np.abs(desired)), 1e-9))
        previous = np.asarray(previous)
        delta = desired-previous
        interval = self.dt if dt is None else dt
        response = -math.expm1(-interval/self.wheel_response_time_s)
        slew = self.wheel_acceleration*interval/max(np.max(np.abs(delta)), 1e-9)
        # Use the same blend for both wheels to approach the target together.
        return previous + min(response, slew)*delta

    def integrate(self, pose, wheels, dt):
        v = self.wheel_radius * (wheels[0]+wheels[1])/2
        w = self.wheel_radius * (wheels[1]-wheels[0])/self.axle_track
        x,y,h = pose
        if abs(w) < 1e-8:
            x += v*math.cos(h)*dt; y += v*math.sin(h)*dt
        else:
            x += v/w*(math.sin(h+w*dt)-math.sin(h))
            y -= v/w*(math.cos(h+w*dt)-math.cos(h))
        return np.array([x,y,wrap(h+w*dt)])


class NavigationMap:
    """Continuous observations and geometry shared by training and Webots.

    The hidden-goal mode exposes only continuous pose, wheel/sensor readings,
    and a yellow-camera flag. Its goal-cell geometry stays inside the sensor
    simulator and is omitted from observations, rewards and policy metadata.
    """
    ray_angles = np.linspace(-math.pi, math.pi, 16, endpoint=False)
    ray_range = .8

    def __init__(self, start, goal, walls, bounds, cell_size, config=None, *, guided=True,
                 hidden_goal=False, goal_cells=None):
        self.config = config or ControlConfig()
        self.hidden_goal = bool(hidden_goal)
        self.guided = bool(guided) and not self.hidden_goal
        self.start = np.array(start, dtype=float)
        self.goal = None if self.hidden_goal else np.array(goal, dtype=float)
        self.goal_cells = frozenset(tuple(map(int, cell)) for cell in (goal_cells or ()))
        self.walls, self.bounds, self.cell_size = walls, bounds, cell_size
        # A dummy start pose keeps the compatibility signature independent of
        # the hidden target position. The actor never receives this value.
        self.maze = ContinuousMaze(start, start if self.hidden_goal else goal, walls, bounds, cell_size)
        self.signature = scene_signature(self.maze)
        self.wall_centers = np.array([w.center for w in walls], dtype=float).reshape(-1,2)
        self.wall_halves = np.array([w.size for w in walls], dtype=float).reshape(-1,2)/2
        if any(w.circular for w in walls):
            raise ValueError('Continuous navigation currently supports rectangular maze walls')
        self.observation_size = 13 if self.hidden_goal else 26
        if not guided:
            if not self.hidden_goal:
                return
        if self.hidden_goal:
            return
        self.centers = np.array([self.maze.center(i) for i in range(self.maze.state_count)])
        self.neighbors = [[] for _ in self.centers]
        for i,p in enumerate(self.centers):
            ix,iy = i % self.maze.width, i // self.maze.width
            for dx,dy in ((1,0),(0,1),(-1,0),(0,-1)):
                x,y = ix+dx,iy+dy
                if 0 <= x < self.maze.width and 0 <= y < self.maze.height:
                    j = y*self.maze.width+x
                    if segment_is_free(p,self.centers[j],walls,bounds):
                        self.neighbors[i].append(j)
        self.costs = np.full(len(self.centers), np.inf)
        self.next = np.full(len(self.centers), -1, dtype=int)
        queue = []
        for i,p in enumerate(self.centers):
            if np.linalg.norm(p-self.goal) <= cell_size*1.5 and segment_is_free(p,self.goal,walls,bounds):
                self.costs[i] = np.linalg.norm(p-self.goal)
                heapq.heappush(queue,(self.costs[i],i))
        while queue:
            cost,i = heapq.heappop(queue)
            if cost > self.costs[i]+1e-9: continue
            for j in self.neighbors[i]:
                new = cost+cell_size
                if new < self.costs[j]-1e-9:
                    self.costs[j],self.next[j] = new,i
                    heapq.heappush(queue,(new,j))
        self.reachable = np.flatnonzero(np.isfinite(self.costs))
        if len(self.reachable) != len(self.centers):
            raise ValueError('Continuous training currently requires reachable cell centres')

    def local_geometry(self, position):
        nearby = np.all(np.abs(self.wall_centers-position) <= self.wall_halves+self.ray_range+.1,axis=1)
        return self.wall_centers[nearby], self.wall_halves[nearby]

    def free(self, a, b, radius=ROBOT_RADIUS+SAFETY_MARGIN):
        """Vectorized swept-disc conservative AABB test, including boundaries."""
        a,b = np.asarray(a),np.asarray(b)
        xmin,xmax,ymin,ymax = self.bounds
        if np.any(np.minimum(a,b) < np.array([xmin,ymin])+radius) or np.any(np.maximum(a,b) > np.array([xmax,ymax])-radius):
            return False
        nearby = np.all(np.abs(self.wall_centers-(a+b)/2) <= self.wall_halves+np.abs(b-a)/2+radius,axis=1)
        centers,halves = self.wall_centers[nearby],self.wall_halves[nearby]
        if not len(centers):
            return True
        lower,upper = centers-halves-radius,centers+halves+radius
        delta = b-a
        safe_delta = np.where(np.abs(delta)<1e-12,1.,delta)
        first,last = (lower-a)/safe_delta,(upper-a)/safe_delta
        lo,hi = np.minimum(first,last),np.maximum(first,last)
        for axis in (0,1):
            if abs(delta[axis])<1e-12:
                inside = (a[axis]>=lower[:,axis]) & (a[axis]<=upper[:,axis])
                lo[:,axis] = np.where(inside,-np.inf,np.inf)
                hi[:,axis] = np.where(inside,np.inf,-np.inf)
        return not np.any(np.maximum(0,lo.max(axis=1)) <= np.minimum(1,hi.min(axis=1)))

    def guidance(self, position):
        state = self.maze.state(position)
        # Within a cell, use visible neighbour cost for a continuous distance.
        candidates = [state]+self.neighbors[state]
        valid = [i for i in candidates if self.free(position,self.centers[i])]
        if not valid:
            return self.centers[state], float(self.costs[state]+np.linalg.norm(position-self.centers[state]))
        i = min(valid,key=lambda s:self.costs[s]+np.linalg.norm(position-self.centers[s]))
        potential = self.costs[i]+np.linalg.norm(position-self.centers[i])
        target = self.centers[i]
        if self.free(position,self.goal) and np.linalg.norm(position-self.goal)<.8:
            return self.goal, float(np.linalg.norm(position-self.goal))
        # Look ahead around consecutive straight cells and shallow corners.
        for _ in range(4):
            j = self.next[i]
            if j < 0 or np.linalg.norm(self.centers[j]-position) > .65 or not self.free(position,self.centers[j],ROBOT_RADIUS+SAFETY_MARGIN+.012):
                break
            target,i = self.centers[j],j
        return target,float(potential)

    def rays(self, pose):
        pose = np.asarray(pose)
        angles = pose[2]+self.ray_angles
        directions = np.stack([np.cos(angles),np.sin(angles)],axis=1)
        centers,halves = self.local_geometry(pose[:2])
        xmin,xmax,ymin,ymax = self.bounds
        # Include arena border as four thin slabs.
        centers = np.concatenate([centers,[[xmin-.05,(ymin+ymax)/2],[xmax+.05,(ymin+ymax)/2],[(xmin+xmax)/2,ymin-.05],[(xmin+xmax)/2,ymax+.05]]])
        halves = np.concatenate([halves,[[.05,(ymax-ymin)/2],[.05,(ymax-ymin)/2],[(xmax-xmin)/2,.05],[(xmax-xmin)/2,.05]]])
        d = np.where(np.abs(directions)<1e-10,1e-10,directions)
        first = (centers[None,:,:]-halves[None,:,:]-pose[None,None,:2])/d[:,None,:]
        last = (centers[None,:,:]+halves[None,:,:]-pose[None,None,:2])/d[:,None,:]
        near = np.minimum(first,last).max(axis=2)
        far = np.maximum(first,last).min(axis=2)
        hits = np.where((far>=near)&(far>=0),np.maximum(near,0),self.ray_range)
        return np.clip(hits.min(axis=1),0,self.ray_range)/self.ray_range

    def ir_ranges(self, pose, maximum=.4):
        """Continuous range readings for the five physical IR sensor poses."""
        pose = np.asarray(pose, dtype=float)
        offsets = np.array([[.052, .026], [.052, -.026], [0., .053],
                            [-.052, 0.], [0., -.053]])
        relative_angles = np.array([-.20, .20, math.pi/2, math.pi, -math.pi/2])
        c, s = math.cos(pose[2]), math.sin(pose[2])
        rotation = np.array([[c, -s], [s, c]])
        origins = pose[:2] + offsets @ rotation.T
        angles = pose[2] + relative_angles
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        centers, halves = self.local_geometry(pose[:2])
        xmin, xmax, ymin, ymax = self.bounds
        centers = np.concatenate([centers, [[xmin-.05, (ymin+ymax)/2],
            [xmax+.05, (ymin+ymax)/2], [(xmin+xmax)/2, ymin-.05],
            [(xmin+xmax)/2, ymax+.05]]])
        halves = np.concatenate([halves, [[.05, (ymax-ymin)/2], [.05, (ymax-ymin)/2],
            [(xmax-xmin)/2, .05], [(xmax-xmin)/2, .05]]])
        safe = np.where(np.abs(directions) < 1e-10, 1e-10, directions)
        first = (centers[None, :, :] - halves[None, :, :] - origins[:, None, :]) / safe[:, None, :]
        last = (centers[None, :, :] + halves[None, :, :] - origins[:, None, :]) / safe[:, None, :]
        near = np.minimum(first, last).max(axis=2)
        far = np.maximum(first, last).min(axis=2)
        hits = np.where((far >= near) & (far >= 0), np.maximum(near, 0), maximum)
        return np.clip(hits.min(axis=1), 0, maximum) / maximum

    def yellow_detected(self, pose):
        if not self.goal_cells:
            return False
        pose = np.asarray(pose, dtype=float)
        x = pose[0] + .045 * math.cos(pose[2])
        y = pose[1] + .045 * math.sin(pose[2])
        ix = math.floor((x - self.bounds[0]) / self.cell_size)
        iy = math.floor((y - self.bounds[2]) / self.cell_size)
        return (ix, iy) in self.goal_cells

    def observation(self, pose, wheels, *, ir_values=None, yellow_detected=None,
                    motion_stagnation=0.):
        pose = np.asarray(pose)
        if self.hidden_goal:
            xmin, xmax, ymin, ymax = self.bounds
            xy = 2*(pose[:2]-[xmin, ymin])/np.array([xmax-xmin, ymax-ymin])-1
            ir = self.ir_ranges(pose) if ir_values is None else np.asarray(ir_values, dtype=float)
            yellow = self.yellow_detected(pose) if yellow_detected is None else bool(yellow_detected)
            return np.concatenate([np.asarray(wheels)/self.config.wheel_limit, ir, xy,
                                   [math.sin(pose[2]), math.cos(pose[2]),
                                    np.clip(motion_stagnation, 0., 1.), float(yellow)]])\
                .astype(np.float32)
        target = self.guidance(pose[:2])[0] if self.guided else self.goal
        delta = target-pose[:2]
        c,s = math.cos(pose[2]),math.sin(pose[2])
        body = np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]])/.8
        xmin,xmax,ymin,ymax = self.bounds
        xy = 2*(pose[:2]-[xmin,ymin])/np.array([xmax-xmin,ymax-ymin])-1
        goal_delta = self.goal-pose[:2]
        goal_body = np.array([c*goal_delta[0]+s*goal_delta[1],-s*goal_delta[0]+c*goal_delta[1]])/math.hypot(xmax-xmin,ymax-ymin)
        return np.concatenate([np.clip(body,-1,1),np.asarray(wheels)/self.config.wheel_limit,self.rays(pose),
                               xy,[s,c],goal_body]).astype(np.float32)

    def metadata(self):
        if self.hidden_goal:
            return dict(format_version=1, algorithm='hidden_goal', signature=self.signature,
                        config=asdict(self.config), observation_version=5,
                        observation=('continuous normalized wheel speeds, five continuous IR ranges, '
                                     'continuous GPS x/y, sin/cos heading, continuous stagnation signal, '
                                     'yellow camera flag'),
                        observation_fields=('wheel_left_norm','wheel_right_norm',
                                            'ir_front_left_norm','ir_front_right_norm',
                                            'ir_left_norm','ir_rear_norm','ir_right_norm',
                                            'x_norm','y_norm','heading_sin','heading_cos',
                                            'stagnation_fraction','yellow_detected'),
                        observation_size=self.observation_size,
                        action='two continuous normalized values: forward speed and yaw rate',
                        guidance='none; terminal success comes from the yellow color sensor',
                        scope='continuous SAC; kinematic sensor simulation; validate Webots physics separately')
        return dict(format_version=1,algorithm='sac' if self.guided else 'rl',signature=self.signature,
                    config=asdict(self.config),observation_version=2 if self.guided else 3,
                    action='continuous normalized forward velocity and yaw rate',
                    guidance='known-map Dijkstra local bearing; no prescribed motor trajectory' if self.guided else
                    'no route guidance; goal vector, pose, wheel speeds and simulated local range rays',
                    scope='kinematic pretraining; Webots physics must be validated separately')


def read_policy(path, navigation, algorithm='sac'):
    """Load the exported SAC actor or the greedy mean of a NAF Q-network."""
    raw = Path(path).read_text()
    data = json.loads(raw)
    data['artifact_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
    expected = navigation.metadata()
    expected['algorithm'] = algorithm
    for key in ('format_version','algorithm','signature','config','observation_version'):
        if data.get(key) != expected[key]:
            raise ValueError(f'Incompatible continuous policy ({key}); retrain it')
    layers = [(np.asarray(x['weight']),np.asarray(x['bias'])) for x in data['layers']]
    if not layers or layers[0][0].shape[1] != navigation.observation_size or layers[-1][0].shape[0] != 2:
        raise ValueError('Malformed continuous actor dimensions')
    previous = navigation.observation_size
    for weight,bias in layers:
        if weight.ndim!=2 or weight.shape[1]!=previous or bias.shape!=(weight.shape[0],) or not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise ValueError('Malformed continuous actor layer')
        previous = weight.shape[0]
    def actor(observation):
        x = np.asarray(observation)
        for index,(weight,bias) in enumerate(layers):
            x = weight@x+bias
            if index<len(layers)-1: x = np.maximum(x,0)
        return np.tanh(x)
    return actor,data
