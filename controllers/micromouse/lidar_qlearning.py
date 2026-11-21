"""Sparse tabular Q-learning from a 360-degree lidar and a known goal pose."""
from collections import deque
import json
import math
from pathlib import Path

import numpy as np


ACTION_NAMES = ('Maju', 'Lengkung kiri', 'Lengkung kanan',
                'Belok kiri', 'Belok kanan', 'Mundur', 'Berhenti')
ACTION_VW = ((.23, 0.), (.19, 1.15), (.19, -1.15),
             (.12, 2.7), (.12, -2.7), (-.11, 0.), (0., 0.))
LIDAR_SECTORS = ('Depan', 'Depan-kiri', 'Kiri', 'Belakang-kiri',
                 'Belakang', 'Belakang-kanan', 'Kanan', 'Depan-kanan')
SAFE_DISTANCE = .12
STATE_VERSION = 3


def safe_actions(lidar_sectors):
    """Mask only maneuvers whose travel direction is blocked by the scan.

    Each sector is already the minimum over a 45-degree wedge. Requiring three
    adjacent wedges to be clear was too conservative and could deadlock the
    robot at ordinary maze corners. Stop is an emergency fallback only.
    """
    sectors=np.asarray(lidar_sectors,dtype=float)
    allowed=[]
    if sectors[0]>=SAFE_DISTANCE:
        allowed.append(0)
    if sectors[1]>=SAFE_DISTANCE:
        allowed.extend((1,3))
    if sectors[7]>=SAFE_DISTANCE:
        allowed.extend((2,4))
    if sectors[4]>=SAFE_DISTANCE:
        allowed.append(5)
    return tuple(allowed) if allowed else (6,)


def aggregate_lidar(scan, maximum=4.):
    """Reduce hundreds of range returns to minimum distance in eight sectors.

    The Webots scanner starts on the robot's right and proceeds
    counter-clockwise. Sector zero is the front, then front-left, left, etc.
    """
    values = np.asarray(scan, dtype=float).reshape(-1)
    values = np.nan_to_num(values, nan=maximum, posinf=maximum, neginf=0.)
    values = np.clip(values, 0., maximum)
    angles = -math.pi/2 + np.arange(len(values))*2*math.pi/max(1, len(values))
    sectors = np.floor(angles/(math.pi/4)+.5).astype(int) % 8
    minima = np.full(8, maximum, dtype=float)
    np.minimum.at(minima, sectors, values)
    return minima.astype(np.float32)


def aggregate_point_cloud(points, maximum=4.):
    """Aggregate Webots local-frame lidar points into the same 8 sectors."""
    minima=np.full(8,float(maximum),dtype=float)
    for point in points:
        x,y=float(point.x),float(point.y)
        distance=math.hypot(x,y)
        if not math.isfinite(distance):
            continue
        sector=int(math.floor(math.atan2(y,x)/(math.pi/4)+.5))%8
        minima[sector]=min(minima[sector],distance)
    return np.clip(minima,0.,maximum).astype(np.float32)


def discretize_state(pose, goal, lidar_sectors, previous_action=None):
    """Create a compact Markov state; no map or route labels are consumed."""
    pose = np.asarray(pose, dtype=float)
    goal = np.asarray(goal, dtype=float)
    delta = goal-pose[:2]
    distance = float(np.linalg.norm(delta))
    bearing = (math.atan2(delta[1], delta[0])-pose[2]+math.pi)%(2*math.pi)-math.pi
    bearing_bin = min(15, int((bearing+math.pi)/(2*math.pi/16)))
    distance_bin = int(np.digitize(distance, [.15, .35, .7, 1.2, 2., 3.5]))
    lidar_bins = tuple(int(x) for x in np.digitize(np.asarray(lidar_sectors), [SAFE_DISTANCE, .45]))
    last_action = len(ACTION_NAMES) if previous_action is None else int(previous_action)
    return (bearing_bin, distance_bin, *lidar_bins, last_action)


class TabularLidarQ:
    """Sparse Q table with standard off-policy Q-learning updates."""
    def __init__(self, alpha=.16, gamma=.995, seed=42):
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError('alpha must be in (0, 1], gamma in [0, 1]')
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.rng = np.random.default_rng(seed)
        self.q = {}
        self.visits = {}
        self.state_order = []
        self.recent = deque(maxlen=10000)

    @staticmethod
    def key(state):
        return ','.join(map(str, state))

    @staticmethod
    def unkey(key):
        return tuple(int(value) for value in key.split(','))

    def values(self, state):
        key = self.key(state)
        if key not in self.q:
            self.q[key] = np.zeros(len(ACTION_NAMES), dtype=np.float32)
            self.visits[key] = 0
            self.state_order.append(key)
        return self.q[key]

    def choose(self, state, epsilon=0., valid_actions=None):
        values = self.values(state)
        key = self.key(state)
        self.visits[key] = self.visits.get(key, 0)+1
        self.recent.append(key)
        valid=np.arange(len(ACTION_NAMES)) if valid_actions is None else np.asarray(valid_actions,dtype=int)
        if not len(valid) or np.any(valid<0) or np.any(valid>=len(ACTION_NAMES)):
            raise ValueError('At least one valid Q action is required')
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        allowed_values=values[valid]
        best=valid[allowed_values>=allowed_values.max()-1e-8]
        return int(self.rng.choice(best))

    def update(self, state, action, reward, next_state, terminal, valid_next_actions=None):
        current = self.values(state)
        if terminal:
            target=float(reward)
        else:
            following=self.values(next_state)
            valid=(np.arange(len(ACTION_NAMES)) if valid_next_actions is None
                   else np.asarray(valid_next_actions,dtype=int))
            if not len(valid) or np.any(valid<0) or np.any(valid>=len(ACTION_NAMES)):
                raise ValueError('At least one valid next Q action is required')
            target=float(reward)+self.gamma*float(following[valid].max())
        old = float(current[action])
        current[action] += self.alpha*(target-old)
        return old, float(current[action]), target

    def recent_rows(self, limit=12):
        rows=[]; seen=set()
        for key in reversed(self.recent):
            if key in seen:
                continue
            seen.add(key)
            rows.append((self.unkey(key), self.q[key].copy(), self.visits.get(key, 0)))
            if len(rows)>=limit:
                break
        return rows

    def save(self, path, metadata=None):
        payload=dict(format_version=STATE_VERSION, algorithm='tabular_q_lidar',
                     alpha=self.alpha, gamma=self.gamma,
                     state=('relative goal bearing 16 bins, goal distance 7 bins, '
                            '8 lidar sector minimums quantized to 3 levels, previous action'),
                     lidar_sector_order=LIDAR_SECTORS,
                     lidar_bins={'near_m':SAFE_DISTANCE,'medium_m':.45,'maximum_range_m':4.},
                     actions=[dict(name=name,linear_m_s=vw[0],yaw_rad_s=vw[1])
                              for name,vw in zip(ACTION_NAMES,ACTION_VW)],
                     rows={key:[float(value) for value in values]
                           for key,values in self.q.items()},
                     visits=self.visits)
        if metadata:
            payload['training']=metadata
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(payload,separators=(',',':'))+'\n')
        temporary.replace(path)

    @classmethod
    def load(cls, path, seed=0):
        payload=json.loads(Path(path).read_text())
        if payload.get('format_version')!=STATE_VERSION or payload.get('algorithm')!='tabular_q_lidar':
            raise ValueError('This LiDAR Q-table uses an older state schema; retrain with `bash run.sh train` before simulation')
        if tuple(item['name'] for item in payload['actions'])!=ACTION_NAMES:
            raise ValueError('Q-table action schema does not match this robot')
        agent=cls(payload['alpha'],payload['gamma'],seed)
        expected=[(item['linear_m_s'],item['yaw_rad_s']) for item in payload['actions']]
        if any(not np.allclose(a,b,atol=1e-9) for a,b in zip(expected,ACTION_VW)):
            raise ValueError('Q-table action velocities do not match this robot')
        agent.q={key:np.asarray(row,dtype=np.float32) for key,row in payload['rows'].items()}
        agent.visits={key:int(value) for key,value in payload.get('visits',{}).items()}
        agent.state_order=list(agent.q)
        return agent,payload
