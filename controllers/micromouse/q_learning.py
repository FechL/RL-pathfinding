"""Tabular Q-learning over continuous XY motion with cell-sized options.

Only geometry helpers are reused from planner.py. No search, demonstrations,
route labels, action masks, or A* fallback are used by this learner.
"""
import hashlib
import json
import math
import random
from pathlib import Path

from planner import ROBOT_RADIUS, SAFETY_MARGIN, distance, segment_is_free

ACTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
ACTION_NAMES = ('EAST', 'NORTH', 'WEST', 'SOUTH')


class ContinuousMaze:
    """Continuous point robot with swept-disc collision and finite motion options.

    A decision requests the next cell's centre plus continuous landing noise.
    A low-level servo is abstracted during training. Webots later executes
    these options with wheel motors, GPS and IMU. Heading is handled by that
    servo, so it is not a tabular state dimension. This is not wheel-level RL.
    """
    def __init__(self, start, goal, walls, bounds, cell_size):
        self.start, self.goal = tuple(start), tuple(goal)
        self.walls, self.bounds, self.cell_size = list(walls), tuple(bounds), cell_size
        self.width = round((bounds[1]-bounds[0])/cell_size)
        self.height = round((bounds[3]-bounds[2])/cell_size)
        self.state_count = self.width*self.height
        self.position = self.start
        # Spatial broad phase only: exact segment tests still decide collision.
        reach = cell_size*1.6 + ROBOT_RADIUS + SAFETY_MARGIN
        self.local_walls = []
        for state in range(self.state_count):
            p = self.center(state)
            self.local_walls.append([w for w in self.walls
                if abs(w.center[0]-p[0]) <= reach+w.size[0]/2
                and abs(w.center[1]-p[1]) <= reach+w.size[1]/2])

    def state(self, position):
        x = math.floor((position[0]-self.bounds[0])/self.cell_size)
        y = math.floor((position[1]-self.bounds[2])/self.cell_size)
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError('Position lies outside arena')
        return y*self.width+x

    def center(self, state):
        return (self.bounds[0]+(state % self.width+.5)*self.cell_size,
                self.bounds[2]+(state // self.width+.5)*self.cell_size)

    def terminal(self, position):
        # The existing goal is the open 2 x 2 cell zone, not a single bin.
        return (abs(position[0]-self.goal[0]) < self.cell_size
                and abs(position[1]-self.goal[1]) < self.cell_size)

    def reset(self, rng, state=None, jitter=.015):
        p = self.start if state is None else self.center(state)
        for _ in range(100):
            candidate = (p[0]+rng.uniform(-jitter,jitter),
                         p[1]+rng.uniform(-jitter,jitter))
            if segment_is_free(candidate,candidate,self.walls,self.bounds):
                self.position = candidate
                return self.position
        raise ValueError('Could not find a safe reset near the requested position')

    def step(self, action, rng, jitter=.015):
        if not 0 <= action < 4:
            raise ValueError('Unknown action')
        if self.terminal(self.position):
            return self.position, 0., True, {'collision': False}
        state = self.state(self.position)
        dx,dy = ACTIONS[action]
        ix,iy = state % self.width+dx, state // self.width+dy
        target = (self.bounds[0]+(ix+.5)*self.cell_size+rng.uniform(-jitter,jitter),
                  self.bounds[2]+(iy+.5)*self.cell_size+rng.uniform(-jitter,jitter))
        collision = not segment_is_free(self.position,target,self.local_walls[state],self.bounds)
        if collision:
            # Rejected options leave pose unchanged, just like a blocked command.
            return self.position, -10., False, {'collision': True}
        self.position = target
        done = self.terminal(target)
        return target, 100. if done else -1., done, {'collision': False}


def scene_signature(env):
    data = {'start':env.start,'goal':env.goal,'bounds':env.bounds,'cell':env.cell_size,
            'radius':ROBOT_RADIUS,'margin':SAFETY_MARGIN,
            'walls':sorted((w.name,*w.center,*w.size) for w in env.walls)}
    # World serialization and Supervisor can differ in floating point tails.
    def normalize(value):
        if isinstance(value,float): return round(value,6)
        if isinstance(value,(tuple,list)): return [normalize(v) for v in value]
        if isinstance(value,dict): return {k:normalize(v) for k,v in value.items()}
        return value
    return hashlib.sha256(json.dumps(normalize(data),sort_keys=True).encode()).hexdigest()


def q_update(q, state, action, reward, next_state, terminated, alpha=.2, gamma=.99):
    target = reward if terminated else reward+gamma*max(q[next_state])
    old = q[state][action]
    q[state][action] += alpha*(target-old)
    return old,q[state][action],target


def train(env, episodes=6000, seed=42, max_steps=600, alpha=.2, gamma=.99):
    if episodes < 1 or max_steps < 1:
        raise ValueError('Episode and step limits must be positive')
    rng = random.Random(seed)
    q = [[0.]*4 for _ in range(env.state_count)]
    resets = [s for s in range(env.state_count) if not env.terminal(env.center(s))]
    history, examples = [], []
    for episode in range(episodes):
        epsilon = max(.03, .999**episode)
        # Exploring starts propagate sparse goal rewards through a long maze.
        # Last 20% of training always begins at the actual mission start.
        random_start = episode < int(episodes*.8) and rng.random() < .8
        position = env.reset(rng,rng.choice(resets) if random_start else None)
        total,collisions,done = 0.,0,False
        for step in range(1,max_steps+1):
            state = env.state(position)
            if rng.random() < epsilon:
                action = rng.randrange(4)
            else:
                best = max(q[state]); action = rng.choice([a for a,v in enumerate(q[state]) if v == best])
            next_position,reward,done,info = env.step(action,rng)
            old,new,target = q_update(q,state,action,reward,env.state(next_position),done,alpha,gamma)
            if episode == 0 and step <= 8:
                examples.append(dict(step=step,state=state,action=ACTION_NAMES[action],reward=reward,
                                     old_q=old,new_q=new,target=target,terminated=done))
            position = next_position; total += reward; collisions += info['collision']
            if done: break
        # Time-limit truncation is not an absorbing MDP terminal: keep bootstrap.
        history.append(dict(episode=episode+1,epsilon=epsilon,reward=total,steps=step,
                            success=done,collisions=collisions,random_start=random_start))
    return q,history,examples


def rollout(env,q,start=None,seed=0,jitter=0.,max_steps=600):
    rng=random.Random(seed)
    if start is None: env.reset(rng,jitter=jitter)
    else:
        if not segment_is_free(start,start,env.walls,env.bounds):
            raise ValueError('Unsafe rollout start')
        env.position=tuple(start)
    path=[env.position]; collisions=0; total=0.
    done=env.terminal(env.position)
    for _ in range(0 if done else max_steps):
        state=env.state(env.position)
        action=max(range(4),key=lambda a:q[state][a])
        p,reward,done,info=env.step(action,rng,jitter)
        path.append(p); total+=reward; collisions+=info['collision']
        if done: break
    return dict(success=done,steps=len(path)-1,reward=total,collisions=collisions,path=path)


def save_policy(path,env,q,metadata):
    data=dict(format_version=1,algorithm='tabular_q_learning',signature=scene_signature(env),
              actions=ACTION_NAMES,q_table=q,metadata=metadata)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2)+'\n')


def load_policy(path,env):
    data=json.loads(Path(path).read_text())
    if data.get('format_version') != 1 or data.get('signature') != scene_signature(env):
        raise ValueError('Policy is incompatible with this scene; rerun tools/train_q_learning.py')
    q=data['q_table']
    if len(q) != env.state_count or any(len(row)!=4 or any(not math.isfinite(v) for v in row) for row in q):
        raise ValueError('Malformed Q-table')
    return q,data['metadata']


def policy_path(start,goal,walls,bounds,cell_size,policy_file):
    env=ContinuousMaze(start,goal,walls,bounds,cell_size)
    q,metadata=load_policy(policy_file,env)
    result=rollout(env,q,start=start)
    if not result['success'] or result['collisions']:
        raise RuntimeError('Learned greedy policy did not safely reach the goal zone')
    path=result['path']
    # Final docking inside the open goal room uses the existing pose servo.
    if distance(path[-1],goal)>1e-9:
        if not segment_is_free(path[-1],goal,walls,bounds):
            raise RuntimeError('Goal centre docking segment is blocked')
        path.append(tuple(goal))
    return path,set(path),metadata
