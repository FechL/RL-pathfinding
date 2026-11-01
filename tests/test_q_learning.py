"""Checks of collision semantics, Bellman targets and learned continuous policy."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'controllers/micromouse'))
sys.path.insert(0,str(ROOT/'tools'))
from planner import Obstacle,segment_is_free
from q_learning import ContinuousMaze,q_update,load_policy,save_policy,rollout
from train_q_learning import read_world

class QLearningTests(unittest.TestCase):
    def test_terminal_target_does_not_bootstrap(self):
        q=[[0.]*4,[999.]*4]
        old,new,target=q_update(q,0,0,100,1,True)
        self.assertEqual(target,100);self.assertEqual(new,20)

    def test_truncation_keeps_bootstrap(self):
        q=[[0.]*4,[10.]*4]
        _,new,target=q_update(q,0,0,-1,1,False)
        self.assertAlmostEqual(target,8.9);self.assertAlmostEqual(new,1.78)

    def test_swept_motion_cannot_jump_through_thin_wall(self):
        env=ContinuousMaze((.2,.2),(.8,.8),[Obstacle('bar','wall',(.4,.2),(.02,.4))],(0,1.6,0,1.6),.4)
        env.position=(.2,.2)
        p,r,done,info=env.step(0,random.Random(0),jitter=0)
        self.assertEqual(p,(.2,.2));self.assertEqual(r,-10)
        self.assertFalse(done);self.assertTrue(info['collision'])

    def test_continuous_positions_are_not_snapped_to_centres(self):
        env=ContinuousMaze((.2,.2),(1.4,1.4),[],(0,1.6,0,1.6),.4)
        env.reset(random.Random(1))
        p,_,_,_=env.step(0,random.Random(2))
        self.assertNotEqual(p,env.center(env.state(p)))

    def test_saved_policy_reaches_goal_and_all_segments_are_safe(self):
        env=read_world();q,_=load_policy(ROOT/'models/q_learning.json',env)
        result=rollout(env,q,seed=12,jitter=.015)
        self.assertTrue(result['success']);self.assertEqual(result['collisions'],0)
        for a,b in zip(result['path'],result['path'][1:]):
            self.assertTrue(segment_is_free(a,b,env.walls,env.bounds))

    def test_stale_map_policy_is_rejected(self):
        env=read_world();env.walls=env.walls[:-1]
        with self.assertRaisesRegex(ValueError,'incompatible'):
            load_policy(ROOT/'models/q_learning.json',env)

if __name__=='__main__':unittest.main()
