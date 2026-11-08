"""Continuous dynamics, collision and episode semantics regressions."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'controllers/micromouse'))
from continuous_control import ControlConfig,NavigationMap,read_policy
from planner import Obstacle,segment_is_free


class ContinuousTests(unittest.TestCase):
    def setUp(self):
        self.config=ControlConfig()
        self.nav=NavigationMap((.2,.2),(1.2,1.2),[],(0,1.6,0,1.6),.4)

    def test_actions_are_real_velocities_with_simultaneous_turning(self):
        wheels=self.config.wheels([.17,.31],[0,0])
        pose=self.config.integrate([.2,.2,0],wheels,.064)
        self.assertGreater(pose[0],.2)
        self.assertGreater(pose[1],.2)
        self.assertGreater(pose[2],0)
        other=self.config.wheels([.18,.32],[0,0])
        self.assertFalse(np.allclose(wheels,other))

    def test_slew_and_speed_limits(self):
        previous=np.zeros(2)
        for action in ([1,1],[-1,-1],[1,-1],[0,.35]):
            wheels=self.config.wheels(action,previous)
            self.assertLessEqual(float(np.max(np.abs(wheels))),17.+1e-9)
            self.assertLessEqual(float(np.max(np.abs(wheels-previous))),45*.064+1e-9)
            previous=wheels
        for bad in ([float('nan'),0],[0], [0,float('inf')]):
            with self.assertRaises(ValueError): self.config.wheels(bad,previous)

    def test_exact_arc_motion(self):
        pose=self.config.integrate([0,0,0],[3,9],.1)
        v=.027*6;w=.027*6/.104
        self.assertAlmostEqual(pose[0],v/w*math.sin(w*.1))
        self.assertAlmostEqual(pose[1],v/w*(1-math.cos(w*.1)))

    def test_wheels_approach_small_targets_without_a_step(self):
        action = [-.95, .05]
        start = np.zeros(2)
        first = self.config.wheels(action, start, .016)
        second = self.config.wheels(action, first, .016)
        whole = self.config.wheels(action, start, .032)
        np.testing.assert_allclose(whole, second)
        self.assertGreater(np.linalg.norm(first), np.linalg.norm(second-first))
        self.assertGreater(np.linalg.norm(second-first), 0.)

    def test_observations_change_with_heading_and_velocity(self):
        a=self.nav.observation([.2,.2,0],[0,0])
        b=self.nav.observation([.2,.2,math.pi/2],[2,3])
        self.assertEqual(a.shape,(26,))
        self.assertTrue(np.all(np.abs(a)<=1))
        self.assertFalse(np.allclose(a,b))

    def test_vectorized_collision_matches_reference(self):
        nav=NavigationMap((.2,.2),(1.4,1.4),[Obstacle('wall','wall',(.8,.8),(.05,.4))],(0,1.6,0,1.6),.4)
        rng=np.random.default_rng(5)
        for _ in range(100):
            a=rng.uniform(.1,1.5,2);b=rng.uniform(.1,1.5,2)
            self.assertEqual(nav.free(a,b),segment_is_free(a,b,nav.walls,nav.bounds))

    def test_numpy_actor_validates_map_and_config(self):
        data=self.nav.metadata()
        data['layers']=[dict(weight=np.zeros((2,26)).tolist(),bias=[.1,.2])]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'actor.json';path.write_text(json.dumps(data))
            actor,_=read_policy(path,self.nav)
            np.testing.assert_allclose(actor(np.zeros(26)),np.tanh([.1,.2]))
            data['signature']='stale';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):read_policy(path,self.nav)


try:
    from continuous_env import ContinuousDriveEnv
except ImportError:
    ContinuousDriveEnv=None


@unittest.skipIf(ContinuousDriveEnv is None,'Install requirements-training.txt for environment tests')
class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.nav=NavigationMap((.2,.2),(1.2,1.2),[],(0,1.6,0,1.6),.4)

    def test_time_limit_is_truncation(self):
        env=ContinuousDriveEnv(self.nav,training=False,max_steps=3)
        env.reset(seed=1)
        for _ in range(2):
            _,_,terminated,truncated,_=env.step([-1,0])
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        _,_,terminated,truncated,info=env.step([-1,0])
        self.assertFalse(terminated);self.assertTrue(truncated);self.assertFalse(info['is_success'])
        self.assertAlmostEqual(info['time_s'], .192)
        with self.assertRaises(RuntimeError):env.step([0,0])

    def test_collision_terminates_without_teleporting(self):
        env=ContinuousDriveEnv(self.nav,training=False)
        env.reset(options={'position':[.09,.2],'heading':math.pi})
        for _ in range(30):
            _,_,terminated,truncated,info=env.step([1,0])
            if terminated:break
        self.assertTrue(info['collision']);self.assertTrue(terminated);self.assertFalse(truncated)
        self.assertGreaterEqual(env.pose[0],.087)

    def test_unguided_motion_can_approach_wall_until_contact(self):
        nav = NavigationMap((.2,.2),(1.2,1.2),[],(0,1.6,0,1.6),.4,guided=False)
        env = ContinuousDriveEnv(nav,training=False)
        env.reset(options={'position':[.09,.2],'heading':math.pi})
        _,_,terminated,truncated,_=env.step([1,0])
        self.assertFalse(terminated or truncated)
        self.assertLess(env.pose[0],.087)
        for _ in range(4):
            _,_,terminated,_,info=env.step([1,0])
            if terminated:
                break
        self.assertTrue(terminated and info['collision'])
        self.assertGreaterEqual(env.pose[0],.075)

    def test_goal_termination(self):
        env=ContinuousDriveEnv(self.nav,training=False)
        env.reset(options={'position':[1.2,1.2]})
        _,_,terminated,truncated,info=env.step([-1,0])
        self.assertTrue(terminated);self.assertTrue(info['is_success']);self.assertFalse(truncated)

    def test_unguided_environment_uses_euclidean_progress_without_graph(self):
        nav = NavigationMap((.2,.2),(1.2,1.2),[],(0,1.6,0,1.6),.4,guided=False)
        self.assertFalse(hasattr(nav, 'next'))
        env = ContinuousDriveEnv(nav, training=False)
        observation, _ = env.reset(seed=2010, options={'local': True})
        self.assertTrue(env.observation_space.contains(observation))
        env.step([.5, .1])
        self.assertAlmostEqual(env.potential, np.linalg.norm(env.pose[:2]-nav.goal))
        self.assertEqual(nav.metadata()['observation_version'], 3)

    def test_unguided_fixed_resets_use_exact_start(self):
        nav = NavigationMap((.2,.2),(1.2,1.2),[],(0,1.6,0,1.6),.4,guided=False)
        env = ContinuousDriveEnv(nav, training=False, max_steps=1)
        for seed in (10, 20, 30):
            env.reset(seed=seed)
            np.testing.assert_array_equal(env.pose, [.2, .2, 0.])
            env.step([1., .7])

    def test_gym_api(self):
        from stable_baselines3.common.env_checker import check_env
        check_env(ContinuousDriveEnv(self.nav),warn=True)

    def test_export_matches_sac_actor(self):
        import torch
        from stable_baselines3 import SAC
        sys.path.insert(0,str(ROOT/'tools'))
        from train_continuous import export_actor
        torch.set_num_threads(1)
        env=ContinuousDriveEnv(self.nav)
        model=SAC('MlpPolicy',env,policy_kwargs=dict(net_arch=[16,16]),buffer_size=128,seed=9,device='cpu')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'actor.json'
            export_actor(model,self.nav,path,{'test':True})
            actor,_=read_policy(path,self.nav)
            for seed in range(8):
                obs,_=env.reset(seed=seed)
                expected,_=model.predict(obs,deterministic=True)
                np.testing.assert_allclose(actor(obs),expected,atol=2e-6)

    def test_included_continuous_actor_completes_mission(self):
        sys.path.insert(0,str(ROOT/'tools'))
        from train_continuous import navigation
        nav=navigation()
        actor,_=read_policy(ROOT/'models/continuous_actor.json',nav)
        env=ContinuousDriveEnv(nav,training=False)
        obs,_=env.reset(seed=1000)
        for _ in range(env.max_steps):
            obs,_,terminated,truncated,info=env.step(actor(obs))
            if terminated or truncated:break
        self.assertTrue(info['is_success'])
        self.assertFalse(info['collision'])


if __name__=='__main__':unittest.main()
