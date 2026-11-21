"""Regression checks for the masked Double DQN and real maze adapter."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'controllers/micromouse'),str(ROOT/'tools')]

from continuous_control import ControlConfig
from evaluate_lidar_dqn import maze_mission, rollout
from lidar_dqn import FEATURES, encode_observation, get_action_mask
from lidar_dqn_agent import DQNAgent, double_dqn_target, load_inference
from lidar_dqn_network import QNetwork
from lidar_dqn_replay import ReplayBuffer


def config():
    return dict(device='cpu', seed=42, hidden=[128,128], learning_rate=.001,
                capacity=8, warmup=2, batch_size=2, gamma=.98, gradient_clip=10.,
                target_interval=2)


class DQNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.state=np.zeros(len(FEATURES),np.float32)
        self.mask=np.array([False,True,False,True,False,False])

    def test_encoder_and_velocity(self):
        cfg=ControlConfig()
        obs=encode_observation([0,0,0],[-1,0],[.16,4,2,3,4],
                               [-cfg.wheel_limit,cfg.wheel_limit],cfg,4)
        self.assertEqual(obs.dtype,np.float32)
        self.assertEqual(obs.shape,(12,))
        np.testing.assert_allclose(obs[:3],[0,-1,.25],atol=1e-6)
        np.testing.assert_allclose(obs[8:],[0,1,0,0])
        self.assertTrue(np.isfinite(obs).all())
        with self.assertRaises(ValueError):
            encode_observation([0,0,0],None,[np.nan]*5,[0,0],cfg,4)

    def test_lidar_mask_fallback(self):
        state=self.state.copy()
        self.assertEqual(np.flatnonzero(get_action_mask(state)).tolist(),[5])
        state[3:8]=np.array([.2,.1,1,1,.4])/4
        self.assertEqual(np.flatnonzero(get_action_mask(state)).tolist(),[0,2,3,4])
        state[-1]=1
        self.assertEqual(np.flatnonzero(get_action_mask(state)).tolist(),[5])

    def test_forward_and_backward(self):
        network=QNetwork()
        x=network(torch.ones(3,len(FEATURES)))
        self.assertEqual(x.shape,(3,6))
        self.assertEqual(sum(p.numel() for p in network.parameters()),18950)
        self.assertTrue(torch.isfinite(x).all())
        x.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in network.parameters()))

    def test_mask_applies_to_greedy_and_epsilon(self):
        agent=DQNAgent(config())
        agent.values=lambda _: np.array([99,1,45,3,4,8])
        self.assertEqual(agent.choose(self.state,self.mask,0),3)
        self.assertEqual({agent.choose(self.state,self.mask,1) for _ in range(80)},{1,3})

    def test_double_target_uses_online_choice_and_terminal_bit(self):
        online=torch.tensor([[100.,3.,0.,2.,0.,0.]]).expand(2,-1)
        target=torch.tensor([[90.,5.,0.,70.,0.,0.]]).expand(2,-1)
        masked=double_dqn_target(online,target,torch.from_numpy(np.tile(self.mask,(2,1))),
                                torch.ones(2),torch.tensor([False,True]),.98)
        torch.testing.assert_close(masked,torch.tensor([5.9,1.]))

    def test_replay_wrap_and_sample(self):
        replay=ReplayBuffer(3)
        for step in range(5):
            replay.push(self.state,1,step,self.state,False,step==4,self.mask)
        self.assertEqual(replay.size,3)
        batch=replay.sample(3,np.random.default_rng(10),'cpu')
        self.assertEqual(set(batch['reward'].tolist()),{2,3,4})
        self.assertEqual(batch['state'].dtype,torch.float32)
        self.assertEqual(batch['action'].dtype,torch.int64)
        self.assertEqual(batch['truncated'].sum().item(),1)
        copy=ReplayBuffer(3)
        copy.load_state_dict(replay.state_dict())
        self.assertEqual(copy.cursor,replay.cursor)

    def test_optimizer_target_sync_save_load_resume(self):
        agent=DQNAgent(config())
        self.assertIsNone(agent.update())
        for _ in range(4):
            agent.replay.push(self.state,1,2,self.state,False,False,self.mask)
        initial=agent.target.layers[-1].bias.clone()
        self.assertTrue(math.isfinite(agent.update()[0]))
        self.assertFalse(torch.equal(agent.online.layers[-1].bias,initial))
        self.assertTrue(torch.equal(agent.target.layers[-1].bias,initial))
        agent.update()
        self.assertEqual(agent.target_updates,1)
        self.assertTrue(torch.equal(agent.online.layers[-1].bias,agent.target.layers[-1].bias))
        self.assertTrue(all(p.grad is None for p in agent.target.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'checkpoint.pt'
            agent.save(checkpoint,dict(episode=1,global_step=4),full=True)
            actor,_=load_inference(checkpoint)
            np.testing.assert_array_equal(actor(torch.from_numpy(self.state)).numpy(),agent.values(self.state))
            restored=DQNAgent(config())
            self.assertEqual(restored.resume(checkpoint)['episode'],1)
            self.assertEqual(restored.replay.size,4)
            np.testing.assert_array_equal(agent.update(),restored.update())
            np.testing.assert_array_equal(agent.values(self.state),restored.values(self.state))

    def test_real_maze_timeout_and_deterministic_evaluation(self):
        mission=maze_mission(.64)
        agent=DQNAgent(config())
        before=agent.online.layers[-1].bias.detach().clone()
        result=rollout(mission,agent.online,heading=0,seed=4)
        self.assertEqual(result,rollout(mission,agent.online,heading=0,seed=4))
        self.assertTrue(result['timeout'])
        self.assertEqual(result['steps'],4)
        self.assertTrue(torch.equal(before,agent.online.layers[-1].bias))
        self.assertEqual(agent.updates,0)

    def test_original_collision_and_recovery(self):
        motion=maze_mission(300).motion
        motion.reset(heading=math.pi)
        for _ in range(15):
            state,reward,terminal,truncated,info=motion.step(0)
            if info['collision']:
                break
        self.assertTrue(info['collision'])
        self.assertFalse(terminal or truncated)
        self.assertEqual(info['reward_parts']['collision'],-100.)
        step=motion.steps
        while motion.collision_recovery:
            _,_,terminated,truncated,info=motion.step(5)
            self.assertFalse(terminated or truncated or info['collision'])
        self.assertEqual(motion.steps-step,5)


if __name__=='__main__':
    unittest.main()
