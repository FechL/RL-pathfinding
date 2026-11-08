"""Greedy continuous Q-learning and NumPy deployment regressions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'controllers/micromouse'))
sys.path.insert(0, str(ROOT/'tools'))
from continuous_control import NavigationMap, read_policy

try:
    import torch
    from naf import NAF, NormalizedAdvantage, Replay
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install requirements-training.txt for NAF tests')
class NafTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        self.model = NormalizedAdvantage()

    def test_continuous_greedy_action_maximizes_q(self):
        observations = torch.rand(128, 26) * 2 - 1
        actions = torch.rand(128, 2) * 2 - 1
        greedy_actions = self.model.mean(observations)
        greedy_values = self.model(observations, greedy_actions)
        torch.testing.assert_close(greedy_values, self.model.value(observations))
        self.assertTrue(torch.all(torch.abs(greedy_actions) <= 1))
        self.assertTrue(torch.all(greedy_values > self.model(observations, actions)))

    def test_terminal_target_cuts_bootstrap(self):
        learner = NAF(self.model, gamma=.5, tau=0.)
        for parameter in self.model.parameters():
            parameter.data.zero_()
        for parameter in learner.target_value.parameters():
            parameter.data.zero_()
        learner.target_value[-1].bias.data[0] = 2.
        obs = np.zeros(26, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)
        for terminated, expected in ((True, 0.), (False, .5)):
            replay = Replay(1)
            replay.add(obs, action, 0., obs, terminated=terminated)
            loss = learner.update(replay.sample(np.random.default_rng(42), 1))
            self.assertAlmostEqual(loss, expected)

    def test_numpy_export_matches_greedy_network(self):
        nav = NavigationMap((.2,.2), (1.2,1.2), [], (0,1.6,0,1.6), .4)
        data = nav.metadata()
        data.update(algorithm='naf', layers=self.model.export_layers())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'naf.json'
            path.write_text(json.dumps(data))
            actor, _ = read_policy(path, nav, algorithm='naf')
            for observation in np.random.default_rng(43).uniform(-1., 1., size=(12, 26)).astype(np.float32):
                np.testing.assert_allclose(actor(observation), self.model.action(observation), atol=2e-6)
