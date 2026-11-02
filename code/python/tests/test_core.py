import unittest

import numpy as np

from phm_core.decision import DecisionConfig, DecisionState, step
from phm_core.features import RobustBaseline, cycle_features


class CoreTests(unittest.TestCase):
    def test_feature_layout(self):
        actual = cycle_features(np.array([[[1.0], [-1.0]]]))
        np.testing.assert_allclose(actual, [[0, 1, 1, 2]])

    def test_baseline_direction_and_constant(self):
        model = RobustBaseline().fit(np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]))
        scores = model.score(np.array([[1.0, 1.0], [10.0, 1.0]]))
        self.assertGreater(scores[1], scores[0])

    def test_hysteresis_reset_and_no_mutation(self):
        config = DecisionConfig(enter=2, exit=1, required=2, window=3, recovery=2)
        original = DecisionState()
        _, state = step(3, original, config)
        event, state = step(3, state, config)
        self.assertEqual(event, "opened")
        self.assertEqual(original.hits, [])
        _, state = step(0, state, config)
        event, state = step(0, state, config)
        self.assertEqual(event, "closed")
        self.assertFalse(state.active)
        self.assertEqual(step(None, state, config)[0], "reset")


if __name__ == "__main__":
    unittest.main()
