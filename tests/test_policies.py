import unittest

from pcc_poker.engine import initial_state
from pcc_poker.policies import MODES, PCCPolicy, component_scores


class PolicyTests(unittest.TestCase):
    def test_every_mode_scores_every_legal_action(self):
        state=initial_state([0,1,2,0,1,2]);policy=PCCPolicy((1,1,1));scores=component_scores(state,policy.opponent_model)
        for mode in MODES:self.assertEqual(set(scores[mode]),set(state.legal_actions()))

    def test_probabilities_normalize(self):
        decision=PCCPolicy((0.8,0.1,0.1),seed=3).decide(initial_state([0,1,2,0,1,2]));self.assertAlmostEqual(sum(decision.probabilities.values()),1)

    def test_mixture_is_normalized(self):
        policy=PCCPolicy((8,1,1));self.assertAlmostEqual(float(policy.weights.sum()),1)


if __name__ == "__main__": unittest.main()
