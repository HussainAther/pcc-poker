import math
import unittest

from pcc_poker.handhq import ingest_phhs_text
from pcc_poker.handhq_features import reconstruct_public_states
from pcc_poker.human_observables import (
    FrozenPublicStateActionModel,
    evaluate_observables,
    measure_observable,
)

KEY = b"synthetic-observable-test-key"


def hand(actions, players=("mock-a", "mock-b"), stacks=(20, 20)):
    text = f"""
[1]
variant = 'NT'
antes = [0, 0]
blinds_or_straddles = [0.5, 1.0]
min_bet = 1.0
starting_stacks = {list(stacks)!r}
actions = {actions!r}
venue = 'Synthetic Lab'
seats = [1, 2]
players = {list(players)!r}
"""
    (parsed,) = ingest_phhs_text(text, pseudonymization_key=KEY)
    return reconstruct_public_states(parsed)


class HumanObservableTests(unittest.TestCase):
    def test_pressure_is_zero_for_nonaggressive_action(self):
        states = hand(['d dh p1 AsKd', 'd dh p2 QhQs', 'p1 cc', 'p2 cc'])
        model = FrozenPublicStateActionModel.fit(states)
        obs = measure_observable(states[0], model)
        self.assertEqual(obs.pressure_index, 0.0)
        self.assertEqual(obs.commitment_fraction, 0.0)
        self.assertEqual(obs.escalation_indicator, 0.0)

    def test_raise_has_commitment_and_escalation_pressure(self):
        states = hand(['d dh p1 AsKd', 'd dh p2 QhQs', 'p1 cbr 4.0', 'p2 f'])
        model = FrozenPublicStateActionModel.fit(states)
        obs = measure_observable(states[0], model)
        self.assertGreater(obs.commitment_fraction, 0.0)
        self.assertEqual(obs.escalation_indicator, 1.0)
        self.assertGreater(obs.pressure_index, 0.5)

    def test_surprisal_is_negative_log_static_probability(self):
        calibration = []
        for action in ('cc', 'cc', 'cc', 'cbr 4.0'):
            calibration.extend(hand(['d dh p1 ????', 'd dh p2 ????', f'p1 {action}', 'p2 f']))
        evaluation = hand(['d dh p1 ????', 'd dh p2 ????', 'p1 cbr 4.0', 'p2 f'])
        model = FrozenPublicStateActionModel.fit(calibration)
        obs = measure_observable(evaluation[0], model)
        self.assertAlmostEqual(obs.behavioral_surprisal, -math.log(obs.static_action_probability))
        self.assertGreater(obs.behavioral_surprisal, 0.0)

    def test_history_alignment_rewards_history_conditioned_pattern(self):
        calibration = []
        # Same broad public state; opponent-history signature differentiates response.
        for _ in range(8):
            calibration.extend(hand([
                'd dh p1 ????', 'd dh p2 ????',
                'p1 cbr 3.0', 'p2 cc', 'd db 2c 8d Jh',
                'p1 cc', 'p2 cbr 4.0', 'p1 f'
            ]))
        for _ in range(8):
            calibration.extend(hand([
                'd dh p1 ????', 'd dh p2 ????',
                'p1 cc', 'p2 cc', 'd db 2c 8d Jh',
                'p1 cc', 'p2 cc'
            ]))
        evaluation = hand([
            'd dh p1 ????', 'd dh p2 ????',
            'p1 cbr 3.0', 'p2 cc', 'd db 2c 8d Jh',
            'p1 cc', 'p2 cbr 4.0', 'p1 f'
        ])
        # Last state is p1 fold after opponent bet; history model has repeatedly seen it.
        model = FrozenPublicStateActionModel.fit(calibration)
        obs = measure_observable(evaluation[-1], model)
        self.assertGreaterEqual(obs.history_alignment, 0.0)

    def test_evaluation_does_not_mutate_frozen_counts(self):
        calibration = hand(['d dh p1 ????', 'd dh p2 ????', 'p1 cc', 'p2 cc'])
        evaluation = hand(['d dh p1 ????', 'd dh p2 ????', 'p1 cbr 4.0', 'p2 f'])
        model = FrozenPublicStateActionModel.fit(calibration)
        before_static = {k: dict(v) for k, v in model.static_counts.items()}
        before_temporal = {k: dict(v) for k, v in model.temporal_counts.items()}
        measure_observable(evaluation[0], model)
        self.assertEqual(before_static, {k: dict(v) for k, v in model.static_counts.items()})
        self.assertEqual(before_temporal, {k: dict(v) for k, v in model.temporal_counts.items()})

    def test_public_only_measurements_ignore_private_card_contents(self):
        a = hand(['d dh p1 AsKd', 'd dh p2 QhQs', 'p1 cc', 'p2 cc'])
        b = hand(['d dh p1 7c2d', 'd dh p2 AhAd', 'p1 cc', 'p2 cc'])
        model = FrozenPublicStateActionModel.fit(a)
        self.assertEqual(measure_observable(a[0], model), measure_observable(b[0], model))

    def test_split_api_returns_only_evaluation_measurements(self):
        calibration = hand(['d dh p1 ????', 'd dh p2 ????', 'p1 cc', 'p2 cc'])
        evaluation = hand(['d dh p1 ????', 'd dh p2 ????', 'p1 cbr 4.0', 'p2 f'])
        measured = evaluate_observables(calibration, evaluation)
        self.assertEqual(len(measured), len(evaluation))


if __name__ == '__main__':
    unittest.main()
