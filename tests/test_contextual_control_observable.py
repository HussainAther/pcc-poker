import unittest
from pcc_poker.contextual_control_observable import FrozenAlignedYokedHistoryModel

class ContextualControlObservableTests(unittest.TestCase):
    def test_yoking_preserves_static_action_margins(self):
        base = {"actor":0,"round_index":0,"to_call":0,"pot":2,"legal_actions":["check","bet"]}
        records=[]
        for history, action in [([],"check"),([],"bet"),(["check"],"check"),(["check"],"bet")]:
            r=dict(base); r["history"]=history; r["action"]=action; records.append(r)
        model=FrozenAlignedYokedHistoryModel.from_records(records,seed=7)
        self.assertTrue(model.margin_checks()["static_context_action_margins_preserved"])
        self.assertTrue(model.margin_checks()["global_action_margins_preserved"])

if __name__ == "__main__": unittest.main()
