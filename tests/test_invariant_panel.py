import unittest
from pcc_poker.invariant_panel import summarize_invariant_panel

class InvariantPanelTests(unittest.TestCase):
    def test_requires_both_families(self):
        with self.assertRaises(ValueError): summarize_invariant_panel([])

    def test_selection_requires_cross_family_discrimination(self):
        # Construct aggregated-looking decision records where Pressure exposure is
        # monotone with Pressure in both families while Chaos metrics are not.
        records=[]
        for family in ("score","independent"):
            for i,p in enumerate((0.1,0.3,0.6,0.9)):
                weights={"pressure":p,"control":(1-p)*0.6,"chaos":(1-p)*0.4}
                m={"pressure_exposure":p,"response_compression":p,"predicted_fold_probability":p,
                   "commitment_ratio":p,"normalized_surprisal":0.5,"effective_surprisal":0.5}
                records.append({"is_focal_policy":True,"behavioral_measurements":m,"policy_family":family,
                                "mixture_id":f"{family}-{i}","focal_seat":0,"target_pcc_weights":weights})
        report=summarize_invariant_panel(records)
        self.assertIn("pressure_exposure",report["selected_invariant_components"])
        self.assertNotIn("effective_surprisal",report["selected_invariant_components"])
        self.assertEqual(report["axis_coverage"]["control"],[])

if __name__ == "__main__": unittest.main()
