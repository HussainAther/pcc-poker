import unittest
from pcc_poker.pressure_surprise_decomposition import PressureSurpriseOracle, summarize_pressure_surprise
from pcc_poker.behavioral import PublicActionModel
from pcc_poker.engine import initial_state

class PressureSurpriseTests(unittest.TestCase):
    def test_measurement_has_bounded_pressure_exposure(self):
        model = PublicActionModel.from_records([])
        oracle = PressureSurpriseOracle(model)
        state = initial_state([0,1,2,0,1,2])
        m = oracle.measure(state, state.legal_actions()[0])
        self.assertGreaterEqual(m.pressure_exposure, 0.0)
        self.assertLessEqual(m.pressure_exposure, 1.0)
        self.assertGreaterEqual(m.effective_surprisal, 0.0)

    def test_summary_adjustment_does_not_require_labels_to_fit(self):
        records=[]
        for i,(p,c,h) in enumerate(((.8,.1,.1),(.1,.8,.1),(.1,.1,.8),(.4,.2,.4))):
            for seat in (0,1):
                exposure=p
                effective=1.0-0.7*exposure+0.5*h
                records.append({"policy_family":"independent","mixture_id":str(i),"focal_seat":seat,"is_focal_policy":True,"target_pcc_weights":{"pressure":p,"control":c,"chaos":h},"behavioral_measurements":{"effective_surprisal":effective,"pressure_exposure":exposure}})
        report=summarize_pressure_surprise(records)
        fam=report["families"]["independent"]
        self.assertGreater(fam["pressure_correlation_reduction"],0)
        self.assertIn("chaos",fam["pressure_adjusted_effective_surprisal_correlations"])

if __name__ == '__main__': unittest.main()
