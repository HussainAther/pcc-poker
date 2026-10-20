"""Focused safeguards for prospective validation; stdlib unittest runnable."""
import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import numpy as np
from pcc_poker import blind_discriminant as b
from pcc_poker import blind_leduc_reference as ref

class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s=b.LeducCFR();cls.p=b.conventional_policy(cls.s,seed=123)
        cls.reference=b.compile_measurements(cls.s)

    def test_blind_input_boundary_and_labels(self):
        first=b.measure(self.s,self.p,self.reference)
        packed=dict(id='x',family='CFR',target_pcc_weights=[999,999,999],probabilities=b.pack(self.p))
        packed['family']='misleading label';packed['id']='different'
        second=b.measure(self.s,b.unpack(packed['probabilities']),self.reference)
        for x,y in zip(first,second):np.testing.assert_array_equal(x,y)
        self.assertNotIn('family',b.measure.__code__.co_varnames)

    def test_frozen_code_guard(self):
        frozen=dict(sources=b.source_hashes(),config=copy.deepcopy(b.CONFIG))
        b.check_freeze(frozen)
        frozen['sources']['engine.py']='tampered'
        with self.assertRaises(RuntimeError):b.check_freeze(frozen)

    def test_early_reveal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);b.event(path,'pipeline_frozen')
            frozen=dict(sources=b.source_hashes(),config=copy.deepcopy(b.CONFIG))
            with self.assertRaises(RuntimeError):b.reveal(path,frozen)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'artifact.json';b.write_new(p,{'a':1})
            with self.assertRaises(FileExistsError):b.write_new(p,{'a':2})

    def test_outcome_panel_cannot_change_candidate_measurement(self):
        p={k:v.copy() for k,v in self.p.items()}
        first=b.measure(self.s,p,self.reference)
        for k in p:
            if b.panel(k)==1:p[k]=p[k][::-1]
        second=b.measure(self.s,p,self.reference)
        # Exact exploitability intentionally uses full policy as a conventional
        # baseline; all measured features except its final 3 entries are isolated.
        np.testing.assert_array_equal(first[0][:-3],second[0][:-3])
        np.testing.assert_array_equal(first[1],second[1])
        self.assertTrue(set(self.reference).isdisjoint({k for k in p if b.panel(k)==1}))

    def test_hidden_state_and_order_invariance(self):
        reversed_policy=dict(reversed(list(self.p.items())))
        for x,y in zip(b.measure(self.s,self.p,self.reference),b.measure(self.s,reversed_policy,self.reference)):
            np.testing.assert_array_equal(x,y)
        for k,ids in self.s.infos.items():
            a=self.s.nodes[ids[0]].state
            for i in ids[1:]:
                s=self.s.nodes[i].state
                self.assertEqual(b.equity(a,a.actor),b.equity(s,s.actor))
        # Hidden opponent card/deck never affect fixed Chaos snapshot.
        b.validate_policy(self.s,b.fixed_chaos(self.s))

    def test_train_only_preprocessing_and_separation(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(20,4));y=rng.normal(size=(20,3));t=rng.normal(size=(4,4))
        pred=b.fit_predict(x,y,t)
        # Extra test population rows cannot affect existing predictions.
        np.testing.assert_allclose(pred,b.fit_predict(x,y,np.r_[t,np.full((1,4),1e6)])[:4],rtol=1e-10,atol=1e-10)
        # Affine feature unit changes preserve standardized model predictions.
        np.testing.assert_allclose(pred,b.fit_predict(x*7+3,y,t*7+3),atol=1e-12)
        # Reordering whole policy replicates preserves inference's point estimate.
        np.testing.assert_allclose(pred,b.fit_predict(x[::-1],y[::-1],t),atol=1e-12)
        aa={730001+101*i for i in range(b.CONFIG['replicates'])}
        bb={930001+101*i for i in range(b.CONFIG['replicates'])}
        self.assertFalse(aa&bb)

    def test_policy_replication_not_decision_pseudoreplication(self):
        # One vector per policy; duplicating hidden chance copies cannot create
        # new observations, because aggregation indexes information sets.
        old=self.s.infos
        original=b.measure(self.s,self.p,self.reference)
        self.s.infos={k:ids*2 for k,ids in old.items()}
        try:duplicate=b.measure(self.s,self.p,self.reference)
        finally:self.s.infos=old
        for x,y in zip(original,duplicate):np.testing.assert_array_equal(x,y)

    def test_holm_prespecified_family(self):
        ps=[.001]+[1.]*11
        self.assertAlmostEqual(b.holm(ps)[0],.012)
        self.assertEqual(b.CONFIG['family_size'],len(b.ENDPOINTS)*2*2)

    def test_independent_rules_exhaustive(self):
        report=ref.verify_tree(self.s)
        self.assertEqual(report['nodes_checked'],14490)
        self.assertAlmostEqual(ref.profile_value(self.p),self.s.profile_value(self.p),places=11)

    def test_policy_conditioned_response(self):
        result=b.evaluate_chaos(self.s)
        p=b.unpack(result['target_table'])
        for player in (0,1):
            response=b.unpack(result['br_tables'][player])
            profile={k:response[k] if k[0]==player else p[k] for k in p}
            self.assertAlmostEqual(ref.profile_value(profile,player),result[f'br_player{player}'],places=10)
        self.assertAlmostEqual(result['exploitability'],result['nash_conv']/2)
        self.assertGreaterEqual(result['br_player0'],result['profile_value_player0'])
        self.assertGreaterEqual(result['br_player1'],-result['profile_value_player0'])

if __name__=='__main__':unittest.main()
