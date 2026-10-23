from pcc_poker import cross_experiment_evidence_synthesis as s
import pytest


def test_names_unique():
    assert len(s.NAMES) == len(set(s.NAMES)) == 4


def test_synthesis_calls_verifiers(monkeypatch, tmp_path):
    p={'status':'verified','frozen_evidence_modified':False,'primary_point_estimate_criterion_met':False,
       'prospective':{d:{e:{'normalized_improvement':-1.0} for e in ('value_vs_tight','value_vs_loose','heldout_card_response')} for d in ('A_to_B','B_to_A')},'integrity':{}}
    a={'status':'verified','frozen_evidence_modified':False,'primary_endpoint':'value_vs_loose','primary_asymmetry_mean':3.0,
       'primary_cohort_bootstrap_ci95':[1,5],'primary_positive_cohort_fraction':1.0,'primary_point_estimate_criterion_met':True,'descriptive_table':[]}
    m={'status':'verified','frozen_evidence_modified':False,'primary_endpoint':'value_vs_loose','primary_matched_incremental_mean':-0.2,
       'primary_cohort_bootstrap_ci95':[-1,1],'primary_positive_cohort_fraction':0.5,'primary_criterion_met':False,'descriptive_table':[]}
    monkeypatch.setattr(s.p30,'audit',lambda _:p)
    monkeypatch.setattr(s.p32,'audit',lambda _:a)
    monkeypatch.setattr(s.p35,'audit',lambda _:m)
    monkeypatch.setattr(s,'_read_intervention',lambda _: {'primary_clipping_improvement_mean':2,'primary_clipping_ci95':[1,3]})
    result=s.synthesize(tmp_path)
    assert result['status']=='verified_synthesis'
    assert 'Not met' in s.markdown(result)
    assert result['artifacts']['patch34_35_matched_control']['primary_criterion_met'] is False


def test_rejects_unverified(monkeypatch,tmp_path):
    monkeypatch.setattr(s.p30,'audit',lambda _: {'status':'unverified','frozen_evidence_modified':False})
    monkeypatch.setattr(s.p32,'audit',lambda _: {'status':'verified','frozen_evidence_modified':False})
    monkeypatch.setattr(s.p35,'audit',lambda _: {'status':'verified','frozen_evidence_modified':False})
    monkeypatch.setattr(s,'_read_intervention',lambda _: {})
    with pytest.raises(ValueError,match='not verified'):
        s.synthesize(tmp_path)
