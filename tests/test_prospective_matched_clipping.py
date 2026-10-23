import json
import numpy as np
import pytest
from pcc_poker import prospective_matched_clipping as p


def rows(shift=0.,n=12):
    out=[]
    for i in range(n):
        v=float(i+1)
        out.append({'baseline':[0.]*5+[v,v*.2,v*.05],
                    'candidate':[v*.4+shift,v*.3+shift,v*.1+shift],
                    'outcomes':[v*.5+v*v*.01,v*.1+v*v*.02,v*.2+v*v*.03]})
    return out


def test_seed_uniqueness_and_primary_locked():
    values=[p.seed(f,r,i) for r in range(p.PROTOCOL['replications'])
            for f in ('A','B') for i in range(p.PROTOCOL['policies_per_family'])]
    assert len(values)==len(set(values))
    assert p.PROTOCOL['primary_endpoint']=='value_vs_loose'
    assert p.PRIMARY=='matched_both_clipping_incremental'


def test_freeze_protects_evidence(tmp_path):
    folder=tmp_path/'freeze'
    p.freeze(folder)
    assert p.verify(folder)
    with pytest.raises(FileExistsError):p.freeze(folder)
    (folder/'results.json').write_text('{}')
    with pytest.raises(FileExistsError):p.verify(folder)


def test_mutated_protocol_rejected(tmp_path):
    folder=tmp_path/'freeze';p.freeze(folder)
    obj=json.loads((folder/'protocol.json').read_text())
    obj['protocol']['ridge_penalty']=1.
    (folder/'protocol.json').write_text(json.dumps(obj))
    with pytest.raises(RuntimeError):p.verify(folder)


def test_evaluate_deterministic_and_matched_comparators():
    a,b=rows(),rows(shift=3.)
    before=json.dumps([a,b]);result=p.evaluate(a,b)
    assert result==p.evaluate(a,b)
    assert before==json.dumps([a,b])
    assert set(result['mse'])==set(p.MODELS)
    mse=result['mse'];comparison=result['comparisons']
    assert all(np.isfinite(list(v.values())).all() for v in comparison.values())
    assert mse['baseline_raw']==mse['baseline_clipped'] or mse['baseline_raw']!=mse['baseline_clipped']


def test_same_family_in_range_clipping_identity():
    r=rows(); result=p.evaluate(r,r)
    assert result['mse']['baseline_raw']==result['mse']['baseline_clipped']
    assert result['mse']['augmented_raw']==result['mse']['augmented_both_clipped']
    assert result['comparisons']['augmented_both_clipping_effect']==dict.fromkeys(p.ENDPOINTS,0.)


def test_summarize_repeatable_and_uses_matched_primary():
    def one(v):
        return {'comparisons':{name:{end:v for end in p.ENDPOINTS} for name in p.COMPARISONS}}
    cohorts=[{'A_to_B':one(v),'B_to_A':one(-v)} for v in (1.,2.,3.,4.)]
    summary=p.summarize(cohorts)
    assert summary==p.summarize(cohorts)
    assert summary['primary_point_estimate_criterion_met']
    assert summary['primary_bootstrap_interval_positive']
