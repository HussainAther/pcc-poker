import copy
import json
import pytest

from pcc_poker import matched_clipping_evidence_audit as a
from pcc_poker import prospective_matched_clipping as experiment
from pcc_poker.prospective_pcc_replication import sha


def small_records():
    # Six observations per family, eight replications; enough for ridge fitting.
    records=[]
    for rep in range(experiment.PROTOCOL['replications']):
        for family in ('A','B'):
            for i in range(experiment.PROTOCOL['policies_per_family']):
                seed=experiment.seed(family,rep,i)
                v=(i+1)/25.0 + rep*0.01 + (0.15 if family=='B' else 0)
                records.append(dict(id=f'r{rep:02d}-{family}-{i:02d}',replication=rep,
                                    family=family,seed=seed,
                                    baseline=[0.,0.,0.,0.,0.,v,v*v, v*.2],
                                    candidate=[v*.5,v*.8,v*v],
                                    outcomes=[v, v*v+.01*i, v*.4+.03*i]))
    return records


def example(tmp_path,monkeypatch):
    # Replace a frozen protocol's source commitments with the active code ones.
    # These records are synthetic fixtures, not an actual prospective replication.
    records=small_records()
    groups={r:{f:[] for f in ('A','B')} for r in range(experiment.PROTOCOL['replications'])}
    for r in records:groups[r['replication']][r['family']].append(r)
    cohorts=[{'A_to_B':experiment.evaluate(g['A'],g['B']),
              'B_to_A':experiment.evaluate(g['B'],g['A'])} for g in groups.values()]
    protocol=experiment.frozen_object()
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'measurements.json').write_text(json.dumps(records))
    report=dict(protocol_sha256=sha((tmp_path/'protocol.json').read_bytes()),
                measurements_sha256=sha((tmp_path/'measurements.json').read_bytes()),
                cohort_results=cohorts,summary=experiment.summarize(cohorts))
    (tmp_path/'results.json').write_text(json.dumps(report))
    return records,report


def test_audit_recomputes_synthetic_evidence(tmp_path):
    example(tmp_path,None)
    result=a.audit(tmp_path)
    assert result['status']=='verified'
    assert result['policies_verified']==384
    assert result['frozen_evidence_modified'] is False


def test_rejects_measurement_tampering(tmp_path):
    records,report=example(tmp_path,None)
    records[0]['outcomes'][0]+=0.5
    (tmp_path/'measurements.json').write_text(json.dumps(records))
    with pytest.raises(ValueError,match='measurement hash'):
        a.audit(tmp_path)


def test_rejects_invalid_seed_even_with_updated_hash(tmp_path):
    records,report=example(tmp_path,None)
    records[0]['seed']+=1
    (tmp_path/'measurements.json').write_text(json.dumps(records))
    report['measurements_sha256']=sha((tmp_path/'measurements.json').read_bytes())
    (tmp_path/'results.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='seed'):
        a.audit(tmp_path)


def test_rejects_changed_summary(tmp_path):
    records,report=example(tmp_path,None)
    report['summary']['directions']['A_to_B'][experiment.PRIMARY]['value_vs_loose']['mean']+=1
    (tmp_path/'results.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='summary'):
        a.audit(tmp_path)


def test_rejects_changed_cohort_mse(tmp_path):
    records,report=example(tmp_path,None)
    report['cohort_results'][0]['A_to_B']['mse']['baseline_raw']['value_vs_loose']+=0.1
    (tmp_path/'results.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='cohort'):
        a.audit(tmp_path)


def test_rejects_source_commitment_tampering(tmp_path):
    example(tmp_path,None)
    payload=json.loads((tmp_path/'protocol.json').read_text())
    payload['sources'][next(iter(payload['sources']))]='0'*64
    (tmp_path/'protocol.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='protocol'):
        a.audit(tmp_path)
