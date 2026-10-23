"""Patch 34: prospective matched extrapolation controls.

The primary comparison is augmented versus behavioral baseline under the SAME
training-range clipping of behavioral features. Clipping is an input operation,
not causal manipulation of PCC or the poker environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from . import blind_discriminant as experiment
from .leduc_cfr import LeducCFR
from .low_dimensional_pcc_audit import ENDPOINTS, ridge_fit_predict, standardize
from .prospective_pcc_replication import SOURCES, sha, write_exclusive
from .prospective_representation_intervention import _clamp_to_training

PROTOCOL = {
    'id': 'pcc-matched-clipping-v5', 'replications': 8,
    'policies_per_family': 24, 'seed_A': 8300003, 'seed_B': 10300003,
    'replication_stride': 110017, 'policy_stride': 157,
    'baseline_columns': [5, 6, 7], 'candidate_columns': [0, 1, 2],
    'ridge_penalty': 10.0, 'endpoints': list(ENDPOINTS),
    'primary_direction': 'A_to_B', 'primary_endpoint': 'value_vs_loose',
    'primary_statistic': 'mean cohort normalized MSE improvement of augmented both-clipped over baseline clipped',
    'primary_criterion': 'mean matched incremental improvement > 0',
    'bootstrap_seed': 34034, 'bootstrap_repetitions': 5000,
    'confidence_percentiles': [2.5, 97.5],
    'comparison': 'fixed models; training-only scale and support; no heldout model selection',
}
SOURCE_NAMES = tuple(sorted(set(SOURCES + (
    'prospective_representation_intervention.py', 'prospective_matched_clipping.py'))))
MODELS = ('baseline_raw', 'baseline_clipped', 'augmented_raw',
          'augmented_baseline_clipped', 'augmented_pcc_clipped', 'augmented_both_clipped')
COMPARISONS = {
    'raw_incremental': ('baseline_raw', 'augmented_raw'),
    'matched_baseline_clipping_incremental': ('baseline_clipped', 'augmented_baseline_clipped'),
    'matched_both_clipping_incremental': ('baseline_clipped', 'augmented_both_clipped'),
    'pcc_clipping_effect': ('augmented_raw', 'augmented_pcc_clipped'),
    'baseline_clipping_effect': ('baseline_raw', 'baseline_clipped'),
    'augmented_baseline_clipping_effect': ('augmented_raw', 'augmented_baseline_clipped'),
    'augmented_both_clipping_effect': ('augmented_raw', 'augmented_both_clipped'),
}
PRIMARY = 'matched_both_clipping_incremental'


def frozen_object():
    directory = Path(__file__).parent
    return {'protocol': PROTOCOL,
            'sources': {name: sha((directory/name).read_bytes()) for name in SOURCE_NAMES},
            'status': 'frozen_before_policy_generation'}


def freeze(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_exclusive(directory/'protocol.json', frozen_object())
    return {'status':'frozen', 'protocol_file_sha256':sha((directory/'protocol.json').read_bytes())}


def verify(directory, require_fresh=True):
    directory = Path(directory)
    observed = json.loads((directory/'protocol.json').read_text(encoding='utf8'))
    if observed != frozen_object():
        raise RuntimeError('frozen protocol or source commitments mismatch')
    if require_fresh and any((directory/n).exists() for n in ('measurements.json','results.json')):
        raise FileExistsError('experiment already generated; refusing overwrite')
    return observed


def seed(family, replication, index):
    return (PROTOCOL['seed_A' if family=='A' else 'seed_B'] +
            replication * PROTOCOL['replication_stride'] +
            index * PROTOCOL['policy_stride'])


def _array(rows, name, columns=None):
    a = np.asarray([row[name] for row in rows], dtype=float)
    return a if columns is None else a[:, columns]


def evaluate(train, test):
    if len(train) < 3 or len(test) < 3:
        raise ValueError('not enough policies')
    xb = _array(train, 'baseline', PROTOCOL['baseline_columns'])
    tb = _array(test, 'baseline', PROTOCOL['baseline_columns'])
    xc = _array(train, 'candidate', PROTOCOL['candidate_columns'])
    tc = _array(test, 'candidate', PROTOCOL['candidate_columns'])
    y, truth = _array(train, 'outcomes'), _array(test, 'outcomes')
    if any(a.ndim != 2 or a.shape[1] != 3 for a in (xb,tb,xc,tc,y,truth)):
        raise ValueError('unexpected shape')
    if not all(np.isfinite(a).all() for a in (xb,tb,xc,tc,y,truth)):
        raise ValueError('nonfinite data')
    b, bt = standardize(xb,tb)
    c, ct = standardize(xc,tc)
    bt_clip = _clamp_to_training(b,bt)
    ct_clip = _clamp_to_training(c,ct)
    penalty = PROTOCOL['ridge_penalty']
    bc = np.c_[b,c]
    prediction = {
        'baseline_raw': ridge_fit_predict(b,y,bt,penalty),
        'baseline_clipped': ridge_fit_predict(b,y,bt_clip,penalty),
        'augmented_raw': ridge_fit_predict(bc,y,np.c_[bt,ct],penalty),
        'augmented_baseline_clipped': ridge_fit_predict(bc,y,np.c_[bt_clip,ct],penalty),
        'augmented_pcc_clipped': ridge_fit_predict(bc,y,np.c_[bt,ct_clip],penalty),
        'augmented_both_clipped': ridge_fit_predict(bc,y,np.c_[bt_clip,ct_clip],penalty),
    }
    variance = y.var(axis=0)
    if np.any(variance <= 1e-12):
        raise ValueError('degenerate training outcomes')
    mse = {name: np.mean((p-truth)**2,axis=0) for name,p in prediction.items()}
    scores = {name: dict(zip(ENDPOINTS,((mse[left]-mse[right])/variance).tolist()))
              for name,(left,right) in COMPARISONS.items()}
    return {
        'n_train':len(train), 'n_test':len(test),
        'mse':{name:dict(zip(ENDPOINTS,a.tolist())) for name,a in mse.items()},
        'comparisons':scores,
        'out_of_range_fraction':{
            'baseline':np.mean(bt != bt_clip,axis=0).tolist(),
            'pcc':np.mean(ct != ct_clip,axis=0).tolist(),
        },
    }


def summarize(cohorts):
    if len(cohorts)<2:
        raise ValueError('need at least two cohorts')
    rng = np.random.default_rng(PROTOCOL['bootstrap_seed'])
    draws = rng.integers(0,len(cohorts),size=(PROTOCOL['bootstrap_repetitions'],len(cohorts)))
    directions={}
    for direction in ('A_to_B','B_to_A'):
        comparisons={}
        for comp in COMPARISONS:
            endpoints={}
            for endpoint in ENDPOINTS:
                v=np.array([cohort[direction]['comparisons'][comp][endpoint] for cohort in cohorts])
                if not np.isfinite(v).all():
                    raise ValueError('nonfinite cohort score')
                lo,hi=np.percentile(v[draws].mean(axis=1),PROTOCOL['confidence_percentiles'])
                endpoints[endpoint]={
                    'mean':float(v.mean()), 'cohort_values':v.tolist(),
                    'cohort_bootstrap_ci95':[float(lo),float(hi)],
                    'fraction_positive':float(np.mean(v>0)),
                }
            comparisons[comp]=endpoints
        directions[direction]=comparisons
    primary=directions[PROTOCOL['primary_direction']][PRIMARY][PROTOCOL['primary_endpoint']]
    return {'n_cohorts':len(cohorts),'directions':directions,
            'primary_point_estimate_criterion_met':primary['mean']>0,
            'primary_bootstrap_interval_positive':primary['cohort_bootstrap_ci95'][0]>0}


def run(directory):
    directory=Path(directory)
    verify(directory)
    solver=LeducCFR()
    reference=experiment.compile_measurements(solver)
    opponents=[experiment.conventional_policy(solver,parameters=p)
               for p in ((5.,-1.5,.5,0.),(2.,1.5,.8,0.))]
    records=[];cohorts=[];all_seeds=set()
    for rep in range(PROTOCOL['replications']):
        group={'A':[],'B':[]}
        for family in ('A','B'):
            for i in range(PROTOCOL['policies_per_family']):
                policy_seed=seed(family,rep,i)
                if policy_seed in all_seeds:
                    raise RuntimeError('duplicate seed')
                all_seeds.add(policy_seed)
                policy=(experiment.sampled_cfr(solver,policy_seed) if family=='A'
                        else experiment.conventional_policy(solver,policy_seed))
                experiment.validate_policy(solver,policy)
                baseline,candidate=experiment.measure(solver,policy,reference)
                outcomes=[experiment.cross_value(solver,policy,opp) for opp in opponents]
                outcomes.append(experiment.matched_response(solver,policy))
                row={'id':f'r{rep:02d}-{family}-{i:02d}','replication':rep,'family':family,
                     'seed':policy_seed,'baseline':baseline.tolist(),
                     'candidate':candidate.tolist(),'outcomes':outcomes}
                records.append(row);group[family].append(row)
        cohorts.append({'A_to_B':evaluate(group['A'],group['B']),
                        'B_to_A':evaluate(group['B'],group['A'])})
        print(f'generated replication {rep+1}/{PROTOCOL["replications"]}',flush=True)
    verify(directory)
    write_exclusive(directory/'measurements.json',records)
    report={'scope':'fresh-seed matched input clipping experiment',
            'protocol_sha256':sha((directory/'protocol.json').read_bytes()),
            'measurements_sha256':sha((directory/'measurements.json').read_bytes()),
            'cohort_results':cohorts,'summary':summarize(cohorts),
            'frozen_evidence_modified':False,
            'limitations':[
                'Hypothesis motivated by previous experiments, not independent preregistration.',
                'Input clipping does not establish causality of PCC constructs.',
                'Eight cohort bootstrap intervals are descriptive.',
                'Clipped test-time model inputs differ from the original feature distribution.',
                'No policy-family-independent simulator or external validation.',
                'No tuning or selection using held-out results is permitted.',
            ]}
    write_exclusive(directory/'results.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','run','verify'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='freeze':result=freeze(args.output)
    elif args.command=='run':
        report=run(args.output)
        primary=report['summary']['directions'][PROTOCOL['primary_direction']][PRIMARY][PROTOCOL['primary_endpoint']]
        result={'primary':primary,'primary_criterion_met':report['summary']['primary_point_estimate_criterion_met'],
                'results_file':str(args.output/'results.json')}
    else:
        verify(args.output,require_fresh=False)
        result={'status':'protocol_and_source_verified'}
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
