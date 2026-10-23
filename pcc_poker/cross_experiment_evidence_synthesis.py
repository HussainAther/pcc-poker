"""Patch 36: auditable, read-only synthesis of prospective PCC evidence.

Every reported numerical result comes from a verified frozen study; this module
never reruns policy generation or alters study artifacts. It is not an
independent preregistration, external replication, or causal identification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import prospective_evidence_synthesis as p30
from . import transfer_asymmetry_evidence_audit as p32
from . import matched_clipping_evidence_audit as p35
from .prospective_pcc_replication import sha

NAMES = ('pcc-prospective-v2', 'pcc-transfer-asymmetry-v3',
         'pcc-representation-intervention-v4', 'pcc-matched-clipping-v5')


def _read_intervention(directory):
    """Artifact hash checks only: no bespoke recomputation audit for Patch 33."""
    paths = {k: Path(directory) / (k + '.json') for k in ('protocol', 'measurements', 'results')}
    data = {k: p.read_bytes() for k, p in paths.items()}
    report = json.loads(data['results'])
    if report['protocol_sha256'] != sha(data['protocol']):
        raise ValueError('representation protocol hash mismatch')
    if report['measurements_sha256'] != sha(data['measurements']):
        raise ValueError('representation measurements hash mismatch')
    frozen = json.loads(data['protocol'])
    if not isinstance(frozen, dict) or report.get('scope') != 'fresh-seed computational input intervention; not causal identification of PCC constructs':
        raise ValueError('unexpected representation study artifact')
    primary = report['summary']['directions']['A_to_B']['clip_pcc_to_training_range']['value_vs_loose']
    if len(primary['cohort_values']) != 8:
        raise ValueError('unexpected representation cohort count')
    return {'verification_level':'stored-hash checks only; no independent result recomputation',
            'protocol_sha256':sha(data['protocol']),
            'measurements_sha256':sha(data['measurements']),
            'results_sha256':sha(data['results']),
            'primary_clipping_improvement_mean':primary['mean'],
            'primary_clipping_ci95':primary['cohort_bootstrap_ci95'],
            'positive_cohorts':sum(v>0 for v in primary['cohort_values']),
            'cohorts':len(primary['cohort_values'])}


def synthesize(root):
    root=Path(root)
    dirs={name:root/name for name in NAMES}
    prospective=p30.audit(dirs[NAMES[0]])
    asymmetry=p32.audit(dirs[NAMES[1]])
    intervention=_read_intervention(dirs[NAMES[2]])
    matched=p35.audit(dirs[NAMES[3]])
    for label, a in (('prospective',prospective),('asymmetry',asymmetry),('matched',matched)):
        if a['status'] != 'verified' or a['frozen_evidence_modified']:
            raise ValueError(f'{label} evidence not verified')
    return {
      'status':'verified_synthesis',
      'scope':'PCC-Poker within-simulator cross-family transfer; no external replication',
      'artifacts':{
        'patch29_30_prospective':{
          'verification':'recomputed via Patch 30 audit',
          'primary_bidirectional_improvement_met':prospective['primary_point_estimate_criterion_met'],
          'directions':{d:{endpoint:prospective['prospective'][d][endpoint]['normalized_improvement']
                            for endpoint in ('value_vs_loose','value_vs_tight','heldout_card_response')}
                        for d in ('A_to_B','B_to_A')},
          'integrity':prospective['integrity']},
        'patch31_32_asymmetry':{
          'verification':'recomputed via Patch 32 audit',
          'primary_endpoint':asymmetry['primary_endpoint'],
          'primary_asymmetry_mean':asymmetry['primary_asymmetry_mean'],
          'primary_ci95':asymmetry['primary_cohort_bootstrap_ci95'],
          'positive_cohort_fraction':asymmetry['primary_positive_cohort_fraction'],
          'primary_criterion_met':asymmetry['primary_point_estimate_criterion_met'],
          'descriptive_table':asymmetry['descriptive_table']},
        'patch33_representation':intervention,
        'patch34_35_matched_control':{
          'verification':'recomputed via Patch 35 audit',
          'primary_endpoint':matched['primary_endpoint'],
          'primary_incremental_mean':matched['primary_matched_incremental_mean'],
          'primary_ci95':matched['primary_cohort_bootstrap_ci95'],
          'positive_cohort_fraction':matched['primary_positive_cohort_fraction'],
          'primary_criterion_met':matched['primary_criterion_met'],
          'descriptive_table':matched['descriptive_table']}},
      'research_conclusions':[
        {'finding':'Universal bidirectional predictive improvement of PCC augmentation',
         'status':'not supported by prospective Patch 29 criterion'},
        {'finding':'Directional cross-family transfer asymmetry',
         'status':'supported across eight simulated cohorts in Patch 31; motivated by earlier outcomes'},
        {'finding':'Input-range extrapolation contributes to A-to-B failure',
         'status':'supported by input clipping intervention in Patch 33; not causal identification of PCC constructs'},
        {'finding':'Incremental PCC advantage under matched clipping',
         'status':'not demonstrated on primary endpoint in Patch 34'},
      ],
      'limits':[
        'Hypotheses were developed sequentially after earlier results; not independently preregistered.',
        'The audits share portions of the original experimental evaluators.',
        'Only eight in-silico cohorts for repeated-cohort experiments; bootstrap intervals are descriptive.',
        'Different studies use distinct fresh seeds and designs; estimates must not be pooled.',
        'Secondary endpoints lack multiplicity-controlled confirmatory inference.',
        'Failure to demonstrate incremental PCC benefit does not prove equivalence or impossibility.',
        'Results do not validate PCC constructs or extend automatically to human or organizational behavior.',
      ],
      'frozen_artifacts_modified':False,
    }


def markdown(report):
    a=report['artifacts']; x=a['patch31_32_asymmetry']; y=a['patch34_35_matched_control']; z=a['patch33_representation']
    lines=['# PCC-Poker: Cross-Experiment Evidence Synthesis','',
           'Scope: computational policy-family transfer in the current Leduc simulator; not external validation.','',
           '## Primary findings','',
           '| Study | Prespecified result | Outcome |','|---|---|---|',
           f"| Patch 29/30 | Positive improvement both directions, all endpoints | {'Met' if a['patch29_30_prospective']['primary_bidirectional_improvement_met'] else 'Not met'} |",
           f"| Patch 31/32 | Directional asymmetry on {x['primary_endpoint']} | {x['primary_asymmetry_mean']:.6f}; CI {x['primary_ci95']} |",
           f"| Patch 33 | PCC clipping reduces A-to-B error on value_vs_loose | {z['primary_clipping_improvement_mean']:.6f}; CI {z['primary_clipping_ci95']} (hash checks only) |",
           f"| Patch 34/35 | Matched-clipping incremental PCC improvement | {y['primary_incremental_mean']:.6f}; CI {y['primary_ci95']}; criterion {'met' if y['primary_criterion_met'] else 'not met'} |",
           '', '## Research interpretations','']
    lines += [f"- **{r['finding']}:** {r['status']}" for r in report['research_conclusions']]
    lines += ['', '## Important limitations','']+[f'- {v}' for v in report['limits']]
    lines += ['', 'All numerical statements are generated from local frozen evidence; see JSON report for provenance.', '']
    return chr(10).join(lines)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-root',type=Path,default=Path('validation'))
    parser.add_argument('--format',choices=('json','markdown'),default='json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    result=synthesize(args.validation_root)
    output=markdown(result) if args.format=='markdown' else json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x',encoding='utf-8') as f: f.write(output)
    else: print(output)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
