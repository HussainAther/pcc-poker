"""Exploratory, read-only PCC feature ablations against frozen policy measurements.

New model fits are post-hoc, not part of the preregistered Beyond Strategy test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .blind_discriminant import fit_predict, CONFIG, ENDPOINTS
from .cross_family_failure_audit import audit as verify_alignment


MODELS = ('baseline', 'pressure', 'control', 'chaos', 'all',
          'omit_pressure', 'omit_control', 'omit_chaos', 'residualized_all')


def _data(rows, cohort, field):
    selected = sorted((r for r in rows if r['cohort'] == cohort), key=lambda r: r['replicate'])
    if len(selected) != 32 or [r['replicate'] for r in selected] != list(range(32)):
        raise ValueError('invalid cohort')
    arr = np.asarray([r[field] for r in selected], dtype=float)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError('invalid measurement array')
    return arr


def _residualize(x, c, tx, tc, ridge=1e-6):
    """Project candidates off conventional features using training samples ONLY."""
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-10, 1., sd)
    z, tz = (x-mu)/sd, (tx-mu)/sd
    cm, cs = c.mean(0), c.std(0)
    cs = np.where(cs < 1e-10, 1., cs)
    v, tv = (c-cm)/cs, (tc-cm)/cs
    beta = np.linalg.solve(z @ z.T + ridge*np.eye(len(z)), v)
    return v-z @ (z.T @ beta), tv-tz @ (z.T @ beta)


def _design(name, x, c, tx, tc):
    if name == 'baseline':
        return x, tx
    if name == 'residualized_all':
        r, tr = _residualize(x, c, tx, tc)
        return np.column_stack((x, r)), np.column_stack((tx, tr))
    if name == 'all':
        inds = (0, 1, 2)
    elif name.startswith('omit_'):
        inds = tuple(i for i, col in enumerate(('pressure', 'control', 'chaos')) if col != name[5:])
    else:
        inds = (('pressure', 'control', 'chaos').index(name),)
    return np.column_stack((x, c[:, inds])), np.column_stack((tx, tc[:, inds]))


def _condition(x, c):
    combined = np.column_stack((x, c))
    sd = combined.std(0)
    active = sd > 1e-10
    z = (combined[:, active]-combined[:, active].mean(0))/sd[active]
    s = np.linalg.svd(z, compute_uv=False)
    rank = int(np.sum(s > 1e-8))
    return {'active_features': int(active.sum()), 'numerical_rank': rank,
            'condition_number_nonzero_singular': float(s[0]/s[rank-1]) if rank else None,
            'rank_deficient': bool(rank < active.sum())}


def audit(directory, penalties=(1., 10., 100.)):
    root = Path(directory)
    # Also checks original frozen predictions against policy ordering and outcomes.
    verify_alignment(root)
    if not penalties or any(not np.isfinite(p) or p <= 0 for p in penalties):
        raise ValueError('penalties must be finite and positive')
    rows = json.loads((root/'measurements.json').read_text())
    result = {'scope': 'exploratory post-hoc fits; frozen predictions are not replaced',
              'frozen_evidence_modified': False, 'ridge_penalties': list(penalties), 'directions': {},
              'limitations': ['No held-out model selection: all penalties and ablations are descriptive.',
                              'Residualization uses training cohorts only; no causal identification.',
                              'No confirmatory significance tests or independently generated policies.']}
    for key, a, b in (('A_to_B', 'A', 'B'), ('B_to_A', 'B', 'A')):
        x, tx = _data(rows, a, 'baseline'), _data(rows, b, 'baseline')
        c, tc = _data(rows, a, 'candidate'), _data(rows, b, 'candidate')
        y, ty = _data(rows, a, 'outcomes'), _data(rows, b, 'outcomes')
        if c.shape != (32, 3) or y.shape != (32, 3) or tx.shape[1] != x.shape[1]:
            raise ValueError('unexpected measurement dimensions')
        scale = np.var(y, axis=0)
        if np.any(scale <= 0):
            raise ValueError('zero training outcome variance')
        fits = {}
        for penalty in penalties:
            pkey = str(float(penalty))
            preds = {}
            for model in MODELS:
                dx, dtx = _design(model, x, c, tx, tc)
                preds[model] = fit_predict(dx, y, dtx, penalty=float(penalty))
            base_sq = (preds['baseline']-ty)**2
            fits[pkey] = {}
            for model in MODELS:
                gain = np.mean((base_sq-(preds[model]-ty)**2)/scale, axis=0)
                fits[pkey][model] = {endpoint: float(gain[j]) for j, endpoint in enumerate(ENDPOINTS)}
        result['directions'][key] = {'training_condition': _condition(x, c), 'n_train': len(x),
                                     'n_test': len(tx), 'improvement_vs_refit_baseline': fits}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default='validation/beyond-strategy-blind-v1')
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.directory), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
