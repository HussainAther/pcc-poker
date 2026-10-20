"""Patch 23: read-only integrity and claim audit for sealed Beyond Strategy v1.

This audit cannot establish methodological validity or external replication.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory: Path) -> dict:
    directory = Path(directory)
    errors = []
    def load(name):
        try:
            return json.loads((directory / name).read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            errors.append(f'{name}: {type(exc).__name__}')
            return None
    frozen = load('freeze.json')
    report = load('report.json')
    predictions = load('predictions_frozen.json')
    chaos = load('chaos_best_response.json')
    historical = load('historical_before.json')
    try:
        lines = (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines(keepends=True)
        events = [json.loads(line) for line in lines]
    except (OSError, ValueError) as exc:
        errors.append(f'events.jsonl: {type(exc).__name__}')
        lines, events = [], []
    expected = [
        ('pipeline_frozen', 'freeze_sha256', 'freeze.json'),
        ('opaque_policies_generated', 'identity_commitment', 'sealed_identity.json'),
        ('measurement_frozen', 'measurements_sha256', 'measurements.json'),
        ('predictions_frozen', 'predictions_sha256', 'predictions_frozen.json'),
        ('identities_revealed', 'identity_sha256', 'sealed_identity.json'),
        ('completed', 'report_sha256', 'report.json'),
    ]
    if [e.get('stage') for e in events] != [e[0] for e in expected]:
        errors.append('event stage sequence differs from frozen protocol')
    for index, event in enumerate(events):
        prior = hashlib.sha256(''.join(lines[:index]).encode('utf-8')).hexdigest() if index else None
        if event.get('sequence') != index or event.get('previous_sha256') != prior:
            errors.append(f'event {index}: invalid hash chain or sequence')
        if index < len(expected):
            stage, field, filename = expected[index]
            if event.get('stage') == stage:
                path = directory / filename
                if not path.is_file() or event.get(field) != sha(path):
                    errors.append(f'event {index}: {filename} commitment mismatch')
    if frozen is not None:
        source_dir = directory.parent.parent / 'pcc_poker'
        for name, digest in frozen.get('sources', {}).items():
            path = source_dir / name
            if not path.is_file() or sha(path) != digest:
                errors.append(f'frozen source mismatch: {name}')
    if historical is not None:
        root = directory.parent
        for name, digest in historical.items():
            path = root / name
            if not path.is_file() or sha(path) != digest:
                errors.append(f'historical evidence mismatch: {name}')
    if report is not None and predictions is not None:
        if report.get('results') != predictions:
            errors.append('published report results differ from frozen predictions')
        if report.get('status') != 'completed':
            errors.append('report not completed')
        if report.get('config') != (frozen or {}).get('config'):
            errors.append('report configuration differs from freeze')
        if report.get('historical_files_verified') != len(historical or {}):
            errors.append('historical file count mismatch')
        if set(predictions) != {'A_to_B', 'B_to_A'}:
            errors.append('missing or extra transfer direction')
        else:
            for key, data in predictions.items():
                for field in ('improvement', 'holm_superiority', 'holm_no_meaningful_benefit'):
                    values = data.get(field)
                    if not isinstance(values, list) or len(values) != 3:
                        errors.append(f'{key}: missing three-endpoint {field}')
    if report is not None and chaos is not None:
        if report.get('chaos_exploitability') != chaos.get('exploitability'):
            errors.append('Chaos exploitability mismatch')
        if chaos.get('external_solver_validated') is not False:
            errors.append('unexpected external validation claim')
    return {'status': 'PASS' if not errors else 'FAIL', 'errors': errors,
            'events_checked': len(events), 'historical_files_checked': len(historical or {}),
            'scope': 'artifact integrity and consistency only; not scientific replication'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    result = audit(args.directory)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
