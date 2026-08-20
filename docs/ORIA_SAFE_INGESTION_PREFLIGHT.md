# ORIA-safe ingestion preflight

This engineering check exists so the future HandHQ ingestion boundary can be
exercised **without opening or analyzing human-source HandHQ records** while the
Georgia Tech ORIA/IRB determination is pending.

## Safety boundary

`python -m pcc_poker oria-ingestion-preflight` accepts only a fixture inside
`tests/fixtures/` whose first line is:

```text
# SYNTHETIC_FIXTURE_ONLY
```

Any path outside `tests/fixtures/` is rejected before its contents are read.
This includes a downloaded `data/handhq/...` file. The command does not provide
an override flag for that restriction.

The default fixture is entirely invented. It is used to exercise:

- PHHS block/schema parsing;
- player-vector and seat consistency;
- immediate HMAC study-ID transformation;
- removal of source player strings, table names, hand IDs, exact time/date,
  timezone, and currency metadata from sanitized objects;
- exclusion of private-card contents;
- exclusion of outcomes and future actions from decision-time rows;
- comparison of observed fixture field names against the planned ORIA field
  inventory; and
- output isolation under `build/audit/` rather than the frozen `validation/`
  bundle.

The audit report contains field **names**, counts, hashes, and pass/fail checks.
It does not serialize the raw player identifiers, table names, hand numbers, or
other prohibited source values.

## Run

```bash
python -m pcc_poker oria-ingestion-preflight
```

or:

```bash
make oria-preflight
```

Default audit output:

```text
build/audit/oria-ingestion-preflight.json
```

A passing result means only that the synthetic engineering boundary behaves as
specified. It is **not** an ORIA/IRB determination and does not authorize human
HandHQ analysis.
