# Reproducibility and audit manifest

This repository keeps negative, partial, and positive synthetic results rather
than rewriting history after evaluation. The `reproduce` command adds a compact
audit layer over that practice.

It does **not** reinterpret failed constructs and it does **not** touch the
human HandHQ dataset. Instead it records:

- SHA-256 hashes for the frozen validation artifacts that currently define the
  Pressure/Control/Chaos measurement status;
- SHA-256 hashes for source, tests, protocol documents, and package metadata;
- a combined source fingerprint and a combined validation fingerprint;
- Python, NumPy, pytest, package, platform, and Git-commit metadata where
  available; and
- optionally, the full pytest result.

Run:

```bash
python -m pcc_poker reproduce --run-tests
```

The default output is `validation/reproducibility-manifest.json`.

A manifest is marked `reproducibility_ready=true` only when every prespecified
frozen validation artifact is present and, if `--run-tests` is requested, the
test suite passes. A missing artifact is reported explicitly; it is never
silently regenerated or substituted.

The manifest is an engineering provenance record. It is not empirical evidence
about humans and is not an IRB-relevant analysis of HandHQ data.
