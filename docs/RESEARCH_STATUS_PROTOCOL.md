# Research-status reporting protocol

`python -m pcc_poker research-status` converts the frozen synthetic validation artifacts into a compact publication-oriented status table.

The generator does not rerun experiments, change thresholds, infer human behavior, or promote a failed construct merely because some secondary statistic is favorable. It uses four conservative labels:

- **confirmed** — the source artifact's direct preregistered acceptance criterion passed;
- **partial** — the primary confirmation failed, but one or more prespecified component checks passed;
- **failed** — the represented claim failed its direct criterion without qualifying component support;
- **unresolved** — the conservative family-invariant panel currently has no selected observable for that axis.

The generated outputs are:

- `validation/research-status.json` — machine-readable report;
- `validation/research-status.csv` — flat table for analysis/manuscript workflows;
- `validation/RESEARCH_STATUS.md` — human-readable publication-style table.

All rows describe synthetic evidence or engineering/reproducibility status. No row is evidence that human poker behavior follows PCC.
