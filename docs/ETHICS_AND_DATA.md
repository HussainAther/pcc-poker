# Ethics and data boundary

The current repository contains synthetic AI-versus-AI Leduc data only. It does
not contain human participants, private records, usernames, or claims about
specific players.

Before adding public human poker histories, the project will:

1. document the source, license or terms, and reasonable privacy expectations;
2. retain no handles or direct identifiers in the analytic dataset;
3. report aggregate or decision-level findings, not profiles of named players;
4. minimize quoted hand histories and guard against re-identification;
5. obtain an institutional NHSR/IRB determination before treating the work as
   exempt or outside human-subjects research oversight; and
6. keep any later consented study in a separate protocol and data area.

This is a research-plan safeguard, not an institutional determination or legal
advice.

## Prepared ingestion layer

A mock-only HandHQ/PHH parser and leakage-validation layer is implemented in
`pcc_poker/handhq.py`; its governing boundary is documented in
`docs/HUMAN_DATA_INGESTION_PROTOCOL.md`. This preparation does not authorize or
constitute confirmatory analysis of human poker histories.
