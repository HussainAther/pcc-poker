# Human-data ingestion protocol

## Status

Implementation-only preparation. No confirmatory human-data analysis is authorized by this document.
The parser and tests in this repository are developed and validated using synthetic/mock PHHS records only.

## Scope boundary

The future human-data pathway is restricted to the anonymized/obfuscated HandHQ portion of the public poker-hand-history dataset after the appropriate institutional determination is obtained. Televised WSOP data, named historical examples, and Pluribus data are outside the planned human-study scope.

## Safe ingestion contract

`pcc_poker.handhq.ingest_phhs_text` is the boundary between source records and analytical records.
It must immediately:

1. replace source player strings with keyed study-specific pseudonyms;
2. replace source hand numbers with study-local sequential identifiers;
3. omit table names, exact time/date metadata, timezone, and currency metadata;
4. discard private-hole-card deal contents from the analytical representation;
5. retain public board cards only after they are publicly dealt;
6. make hand outcomes opt-in rather than part of the default analytical record; and
7. keep outcomes and future actions out of decision-time feature rows.

The pseudonymization key must not be committed to the repository or written into analytical outputs.

## Decision-time leakage rule

For decision index `t`, model features may contain only public state available strictly before the focal action at `t` plus permitted static game metadata. They must not contain:

- the focal action itself as an input feature;
- later actions;
- unrevealed/private cards;
- final winnings or terminal outcome;
- source player strings;
- source hand/table identifiers; or
- exact source timestamps/dates/timezones.

## Current validation

`tests/test_handhq.py` uses an entirely invented PHHS fixture. It checks:

- PHH action-code normalization;
- seat-to-player reconstruction;
- deterministic keyed pseudonymization;
- removal of source metadata and raw player strings;
- suppression of private-card contents;
- availability timing of public board cards;
- outcome isolation; and
- prior-only public action history.

No real HandHQ record is included in the tests or repository fixtures.

## Before any human-data run

Before this module is pointed at the public HandHQ files, the project must record the institutional determination, freeze the retained/excluded field list, specify the pseudonymization-key handling procedure, and preregister the confirmatory feature/analysis contract. Any change after looking at human results must be labeled exploratory.

## Public game-state reconstruction layer

After sanitization, downstream code must consume `SanitizedHand` rather than raw PHHS records. The public-state layer in `pcc_poker/handhq_features.py` reconstructs only information available before each focal decision:

- street and public board progression;
- pot size from antes, blinds, calls, bets, and raises already observed;
- current street bet and amount to call;
- actor street/total contribution and remaining stack;
- multiway effective stack using active opponents;
- active-player count and number of prior bets/raises on the street;
- prior public action sequence and current-street action sequence; and
- a conservative legal-action context (`fold`, `check_call`, `bet_raise`).

The feature API deliberately excludes study/player identifiers, source seats as identity features, venue, the focal action label and amount, outcomes, winnings, private cards, exact public-card identities, and future actions. Public board *size/street* may be used as a modeling feature; public card identities remain available only in the reconstructed audit state for future preregistered analyses and are not emitted by the default `modeling_features()` function.

All tests for this layer use invented PHHS fixtures only. No HandHQ/Zenodo human records are included in the test suite or repository artifacts.
