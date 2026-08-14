# Frozen Control-over-Pressure Mechanism Test

## Question

Does Control benefit from correctly timed, context-specific knowledge of a
Pressure opponent, or merely from that opponent's aggregate action rates?

The v0.6 counterfactual experiment found a large aligned-model effect for
Control against Pressure, alongside negative effects against Control and
Chaos. Because that target-specific result was not the original aggregate
confirmation rule, it is treated as hypothesis-generating here and tested on
fresh seeds without changing the frozen v0.3 policy.

## Matched interventions

A balanced probe learns an opponent model. During evaluation the same
Control-heavy focal policy receives one of three read-only models:

1. `aligned`: the observed actions remain in their correct public contexts;
2. `round_swapped`: facing-action counts from rounds zero and one are swapped;
3. `context_yoked`: observed action labels are randomly reassigned across
   rounds within the same legal-action stratum (`open` or `facing`).

The two controls preserve the total number of observations and global action
counts. The yoked condition additionally preserves each context's sample size
and never moves actions between incompatible legal-action strata.
Only action-to-context alignment changes. Policy code, mixture, temperature,
deck stream, policy random seeds, opponent, and seat balancing are identical
within each comparison.

## Frozen default design

- 16 independent replicates;
- 250 calibration hands per seat order;
- 500 evaluation hands per seat order and condition;
- disjoint seed families `91001` and `101001`, stride `2000`;
- purities `.70`, `.80`, and `.90`;
- temperatures `.25`, `.35`, and `.50`;
- Pressure is the primary target and Chaos is the discriminant target;
- replicate-level averages are the inferential units.

No result from v0.6 is reused as an evaluation observation.

## Implementation audit

An initial engineering smoke run randomized actions across both `open` and
`facing` contexts. Post-run validation identified that this could place actions
outside their legal-action stratum, violating the intended matched-control
design. That output was discarded before acceptance, the yoke was restricted
to randomization within legal-action strata, and the unchanged seeds, surface,
sample sizes, and confirmation thresholds were rerun. The support-preserving
run below is the frozen scientific result.

## Prespecified confirmation rule

The mechanism is confirmed only if all conditions pass:

1. against Pressure, aligned minus round-swapped has a 95% interval above zero;
2. against Pressure, aligned minus context-yoked has a 95% interval above zero;
3. each effect is larger against Pressure than Chaos with a 95% interval above
   zero;
4. both contrasts have positive means in at least six of nine
   purity-temperature cells; and
5. every programmed margin-preservation check passes.

The policies, seeds, surface, contrasts, and thresholds are frozen before the
default result is examined. Failure is retained rather than tuned away.

## Interpretation boundary

A positive result would support a causal contextual-prediction mechanism for
the engineered Control-over-Pressure edge. A null would show that the earlier
effect depended on aggregate opponent propensities or another feature rather
than contextual prediction. Neither outcome establishes PCC in humans.

## Frozen result

The default run confirmed every prespecified condition without modifying the
frozen v0.3 policy. Against Pressure, the aligned model improved Control payoff
by `.2841` chips per hand relative to the round-swapped model, with a normal
95% interval of `[.2547, .3134]`. Relative to the support-preserving
context-yoked model, the improvement was `.2065`, with an interval of
`[.1814, .2316]`.

Both effects were target-specific. The Pressure-minus-Chaos difference was
`.3202` for round timing (`[.2759, .3644]`) and `.2336` for context alignment
(`[.1972, .2699]`). Both contrasts had positive means in all nine
purity-temperature cells, exceeding the frozen six-of-nine rule. Every
action-count, context-size, seat-balance, and common-random-number check passed.

Against Chaos, correct context alignment was slightly harmful (`-.0270`,
`[-.0439, -.0101]`). The benefit is therefore not merely larger against
Pressure; it reverses direction for the discriminant target. This establishes
the engineered mechanism-level result; it does not establish a human cognitive
state.
