# Poker Control Architecture Falsification Protocol

## Question

Does the cross-game architectural hypothesis observed in Colonel Blotto transfer to Leduc poker: are Pressure and Chaos comparatively state-like signatures while Control is disproportionately expressed through context-dependent modulation?

## Frozen design

- Twelve fixed synthetic PCC-mixture agents are sampled once from a symmetric Dirichlet distribution.
- The latent PCC weights define the synthetic agent generator but are **never regression predictors**.
- Behavioral signatures are estimated on seeds disjoint from the outcome-evaluation seeds.
- Four opponent contexts are frozen: balanced, Pressure-heavy, Control-heavy, and Chaos-heavy.
- Every evaluation is seat-balanced.
- The Control signature is the already established aligned-vs-context-yoked public-history log-likelihood advantage.
- Pressure is measured as observed aggressive-action rate.
- Chaos is measured as observed action entropy.

## Model comparison

The common cross-game comparison is evaluated with leave-one-agent-out prediction:

1. additive: `behavior ~ Pressure + Control + Chaos + context`
2. Control-modulatory: additive + `Control x context`
3. Pressure control: additive + `Pressure x context`
4. Chaos control: additive + `Chaos x context`

Behavioral targets are mean payoff, aggression rate, fold rate, and action entropy.

## Prospective primary rule

Poker passes its game-native Control-modulation test only if:

- `Control x context` reduces pooled standardized LOAO MAE by at least 5% relative to the additive model; and
- at least two of four behavioral targets improve.

A separate discriminant asks whether the Control interaction improvement exceeds both Pressure and Chaos interaction improvements. This discriminant is reported but is not used to rewrite the primary rule after the result is known.

## Guardrails

- No post-result retuning of contexts, thresholds, target list, or signature definitions.
- Failure is retained as evidence against a universal Blotto-like architecture.
- Because latent PCC weights still exist in the Poker generator, this experiment tests architecture in engineered synthetic agents; it is not equivalent to Blotto v1.1's emergence-without-latent-PCC result.
