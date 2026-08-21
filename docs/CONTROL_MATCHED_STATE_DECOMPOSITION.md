# Poker Control matched-state decomposition (post-v0.8 synthetic extension)

This read-only diagnostic asks why Adaptive-Control converts opponent context into value-sensitive intervention while contextual Score-Control does not. It changes no policy and does not modify the frozen v0.8 human-facing panel.

## Frozen comparison

The same representative Leduc decision states are evaluated at learned opponent fold probabilities 0.10 and 0.90. Score-Control and Adaptive-Control receive the same state/context. We compare action distributions and a common one-step action-value proxy.

The prespecified candidate architectural ingredients are: aggressive response timing, passive/resistance preservation, and card/value safety.

## Result

All checks pass. Adaptive-Control has a mean matched-state expected-value advantage of about 0.460. At low fold probability (a resistant opponent), Adaptive preserves about 0.170 more probability mass on check/call. Its opponent context is explicitly two-sided in every representative state: fold vulnerability times aggression, while fold resistance boosts passive/optionality branches. Card safety is state-dependent but is not itself the opponent-context signal.

## Interpretation

The remaining Score-Control bottleneck is not simply insufficient aggression gain. The smallest supported architectural difference is **two-sided context allocation**: Control should time aggression when intervention is valuable and explicitly preserve optionality when the opponent resists. This diagnosis does not resolve Control and does not authorize automatic retuning.
