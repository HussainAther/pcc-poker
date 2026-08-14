# Pre-human analysis preregistration

## Administrative gate

This protocol is written before confirmatory analysis of the human HandHQ records. It becomes executable only after the applicable Georgia Tech ORIA/IRB determination or approval. Until then, development and tests use synthetic or invented PHH fixtures only.

## Data source and exclusions

The intended source is the HandHQ online-hand-history portion of Zenodo record `13997158`. The analysis scope is restricted to the online HandHQ records. Televised WSOP data, Pluribus data, and named/historical example hands are excluded.

During ingestion, persistent source player strings are replaced with study-specific pseudonymous IDs where grouping is necessary. No reverse lookup is retained. Source table names, source hand IDs, exact date/time fields, timezone metadata, and other nonessential source metadata are excluded from modeling data.

## Unit of analysis

The elementary unit is a legal player decision reconstructed from a complete hand. Decisions remain nested within hands and study-specific player IDs for grouping and inference.

## Calibration/evaluation split

The confirmatory split is made before model fitting. When persistent obfuscated player identifiers are available, all decisions from a player belong to only one side of the calibration/evaluation boundary. Hands never cross the boundary. If player grouping is unavailable for a subset, complete hands are the minimum grouping unit and that subset is reported separately as a sensitivity analysis rather than silently mixed into the primary analysis.

No confirmatory evaluation decision is used to tune feature definitions, thresholds, smoothing constants, model class, or inclusion criteria.

## Primary confirmatory question

The human confirmatory phase evaluates only the cross-family-invariant **Pressure** measurement panel frozen in `HUMAN_MEASUREMENT_CONTRACT.md`.

### Endpoint P1 — held-out fold discrimination

For evaluation decisions where an opponent faces a wager and fold is legal, test whether frozen-model `predicted_fold_probability` discriminates the observed fold response above chance. The primary statistic is ROC AUC. Confirmation requires a player-cluster bootstrap 95% confidence interval whose lower bound is greater than `0.50`.

### Endpoint P2 — held-out Pressure exposure association

For the same evaluation response states, fit a prespecified logistic model for observed opponent fold with `pressure_exposure` as the focal predictor and the frozen public state covariates already permitted by the sanitized feature contract as adjustment variables. The primary statistic is the standardized `pressure_exposure` coefficient. Confirmation requires its player-cluster bootstrap 95% confidence interval to be entirely above zero.

The two Pressure endpoints are co-primary. Family-wise type-I error is controlled by Holm correction at two-sided alpha `0.05`; both raw and adjusted p-values are reported. Effect sizes and confidence intervals are reported regardless of significance.

## Secondary descriptive analyses

Report distributions of the two Pressure components by street, wager-size/commitment bins, and legal-action context using prespecified public-state strata. These analyses are descriptive and do not replace the co-primary endpoints.

## Exploratory Control and Chaos analyses

Control- and Chaos-adjacent observables may be computed only under the frozen definitions documented in the repo and are labeled exploratory. They are not used to declare confirmation of the Control or Chaos axes. No multiplicity-adjusted confirmatory claim is made from them in this release.

## Missingness and exclusions

A decision is excluded only when the public state cannot be reconstructed unambiguously, the acting/responding player mapping is invalid, the action is unsupported by the parser, or required stack/contribution information is internally inconsistent. Counts and reasons are reported. Exclusion rules are not altered after evaluation outcomes are examined.

## Inference and clustering

Primary uncertainty uses nonparametric bootstrap resampling at the study-specific player level. Complete hands remain intact within resamples. A hand-cluster bootstrap is reported as a sensitivity analysis. If player-level grouping is unavailable for a material fraction of data, that limitation is disclosed and the primary inference is not silently replaced.

## Robustness checks

Prespecified sensitivity analyses are:

- evaluation by poker network/venue, without naming individual players;
- alternative hand-cluster bootstrap;
- exclusion of records with ambiguous stack or action reconstruction;
- results by broad wager/commitment bins; and
- a negative-control analysis in which evaluation response labels are permuted within public-state strata.

These checks do not change the primary endpoint definitions.

## Stopping rule

The confirmatory evaluation is run once after the calibration pipeline and QA checks are frozen. Unexpected engineering failures may be corrected only if the correction is outcome-blind; the change, reason, affected files, and new checksum must be documented in a dated amendment before rerunning the affected confirmatory endpoint.

## Reporting rule

Null, partial, and contradictory findings are retained. The publication must distinguish: (a) synthetic construct evidence, (b) human confirmatory Pressure results, and (c) exploratory Control/Chaos diagnostics.
