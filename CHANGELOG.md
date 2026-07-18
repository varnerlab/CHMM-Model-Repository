# Changelog

Development history of the review-response revisions. This file is the
canonical record of what changed in each round; runner sources point here
("see CHANGELOG.md") instead of citing review rounds inline (the older journal
peer-review item tags such as "Peer-review item 4 (R1 Q4)" are a separate,
stable cross-reference system and are retained). The manuscript itself
describes the current methodology without this history.

## Ninth-review response (2026-07)

- **Frontier comparator aligned with the published fits** (finding 1): the
  frontier's "likelihood fit" was a fresh single-start Baum-Welch reference,
  while the surrounding paper compares against the converged multistart fits
  of `run_spectral_rank_cross_ticker.jl` (canonical start best at only 12/31
  K = 3 tickers; max cross-start LL spread 31.1 nats). Those exact fits were
  not persisted anywhere, so `run_hmm_acf_frontier.jl` now recomputes them
  deterministically (same constants and seed formula, identical data path),
  evaluates the full frontier metric set on them, writes them as `ml_multi`
  CSV rows, and persists their parameters in every frontier JLD2; the
  certificate test ties each recomputed fit's log-likelihood back to the
  spectral artifact. The single-start `ml_ref` remains as a labeled internal
  row (it seeds the optimizer and sets the balance scale s, which is kept at
  its original provenance so the arm objectives stay byte-deterministic).
  Paper comparisons now quote `ml_multi`.
- **Per-ticker joint regret** (finding 3): the declared reading rule
  aggregated the two axes as separate cross-ticker medians, which need not be
  attained by the same tickers. The runner now also computes, per arm, the
  median over tickers of max(ACF regret, CvM regret) and the count of tickers
  with both regrets <= 1.5, writes a per-ticker
  `hmm_acf_frontier_regret.csv`, and reports this joint statistic as the
  primary panel summary (the original two-median rule is retained, labeled
  secondary). A test recomputes the regret CSV from the main CSV; the paper
  gate recomputes the quoted median and count as aggregates.
- **Achieved-feasible terminology** (finding 2): weighted-arm winners mostly
  hit the 4,000-iteration cap (lambda = 0.1: 29/31; pure-marginal: 19/31),
  and objective stall is not a stationarity certificate, so "near-optimal"
  and "multistart optima" language was replaced throughout the runner txt and
  the manuscript by achieved-feasible phrasing, with per-arm stop-reason
  counts now printed in the artifact.
- **Objective field names** (finding 7): under a custom frontier objective the
  `fit_acf_hmm` diagnostics fields `sse_init`/`sse_final` hold the weighted
  objective, not ACF SSE; parallel `objective_init`/`objective_final` fields
  and an `objective_value` return alias now carry the honest name (capacity
  certificates, fitted under the default pure-ACF objective where the sse
  names are accurate, are unchanged), the frontier JLD2s store
  `objective_value` + `lambda_scaled`, the certificate test recomputes each
  arm's objective from the stored model, and the stale `converged` docstring
  reference is fixed to `stop_reason`.
- **Runner ergonomics**: `run_hmm_acf_frontier.jl --summary-only` rebuilds the
  txt and regret CSV from the main CSV without refits (the summary layer is a
  pure function of the CSV), mirroring the capacity runner.
- **Typography gate**: the paper checker gains a section that fails on any em
  dash (TeX `---` or Unicode) in `paper.tex`/`sections/*.tex` (author
  directive; en dashes for ranges are untouched).
- Determinism: the rerun reproduces every pre-existing arm and ml_ref CSV row
  byte-identically; `ml_multi` rows, the regret CSV, and the enriched JLD2
  keys are the only additions.

## Eighth-review response (2026-07)

- **Pareto-frontier experiment** (finding 1): the seventh round's realizable
  experiment proved a valid three-state HMM can fit the finite-band ACF in
  isolation, but the paper claimed categorically that the marginal and ACF
  axes COMPETE and that ML "sides with the marginal" — a mechanism one
  unconstrained ACF solution cannot identify. New
  `run_hmm_acf_frontier.jl` sweeps the weighted objective
  ACF_SSE + lambda * s * CvM (stationary-mixture Cramer-von-Mises distance on
  a 500-point empirical-quantile grid; s = SPY likelihood-fit balance scale)
  over lambda in {0, 0.1, ..., 100} plus a pure-marginal arm, 6 starts per
  arm, full 31-ticker panel at K = 3, with a reading rule (regret-based
  competition/joint-attainability criteria) declared in the runner header
  before the run. Per-arm marginal metrics: CvM, mixture log-likelihood,
  tail-quantile errors, exact kurtosis. Full per-arm models persisted to JLD2.
  The manuscript's mechanism prose was rewritten from the outcome.
- **Capacity certificates** (finding 3): `run_hmm_acf_capacity.jl` previously
  discarded the fitted models; the txt claimed per-start diagnostics lived in
  the CSV (false). Now every (ticker, K) winner persists to
  `results/hmm_acf_capacity/*.jld2` (T, pi, mu, sigma, fitted + target ACF
  curves, per-start diagnostics with stop reasons, likelihood-seed SSE), the
  misnamed `converged` flag is `stop_reason` in {stall, iter_cap} (an
  objective-stall stop, not a stationarity certificate), and a test reloads
  every certificate and re-verifies stochasticity, stationarity, the
  recomputed population ACF, and the CSV row. The K = 3 vs K = 18 comparison
  is recast as ACHIEVED accuracy under the declared optimizer (all 31 K = 18
  winners hit the iteration cap; near-degenerate stationary laws disclosed);
  the likelihood seed is labeled precisely as a fresh single-start
  Baum-Welch refit, not the published converged multistart fit.
- **Exponential diagnostic containment claims removed** (finding 5): the
  seven-angle fixed grid does not contain every HMM ACF curve (continuous
  angles; Jordan terms absent) and the realizable K = 3 result beat the
  "containing" heuristic at equal budget, so the containment and headroom
  sentences were deleted; the dictionary is described as representative
  shapes for diagonalizable chains; artifact regenerated.
- **Accuracy/wording** (findings 6-9): HSMM sensitivity range corrected to
  0.039-0.046 (grid max 0.046037); "exact censored duration update" renamed
  grid-bracketed censored update; the initial-segment (no left-censoring)
  convention stated in the manuscript; consistency gate gained an AGGREGATE
  layer (min/max/median/count over all artifact rows vs the paper's quoted
  ranges, catching range drift mechanically); review-round tags in runner
  sources replaced with CHANGELOG pointers (four artifacts whose printed
  headers embedded tags were regenerated verbatim-identical in values).

## Seventh-review response (2026-07)

- **Capacity-ceiling claim retracted and replaced** (finding 1): the sixth
  round's `run_mode_capacity_ceiling.jl` fit free-sign positive-real
  exponentials by greedy OMP and presented the result as an HMM capacity
  bound. It is not one: the curve class is neither a superset of HMM ACFs
  (which include signed and damped-oscillatory modes) nor globally optimized,
  and a relaxation gap does not measure the feasible-set gap. The runner was
  replaced by two separated instruments: (a) `run_exp_mode_diagnostic.jl`,
  an explicitly exploratory unrestricted approximation diagnostic whose
  dictionary now spans every HMM component shape (positive/negative real +
  damped-oscillatory pairs at 2-mode budget cost), with paired per-ticker
  median gaps, persisted coefficients and fitted curves, and
  dictionary-resolution sensitivity; and (b) `run_hmm_acf_capacity.jl`, a
  realizable experiment that optimizes ACTUAL valid K-state Gaussian HMMs
  (softmax-stochastic transition rows, exact stationary law, analytic
  folded-normal moments) against the sample ACF by multistart
  finite-difference Adam — every reported error is attained by a valid HMM
  (attainability certificate; one start is seeded at the likelihood fit).
  Optimizer core in `acf_capacity_common.jl` / `exp_mode_common.jl`, both
  unit-tested (recovery, parametrization roundtrip, budget accounting).
  All "binding constraint" / "best attainable" / "ceiling" manuscript claims
  were rewritten from the realizable artifact.
- **HSMM right-censored terminal segment** (findings 2-4): the ML HSMM's
  likelihood conditioned every segmentation on a sojourn boundary exactly at
  T while its simulator right-truncated an ongoing final sojourn. The EM core
  (extracted to `hsmm_core.jl`) now marginalizes the terminal segment through
  the duration survival function (start-forward restructure, censored
  posterior eta_c, boundary-correct gamma), the duration M-step includes the
  expected censored counts (library `fit_truncated_pareto_alpha` gains
  `censored_counts` + `truncated_pareto_logsf`; grid-bracketed golden section
  since censoring breaks concavity), and the headline fits are multistart
  local-EM estimates (5 starts at K = 3, 3 at K = 18) with per-start
  diagnostics. A K = 3 sensitivity grid over D_max in {100, 200, 400} x
  alpha lower bound in {0.02, 0.05, 0.2} quantifies the boundary/truncation
  dependence of the fitted duration law. Brute-force tiny-window enumeration
  tests pin the censored E-step exactly (likelihood both recursion
  directions, occupancy, transitions, completed + censored duration counts).
  `run_hsmm_gamma.jl` was rewritten as a thin driver over the shared core
  (pluggable duration M-step; its Gamma moment update is disclosed as
  completed-counts-only), removing another private EM copy.
- **Wording/scoping** (findings 5-8, 10): "pre-specified" for the held-out
  ACF criterion replaced with declared-criterion language; the absolute
  "not used for any spectral or ACF claim" sentence scoped to the subsection;
  "any of the likelihood fits" corrected to median errors; K = 2 and K-sweep
  "confirms" language reframed as fitted-outcome evidence with the lambda =
  0.942 half-life (~11.6 days) stated; paper README title/BibTeX synced to
  "Continuous-Emission"; vendor-stitch history note reworded ahistorically.

## Sixth-review response (2026-07)

- **Converged multistart spectral panel** (findings 1-2): the previous
  cross-ticker K = 3 / K = 18 comparison used a 60-iteration cap that left the
  K = 18 fits unconverged (final likelihood increment ~5,800x the tolerance on
  SPY) from one deterministic start. New `baum_welch_multistart` library
  routine (per-start diagnostics: n_evals, final increment, converged flag,
  cross-start LL spread); the cross-ticker runner now fits to tol = 1e-4 with
  max_iter = 4000, 3 starts at K = 3 and 5 at K = 18, saves a per-ticker
  optimizer-evidence CSV, and adds a held-out OoS ACF criterion.
- **Mode-capacity ceiling** (finding 1) — SUPERSEDED by the seventh-review
  response: `run_mode_capacity_ceiling.jl` measured the best positive-real
  free-coefficient m-mode approximation of the sample ACF and presented it as
  an attainability ceiling identifying the ML objective as the binding
  constraint. The seventh review showed this inference invalid (the curve
  class is not a superset of HMM ACFs and the greedy fit carries no global
  certificate); the runner was replaced by `run_exp_mode_diagnostic.jl` +
  `run_hmm_acf_capacity.jl` above. The asymptotic-vs-finite-horizon
  duration-law correction from this round stands.
- **ML HSMM duration update** (findings 4-5): the truncated discrete Pareto
  M-step previously used the continuous untruncated formula
  alpha = 1/E_w[log d]; replaced with the exact concave-likelihood maximizer
  `fit_truncated_pareto_alpha` (library, tested), the EM driver now follows
  the evaluate-before-update contract, the runner moved from `_attic/` back
  to `runners/baselines/`, the artifact was regenerated, and the manifest +
  keyed checks now cover the main-table HSMM row. (Seventh review: the
  terminal-segment convention was subsequently corrected to right-censored,
  the fit made multistart, and the EM core extracted to `hsmm_core.jl`;
  this round's artifact numbers are superseded.)
- **Numerics/statistics polish** (findings 6-8): stationary distributions by
  checked left-eigenvector solve (uniqueness, non-negativity, residual
  asserts) instead of fixed matrix powers; analytic folded-normal moments in
  the spectral runners; winsorized-kurtosis inference scoped by explicit
  stationarity/quantile-regularity assumptions with the ~6-obs-per-tail
  q = 0.01 caveat.

## Fifth-review response (2026-07)

- **Copula 2×2 strict CRN** (finding 2): the headline quarter-level
  `run_cross_asset_rolling_copula.jl` now calls the four-argument
  `simulate(model, T, n_paths, crn_seed)` interface in all four arms, so the
  Gaussian and Student-t arms consume identical base normals and marginal
  draws (previously seed-paired only, which let the t-arm chi-square stream
  shift subsequent draws). Regenerated: arm means essentially unchanged
  (0.264/0.186 t, 0.261/0.179 G); family contrasts now precisely resolved
  (0.003/0.007, slightly favouring Gaussian, nominally significant) and still
  an order of magnitude below the refit effects (0.077/0.082). Source-level
  test guards the call sites.
- **Cross-ticker spectral diagnostic at K = 3 + horizon-aware norm**
  (finding 1): `run_spectral_rank_cross_ticker.jl` now runs at K in {3, 18},
  ranks components under both the lag-1 and an integrated lags-1..252 norm,
  and adds the direct binding-ness check (model-vs-sample |G| ACF MAE at
  K = 3 vs K = 18 per ticker). The specific share and 1/31 figures of this
  round came from 60-iteration-capped K = 18 fits and were superseded by the
  sixth-review converged multistart rerun below; the qualitative reading
  (extra states do not improve the ACF match at the typical ticker) stands.
- **ACF bootstrap band restricted to lags 1–5** (finding 3): block-boundary
  contamination is gradual (fraction ~h/L at lag h), so the L = 20 band is
  displayed only over a conservative low-lag range instead of through lag 20.
- **Tail-robust kurtosis inference** (finding 4): raw excess-kurtosis
  resampling bands relabeled descriptive (Hill ~3.15 < 4 means the population
  fourth moment is plausibly infinite); the inferential object is now the
  winsorized excess kurtosis (trim q per tail, thresholds recomputed per
  resample), whose IS−OoS difference CI covers zero at every block length.
- **Hill "matched" language** (finding 5) replaced with
  compatibility-at-threshold language; **"at the i.i.d. floor"** (finding 6)
  replaced with "does not beat the floor (0.041 vs 0.036)"; **"matches the
  heavy-tailed marginal"** (finding 7) narrowed to "substantially improves".
- **Hygiene** (code findings): `_auc` precedence bug fixed and tested;
  README claims synced with the paper; RUNNERS.md kurtosis path corrected;
  Types.jl GED identifiability docstring aligned with the paper's caveat.
- **Provenance**: remaining `unverified` manifest rows resolved; additional
  keyed checks in `run_paper_artifact_check.jl`.

## Fourth-review response (2026-07, commit b281f09)

Shared-ν Student-t fitter moved into the library under the
best-evaluated-iterate contract; ACF band dropped beyond the block scale;
two-independent-resample KS null (flipped the "stricter" narrative); 2×2
family/refit copula experiment on identical quarter targets; kurtosis
difference CIs replacing a p-value mislabel; DQ bootstrap B = 2000 with
two-row scope; manifest status gate + keyed checks in the consistency runner.

## Third-review response (2026-07, commit 02d3688)

Honest vendor provenance + feed-boundary appendix; paired K = 3
rolling-copula design; 573-aligned MS-GARCH VaR block with finite-sample DQ
calibration; single-convention per-path ACF column; grouped-component
spectral diagnostics; five-row Holm-corrected CRPS/DM panel; ACF horizon
figure; evaluate-before-update EM contract with explicit T−1 limits.

## Earlier rounds

First and second third-party reviews: EM checkpoint/return-contract repairs
across the four emission families, VaR breach-flag and forecast-alignment
fixes, artifact/manifest introduction. See git history for details.
