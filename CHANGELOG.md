# Changelog

Development history of the review-response revisions. Some artifact headers and
runner comments cite review items by round and number ("third-review item 7",
"fifth-review finding 2"); this file is the canonical map from those tags to
what changed. The manuscript itself describes the current methodology without
this history.

## Sixth-review response (2026-07)

- **Converged multistart spectral panel** (findings 1-2): the previous
  cross-ticker K = 3 / K = 18 comparison used a 60-iteration cap that left the
  K = 18 fits unconverged (final likelihood increment ~5,800x the tolerance on
  SPY) from one deterministic start. New `baum_welch_multistart` library
  routine (per-start diagnostics: n_evals, final increment, converged flag,
  cross-start LL spread); the cross-ticker runner now fits to tol = 1e-4 with
  max_iter = 4000, 3 starts at K = 3 and 5 at K = 18, saves a per-ticker
  optimizer-evidence CSV, and adds a held-out OoS ACF criterion.
- **Mode-capacity ceiling** (finding 1): new
  `run_mode_capacity_ceiling.jl` measures the best attainable m-mode geometric
  approximation of the observed sample ACF (OMP over a 400-atom half-life
  dictionary + LLS refit + local refinement). Result: the m = 2 ceiling is
  within ~0.004 MAE of the m = 17 ceiling in both bands and ~3x better than
  what likelihood fits deliver, identifying the ML objective, not the mode
  budget, as the binding constraint; the paper claim was rewritten
  accordingly, and the categorical "requires a different duration law, not a
  larger state count" was corrected to the asymptotic-vs-finite-horizon form.
- **ML HSMM duration update** (findings 4-5): the truncated discrete Pareto
  M-step previously used the continuous untruncated formula
  alpha = 1/E_w[log d]; replaced with the exact concave-likelihood maximizer
  `fit_truncated_pareto_alpha` (library, tested), the EM driver now follows
  the evaluate-before-update contract, the runner moved from `_attic/` back
  to `runners/baselines/`, the artifact was regenerated, and the manifest +
  keyed checks now cover the main-table HSMM row.
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
