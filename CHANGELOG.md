# Changelog

Development history of the review-response revisions. Some artifact headers and
runner comments cite review items by round and number ("third-review item 7",
"fifth-review finding 2"); this file is the canonical map from those tags to
what changed. The manuscript itself describes the current methodology without
this history.

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
  K = 3 vs K = 18 per ticker). Result: horizon-aware median dominant share
  0.935 (n95 = 2) at K = 18, and K = 18 improves the near-band ACF match over
  K = 3 at only 1/31 tickers — the mode budget is not binding at the typical
  ticker; the paper's claim is now supported at K = 3 cross-ticker rather
  than resting on SPY alone.
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
