# `_attic_v10/runners/` — archived runners

This directory holds runners whose outputs are no longer cited by any current
paper artefact. They are preserved for full historical reproducibility of the
pre-arXiv revisions; nothing in `run_full_rebuild.jl` references them.

## Pre-arXiv-prep archive (older revisions)

| Runner | Original purpose |
|---|---|
| `regen_var_es_fig.jl` | One-shot VaR/ES figure regeneration; superseded by `run_figures.jl` |
| `run_copula_profile_ci.jl` | First-pass profile-LL CI; superseded by `run_copula_profile_ci_halfunit.jl` |
| `run_crps_dm.jl` | First-pass CRPS DM test; superseded by `run_crps_dm_bandwidth.jl` |
| `run_crps_extra_rows.jl` | One-shot CRPS row addendum; superseded by `run_baselines_and_cross_asset.jl` |
| `run_ged_bracket_sensitivity.jl` | Earlier-revision GED bracket sweep; superseded by `run_ged_diagnostics.jl` |
| `run_ged_robustness.jl` | Earlier-revision GED multiseed; superseded by `run_ged_diagnostics.jl` |
| `run_gru_baseline.jl` | GRU + Gaussian-head deep-generative baseline (replaced by QuantGAN row in body) |
| `run_hsmm_ml.jl` | First-pass ML HSMM; superseded by `run_smchmm_baseline.jl` |
| `run_multiseed_headline.jl` | Multiseed Monte Carlo on headline metrics (results kept in CHMM-Model/results/multiseed_headline/) |
| `run_track_a_metrics.jl`, `run_track_a_utility.jl`, `run_track_b3_diffusion.jl`, `run_track_c3_*.jl`, `run_track_c4_leverage_emission.jl` | Earlier "track" exploration runners that were later folded into the diagnostics pipeline |

## arXiv-prep archive (this pass)

Moved to this directory because the corresponding paper sections were trimmed
during the arXiv-readiness pass; the underlying results files in `results/` are
unchanged and still present, only the runner is no longer in the active set.

| Runner | Why archived |
|---|---|
| `run_cross_ticker_anova.jl` | Sector-vs-ticker ANOVA; result is now a one-line body sentence ("sector membership explains $\sim 16\%$ of OoS-KS variance under an underpowered $n = 3$ design") |
| `run_walkforward_w7.jl` | W7a/W7b 2018-2019 fold extension; superseded by the body six-fold walk-forward |
| `run_exact_binomial_kupiec.jl` | Exact-binomial Kupiec complement; the asymptotic LR Kupiec already in body Table 5 carries the diagnostic |
| `run_hsmm_ml_intermediate_K.jl` | Intermediate-$K$ HSMM rows; the body $K^\star = 3$ ML HSMM-N is the headline HSMM row |
| `run_emission_family_frobenius.jl` | Pairwise Frobenius distances between emission families; the four-family ablation already separates Gaussian from heavy-tail in Table 2 |
| `run_oos_regime_trajectory.jl` | Volatility-rank band aggregates; now a one-paragraph reference in the body per-state interpretability discussion |
| `run_non_equity_validation.jl` | Earlier-revision non-equity stress test (GLD/SLV); superseded by `run_non_us_asset.jl` (body cross_asset_appendix `sec:non_us_asset_supp`) |
| `run_subdecade_validation.jl` | 2014-2024 5-year sub-window validation; superseded by the cross-decade CRSP run (`run_cross_decade_validation.jl`) |
| `run_sector_panel_n6.jl` | 60-ticker sector ANOVA at $n = 6$; arXiv-prep stub in `sec:sector_panel_n6` retains the headline numbers and points to the archived CSV |
| `run_sector_panel_n6_postprocess.jl` | Companion postprocess for `run_sector_panel_n6.jl`; archived together |
| `run_walkforward_cond_var_refit_cadence.jl` | Walk-forward refit-cadence sweep (monthly / weekly); arXiv-prep stub in `sec:walkforward_refit_cadence` retains the rejection-count summary |
| `run_sector_panel_quarterly_refit_k6.jl` | $K = 6$ quarterly-refit cross-ticker row; the $K = 18$ row in `sec:cross_ticker_quarterly_refit` already carries the body claim |
| `run_figures_ksweep.jl` | Per-$K$ visual sweep figures; the appendix figure section was collapsed to a one-line pointer in the arXiv-prep pass |

## arXiv-prep cleanup pass (2026-05-02)

Second-pass sweep against the trimmed paper appendix. These runners produced
artefacts that no surviving paper section cites, or were one-shot setup utilities
no longer needed for reproduction.

| Runner | Why archived |
|---|---|
| `run_cross_asset_sim_copula_k6.jl` | $K = 6$ Pipeline B variant; `sec:cross_asset_kstar6` was cut in the arXiv-prep appendix trim, and the $K = 3$ / $K = 18$ sandwich already brackets the body claim |
| `run_mssv_baseline.jl` | Markov-switching stochastic-volatility baseline; `sec:mssv_baseline` is on the arXiv-prep cut list and the SV-AR(1) row in `run_sv_msm_jd_baselines.jl` already covers the SV-family comparison |
| _(restored 2026-05-06)_ `build_new_train_oos.jl` was previously listed here; on 2026-05-06 it was moved back to the repo root because `PROFF_PREP.md` §13.4 documents it as part of the reproducer recipe (`julia --project=. build_new_train_oos.jl`). The pre-built JLD2 splits are still committed under `data/`, so re-running the script is optional. |

## 15-page main-body trim (2026-05-03)

Third-pass sweep alongside the 21-pp $\to$ 15-pp main-body cut and the
companion supplementary pruning. The corresponding paper appendix subsections
were dropped, so the runners that produced their numbers are no longer cited
by any surviving paper artefact.

| Runner | Why archived |
|---|---|
| `run_state_distinctness.jl` | `sec:state_distinctness` appendix dropped; the body $K_{\text{eff}}$ claim was removed from the state-selection paragraph and the standalone single-linkage-clustering write-up is no longer cited |
| `run_crps_dm_bandwidth.jl` | `sec:dm_bandwidth` appendix dropped; the body's "within-CHMM family equivalence is not bandwidth-fragile" claim is now carried inline by the headline CRPS DM result without the bandwidth-sweep write-up |

## Archived `src/` modules

Companion to `_attic_v10/runners/` for source modules that load cleanly but are
not reachable from any active runner or test.

| Module | Why archived |
|---|---|
| `src/SkewEmissions.jl` | Skew-$t$ / skew-Normal emission scaffolding for a future leverage-effect CHMM variant; the body "Out of scope" paragraph defers skew emissions to a companion paper, and no current runner or test references the module |

To restore any archived file: `git mv _attic_v10/runners/run_X.jl ./` (or
`_attic_v10/src/Module.jl src/`) and re-add it to `run_full_rebuild.jl` /
`Include.jl` as appropriate.
