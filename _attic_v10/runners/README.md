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

To restore any archived runner: `git mv _attic_v10/runners/run_X.jl ./` and
re-add to `run_full_rebuild.jl` if it should be in the default end-to-end flow.
