# RUNNERS.md — runner-to-paper-artefact map

Each row maps a `runners/<theme>/run_*.jl` script to (a) the artefact it produces
under `results/<subdir>/`, and (b) the paper section, table, or figure that
consumes it. Use this as the reproducibility audit: pick a paper artefact, find
its runner, inspect the script.

The runners are grouped on disk into seven theme folders that mirror the
sections below:

```
runners/
  headline/         body Section 5 (Empirical Study) pipeline
  var_backtest/     body Section 5 (VaR backtest subsection) diagnostics
  robustness/       walk-forward, cross-decade, K-selection sensitivity
  spectral/         body Section 4 (Spectral Mechanism)
  cross_asset/      cross-asset extras (copula CI, non-US stress test)
  baselines/        Appendix B extended baselines (SV / MSM / JD / HSMM, ...)
  diagnostics/      catch-all
```

`run_full_rebuild.jl` (kept at the repo root, since it is the main entry point)
is the end-to-end orchestrator. It excludes the slow QuantGAN stage by default;
see the script's header comment for the inclusion toggle.

All commands below assume the working directory is the CHMM-Model repo root,
e.g. `julia --project=. runners/headline/run_all_analysis.jl`.

## Headline pipeline (body Section 5 — Empirical Study)

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/headline/run_all_analysis.jl` | `results/SPY/{stylized_facts,K*}/...` | Body §5.1 (descriptive); §5.2 (state selection); per-K internals |
| `runners/headline/run_multi_emission_analysis.jl` | `results/SPY/multi_emission/...` | Body Table 2 (CHMM-N/-t/-L/-GED at $K^\star = 3$ block) |
| `runners/headline/run_baselines_and_cross_asset.jl` | `results/SPY/Table-2-Baselines.txt`, `results/cross_asset/...` | Body Table 2 (i.i.d., GARCH, MS-GARCH in-house rows); body Table 4 (cross-asset Pipeline B) |
| `runners/headline/run_msgarch_baselines.jl` | `results/msgarch_baselines/...` | Body Table 2 MS-GARCH $K \in \{2, 3, 6\}$ rows (Nelder-Mead) |
| `runners/headline/run_msgarch_reference.jl` | `results/msgarch_reference/...` | Body Table 2 MS-GARCH ref. Bayesian rows (CRAN `MSGARCH`, requires R) |
| `runners/headline/run_smchmm_baseline.jl` | `results/smchmm_baseline/...` | Body Table 2 ML HSMM-N at $K^\star = 3$ co-headline row |
| `runners/headline/run_quantgan_baseline.jl` | `results/quantgan_baseline/...` | Body Table 2 QuantGAN row + Appendix QuantGAN spec |
| `runners/headline/run_cross_asset_sim_copula.jl` | `results/cross_asset/...` | Body Table 4 (Pipeline B Student-$t$ copula at $\nu^\star = 6$) |
| `runners/headline/run_cross_ticker_penalised.jl` | `results/chmm_t_penalised/...` | Body Table 3 (penalised CHMM-t cross-ticker headline) |
| `runners/headline/run_sector_panel.jl` | `results/sector_panel/sector_panel_summary.{csv,txt}` | Body Table 3 (30-ticker rollup); Appendix sector panel |
| `runners/headline/run_chmm_t_shared_nu.jl` | `results/chmm_t_shared_nu/...` | Body Table 2 footnoted shared-$\nu$ row + Appendix `sec:chmm_t_shared_nu` |
| `runners/headline/run_figures.jl` | `figs/Fig-{1..5}-*.pdf` | Body Figures 1, 2, 3, 4 |

## VaR / conditional-coverage diagnostics (body §5 Empirical Study, VaR Backtest subsection)

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/var_backtest/run_conditional_var_all_families.jl` | `results/conditional_var_all_families/...` | Body Table 5 (regime-conditional VaR Christoffersen-cc); Appendix four-family extension |
| `runners/var_backtest/run_christoffersen_power.jl` | `results/diagnostics/christoffersen_power/...` | Appendix `sec:christoffersen_power` (Monte Carlo power calibration) |
| `runners/var_backtest/run_engle_manganelli_dq.jl` | `results/diagnostics/engle_manganelli_dq.txt` | Appendix `sec:engle_manganelli_dq` (DQ test backstop) |
| `runners/var_backtest/run_quarterly_refit_conditional_var.jl` | `results/quarterly_refit_conditional_var/...` | Appendix `sec:quarterly_refit_cond_var` |
| `runners/var_backtest/run_walkforward_conditional_var.jl` | `results/walkforward/walkforward_conditional_var.{csv,txt}` | Body Table~\ref{tab:walkforward_cond_var} (six-fold walk-forward regime-conditional VaR; Christoffersen-cc passes 19/24) |

## Walk-forward + cross-decade + cross-ticker robustness

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/robustness/run_walkforward_oos.jl` | `results/walkforward/walkforward_summary.{csv,txt}` | Body Table~\ref{tab:walkforward} (six-fold rolling-origin OoS; CHMM-N at $K \in \{3, 18\}$, median KS 62.1\% / 67.7\%) |
| `runners/robustness/run_cross_decade_validation.jl` | `results/cross_decade_validation/...` | Appendix `sec:cross_decade_validation` (CRSP 1994-2006 IS / OoS) |
| `runners/robustness/run_sector_panel_quarterly_refit.jl` | `results/sector_panel/sector_panel_quarterly_refit.{csv,txt}` | Appendix `sec:cross_ticker_quarterly_refit` |
| `runners/robustness/run_kurtosis_bootstrap.jl` | `results/SPY/diagnostics/kurtosis_bootstrap.txt` | Appendix `sec:kurtosis_bootstrap_ci` |
| `runners/robustness/run_kurtosis_ci_placement.jl` | `results/kurtosis_ci_placement/...` | Appendix `sec:kurtosis_ci_placement` |
| `runners/robustness/run_lambda_cv_pre2020.jl` | `results/diagnostics/lambda_cv_pre2020/...` | Appendix `sec:lambda_cv_pre2020` ($1/\nu_k$ penalty CV) |
| `runners/robustness/run_k_selection_kfold_pre2020.jl` | `results/k_selection_validation/...` | Appendix `sec:k_selection_kfold_pre2020` (single + four-fold CV at body $K^\star = 3$) |
| `runners/robustness/run_k_selection_kfold_h12y_pre2020.jl` | `results/k_selection_validation/h12y/...` | Six-fold rolling-origin CV (referenced in body §5.2) |
| `runners/robustness/run_k_selection_hac.jl` | `results/k_selection_hac/...` | Appendix `sec:k_selection_hac` (HAC-corrected K selection) |

## Spectral + theoretical diagnostics (body §4 — Spectral Mechanism)

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/spectral/run_spectral_rank.jl` | `results/diagnostics/spectral_rank.txt` | Body Table 1 + Appendix `sec:spectral_rank` (SPY effective rank diagnostic) |
| `runners/spectral/run_spectral_rank_cross_ticker.jl` | `results/diagnostics/spectral_rank_cross_ticker.txt` | Appendix `sec:spectral_rank_xticker` (cross-ticker dominant-mode share) |

## Cross-asset extras

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/cross_asset/run_copula_profile_ci_halfunit.jl` | `results/copula_profile_ci/...` | Appendix `sec:copula_halfunit` (half-unit-grid Wilks CI) |
| `runners/cross_asset/run_non_us_asset.jl` | `results/non_us_asset/...` | Appendix `sec:non_us_asset_supp` (GLD / SLV stress test) |
| `runners/cross_asset/run_non_us_asset_quarterly_refit.jl` | `results/non_us_asset/Non_US_Asset_QuarterlyRefit.{txt,csv}` | Item 11 of REVIEW_RESPONSE_PLAN.md (GLD quarterly-refit follow-up). Configurable via `GLD_REFIT_K`, `GLD_REFIT_FAMILY` env vars. |

## Auxiliary baselines (Appendix B — extended baselines)

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/baselines/run_sv_msm_jd_baselines.jl` | `results/sv_msm_jd/...` | Appendix `sec:sv_msm_jd_baselines` (SV-AR(1), MSM, Merton-JD) |
| `runners/baselines/run_ged_diagnostics.jl` | `results/ged_diagnostics/...` | Appendix `sec:supp_p_partition` ($\hat p_k$ partition diagnostic) |
| `runners/baselines/run_leverage_effect.jl` | `results/diagnostics/leverage_effect/...` | Appendix `sec:leverage_effect` |
| `runners/baselines/run_ks_block_bootstrap_oos.jl` | `results/ks_block_bootstrap/...` | Appendix `sec:ks_block_bootstrap` (OoS-anchored block-bootstrap KS, including body operating-point summary at $L = 20$) |
| `runners/baselines/run_hsmm_ml_gamma.jl` | `results/hsmm_ml_gamma/...` | Appendix `sec:hsmm_gamma_sojourn` (Gamma-sojourn HSMM at $K = 18$) |
| `runners/baselines/run_filtered_bootstrap_var.jl` | `results/filtered_bootstrap_var/...` | Item 6a of REVIEW_RESPONSE_PLAN.md (Hull-White-style filtered historical-simulation VaR contender for body Section 5). |
| `runners/baselines/run_caviar_var.jl` | `results/caviar_var/...` | Item 6b of REVIEW_RESPONSE_PLAN.md (Engle-Manganelli SAV CAViaR contender for body Section 5). |

## Diagnostics + miscellanea

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/diagnostics/run_diagnostics.jl` | `results/diagnostics/...` (catch-all) | Various appendix diagnostics |
| `runners/diagnostics/run_vendor_stitch_check.jl` | `results/diagnostics/vendor_stitch_check.{txt,csv}` | Item 9 of REVIEW_RESPONSE_PLAN.md (Polygon vs Alpaca stitch sanity check on the OoS overlap window). |

## Archived runners

The following runners were moved to `_attic_v10/runners/` during the arXiv-prep
pass because their outputs are no longer cited in the trimmed paper. They are
preserved for full historical reproducibility of the pre-arXiv revisions.

See `_attic_v10/runners/README.md` for the full archive and the rationale for
each move.
