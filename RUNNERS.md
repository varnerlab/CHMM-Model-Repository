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
| `runners/headline/run_chmm_t_penalised_headline.jl` | `results/chmm_t_penalised/Headline_CHMM_t_Pen.txt` | SPY $K = 18$ penalised CHMM-t headline (IS/OoS KS, kurtosis; shared-$\nu$ comparison block + Appendix penalised-t reference) |
| `runners/headline/run_msgarch_higher_k.jl` | `results/msgarch_baselines/MSGARCH_higher_K.txt` | Body Table 2 / Appendix MS-GARCH higher-$K$ rows (extends `run_msgarch_baselines.jl`) |
| `runners/headline/run_figures.jl` | `figs/Fig-{1..5}-*.pdf` | Body Figures 1, 2, 3, 4 |
| `runners/headline/run_acf_horizon_diagnostics.jl` | `results/acf_horizon/{acf_horizon.txt,acf_curves.csv}`, paper `figs/Fig-ACF-Observed-vs-Simulated.pdf` | Body observed-vs-simulated \|G_t\| ACF figure + horizon-banded ACF-MAE table (lags 1-5, 6-20, 21-63, 64-252) with i.i.d. floors; theoretical population ACF via the matrix formula (third-review item 8) |

## VaR / conditional-coverage diagnostics (body §5 Empirical Study, VaR Backtest subsection)

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/var_backtest/run_conditional_var_all_families.jl` | `results/conditional_var_all_families/...` | Body Table 5 (regime-conditional VaR Christoffersen-cc); Appendix four-family extension |
| `runners/var_backtest/run_christoffersen_power.jl` | `results/diagnostics/christoffersen_power/...` | Appendix `sec:christoffersen_power` (Monte Carlo power calibration) |
| `runners/var_backtest/run_engle_manganelli_dq.jl` | `results/diagnostics/engle_manganelli_dq.txt` | Appendix `sec:engle_manganelli_dq` (DQ test backstop) |
| `runners/var_backtest/run_quarterly_refit_conditional_var.jl` | `results/quarterly_refit_conditional_var/...` | Appendix `sec:quarterly_refit_cond_var` |
| `runners/var_backtest/run_walkforward_conditional_var.jl` | `results/walkforward/walkforward_conditional_var.{csv,txt}` | Body Table~\ref{tab:walkforward_cond_var} (six-fold walk-forward regime-conditional VaR; Christoffersen-cc passes 19/24) |
| `runners/var_backtest/run_msgarch_conditional_var.jl` | `results/msgarch_conditional_var/...` + paper `results/robustness/msgarch_conditional_var.csv` | MS-GARCH conditional-VaR block of the main VaR table ($K \in \{2,3,4\}$, forecast origins aligned at 573 with the CHMM harness; replaces the previously untraced 572-forecast panel) |
| `runners/var_backtest/run_var_inference_upgrade.jl` | `results/var_inference_upgrade/var_inference_upgrade.{csv,txt}` | VaR-table inference upgrades: exact binomial UC p-values, parametric-bootstrap DQ calibration at 1% (CHMM-N $K=3$, MS-GARCH $K=4$), paired pinball-loss comparison with Newey-West t-stats. Run `run_msgarch_conditional_var.jl` first. |

## Walk-forward + cross-decade + cross-ticker robustness

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/robustness/run_walkforward_oos.jl` | `results/walkforward/walkforward_summary.{csv,txt}` | Body Table~\ref{tab:walkforward} (six-fold rolling-origin OoS; CHMM-N at $K \in \{3, 18\}$, median KS 62.1\% / 67.7\%) |
| `runners/robustness/run_cross_decade_validation.jl` | `results/cross_decade_validation/...` | Appendix `sec:cross_decade_validation` (CRSP 1994-2006 IS / OoS) |
| `runners/robustness/run_sector_panel_quarterly_refit.jl` | `results/sector_panel/sector_panel_quarterly_refit.{csv,txt}` | Appendix `sec:cross_ticker_quarterly_refit` |
| `runners/robustness/run_sector_panel_monthly_refit.jl` | `results/sector_panel/sector_panel_monthly_refit.{csv,txt}` | Body monthly-cadence reference ($K = 18$, OoS KS median 86.7\%, 5-of-30) + Appendix stationarity-scope panel. Heavy ($\sim 3\times$ quarterly compute) |
| `runners/robustness/run_kurtosis_bootstrap.jl` | `results/SPY/diagnostics/kurtosis_bootstrap.txt` | Appendix `sec:kurtosis_bootstrap_ci` |
| `runners/robustness/run_kurtosis_ci_placement.jl` | `results/kurtosis_ci_placement/...` | Appendix `sec:kurtosis_ci_placement` |
| `runners/robustness/run_lambda_cv_pre2020.jl` | `results/nu_shrinkage_sweep/lambda_cv_pre2020_k{18,3}.txt` (set `LAMBDA_CV_K`) | Appendix `sec:lambda_cv_pre2020` ($1/\nu_k$ penalty CV; K=18 backs the table, K=3 the re-tuning paragraph) |
| `runners/robustness/run_k_selection_kfold_pre2020.jl` | `results/k_selection_validation/...` | Appendix `sec:k_selection_kfold_pre2020` (single + four-fold CV at body $K^\star = 3$) |
| `runners/robustness/run_k_selection_kfold_h12y_pre2020.jl` | `results/k_selection_validation/h12y/...` | Six-fold rolling-origin CV (referenced in body §5.2) |
| `runners/robustness/run_k_selection_hac.jl` | `results/k_selection_hac/...` | Appendix `sec:k_selection_hac` (HAC-corrected K selection) |
| `runners/robustness/run_k_selection_validation_pre2020.jl` | `results/k_selection_validation/K_Selection_Validation_Pre2020.txt` (paper `k_selection_validation_pre2020.csv`) | Appendix pre-2020 single-split K-selection validation (companion to the kfold runner) |
| `runners/robustness/run_nu_shrinkage_sweep.jl` | `results/nu_shrinkage_sweep/NU_Shrinkage_Sweep.txt` (paper `nu_shrinkage_sweep.csv`) | Appendix $1/\nu_k$ penalty $\lambda$ rate-sweep on the penalised CHMM-t |
| `runners/robustness/run_crps_dm_kstar3.jl` | `results/crps_dm_kstar3/{crps_dm_kstar3.txt,per_day_losses.csv}` (paper `crps_dm_kstar3.csv`) | Body Table 2 CRPS inference: HAC (Newey-West) Diebold-Mariano on all 10 pairs among the five $K^\star = 3$ CHMM rows (N, penalised t, L, GED, shared-$\nu$ t), Holm-corrected |

## Spectral + theoretical diagnostics (body §4 — Spectral Mechanism)

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/spectral/run_spectral_rank.jl` | `results/diagnostics/spectral_rank.txt` | Body Table 1 + Appendix `sec:spectral_rank` (SPY effective rank diagnostic) |
| `runners/spectral/run_spectral_rank_cross_ticker.jl` | `results/diagnostics/spectral_rank_cross_ticker.txt` | Appendix `sec:spectral_rank_xticker` (cross-ticker dominant-mode share) |
| `runners/spectral/run_t_singular_values.jl` | `results/diagnostics/t_singular_values.txt` | Supplementary transition-matrix singular-value diagnostic (per emission family) |

## Cross-asset extras

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/cross_asset/run_copula_profile_ci_halfunit.jl` | `results/copula_profile_ci/...` | Appendix `sec:copula_halfunit` (half-unit-grid Wilks CI) |
| `runners/cross_asset/run_non_us_asset.jl` | `results/non_us_asset/...` | Appendix `sec:non_us_asset_supp` (GLD / SLV stress test) |
| `runners/cross_asset/run_non_us_asset_quarterly_refit.jl` | `results/non_us_asset/Non_US_Asset_QuarterlyRefit.{txt,csv}` | Item 11 of REVIEW_RESPONSE_PLAN.md (GLD quarterly-refit follow-up). Configurable via `GLD_REFIT_K`, `GLD_REFIT_FAMILY` env vars. |
| `runners/cross_asset/run_full_tcopula_mle.jl` | `results/copula_profile_ci/full_tcopula_mle.txt` (paper `full_tcopula_mle.csv`) | Body Table 4 one-shot joint $(\Sigma, \nu)$ MLE confirming the two-step $\nu^\star = 6$ |
| `runners/cross_asset/run_cross_asset_rolling_copula.jl` | `results/cross_asset/Rolling_Copula_OoS.txt` (paper `rolling_copula_oos.csv`) | Appendix `tab:rolling_copula` 2x2 family/refit experiment: {static, rolling} x {Gaussian, Student-t} on identical quarter targets; refit effect 0.263 -> 0.185 (t), 0.260 -> 0.178 (G); family effect 0.003-0.007; partial block excluded from paired inference |

## Auxiliary baselines (Appendix B — extended baselines)

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/baselines/run_sv_msm_jd_baselines.jl` | `results/sv_msm_jd/...` | Appendix `sec:sv_msm_jd_baselines` (SV-AR(1), MSM, Merton-JD) |
| `runners/baselines/run_ged_diagnostics.jl` | `results/ged_diagnostics/...` | Appendix `sec:supp_p_partition` ($\hat p_k$ partition diagnostic) |
| `runners/baselines/run_leverage_effect.jl` | `results/diagnostics/leverage_effect/...` | Appendix `sec:leverage_effect` |
| `runners/baselines/run_ks_block_bootstrap_oos.jl` | `results/ks_block_bootstrap/...` | Appendix `sec:ks_block_bootstrap` (OoS-anchored block-bootstrap KS, including body operating-point summary at $L = 20$) |
| `runners/baselines/run_hsmm_gamma.jl` | `results/hsmm_gamma/...` | Appendix `sec:hsmm_gamma_sojourn` (Gamma-sojourn HSMM at $K = 18$) |
| `runners/baselines/run_filtered_bootstrap_var.jl` | `results/filtered_bootstrap_var/...` | Item 6a of REVIEW_RESPONSE_PLAN.md (Hull-White-style filtered historical-simulation VaR contender for body Section 5). |
| `runners/baselines/run_caviar_var.jl` | `results/caviar_var/...` | Item 6b of REVIEW_RESPONSE_PLAN.md (Engle-Manganelli SAV CAViaR contender for body Section 5). |
| `runners/baselines/run_garch_suite.jl` | `results/garch_suite/GARCH_Suite.txt` (paper `garch_suite.csv`) | Body Table 2 GARCH-$t$ / MS-GARCH per-path ACF rows + Appendix conditional-volatility extended baselines (EGARCH, GJR, HAR-RV, MS-GARCH; reports both pooled and per-path ACF conventions) |
| `runners/baselines/run_garch_canonical_metrics.jl` | `results/garch_canonical/garch_canonical_metrics.{txt,csv}` (paper `garch_canonical_metrics.csv`) | Body Table 2 GARCH(1,1) row: full metric panel under the canonical grid-initialised ML fit, headline per-path conventions |
| `runners/baselines/run_ks_block_bootstrap.jl` | `results/ks_block_bootstrap/KS_Bootstrap_Recalibration.txt` (paper `ks_block_bootstrap.csv`) | Appendix `sec:ks_block_bootstrap` IS-anchored block-bootstrap KS (companion to the OoS-anchored `run_ks_block_bootstrap_oos.jl`) |
| `runners/baselines/run_ks_block_body_kstar.jl` | `results/ks_block_bootstrap/KS_Bootstrap_Body_Kstar.txt` (paper `ks_block_body_kstar.csv`) | Body Table~\ref{tab:ks_block_body} (OoS $L = 20$ block-bootstrap KS at $K^\star = 3$ and $K = 6$) |

## Diagnostics + miscellanea

| Runner | Output | Paper artefact |
|---|---|---|
| `runners/diagnostics/run_diagnostics.jl` | `results/diagnostics/...` (catch-all) | Various appendix diagnostics |
| `runners/diagnostics/run_paper_artifact_check.jl` | (consistency gate; no artifact) | Asserts that every registered paper-facing value appears both in its model-repo artifact and in the paper `.tex` source. Run before any paper submission. Provenance map: `results/artifacts_manifest.csv`. |
| `runners/diagnostics/run_feed_boundary_check.jl` | `results/diagnostics/feed_boundary_check.txt` | Methods data-provenance paragraph (pre/post feed-switch diagnostic at the 2025-01-03 Polygon→Alpaca/IEX boundary). Replaces the retired vendor-stitch check, which compared stitched rows against the same Alpaca rows they were copied from and is archived in `_attic/runners/` as invalid. |

## Archived runners

The following runners were moved to `_attic/runners/` during the arXiv-prep
pass because their outputs are no longer cited in the trimmed paper, or because
they were retired as invalid (`run_vendor_stitch_check.jl`). They are preserved
for full historical reproducibility of the pre-arXiv revisions.

See `_attic/runners/README.md` for the full archive and the rationale for
each move.
