# RUNNERS.md — runner-to-paper-artefact map

Each row maps a top-level `run_*.jl` script to (a) the artefact it produces under
`results/<subdir>/`, and (b) the paper section, table, or figure that consumes it.
Use this as the reproducibility audit: pick a paper artefact, find its runner,
inspect the script.

`run_full_rebuild.jl` is the end-to-end orchestrator (excludes the slow QuantGAN
stage by default; see the script's header comment for the inclusion toggle).

## Headline pipeline (body Section 4 — Empirical Study)

| Runner | Output | Paper artefact |
|---|---|---|
| `run_all_analysis.jl` | `results/SPY/{stylized_facts,K*}/...` | Body §4.1 (descriptive); §4.2 (state selection); per-K internals |
| `run_multi_emission_analysis.jl` | `results/SPY/multi_emission/...` | Body Table 2 (CHMM-N/-t/-L/-GED at $K^\star = 3$ block) |
| `run_baselines_and_cross_asset.jl` | `results/SPY/Table-2-Baselines.txt`, `results/cross_asset/...` | Body Table 2 (i.i.d., GARCH, MS-GARCH in-house rows); body Table 4 (cross-asset Pipeline B) |
| `run_msgarch_baselines.jl` | `results/msgarch_baselines/...` | Body Table 2 MS-GARCH $K \in \{2, 3, 6\}$ rows (Nelder-Mead) |
| `run_msgarch_reference.jl` | `results/msgarch_reference/...` | Body Table 2 MS-GARCH ref. Bayesian rows (CRAN `MSGARCH`, requires R) |
| `run_smchmm_baseline.jl` | `results/smchmm_baseline/...` | Body Table 2 ML HSMM-N at $K^\star = 3$ co-headline row |
| `run_quantgan_baseline.jl` | `results/quantgan_baseline/...` | Body Table 2 QuantGAN row + Appendix QuantGAN spec |
| `run_cross_asset_sim_copula.jl` | `results/cross_asset/...` | Body Table 4 (Pipeline B Student-$t$ copula at $\nu^\star = 6$) |
| `run_cross_ticker_penalised.jl` | `results/chmm_t_penalised/...` | Body Table 3 (penalised CHMM-t cross-ticker headline) |
| `run_sector_panel.jl` | `results/sector_panel/sector_panel_summary.{csv,txt}` | Body Table 3 (30-ticker rollup); Appendix sector panel |
| `run_chmm_t_shared_nu.jl` | `results/chmm_t_shared_nu/...` | Body Table 2 footnoted shared-$\nu$ row + Appendix `sec:chmm_t_shared_nu` |
| `run_figures.jl` | `figs/Fig-{1..5}-*.pdf` | Body Figures 1, 2, 3, 4 |

## VaR / conditional-coverage diagnostics (body §5 — VaR Backtest)

| Runner | Output | Paper artefact |
|---|---|---|
| `run_conditional_var_all_families.jl` | `results/conditional_var_all_families/...` | Body Table 5 (regime-conditional VaR Christoffersen-cc); Appendix four-family extension |
| `run_christoffersen_power.jl` | `results/diagnostics/christoffersen_power/...` | Appendix `sec:christoffersen_power` (Monte Carlo power calibration) |
| `run_engle_manganelli_dq.jl` | `results/diagnostics/engle_manganelli_dq.txt` | Appendix `sec:engle_manganelli_dq` (DQ test backstop) |
| `run_quarterly_refit_conditional_var.jl` | `results/quarterly_refit_conditional_var/...` | Appendix `sec:quarterly_refit_cond_var` |

## Walk-forward + cross-decade + cross-ticker robustness

| Runner | Output | Paper artefact |
|---|---|---|
| `run_cross_decade_validation.jl` | `results/cross_decade_validation/...` | Appendix `sec:cross_decade_validation` (CRSP 1994-2006 IS / OoS) |
| `run_sector_panel_quarterly_refit.jl` | `results/sector_panel/sector_panel_quarterly_refit.{csv,txt}` | Appendix `sec:cross_ticker_quarterly_refit` |
| `run_kurtosis_bootstrap.jl` | `results/SPY/diagnostics/kurtosis_bootstrap.txt` | Appendix `sec:kurtosis_bootstrap_ci` |
| `run_kurtosis_ci_placement.jl` | `results/kurtosis_ci_placement/...` | Appendix `sec:kurtosis_ci_placement` |
| `run_lambda_cv_pre2020.jl` | `results/diagnostics/lambda_cv_pre2020/...` | Appendix `sec:lambda_cv_pre2020` ($1/\nu_k$ penalty CV) |
| `run_k_selection_kfold_pre2020.jl` | `results/k_selection_validation/...` | Appendix `sec:k_selection_kfold_pre2020` (single + four-fold CV at body $K^\star = 3$) |
| `run_k_selection_kfold_h12y_pre2020.jl` | `results/k_selection_validation/h12y/...` | Six-fold rolling-origin CV (referenced in body §4.2) |
| `run_k_selection_hac.jl` | `results/k_selection_hac/...` | Appendix `sec:k_selection_hac` (HAC-corrected K selection) |
| `run_state_distinctness.jl` | `results/diagnostics/state_distinctness/...` | Appendix `sec:state_distinctness` |

## Spectral + theoretical diagnostics (body §3.3 — Spectral Mechanism)

| Runner | Output | Paper artefact |
|---|---|---|
| `run_spectral_rank.jl` | `results/diagnostics/spectral_rank.txt` | Body Table 1 + Appendix `sec:spectral_rank` (SPY effective rank diagnostic) |
| `run_spectral_rank_cross_ticker.jl` | `results/diagnostics/spectral_rank_cross_ticker.txt` | Appendix `sec:spectral_rank_xticker` (cross-ticker dominant-mode share) |

## Cross-asset extras

| Runner | Output | Paper artefact |
|---|---|---|
| `run_copula_profile_ci_halfunit.jl` | `results/copula_profile_ci/...` | Appendix `sec:copula_halfunit` (half-unit-grid Wilks CI) |
| `run_non_us_asset.jl` | `results/non_us_asset/...` | Appendix `sec:non_us_asset_supp` (GLD / SLV stress test) |

## Auxiliary baselines (Appendix B — extended baselines)

| Runner | Output | Paper artefact |
|---|---|---|
| `run_sv_msm_jd_baselines.jl` | `results/sv_msm_jd/...` | Appendix `sec:sv_msm_jd_baselines` (SV-AR(1), MSM, Merton-JD) |
| `run_ged_diagnostics.jl` | `results/ged_diagnostics/...` | Appendix `sec:supp_p_partition` ($\hat p_k$ partition diagnostic) |
| `run_leverage_effect.jl` | `results/diagnostics/leverage_effect/...` | Appendix `sec:leverage_effect` |
| `run_ks_block_bootstrap_oos.jl` | `results/ks_block_bootstrap/...` | Appendix `sec:ks_block_bootstrap` (OoS-anchored block-bootstrap KS) |
| `run_crps_dm_bandwidth.jl` | `results/crps_dm/...` | Appendix `sec:dm_bandwidth` (DM bandwidth sensitivity) |
| `run_hsmm_ml_gamma.jl` | `results/hsmm_ml_gamma/...` | Appendix `sec:hsmm_gamma_sojourn` (Gamma-sojourn HSMM at $K = 18$) |

## Diagnostics + miscellanea

| Runner | Output | Paper artefact |
|---|---|---|
| `run_diagnostics.jl` | `results/diagnostics/...` (catch-all) | Various appendix diagnostics |
| `run_figures_ksweep.jl` | `figs/Fig-3-IS-Comparison-K{3,6,12,21}-*.pdf` | (formerly Appendix per-K figures; removed in arXiv-prep trim, kept for reproducibility) |

## Archived runners

The following runners were moved to `_attic_v10/runners/` during the arXiv-prep
pass because their outputs are no longer cited in the trimmed paper. They are
preserved for full historical reproducibility of the pre-arXiv revisions.

See `_attic_v10/runners/README.md` for the full archive and the rationale for
each move.
