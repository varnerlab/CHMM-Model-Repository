# `results/` directory key

Every table and figure in this project is produced by one of two pipelines.
This file states which, so a reader can open any artifact and know immediately
what it is testing. The authoritative table-by-table provenance map (paper
label -> runner -> artifact -> seed -> status) is `results/artifacts_manifest.csv`,
which the consistency gate `runners/diagnostics/run_paper_artifact_check.jl`
validates; the per-runner catalogue is `RUNNERS.md` at the repo root.

## The two pipelines

**Pipeline A -- single-index trained CHMM (per-ticker independent).**
Each ticker's return series is fit with its own continuous HMM via Baum-Welch.
No cross-asset coupling. Evaluates marginal (univariate) fidelity only.

**Pipeline B -- cross-asset dependence extension.**
Starts from the CHMM-N marginals produced by Pipeline A. Adds one of the
joint dependence constructions: Single Index Model (SIM) with SPY as market
factor, Gaussian copula, or Student-t copula (nu selected by profile MLE).
Evaluates cross-asset correlation reproduction on top of marginal fidelity.

Main tickers: SPY, NVDA, JNJ, JPM, AAPL, QQQ. Main K: K* = 3 (K = 18 retained as the excess kurtosis-fidelity sensitivity reference).

## Where the paper-facing artifacts live

| Subdirectory                           | Pipeline | What it holds |
| -------------------------------------- | -------- | ------------- |
| `kstar3_headline/`                     | A        | Main Table 2 CHMM rows at K* = 3 |
| `garch_suite/`, `garch_canonical/`     | A        | GARCH-family baselines (per-path + pooled ACF conventions) |
| `msgarch_baselines/`, `msgarch_reference/` | A    | In-house and R-reference MS-GARCH rows |
| `msgarch_conditional_var/`             | A        | MS-GARCH conditional VaR on the aligned 573-forecast origin set |
| `conditional_var_all_families/`        | A        | Sixteen-row regime-conditional VaR family panel + per-day series |
| `var_inference_upgrade/`               | A        | Exact binomial, bootstrap DQ (B = 2000), pinball-loss panel |
| `chmm_t_shared_nu/`                    | A        | Shared-nu Student-t ablation (library fitter) |
| `crps_dm_kstar3/`                      | A        | Five-row CRPS Diebold-Mariano panel + per-day losses |
| `acf_horizon/`                         | A        | Observed-vs-simulated ACF curves, banded MAE, main-text figure |
| `ks_block_bootstrap/`                  | A        | Two-sample block-bootstrap KS recalibrations (IS-, OoS-anchored, K*-body) |
| `cross_asset/`                         | B        | Table T3 dependence panel, 2x2 family/refit copula experiment |
| `non_us_asset/`, `copula_profile_ci/`  | B        | GLD stress, per-pair MAE, profile-CI and full-MLE checks |
| `diagnostics/`                         | A        | Spectral rank, tail index, DQ, feed boundary, kurtosis bootstrap, utility VaR/ES |
| `sector_panel/`, `chmm_t_penalised/`   | A        | 30-ticker cross-ticker panel and refit cadences |
| `walkforward/`                         | A        | Six-fold walk-forward OoS + conditional VaR folds |
| `cross_decade_validation/`             | A        | CRSP 1994-2006 transfer |
| `nu_shrinkage_sweep/`, `k_selection_*` | A        | Lambda CV and K-selection designs |

## Table T2 vs. Table T3: quick disambiguation

Both cover the same six tickers. They answer different questions:

- **Table T2** (Pipeline A). Vary the **emission family** (Gaussian, Student-t,
  Laplace) with each ticker fit independently. No dependence. Question: does
  the CHMM reproduce each ticker's **univariate** return distribution across
  emission families?

- **Table T3** (`cross_asset/Table-T3-Cross-Asset-Dependence.txt`, Pipeline B).
  Hold the emission family fixed (Gaussian). Vary the **dependence
  construction** (SIM, Gaussian copula, Student-t copula). Question: given the
  marginals, which dependence mechanism best reproduces the **cross-asset
  correlation matrix**?

Read T2 first (marginals), then T3 (dependence on top of marginals).

## Generating scripts

`run_full_rebuild.jl` (repo root) drives the eight headline stages; every other
paper-facing table has a dedicated runner under `runners/{headline,baselines,
cross_asset,robustness,spectral,var_backtest,diagnostics}/`. The complete
runner-to-artifact mapping, including seeds and status, is
`results/artifacts_manifest.csv`; per-runner descriptions are in `RUNNERS.md`.
Retired exploratory runners (GRU, diffusion, track-* experiments) live under
`_attic/runners/` and produce nothing the paper cites.
