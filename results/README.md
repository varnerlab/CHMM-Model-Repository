# `results/` directory key

Every table and figure in this project is produced by one of two pipelines.
This file states which, so a reader can open any artifact and know immediately
what it is testing.

## The two pipelines

**Pipeline A -- single-index trained CHMM (per-ticker independent).**
Each ticker's return series is fit with its own continuous HMM via Baum-Welch.
No cross-asset coupling. Evaluates marginal (univariate) fidelity only.

**Pipeline B -- cross-asset dependence extension.**
Starts from the CHMM-N marginals produced by Pipeline A. Adds one of three
joint dependence constructions: Single Index Model (SIM) with SPY as market
factor, Gaussian copula, or Student-t copula (nu selected by profile MLE).
Evaluates cross-asset correlation reproduction on top of marginal fidelity.

Main tickers: SPY, NVDA, JNJ, JPM, AAPL, QQQ. Main K: 18.

## Which artifact uses which pipeline

| Subdirectory / file                                            | Pipeline | Scope                        | Paper reference           |
| -------------------------------------------------------------- | -------- | ---------------------------- | ------------------------- |
| `SPY/Table-2-Baselines.txt`                                    | A        | SPY only                     | Table 2                   |
| `SPY/Table-T1-State-Resolution-Sensitivity.txt`                | A        | SPY, K sweep                 | Table T1a                 |
| `SPY/Table-T1-Multi-Emission.txt`                              | A        | SPY, K x (N / t / L) sweep   | Table T1b                 |
| `SPY/Table-T2-Per-Ticker-Emission-Families.txt`                | A        | 6 tickers x 3 families       | Table T2                  |
| `SPY/K{3,6,...,21}/...`                                        | A        | SPY, per-K diagnostics       | Appendix per-K panels     |
| `SPY/multi_emission/K18/{N,t,L}/...`                           | A        | SPY, per-family diagnostics  | Figs 2-5                  |
| `SPY/stylized_facts/...`                                       | A (data) | SPY                          | Fig 1                     |
| `equity_price_sim/Fig-{TICKER}-PriceFan-{N,t,L}.*`             | A        | 6 tickers x 3 families       | Fig 6                     |
| `equity_price_sim/Fig-{TICKER}-TerminalDist-{N,t,L}.*`         | A        | 6 tickers x 3 families       | Fig terminal (main / app) |
| `equity_price_sim/summary.csv`                                 | A        | 6 tickers x 3 families       | Tables in price-sim sect. |
| `cross_asset/Table-T3-Cross-Asset-Dependence.txt`              | B        | 6 tickers, SIM / Gauss / t   | Table T3                  |
| `cross_asset/Fig-Cross-Asset-Correlation.*`                    | B        | 6x6 correlation matrices     | Fig 7                     |
| `cross_asset/Fig-Cross-Asset-KS-Dist.*`                        | B        | 6 tickers, bar chart         | Fig cross-asset KS        |
| `diagnostics/block_bootstrap/...`                                       | A        | SPY                          | Appendix                  |
| `diagnostics/bin_t/...`                                                 | A        | SPY                          | Appendix                  |
| `diagnostics/copula_profile/...`                                        | B        | 6 tickers, profile MLE of nu | Appendix                  |
| `diagnostics/gru/...`                                                   | A        | SPY, neural baseline         | Appendix                  |
| `diagnostics/ks_power/...`                                              | A        | SPY, power simulation        | Appendix                  |
| `diagnostics/nu_diagnostics/...`                                        | A        | SPY, CHMM-t nu behavior      | Appendix                  |
| `diagnostics/ryden_k2/...`                                              | A        | SPY, Ryden K=2 comparison    | Appendix                  |
| `diagnostics/utility/VaR_ES_Backtest.*`                                 | A        | SPY                          | Appendix VaR / ES         |
| `diagnostics/walk_forward/WalkForward.txt`                              | A        | JNJ, JPM                     | Appendix walk-forward     |

## Table T2 vs. Table T3: quick disambiguation

Both cover the same six tickers. They answer different questions:

- **Table T2** (`SPY/Table-T2-Per-Ticker-Emission-Families.txt`). Pipeline A.
  Vary the **emission family** (Gaussian, Student-t, Laplace) with each ticker
  fit independently. No dependence. Question: does the CHMM reproduce each
  ticker's **univariate** return distribution across emission families?

- **Table T3** (`cross_asset/Table-T3-Cross-Asset-Dependence.txt`). Pipeline B.
  Hold the emission family fixed (Gaussian). Vary the **dependence
  construction** (SIM, Gaussian copula, Student-t copula). Question: given the
  marginals, which dependence mechanism best reproduces the **cross-asset
  correlation matrix**?

Read T2 first (marginals), then T3 (dependence on top of marginals).

## Generating scripts

Each script states its pipeline in its header.

- `run_baselines_and_cross_asset.jl` -- Pipeline A. Emits Table 2 and Table T2.
- `run_multi_emission_analysis.jl`   -- Pipeline A. Emits Table T1b and per-(K, family) SPY diagnostics (Figs 2-5).
- `run_equity_price_sim.jl`          -- Pipeline A. Emits Fig 6 price fans and terminal distributions.
- `run_cross_asset_sim_copula.jl`    -- Pipeline B. Emits Table T3 and Fig 7.
- `run_full_rebuild.jl`              -- Master driver, all stages.
- `run_diagnostics.jl`               -- Pipeline A diagnostics (VaR/ES, walk-forward, copula profile, ...).
- `run_gru_baseline.jl`              -- Deep-generative baseline.
- `run_figures.jl`                   -- Re-renders the paper's main-body figures.
