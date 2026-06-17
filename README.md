# CHMM-Model

A Julia framework for modeling equity return dynamics as a **Continuous Hidden Markov Model (CHMM) digital twin**, with a four-family emission ablation under a unified ECM scaffold (Gaussian, Student-t with per-state $\nu_k$, Laplace, Generalised Error Distribution with per-state $p_k$) and cross-asset copula composition. Companion code repository to the working paper in [`../CHMM-paper`](https://github.com/altashly1/CHMM-paper).

The framework compares the CHMM against a broad panel of alternatives:

1. **Continuous Hidden Markov Model** (Baum-Welch / ECM, four emission families CHMM-N / CHMM-t / CHMM-L / CHMM-GED) — the main contribution
2. **ML hidden semi-Markov model** (explicit-duration EM with truncated Pareto and discrete-Gamma sojourns) — co-headline scaffold from [SM-CHMM-AR-Model](https://github.com/altashly1/SM-CHMM-AR-Model)
3. **Discrete HMM with Poisson jumps** — baseline from [JumpHMM.jl](https://github.com/varnerlab/JumpHMM.jl)
4. **GARCH family** — GARCH(1,1) Gaussian, GARCH(1,1)-$t$, EGARCH, GJR-GARCH, HAR-RV
5. **Markov-switching GARCH** at $K \in \{2, 3, 4, 6\}$ via in-house Nelder-Mead and via the reference Bayesian `MSGARCH` R package of Ardia et al. (2019)
6. **Stochastic-volatility, multifractal, jump-diffusion** — SV-AR(1) (Taylor 1982), MSM (Calvet-Fisher 2004), Merton-JD (Merton 1976)
7. **QuantGAN deep-generative baseline** — convolutional WGAN (re-implementation of Wiese et al. 2020)
8. **Bootstrap, stationary-block bootstrap, Gaussian / Laplace i.i.d.** — non-parametric and parametric null generators
9. **Single Index Model, Gaussian copula, Student-t copula, truncated C-vine** — cross-asset dependence generators

At moderate $K$, the CHMM reproduces the three symmetric Cont (2001) stylized facts of financial returns (heavy tails, negligible linear autocorrelation, persistent volatility clustering) without an explicit jump mechanism. CHMM-GED's per-state $\hat p_k$ partitions bimodally into a Gaussian-bulk / Laplace-tail structure that replicates across seeds and tickers. The Student-$t$ copula with $\nu^\ast = 6$ reproduces cross-asset IS dependence; on OoS it is statistically indistinguishable from the Gaussian copula. A regime-conditional Value-at-Risk that propagates the one-step-ahead state forecast through the predictive mixture passes Christoffersen-cc cleanly across emission families.

## Authors

- **Abdulrahman Alswaidan** — Robert Frederick Smith School of Chemical and Biomolecular Engineering, Cornell University, Ithaca, NY, USA. `aa2725@cornell.edu`
- **Cade Jin** — Cornell University, Ithaca, NY, USA. `cj383@cornell.edu`
- **Jeffrey D. Varner** — Robert Frederick Smith School of Chemical and Biomolecular Engineering, Cornell University, Ithaca, NY, USA. `jdv27@cornell.edu`

## Academic Citation

Alswaidan A, Jin C, Varner JD. *Continuous Hidden Markov Models for Equity Returns: Heavy-Tail Emission Families and Regime-Conditional Value-at-Risk.* Working paper, Cornell University, 2026.

```bibtex
@article{alswaidan2026chmm,
  title   = {Continuous Hidden Markov Models for Equity Returns:
             Heavy-Tail Emission Families and Regime-Conditional
             {Value-at-Risk}},
  author  = {Alswaidan, Abdulrahman and Jin, Cade and Varner, Jeffrey D.},
  year    = {2026},
  institution = {Cornell University},
  note    = {Working paper}
}
```

## Overview

Financial markets exhibit volatility clustering, heavy-tailed returns, and regime-dependent dynamics. This framework:

- Trains a **continuous HMM** via Baum-Welch / ECM on observed returns with four emission families (Gaussian; Student-$t$ with per-state $\nu_k$ and an optional $1/\nu_k$ shrinkage prior; Laplace; Generalised Error Distribution with per-state shape $p_k \in [0.5, 3.0]$ that nests Gaussian at $p = 2$ and Laplace at $p = 1$); the M-step is the only architectural difference across families
- Compares against an **ML hidden semi-Markov model** at $K \in \{3, 6\}$ (truncated Pareto and discrete-Gamma sojourns) and a **discrete HMM + Poisson jumps** baseline from JumpHMM.jl
- Benchmarks against the **GARCH family** (GARCH(1,1) Gaussian / Student-$t$, EGARCH, GJR-GARCH, HAR-RV), **Markov-switching GARCH** at $K \in \{2, 3, 4, 6\}$ in both an in-house frequentist fit and the reference Bayesian `MSGARCH` R package, **stochastic-volatility / MSM / Merton jump-diffusion** rows, a **QuantGAN** deep-generative re-implementation, and parametric / non-parametric null generators (Gaussian i.i.d., Laplace i.i.d., stationary block bootstrap)
- Composes marginals into **multi-asset samples** via Single Index Model, Gaussian copula, Student-$t$ copula, and truncated C-vine
- Validates with a seven-metric fidelity panel (KS, AD, kurtosis, $|G_t|$ ACF-MAE, Wasserstein-1, Hellinger, OoS coverage), a regime-conditional VaR back-test (Kupiec, Christoffersen-ind, Christoffersen-cc, Engle--Manganelli DQ), and a six-fold rolling-origin walk-forward

## Quick Start

```julia
include("Include.jl")

# Load SPY returns
dataset = MyPortfolioDataSet()["dataset"]
R = log_growth_matrix(dataset, "SPY"; Δt=1/252, risk_free_rate=0.0)

# Train CHMM-N at K = 18 (Gaussian emissions)
chmm = build(MyContinuousHiddenMarkovModel, (
    observations = R, number_of_states = 18, max_iter = 60))

# Fit GARCH(1,1) benchmark
garch = build(MyGARCHModel, (observations = R,))

# Simulate from the stationary distribution
K = length(chmm.states)
T_mat = zeros(K,K); for i in 1:K; T_mat[i,:] = probs(chmm.transition[i]); end
π_stat = (T_mat^1000)[1,:]; start_dist = Categorical(π_stat)

states = chmm(rand(start_dist), 252)
chmm_returns = [rand(chmm.emission[s]) for s in states]
garch_returns = simulate_garch(garch, 252)
```

To regenerate every table and figure in the companion paper in one pass:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. run_full_rebuild.jl
```

### Optional: reference MS-GARCH baseline (R + RCall)

The reviewer-requested reference MS-GARCH baseline (Ardia et al. 2019, JSS,
CRAN `MSGARCH` package) is driven from Julia via [RCall.jl](https://juliainterop.github.io/RCall.jl/).
This is required only for the `runners/headline/run_msgarch_reference.jl` row in Table 2 and
its companion artefacts under `results/msgarch_reference/`. Everything else
in the paper runs without R.

Prerequisites: R >= 4.2 on `PATH`. One-time setup:

```bash
cd r_msgarch
Rscript setup.R                                  # first time on a fresh checkout
# or, if r_msgarch/renv.lock is already committed:
Rscript -e 'renv::restore(prompt = FALSE)'
```

This pins MSGARCH (currently 2.51) and all transitive R dependencies into a
project-local `renv` library. After setup, run the reference baseline:

```bash
julia --project=. runners/headline/run_msgarch_reference.jl
```

See [`r_msgarch/README.md`](r_msgarch/README.md) for the full version-pinning
contract. The Julia test suite skips the MS-GARCH reference test if R is
unavailable, so `Pkg.test()` works without R.

## Data

| Dataset | Period | Trading Days | Coverage |
|---------|--------|-------------|----------|
| Training (IS) | 2014-01-03 -- 2024-01-03 | 2,516 | Six-ticker panel (SPY, NVDA, JNJ, JPM, AAPL, QQQ) |
| Out-of-Sample (OoS) | 2024-01-04 -- 2026-04-20 | 573 | Same universe |

Returns convention: annualized excess log returns, $G_t = (1/\Delta t)\ln(P_t / P_{t-1}) - r_f$ with $\Delta t = 1/252$ and $r_f = 0$.

## Project Structure

```
.
|-- Include.jl                            # Entry point (sets paths, loads src/)
|-- run_full_rebuild.jl                   # End-to-end rebuild of every paper artefact
|-- build_new_train_oos.jl                # (re)build IS / OoS JLD2 splits from raw OHLC bundles (optional; pre-built splits ship in data/)
|-- runners/                              # All experiment scripts, grouped by paper section
|   |-- headline/                         # Body §5 (Empirical Study) pipeline
|   |   |-- run_all_analysis.jl              # SPY-only stylized facts + per-K internals
|   |   |-- run_multi_emission_analysis.jl   # CHMM-N / -t / -L / -GED at K* block
|   |   |-- run_kstar3_headline.jl           # K* = 3 four-emission Table 2 CHMM rows
|   |   |-- run_baselines_and_cross_asset.jl # Pipeline A baselines + Pipeline B setup
|   |   |-- run_cross_asset_sim_copula.jl    # Pipeline B: SIM, Gaussian / Student-t copula
|   |   |-- run_msgarch_baselines.jl         # MS-GARCH K in {2,3,6} rows (in-house NM)
|   |   |-- run_msgarch_higher_k.jl          # MS-GARCH K in {4,6} rows (in-house NM)
|   |   |-- run_msgarch_reference.jl         # MS-GARCH ref. Bayesian rows (CRAN MSGARCH)
|   |   |-- run_smchmm_baseline.jl           # Semi-Markov CHMM at K*
|   |   |-- run_quantgan_baseline.jl         # QuantGAN deep-generative row
|   |   |-- run_cross_ticker_penalised.jl    # penalised CHMM-t cross-ticker headline
|   |   |-- run_chmm_t_penalised_headline.jl # penalised CHMM-t (λ=20) SPY headline row
|   |   |-- run_sector_panel.jl              # 30-ticker sector rollup
|   |   |-- run_chmm_t_shared_nu.jl          # shared-nu sensitivity
|   |   `-- run_figures.jl                   # Body Figures 1-4
|   |-- var_backtest/                     # body §5 (regime-conditional VaR, Christoffersen, DQ)
|   |-- robustness/                       # walk-forward, cross-decade, K-selection sensitivity
|   |-- spectral/                         # body §4 Spectral Mechanism diagnostics
|   |-- cross_asset/                      # half-unit copula CI, non-US stress test
|   |-- baselines/                        # Appendix: SV-AR(1), MSM, Merton-JD, HSMM-Gamma, filtered bootstrap, CAViaR, ...
|   `-- diagnostics/                      # catch-all
|-- src/                                  # loaded in this order by Include.jl
|   |-- Types.jl                          # HMM / GARCH / copula / semi-Markov type definitions
|   |-- Files.jl                          # JLD2 data loaders
|   |-- Factory.jl                        # build() constructors
|   |-- Compute.jl                        # Baum-Welch / ECM (N/t/L/GED), GARCH(1,1) MLE, simulation
|   |-- CrossAsset.jl                     # SIM, Gaussian / Student-t copula, truncated C-vine generators
|   |-- Visualize.jl                      # Plotting utilities
|   |-- Metrics.jl                        # MMD / path-signature / classifier-AUC + VaR LR diagnostics
|   |-- SemiMarkov.jl                     # ML hidden semi-Markov CHMM (ported from SM-CHMM-AR-Model)
|   |-- MSGARCH.jl                        # In-house MS-GARCH (Hamilton filter + Nelder-Mead)
|   |-- MSGARCHReference.jl               # Reference Bayesian MS-GARCH via RCall (opt-in; not auto-loaded)
|   |-- GARCHFamily.jl                    # GARCH(1,1)-t, EGARCH, GJR-GARCH, HAR-RV
|   `-- SVMSMBaselines.jl                 # SV-AR(1), MSM, Merton jump-diffusion baselines
|-- r_msgarch/                            # R-side scaffolding (renv-pinned, setup.R + fit_msgarch.R)
|-- data/                                 # JLD2 datasets (active IS / OoS bundles)
|-- results/                              # Generated metrics tables (per-ticker, per-K)
|-- figs/                                 # Generated SVG / PDF figures
|-- test/                                 # Test suite
|-- docs/                                 # Documenter.jl docs
`-- _attic/                               # Archived journal-revision-era runners, results, and docs
```

`RUNNERS.md` is the full runner-to-paper-artefact map (every script in
`runners/` is keyed to the table or figure it produces).

## Related Repositories

- [`CHMM-paper`](https://github.com/altashly1/CHMM-paper) — LaTeX source for the working paper this code supports
- [`SM-CHMM-AR-Model`](https://github.com/altashly1/SM-CHMM-AR-Model) / [`SM-CHMM-AR-Paper`](https://github.com/altashly1/SM-CHMM-AR-Paper) — companion VIX / semi-Markov extension
- [JumpHMM.jl](https://github.com/varnerlab/JumpHMM.jl) — discrete HMM core package from the prior paper

### Shared code with `SM-CHMM-AR-Model`

The two companion repositories share code in both directions:

- `src/SemiMarkov.jl` and the `MySemiMarkovContinuousHMM` type in
  `src/Types.jl` are **ported from `SM-CHMM-AR-Model`** (truncated discrete
  Pareto sojourns, plug-in estimator, AR(1) residual fits). Used here only
  as the SPY SM ablation baseline; the joint-EM SM-CHMM-AR is not ported.
- Conversely, the flat-CHMM machinery in this repo's `src/Compute.jl`
  (Baum-Welch for Gaussian / Student-t ECM with per-state $\nu$ /
  closed-form weighted-Laplace M-step, forward/backward, Viterbi, simulate)
  is re-used verbatim in `SM-CHMM-AR-Model/src/Compute.jl` (lines 1--1327
  there).

Each repo is self-contained for reproducibility of its associated paper.

## Disclaimer

For research and educational purposes only. Not financial advice.

## License

MIT License. See [LICENSE](LICENSE).
