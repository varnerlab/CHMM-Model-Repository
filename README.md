# CHMM-Model

A Julia framework for modeling equity return dynamics as a **Continuous Hidden Markov Model (CHMM) digital twin**, with a three-family emission ablation (Gaussian, Student-t with per-state $\nu_k$, Laplace) and cross-asset copula composition. Companion code repository to the working paper in [`../CHMM-paper`](https://github.com/altashly1/CHMM-paper).

The framework compares the CHMM against a broad panel of alternatives:

1. **Continuous Hidden Markov Model** (Baum-Welch / ECM, three emission families) — the main contribution
2. **Discrete HMM with Poisson jumps** — baseline from [JumpHMM.jl](https://github.com/varnerlab/JumpHMM.jl)
3. **GARCH(1,1)** — classical conditional-variance benchmark
4. **GRU + Gaussian head** — deep-generative baseline
5. **Bootstrap, stationary-block bootstrap, Gaussian i.i.d.** — non-parametric and parametric null generators
6. **Single Index Model, Gaussian copula, Student-t copula** — cross-asset dependence generators

At moderate $K$, the CHMM reproduces the three canonical stylized facts of financial returns (heavy tails, negligible linear autocorrelation, persistent volatility clustering) without requiring an explicit jump mechanism, and the Student-t copula with $\nu^\ast \approx 6$ reproduces cross-asset tail dependence missed by SIM and Gaussian copulas.

## Authors

- **Abdulrahman Alswaidan** — Robert Frederick Smith School of Chemical and Biomolecular Engineering, Cornell University, Ithaca, NY, USA. `aa2725@cornell.edu`
- **Cade Jin** — Cornell University, Ithaca, NY, USA. `cj383@cornell.edu`
- **Jeffrey D. Varner** — Robert Frederick Smith School of Chemical and Biomolecular Engineering, Cornell University, Ithaca, NY, USA. `jdv27@cornell.edu`

## Academic Citation

Alswaidan A, Jin C, Varner JD. *Continuous Hidden Markov Models as a Digital Twin for Equity Returns: Gaussian, Student-t, and Laplace Emissions Trained by EM, with Cross-Asset Copula Composition.* Working paper, Cornell University, 2026.

```bibtex
@article{alswaidan2026chmm,
  title   = {Continuous Hidden Markov Models as a Digital Twin for Equity Returns:
             {Gaussian}, {Student}-t, and {Laplace} Emissions Trained by {EM},
             with Cross-Asset Copula Composition},
  author  = {Alswaidan, Abdulrahman and Jin, Cade and Varner, Jeffrey D.},
  year    = {2026},
  institution = {Cornell University},
  note    = {Working paper}
}
```

## Overview

Financial markets exhibit volatility clustering, heavy-tailed returns, and regime-dependent dynamics. This framework:

- Trains a **continuous HMM** via Baum-Welch / ECM on observed returns with three emission families (Gaussian, Student-t with per-state $\nu_k$, Laplace)
- Compares against **discrete HMM + Poisson jumps** (regime teleportation baseline from JumpHMM.jl)
- Benchmarks against **GARCH(1,1)**, a **GRU + Gaussian head** deep-generative model, and parametric / non-parametric null generators (Gaussian i.i.d., bootstrap, stationary block bootstrap)
- Composes marginals into **multi-asset samples** via Single Index Model, Gaussian copula, and Student-t copula
- Validates with a seven-metric fidelity panel: KS, AD, kurtosis, ACF-MAE, Wasserstein-1, Hellinger, quantile-envelope coverage

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
This is required only for the `run_msgarch_reference.jl` row in Table 2 and
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
julia --project=. run_msgarch_reference.jl
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
|-- run_all_analysis.jl                   # SPY-only analysis pipeline
|-- run_multi_emission_analysis.jl        # Pipeline A: three-emission CHMM + benchmarks
|-- run_baselines_and_cross_asset.jl      # Pipeline A baselines + Pipeline B setup
|-- run_cross_asset_sim_copula.jl         # Pipeline B: SIM, Gaussian copula, Student-t copula
|-- run_diagnostics.jl                    # Diagnostic metrics
|-- run_figures.jl                        # Paper figures
|-- run_quantgan_baseline.jl              # QuantGAN deep-generative baseline row
|-- run_msgarch_baselines.jl              # MS-GARCH K=2/3 rows in extended panel (in-house Nelder-Mead)
|-- run_msgarch_reference.jl              # MS-GARCH K=2/3/4 rows via CRAN MSGARCH (RCall)
|-- r_msgarch/                            # R-side scaffolding (renv-pinned, setup.R + fit_msgarch.R)
|-- run_smchmm_baseline.jl                # Semi-Markov CHMM rows in extended panel
|-- run_cross_asset_large_universe.jl     # Large-universe cross-asset scaling
|-- run_garch_suite.jl, run_ks_block_bootstrap.jl, ...   # Standalone CSV producers (results/robustness/*.csv)
|-- src/
|   |-- Types.jl                          # HMM / GARCH / copula type definitions
|   |-- Files.jl                          # JLD2 data loaders
|   |-- Factory.jl                        # build() constructors
|   |-- Compute.jl                        # Baum-Welch, ECM, GARCH MLE, simulation
|   |-- CrossAsset.jl                     # SIM and Gaussian / Student-t copula generators
|   |-- Visualize.jl                      # Plotting utilities
|-- data/                                 # JLD2 datasets (active IS / OoS bundles)
|-- results/                              # Generated metrics tables (per-ticker, per-K)
|-- figs/                                 # Generated SVG / PDF figures
|-- test/                                 # Test suite
|-- docs/                                 # Documenter.jl docs
|-- _attic_v10/                           # Archived journal-revision-era runners, results, and docs
```

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
