# CLAUDE.md -- Development Guidelines for ContinuousHMM

This file provides context for AI-assisted development on this codebase.

## Project Summary
Continuous Hidden Markov Models with Gaussian emissions for financial time series simulation. Trained via Baum-Welch (EM). At small K, the CHMM alone reproduces all three canonical stylized facts (heavy tails, negligible linear ACF, persistent volatility clustering) to some extent.

Includes option pricing via CHMM regime-switching volatility (using VIX regimes) and Heston stochastic volatility benchmark.

## Module Load Order (Critical)
Source files must be loaded in this exact order (defined in `Include.jl`):
1. `Types.jl` -- abstract and concrete type definitions (no dependencies on other src files)
2. `Files.jl` -- data loading (depends on path constants from `Include.jl`)
3. `Factory.jl` -- model constructors (depends on Types)
4. `Compute.jl` -- algorithms and simulation (depends on Types, uses Factory via `build`)
5. `Pricing.jl` -- option pricing engines (depends on Types, Compute)
6. `Visualize.jl` -- plotting utilities (depends on StatsBase for `autocor`)

Rearranging this order will cause `UndefVarError` at load time.

## Type Hierarchy
```
AbstractMarkovModel
  |-- MyContinuousHiddenMarkovModel   (continuous Gaussian, Baum-Welch trained)

AbstractDistributionModel
  |-- StudentTModel    (Bayesian Student-t via Turing)
  |-- LaplaceModel     (Bayesian Laplace via Turing)

AbstractPricingModel
  |-- MyCHMMPricingModel    (regime-switching MC via VIX-trained CHMM)
  |-- MyHestonPricingModel  (Heston stochastic vol benchmark)
```

All model structs are currently `mutable` with empty inner constructors. The `build()` factory pattern populates fields after construction.

## Key Design Decisions

### Baum-Welch (EM) for Parameter Learning
This project trains continuous emission parameters (means, standard deviations) and transition matrices from raw returns via the Baum-Welch algorithm.

### Quantile-Based Initialization
EM is sensitive to initialization. We sort observations into K quantile chunks and use each chunk's mean/std as initial emission parameters. This prevents degenerate solutions.

### Log-Space Numerics
All forward-backward computations use `_logsumexp_vec()` to prevent floating-point underflow. Never convert to probability space during the E-step.

### No Jump Mechanisms
This study focuses on the continuous HMM alone. At small K, the CHMM reproduces all stylized facts to some extent without requiring jump processes. All jump-related code (Poisson teleportation, discrete HMMs) has been removed.

### Option Pricing via VIX Regimes
The CHMM pricer trains a separate HMM on VIX data, decodes regimes via Viterbi, and maps each regime's median VIX level to equity volatility (σ_s = median(VIX|state=s) / 100). This drives regime-switching GBM paths for Monte Carlo option pricing.

### Functor Interface
Models are callable: `model(start_state, n_steps)` dispatches to `_simulate()`. This keeps the public API clean.

## Commands

### Load the framework
```julia
include("Include.jl")
```

### Run tests
```julia
using Pkg; Pkg.test()
```
Or directly:
```julia
include("test/runtests.jl")
```

### Run full analysis pipeline
```julia
include("run_all_analysis.jl")
```

## Conventions

- **Naming**: Types use `My` prefix (e.g., `MyContinuousHiddenMarkovModel`). Factory methods are `build()` with type dispatch. Private methods start with `_`.
- **Data format**: All datasets are JLD2 files. DataFrames have columns: `date`, `open`, `high`, `low`, `close`, `volume`, `volume_weighted_average_price`.
- **Returns convention**: Annualized excess log returns: `G_t = (1/dt) * ln(P_t / P_{t-1}) - r_f`, with `dt = 1/252` (daily) and `r_f = 0` by default.
- **Semicolons**: Julia semicolons at end of lines are used throughout for consistency with professor Varner's style.
- **Figures**: Generated SVGs go in `figs/`, named as `Fig-{TICKER}-{Type}-{Detail}.svg`.

## What NOT to Change Without Discussion
- The Baum-Welch convergence tolerance (`tol=1e-4`) and default iterations (`max_iter=30`) -- these are tuned for the datasets
- The `_logsumexp_vec` implementation -- numerical stability depends on this exact formulation
