# CLAUDE.md -- Development Guidelines for ContinuousHMM

This file provides context for AI-assisted development on this codebase.

## Project Summary
This project compares three approaches to modeling financial time series:
1. **Continuous HMM (Baum-Welch, no jumps)** — the new contribution. Gaussian emissions trained via EM.
2. **Discrete HMM with Poisson jumps** — the baseline from the prior paper (JumpHMM.jl). Regime teleportation.
3. **GARCH(1,1)** — the classical benchmark for conditional variance modeling.

The study demonstrates that the continuous HMM at small K reproduces all three stylized facts (heavy tails, negligible linear ACF, persistent volatility clustering) without requiring jump mechanisms.

## Module Load Order (Critical)
Source files must be loaded in this exact order (defined in `Include.jl`):
1. `Types.jl` -- abstract and concrete type definitions (no dependencies on other src files)
2. `Files.jl` -- data loading (depends on path constants from `Include.jl`)
3. `Factory.jl` -- model constructors (depends on Types)
4. `Compute.jl` -- algorithms and simulation (depends on Types, uses Factory via `build`)
5. `CrossAsset.jl` -- SIM and Gaussian/Student-t copula multi-asset generators
6. `Visualize.jl` -- plotting utilities (depends on StatsBase for `autocor`)

Rearranging this order will cause `UndefVarError` at load time.

## Type Hierarchy
```
AbstractMarkovModel
  |-- MyHiddenMarkovModel                  (discrete, baseline)
  |-- MyHiddenMarkovModelWithJumps         (discrete + Poisson jumps, baseline)
  |-- MyContinuousHiddenMarkovModel        (continuous Gaussian, new contribution)

MyGARCHModel                               (GARCH(1,1) benchmark)

AbstractDistributionModel
  |-- StudentTModel    (Bayesian Student-t via Turing)
  |-- LaplaceModel     (Bayesian Laplace via Turing)
```

All mutable model structs use empty inner constructors. The `build()` factory pattern populates fields after construction.

## Key Design Decisions

### Three-Model Comparison
- **Discrete HMM + Jumps**: From JumpHMM.jl. Discretizes returns into bins, frequency-counts transitions, adds Poisson regime teleportation.
- **Continuous HMM**: Trains Gaussian emission parameters and transition matrices directly from raw returns via Baum-Welch. No jumps needed.
- **GARCH(1,1)**: σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}. Fitted via MLE with Nelder-Mead. Single-regime classical benchmark.

### Baum-Welch (EM) for Parameter Learning
Quantile-based initialization, log-space forward-backward, convergence monitoring.

### Log-Space Numerics
All forward-backward computations use `_logsumexp_vec()` to prevent floating-point underflow.

### Jump Mechanism (Discrete Model Only)
- Jump probability `epsilon` checked at each time step
- Jump duration drawn from `Poisson(lambda)`
- During jumps, coin flip (52/48) selects crash states (1:3) or boom states (N-2:N)
- Used ONLY in the discrete baseline for comparison — not in the continuous model

### Functor Interface
HMM models are callable: `model(start_state, n_steps)` dispatches to `_simulate()`.

## Commands

### Load the framework
```julia
include("Include.jl")
```

### Run tests
```julia
using Pkg; Pkg.test()
```

### Run full analysis pipeline
```julia
include("run_all_analysis.jl")
```

## Conventions

- **Naming**: Types use `My` prefix. Factory methods are `build()` with type dispatch. Private methods start with `_`.
- **Data format**: JLD2 files. DataFrames: `date`, `open`, `high`, `low`, `close`, `volume`, `volume_weighted_average_price`.
- **Returns convention**: Annualized excess log returns: `G_t = (1/dt) * ln(P_t / P_{t-1}) - r_f`, with `dt = 1/252` and `r_f = 0`.
- **Semicolons**: Julia semicolons at end of lines for consistency with professor Varner's style.
- **Figures**: Generated SVGs go in `figs/`, named as `Fig-{TICKER}-{Type}-{Detail}.svg`.

## What NOT to Change Without Discussion
- The Baum-Welch convergence tolerance (`tol=1e-4`) and default iterations (`max_iter=30`)
- The jump coin-flip bias (52/48) in the discrete model — calibrated to empirical crash/boom asymmetry
- Tail state ranges (bottom 3, top 3) in the discrete model
- The `_logsumexp_vec` implementation — numerical stability depends on this exact formulation
