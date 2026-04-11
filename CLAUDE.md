# CLAUDE.md -- Development Guidelines for ContinuousJumpHMM

This file provides context for AI-assisted development on this codebase.

## Project Summary
Continuous Hidden Markov Models with Poisson jump processes for financial time series simulation. Extends the discrete [JumpHMM.jl](https://github.com/varnerlab/JumpHMM.jl) to continuous Gaussian emissions trained via Baum-Welch (EM).

## Module Load Order (Critical)
Source files must be loaded in this exact order (defined in `Include.jl`):
1. `Types.jl` -- abstract and concrete type definitions (no dependencies on other src files)
2. `Files.jl` -- data loading (depends on path constants from `Include.jl`)
3. `Factory.jl` -- model constructors (depends on Types)
4. `Compute.jl` -- algorithms and simulation (depends on Types, uses Factory via `build`)
5. `Visualize.jl` -- plotting utilities (depends on StatsBase for `autocor`)

Rearranging this order will cause `UndefVarError` at load time.

## Type Hierarchy
```
AbstractMarkovModel
  |-- MyHiddenMarkovModel                        (discrete, legacy)
  |-- MyHiddenMarkovModelWithJumps               (discrete + jumps, legacy)
  |-- MyContinuousHiddenMarkovModel              (continuous Gaussian, active)
  |-- MyContinuousHiddenMarkovModelWithJumps     (continuous + jumps, active)

AbstractDistributionModel
  |-- StudentTModel    (Bayesian Student-t via Turing)
  |-- LaplaceModel     (Bayesian Laplace via Turing)
```

All model structs are currently `mutable` with empty inner constructors. The `build()` factory pattern populates fields after construction.

## Key Design Decisions

### Baum-Welch (EM) for Parameter Learning
Unlike JumpHMM.jl which uses direct frequency counting on discretized observations, this project trains continuous emission parameters (means, standard deviations) and transition matrices from raw returns via the Baum-Welch algorithm. This is the core methodological contribution.

### Quantile-Based Initialization
EM is sensitive to initialization. We sort observations into K quantile chunks and use each chunk's mean/std as initial emission parameters. This prevents degenerate solutions.

### Log-Space Numerics
All forward-backward computations use `_logsumexp_vec()` to prevent floating-point underflow. Never convert to probability space during the E-step.

### Jump Mechanism (Regime Teleportation)
- Jump probability `epsilon` is checked at each time step
- Jump duration drawn from `Poisson(lambda)`
- During jumps, coin flip (52/48) selects crash states (1:3) or boom states (N-2:N)
- The coin flip happens **inside** the duration loop to produce volatility without directional bias

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

### Build documentation
```julia
include("docs/make.jl")
```

## Conventions

- **Naming**: Types use `My` prefix (e.g., `MyContinuousHiddenMarkovModel`). Factory methods are `build()` with type dispatch. Private methods start with `_`.
- **Data format**: All datasets are JLD2 files. DataFrames have columns: `date`, `open`, `high`, `low`, `close`, `volume`, `volume_weighted_average_price`.
- **Returns convention**: Annualized excess log returns: `G_t = (1/dt) * ln(P_t / P_{t-1}) - r_f`, with `dt = 1/252` (daily) and `r_f = 0` by default.
- **Semicolons**: Julia semicolons at end of lines are used throughout for consistency with professor Varner's style.
- **Figures**: Generated SVGs go in `figs/`, named as `Fig-{TICKER}-{Type}-{Detail}.svg`.

## What NOT to Change Without Discussion
- The Baum-Welch convergence tolerance (`tol=1e-4`) and default iterations (`max_iter=30`) -- these are tuned for the datasets
- The jump coin-flip bias (52/48) -- calibrated to empirical crash/boom asymmetry
- Tail state ranges (bottom 3, top 3) -- hardcoded but consistent with the paper's methodology
- The `_logsumexp_vec` implementation -- numerical stability depends on this exact formulation
