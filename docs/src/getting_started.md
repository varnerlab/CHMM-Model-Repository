# Getting Started

This guide walks through the complete workflow: loading data, training a model, adding jumps, simulating paths, and validating results.

## Prerequisites

Ensure Julia 1.9+ is installed and the project environment is instantiated:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Step 1: Load the Framework

```julia
include("Include.jl")
```

This loads all dependencies and source files. The following constants are defined:
- `_ROOT` -- project root directory
- `_PATH_TO_SRC` -- path to `src/`
- `_PATH_TO_DATA` -- path to `data/`
- `_PATH_TO_FIGURES` -- path to `figs/`

## Step 2: Load Data

```julia
# Training dataset: SP500 constituents, Jan 2014 -- Dec 2024
dataset = MyPortfolioDataSet()["dataset"]

# Out-of-sample dataset: Jan 2025 -- Nov 2025
oos_dataset = MyOutOfSamplePortfolioDataSet()["dataset"]
```

Each dataset is a `Dict{String, DataFrame}` mapping ticker symbols to OHLCV DataFrames.

## Step 3: Compute Excess Log Returns

```julia
# Single ticker (returns a Vector)
R = log_growth_matrix(dataset, "SPY")

# Multiple tickers (returns a Matrix: time x firms)
R_multi = log_growth_matrix(dataset, ["SPY", "AAPL", "NVDA"])
```

Returns are annualized by default (`Dt = 1/252`) with zero risk-free rate.

## Step 4: Train a Continuous HMM

```julia
base_model = build(MyContinuousHiddenMarkovModel, (
    observations = R,
    number_of_states = 13,
    max_iter = 60
))
```

The Baum-Welch algorithm runs until convergence (`tol = 1e-4`) or `max_iter` iterations. Check convergence:

```julia
plot(base_model.log_likelihood_history,
    xlabel = "Iteration", ylabel = "Log-Likelihood",
    title = "Baum-Welch Convergence")
```

Inspect learned regimes:

```julia
for s in 1:13
    d = base_model.emission[s]
    println("State $s: mean = $(round(mean(d), digits=4)), std = $(round(std(d), digits=4))")
end
```

## Step 5: Add Jump Dynamics

```julia
model = build(MyContinuousHiddenMarkovModelWithJumps, (
    base_model = base_model,
    epsilon = 0.01,   # jump probability per step
    lambda = 3.0      # mean jump duration
))
```

## Step 6: Simulate Paths

```julia
n_paths = 1000
n_steps = 252  # one trading year

# Simulate state chains
state_chains = [model(1, n_steps) for _ in 1:n_paths]

# Convert state chains to returns
simulated_returns = [
    [rand(model.emission[s]) for s in chain]
    for chain in state_chains
]
```

## Step 7: Validate

```julia
# Pick a random simulated path for comparison
idx = rand(1:n_paths)

# ACF of returns (should show no autocorrelation)
plot_acf_comparison(R, simulated_returns[idx], "SPY Returns ACF", idx)

# ACF of |returns| (should show slow decay = volatility clustering)
plot_acf_comparison(R, simulated_returns[idx], "SPY |Returns| ACF", idx; is_absolute=true)
```

For more details on each step, see the dedicated documentation pages.
