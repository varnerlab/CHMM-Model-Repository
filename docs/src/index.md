# ContinuousJumpHMM

A Julia framework for modeling financial time series using **Continuous Hidden Markov Models** with **Poisson jump processes**.

## Overview

This project extends the discrete HMM methods in [JumpHMM.jl](https://github.com/varnerlab/JumpHMM.jl) to continuous Gaussian emissions trained via the Baum-Welch (Expectation-Maximization) algorithm. A regime-teleportation jump mechanism captures sudden market shocks (crashes and booms) while preserving the stylized facts of financial returns.

### Key Capabilities

- **Baum-Welch training** of continuous Gaussian HMMs from raw return data
- **Poisson jump process** with regime teleportation for tail-event modeling
- **Bayesian parameter learning** for Student-t and Laplace emission distributions via Turing.jl
- **Validation tools** including ACF comparison, distribution matching, and KS tests
- **Multiple dispatch** factory pattern for model construction

## Quick Start

```julia
include("Include.jl")

# Load data and compute returns
dataset = MyPortfolioDataSet()["dataset"]
R = log_growth_matrix(dataset, "SPY")

# Train a 13-state continuous HMM
model = build(MyContinuousHiddenMarkovModel, (
    observations = R,
    number_of_states = 13,
    max_iter = 60
))

# Simulate paths
paths = [model(1, 252) for _ in 1:1000]
```

See [Getting Started](@ref) for a complete walkthrough.

## Contents

```@contents
Pages = [
    "getting_started.md",
    "fitting.md",
    "simulation.md",
    "validation.md",
    "finance.md",
    "api.md",
]
Depth = 2
```
