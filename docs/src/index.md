# ContinuousHMM

A Julia framework for modeling financial time series with **continuous-emission Hidden Markov Models**, benchmarked against a discrete jump-HMM baseline and GARCH(1,1).

## Overview

This project compares three approaches to modeling the stylized facts of financial returns (heavy tails, negligible linear autocorrelation, persistent volatility clustering):

1. **Continuous HMM (Baum-Welch, no jumps)** -- the contribution. Gaussian emissions trained directly from raw returns via the Baum-Welch (Expectation-Maximization) algorithm.
2. **Discrete HMM with Poisson jumps** -- the baseline from [JumpHMM.jl](https://github.com/varnerlab/JumpHMM.jl). Returns are binned and a Poisson regime-teleportation mechanism injects crashes and booms.
3. **GARCH(1,1)** -- the classical single-regime conditional-variance benchmark, fitted by MLE.

The central result is that the continuous HMM reproduces all three stylized facts at small state counts **without** any jump mechanism. The Poisson jump process exists only in the discrete baseline (`MyHiddenMarkovModelWithJumps`); the continuous model never teleports.

Beyond the headline Gaussian CHMM, the framework includes heavier-tailed emission families (Student-t, Laplace, GED), a semi-Markov CHMM with explicit sojourn distributions, Markov-switching and asymmetric GARCH variants, stochastic-volatility / MSM / jump-diffusion baselines, a GRU neural generator, and copula-based multi-asset generators.

### Key Capabilities

- **Baum-Welch training** of continuous Gaussian HMMs directly from raw return data -- no jumps required
- **Alternative emission families** (Student-t, Laplace, GED) via ECM, plus a semi-Markov CHMM with state-dependent sojourn distributions
- **Baseline models** for comparison: discrete jump-HMM, GARCH(1,1), MS-GARCH, EGARCH/GJR/GARCH-t/HAR-RV, SV-AR(1), MSM, jump-diffusion, and a GRU generator
- **Bayesian parameter learning** for Student-t and Laplace emission distributions via Turing.jl
- **Multi-asset generation** via single-index, Gaussian/Student-t copula, and truncated C-vine models
- **Validation tools** including ACF comparison, distribution matching, KS tests, and signature/MMD metrics
- **Multiple dispatch** factory pattern (`build`) for model construction

## Quick Start

```julia
include("Include.jl")

# Load data and compute returns
dataset = MyPortfolioDataSet()["dataset"]
R = log_growth_matrix(dataset, "SPY")

# Train a small-K continuous HMM (no jumps)
model = build(MyContinuousHiddenMarkovModel, (
    observations = R,
    number_of_states = 6,
    max_iter = 60
))

# Simulate state chains, then draw returns from the per-state emissions
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
