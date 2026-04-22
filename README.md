# ContinuousHMM

A Julia framework for modeling financial time series, comparing three approaches:

1. **Continuous Hidden Markov Model** (Baum-Welch, no jumps) — the new contribution
2. **Discrete HMM with Poisson Jumps** — baseline from the prior paper ([JumpHMM.jl](https://github.com/varnerlab/JumpHMM.jl))
3. **GARCH(1,1)** — classical conditional variance benchmark

At small K, the continuous HMM alone reproduces all three canonical stylized facts of financial returns (heavy tails, negligible linear autocorrelation, persistent volatility clustering) without requiring jump mechanisms.

## Academic Citation
Alswaidan A, Varner JD. Continuous Hidden Markov Models for Financial Time Series. *In preparation*, Cornell University, 2026.

## Overview

Financial markets exhibit volatility clustering, heavy-tailed returns, and regime-dependent dynamics. This framework:

- Trains **continuous Gaussian HMMs** via Baum-Welch (EM) on observed returns
- Compares against **discrete HMM + Poisson jumps** (regime teleportation baseline)
- Benchmarks against **GARCH(1,1)** fitted via maximum likelihood
- Provides **Bayesian parameter learning** for alternative emissions (Student's t, Laplace) via Turing.jl
- Validates with KS tests, ACF matching, kurtosis, Wasserstein distance, and Hellinger distance

## Quick Start

```julia
include("Include.jl")

# Load SPY returns
dataset = MyPortfolioDataSet()["dataset"]
R = log_growth_matrix(dataset, "SPY"; Δt=1/252, risk_free_rate=0.0)

# Train continuous HMM (6 states)
chmm = build(MyContinuousHiddenMarkovModel, (
    observations = R, number_of_states = 6, max_iter = 60))

# Fit GARCH(1,1) benchmark
garch = build(MyGARCHModel, (observations = R,))

# Simulate and compare
K = length(chmm.states)
T_mat = zeros(K,K); for i in 1:K; T_mat[i,:] = probs(chmm.transition[i]); end
π_stat = (T_mat^1000)[1,:]; start_dist = Categorical(π_stat)

states = chmm(rand(start_dist), 252)
chmm_returns = [rand(chmm.emission[s]) for s in states]
garch_returns = simulate_garch(garch, 252)
```

## Data

| Dataset | Period | Trading Days | Coverage |
|---------|--------|-------------|----------|
| Training | Jan 2014 -- Dec 2024 | 2,515 | 400+ US equities and ETFs |
| Out-of-Sample | Jan 2025 -- Nov 2025 | ~240 | Same universe |

## Project Structure

```
.
|-- Include.jl              # Entry point
|-- run_all_analysis.jl     # Full analysis pipeline
|-- src/
|   |-- Types.jl            # All type definitions (HMM, GARCH)
|   |-- Files.jl            # Data loading (JLD2)
|   |-- Factory.jl          # Model constructors (build methods)
|   |-- Compute.jl          # Baum-Welch, GARCH MLE, simulation, growth calc
|   |-- CrossAsset.jl       # SIM and Gaussian/Student-t copula generators
|   |-- Visualize.jl        # Plotting utilities
|-- Notebooks/              # Interactive analysis notebooks
|   |-- 01-Stylized-Facts.ipynb           # Empirical analysis
|   |-- 02-Discrete-HMM-Jumps.ipynb      # Discrete baseline
|   |-- 03-Continuous-HMM.ipynb           # New contribution
|   |-- 04-GARCH-Benchmark.ipynb          # Classical benchmark
|   |-- 05-Model-Comparison.ipynb         # Head-to-head comparison
|-- data/                   # JLD2 datasets
|-- test/                   # Test suite
|-- docs/                   # Documenter.jl docs
```

## Disclaimer

For research and educational purposes only. Not financial advice.

## License

MIT License. See [LICENSE](LICENSE).
