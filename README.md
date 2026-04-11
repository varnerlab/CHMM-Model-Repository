# ContinuousJumpHMM

A Julia framework for modeling financial time series using **Continuous Hidden Markov Models** with **Poisson jump processes**. This project extends discrete HMM methods (see [JumpHMM.jl](https://github.com/varnerlab/JumpHMM.jl)) to continuous Gaussian emissions learned via the Baum-Welch algorithm, with a regime-teleportation jump mechanism for capturing sudden market shocks.

## Academic Citation
Alswaidan A, Varner JD. Continuous Hidden Markov Models with Poisson Jump Processes for Financial Time Series. *In preparation*, Cornell University, 2026.

## Overview
Financial markets exhibit complex behaviors -- volatility clustering, heavy-tailed return distributions, and regime-dependent dynamics -- that classical models fail to capture. This framework addresses these challenges by:

- Training **continuous Gaussian HMMs** via the Baum-Welch (Expectation-Maximization) algorithm on observed returns
- Augmenting learned dynamics with a **Poisson jump process** that triggers regime teleportation into extreme tail states (crash/boom)
- Providing **Bayesian parameter learning** for alternative emission distributions (Student's t, Laplace) via Turing.jl
- Validating simulated paths against historical data using autocorrelation analysis, distribution comparisons, and statistical tests

## Installation

Ensure Julia 1.9+ is installed. Clone the repository and instantiate the environment:

```bash
git clone https://github.com/varnerlab/ContinuousJumpHMM.jl.git
cd ContinuousJumpHMM.jl
```

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Then load the framework:
```julia
include("Include.jl")
```

## Single-Asset Example

### Step 1: Load data and compute excess log-growth rates
```julia
include("Include.jl")

# Load the training dataset (SP500 constituents, 2014-2024)
dataset = MyPortfolioDataSet()["dataset"]

# Compute annualized excess log returns for SPY
R = log_growth_matrix(dataset, "SPY"; Dt = 1.0/252.0, risk_free_rate = 0.0)
```

### Step 2: Fit a continuous HMM via Baum-Welch
```julia
# Train a 13-state Gaussian HMM
base_model = build(MyContinuousHiddenMarkovModel, (
    observations = R,
    number_of_states = 13,
    max_iter = 60
))

# Inspect convergence
plot(base_model.log_likelihood_history, xlabel="Iteration", ylabel="Log-Likelihood", title="Baum-Welch Convergence")
```

### Step 3: Add jump dynamics
```julia
# Wrap the trained model with Poisson jump parameters
model = build(MyContinuousHiddenMarkovModelWithJumps, (
    base_model = base_model,
    epsilon = 0.01,   # jump probability per step
    lambda = 3.0      # mean jump duration (Poisson rate)
))
```

### Step 4: Simulate and validate
```julia
# Simulate 1000 paths of 252 trading days
n_paths = 1000
paths = [model(1, 252) for _ in 1:n_paths]

# Compare ACF of observed vs simulated returns
idx = rand(1:n_paths)
simulated_returns = [rand(model.emission[s]) for s in paths[idx]]
plot_acf_comparison(R, simulated_returns, "SPY Returns ACF", idx)

# Volatility clustering (absolute returns ACF)
plot_acf_comparison(R, simulated_returns, "SPY |Returns| ACF", idx; is_absolute=true)
```

## How It Works

### Baum-Welch Training
The framework uses a full Expectation-Maximization implementation with:
- **Quantile-based initialization**: sorted observations are split into K chunks to seed regime means and standard deviations, avoiding poor local optima
- **Log-space forward-backward**: all computations use log-sum-exp for numerical stability with long observation sequences
- **Convergence monitoring**: log-likelihood history is tracked and stored for diagnostics

### Jump Mechanism (Regime Teleportation)
At each time step, with probability `epsilon`:
1. A **Poisson-distributed duration** is drawn (mean = `lambda`)
2. For each step in the jump, a **coin flip** (52/48 crash/boom bias) selects the target pool:
   - **Crash states**: bottom 3 regimes (lowest mean returns)
   - **Boom states**: top 3 regimes (highest mean returns)
3. The system is **teleported** into the selected tail pool, overriding normal Markov transitions

This design produces volatility (magnitude) without directional bias (trend), matching the stylized facts of financial returns.

### Bayesian Distribution Learning
For regime-specific emission modeling beyond Gaussians, Turing.jl enables MCMC inference:
- **Student's t-distribution**: captures heavy tails with learned degrees of freedom
- **Laplace distribution**: models peaked, leptokurtic returns

## Data

The framework is validated on a comprehensive dataset:

| Dataset | Period | Trading Days | Coverage |
|---------|--------|-------------|----------|
| Training | Jan 2014 -- Dec 2024 | 2,515 | 400+ US equities and ETFs |
| Out-of-Sample | Jan 2025 -- Nov 2025 | ~240 | Same universe |

Primary validation tickers: **SPY**, **AAPL**, **NVDA**

## Project Structure

```
.
|-- Include.jl              # Entry point: loads dependencies and source files
|-- Project.toml            # Julia package dependencies
|-- src/
|   |-- Types.jl            # Abstract and concrete model type definitions
|   |-- Files.jl            # Data loading utilities (JLD2 datasets)
|   |-- Factory.jl          # Model constructors (build methods)
|   |-- Compute.jl          # Baum-Welch algorithm, simulation, growth calculations
|   |-- Visualize.jl        # ACF comparison plotting
|-- data/                   # JLD2 datasets (training, test, pre-trained models)
|-- figs/                   # Generated figures (SVG)
|-- Notebooks/              # Paper-ready analysis notebooks (run in order)
|   |-- 01-Stylized-Facts.ipynb                     # Empirical analysis (Figure 1, Table 1)
|   |-- 02-Model-Fitting.ipynb                      # Baum-Welch training + model internals
|   |-- 03-Hyperparameter-Tuning.ipynb              # Grid search for jump params (ε, λ)
|   |-- 04-In-Sample-Validation.ipynb               # IS comparison NJ vs WJ (Figure 3, Table 2)
|   |-- 05-Out-of-Sample-Validation.ipynb           # OoS evaluation (Figure 4, Table 2)
|   |-- 06-Sensitivity-and-Cross-Asset.ipynb        # Sensitivity + NVDA/JNJ/JPM (Tables T1, T2)
|-- legacy/                 # Previous notebook versions
|-- docs/                   # Documentation source (Documenter.jl)
|-- test/                   # Test suite
```

## Disclaimer

This project is intended **for research and educational purposes only**. It does **not** constitute financial advice, investment recommendations, or trading strategies.

The models and simulations provided are simplified representations of financial markets and may not capture all real-world complexities. Past performance of any model does not guarantee future results. Users are solely responsible for any decisions made using this code.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
