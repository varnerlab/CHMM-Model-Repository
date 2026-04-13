# ContinuousHMM

A Julia framework for modeling financial time series using **Continuous Hidden Markov Models** with Gaussian emissions learned via the Baum-Welch algorithm. At small K, the CHMM alone reproduces all three canonical stylized facts of financial returns (heavy tails, negligible linear autocorrelation, persistent volatility clustering).

Includes option pricing via **CHMM regime-switching volatility** (using VIX-trained regimes) and a **Heston stochastic volatility** benchmark.

## Academic Citation
Alswaidan A, Varner JD. Continuous Hidden Markov Models for Financial Time Series. *In preparation*, Cornell University, 2026.

## Overview
Financial markets exhibit complex behaviors -- volatility clustering, heavy-tailed return distributions, and regime-dependent dynamics -- that classical models fail to capture. This framework addresses these challenges by:

- Training **continuous Gaussian HMMs** via the Baum-Welch (Expectation-Maximization) algorithm on observed returns
- Providing **Bayesian parameter learning** for alternative emission distributions (Student's t, Laplace) via Turing.jl
- Validating simulated paths against historical data using autocorrelation analysis, distribution comparisons, and statistical tests
- **Option pricing** via regime-switching Monte Carlo (CHMM) and Heston stochastic volatility

## Installation

Ensure Julia 1.9+ is installed. Clone the repository and instantiate the environment:

```bash
git clone https://github.com/varnerlab/ContinuousHMM.jl.git
cd ContinuousHMM.jl
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
# Train a 6-state Gaussian HMM
model = build(MyContinuousHiddenMarkovModel, (
    observations = R,
    number_of_states = 6,
    max_iter = 60
))

# Inspect convergence
plot(model.log_likelihood_history, xlabel="Iteration", ylabel="Log-Likelihood", title="Baum-Welch Convergence")
```

### Step 3: Simulate and validate
```julia
# Compute stationary distribution for initial state sampling
K = length(model.states)
T_mat = zeros(K, K)
for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
π_stat = (T_mat^1000)[1, :]
start_dist = Categorical(π_stat)

# Simulate 1000 paths
n_paths = 1000
s0 = rand(start_dist)
states = model(s0, 252)
simulated_returns = [rand(model.emission[s]) for s in states]

# Compare ACF of observed vs simulated returns
plot_acf_comparison(R, simulated_returns, "SPY |Returns| ACF", 1; is_absolute=true)
```

## How It Works

### Baum-Welch Training
The framework uses a full Expectation-Maximization implementation with:
- **Quantile-based initialization**: sorted observations are split into K chunks to seed regime means and standard deviations, avoiding poor local optima
- **Log-space forward-backward**: all computations use log-sum-exp for numerical stability with long observation sequences
- **Convergence monitoring**: log-likelihood history is tracked and stored for diagnostics

### Bayesian Distribution Learning
For regime-specific emission modeling beyond Gaussians, Turing.jl enables MCMC inference:
- **Student's t-distribution**: captures heavy tails with learned degrees of freedom
- **Laplace distribution**: models peaked, leptokurtic returns

### Option Pricing
- **CHMM regime-switching**: Train a CHMM on VIX data, decode regimes via Viterbi, map median VIX levels to equity volatility per regime. Drives GBM paths for Monte Carlo option pricing.
- **Heston benchmark**: Standard stochastic volatility model with mean-reverting variance (Euler-Maruyama with full truncation).
- **Black-Scholes**: Analytical benchmark for comparison and implied volatility inversion.

## Data

The framework is validated on a comprehensive dataset:

| Dataset | Period | Trading Days | Coverage |
|---------|--------|-------------|----------|
| Training | Jan 2014 -- Dec 2024 | 2,515 | 400+ US equities and ETFs |
| Out-of-Sample | Jan 2025 -- Nov 2025 | ~240 | Same universe |
| VIX Training | ~20 years through Dec 2024 | ~5,000 | VIX volatility index |
| VIX Test | Jan 2025 onward | ~60 | VIX volatility index |

Primary validation tickers: **SPY**, **AAPL**, **NVDA**

## Project Structure

```
.
|-- Include.jl              # Entry point: loads dependencies and source files
|-- Project.toml            # Julia package dependencies
|-- run_all_analysis.jl     # Full analysis pipeline (all K values)
|-- src/
|   |-- Types.jl            # Abstract and concrete model type definitions
|   |-- Files.jl            # Data loading utilities (JLD2 datasets)
|   |-- Factory.jl          # Model constructors (build methods)
|   |-- Compute.jl          # Baum-Welch algorithm, simulation, growth calculations
|   |-- Pricing.jl          # Option pricing (CHMM, Heston, Black-Scholes)
|   |-- Visualize.jl        # Plotting utilities (ACF, regime overlay, pricing)
|-- data/                   # JLD2 datasets (training, test, VIX)
|-- figs/                   # Generated figures (SVG)
|-- Notebooks/              # Interactive analysis notebooks (run in order)
|   |-- 01-Stylized-Facts.ipynb               # Empirical analysis (Figure 1, Table 1)
|   |-- 02-Model-Fitting.ipynb                # Baum-Welch training + model internals
|   |-- 03-In-Sample-Validation.ipynb         # IS validation (Figure 3, metrics)
|   |-- 04-Out-of-Sample-Validation.ipynb     # OoS evaluation (Figure 4)
|   |-- 05-Sensitivity-and-Cross-Asset.ipynb  # K sensitivity + multi-ticker
|   |-- 06-Option-Pricing.ipynb               # CHMM vs Heston option pricing
|-- docs/                   # Documentation source (Documenter.jl)
|-- test/                   # Test suite
```

## Disclaimer

This project is intended **for research and educational purposes only**. It does **not** constitute financial advice, investment recommendations, or trading strategies.

The models and simulations provided are simplified representations of financial markets and may not capture all real-world complexities. Past performance of any model does not guarantee future results. Users are solely responsible for any decisions made using this code.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
