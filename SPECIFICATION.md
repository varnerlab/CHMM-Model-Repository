# SPECIFICATION.md -- Design Philosophy and Architecture

## Motivation

The discrete HMM approach in [JumpHMM.jl](https://github.com/varnerlab/JumpHMM.jl) discretizes continuous returns into bins, fits transition matrices by frequency counting, and models emissions with Student-t distributions. While effective, this introduces quantization error and limits the model's ability to capture fine-grained regime dynamics.

This project extends the methodology to **continuous emissions** -- each hidden state emits from a Gaussian distribution with learned mean and variance -- trained via the **Baum-Welch algorithm**. The Poisson jump mechanism is preserved but operates on the continuous model's state space.

## Architecture

### Separation of Concerns

The codebase follows a strict separation into five modules:

| Module | Responsibility |
|--------|---------------|
| `Types.jl` | Data structures only. No logic, no imports beyond Julia Base. |
| `Files.jl` | Data I/O. Maps file paths to loaded JLD2 datasets. |
| `Factory.jl` | Construction. Every model instance is created through `build()`. |
| `Compute.jl` | Algorithms. Baum-Welch, simulation, growth-rate calculations. |
| `Visualize.jl` | Plotting. Comparison plots for model validation. |

### Factory Pattern

All model construction goes through `build(ModelType, data::NamedTuple)`. This pattern:
- Centralizes validation and initialization logic
- Makes construction self-documenting (NamedTuple keys describe inputs)
- Enables dispatch: the same `build` function handles discrete, continuous, and jump variants

### Functor Interface

Models implement Julia's callable-object protocol:
```julia
(m::MyContinuousHiddenMarkovModel)(start, steps) = _simulate(m, start, steps)
```
This provides a clean public API (`model(1, 252)`) while keeping simulation logic private.

## Algorithm: Baum-Welch (EM)

### Why Baum-Welch Instead of Frequency Counting

JumpHMM.jl uses direct frequency counting on discretized states -- fast and non-iterative, but requires a discretization step (Laplace quantile binning). For continuous emissions:
- **No discretization**: parameters are learned directly from continuous observations
- **Soft assignments**: the E-step computes posterior state probabilities (gamma), allowing fractional state membership
- **Joint optimization**: transition matrix, emission means, and emission variances are updated simultaneously

### Implementation Details

**Initialization**: Quantile-based. Observations are sorted and partitioned into K equal-sized chunks. Each chunk's sample mean and standard deviation seed the corresponding state's emission parameters. This prevents the common EM failure mode of all states collapsing to the global mean.

**Forward-Backward in Log-Space**: The forward variable `alpha[t,k]` can underflow for long sequences (N > 1000). All computations use log-probabilities with `logsumexp` for numerically stable aggregation:
```
log_alpha[t,j] = logsumexp(log_alpha[t-1,:] + log(T[:,j])) + log_B[t,j]
```

**Convergence**: The algorithm terminates when the change in log-likelihood falls below `tol=1e-4` or after `max_iter` iterations. The full log-likelihood history is stored in the model for diagnostic inspection.

## Algorithm: Poisson Jump Process

### Design Rationale

Real financial returns exhibit:
- **Fat tails**: extreme returns occur more often than Gaussian models predict
- **Volatility clustering**: large moves tend to follow large moves
- **No autocorrelation in returns**: but positive autocorrelation in |returns|

The jump mechanism addresses all three by:
1. **Fat tails**: jumps force the system into extreme states (crash/boom), generating outlier returns
2. **Volatility clustering**: jump *duration* (Poisson-distributed) creates consecutive extreme observations
3. **No return autocorrelation**: the coin flip inside the duration loop alternates between crash and boom independently each step, preventing directional runs

### Parameters

| Parameter | Symbol | Role |
|-----------|--------|------|
| Jump probability | `epsilon` | Probability of entering jump mode at any step |
| Jump duration rate | `lambda` | Mean number of steps spent in jump mode (Poisson parameter) |
| Crash/boom bias | 52/48 | Slight asymmetry reflecting empirical crash-boom frequency |
| Tail depth | 3 states | Number of states at each extreme used as teleportation targets |

### Regime Teleportation vs. Inverse Transition

The discrete JumpHMM.jl uses an **inverse transition matrix** during jumps (favoring unlikely transitions). The continuous model uses **regime teleportation** -- directly sampling from tail states -- because:
- Continuous state ordering (by emission mean) makes "tail" semantically clear
- Teleportation is simpler and produces cleaner volatility clustering
- The inverse-transition approach was designed for discrete bins and does not generalize naturally

## Bayesian Extensions

For scenarios where Gaussian emissions are insufficient, the framework supports Bayesian parameter estimation via Turing.jl:

- **Student's t-distribution**: three parameters (location mu, scale sigma, degrees of freedom nu) with weakly informative priors. Captures heavy tails more naturally than Gaussian.
- **Laplace distribution**: two parameters (location mu, scale b). Models peaked, leptokurtic returns.

These are fit per-regime using NUTS sampling and can be used as alternative emission models in future extensions.

## Validation Strategy

Model quality is assessed on multiple axes:

1. **Convergence diagnostics**: log-likelihood history should be monotonically increasing and plateau
2. **Distributional match**: PDF/CDF comparison of simulated vs. observed returns
3. **Autocorrelation of returns**: should show no significant lag-1 autocorrelation (efficient markets)
4. **Autocorrelation of |returns|**: should show slow decay (volatility clustering)
5. **Statistical tests**: Kolmogorov-Smirnov test for distributional fit
6. **Out-of-sample**: models trained on 2014-2024 data validated against 2025 holdout

## Future Directions

- Multi-asset portfolio simulation with dependence modeling (copulas, vine copulas)
- Automatic state-number selection (BIC/AIC criteria)
- Student-t emissions integrated directly into the Baum-Welch loop
- Parameter sensitivity analysis for jump parameters (epsilon, lambda)
- Formal package registration and CI/CD pipeline
