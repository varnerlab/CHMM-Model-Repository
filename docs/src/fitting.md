# Model Fitting

## Continuous HMM (Baum-Welch)

The primary model fitting method uses the Baum-Welch algorithm to learn Gaussian emission parameters and transition probabilities from continuous observations.

### Usage

```julia
model = build(MyContinuousHiddenMarkovModel, (
    observations = returns_vector,
    number_of_states = K,
    max_iter = 60
))
```

### Algorithm Overview

The Baum-Welch algorithm alternates between two steps:

**E-Step (Expectation)**: Compute posterior state probabilities using the forward-backward algorithm in log-space.
- Forward variables `log_alpha[t,k]` propagate evidence forward through time
- Backward variables `log_beta[t,k]` propagate evidence backward
- Posterior `gamma[t,k] = P(state_t = k | observations)` combines both

**M-Step (Maximization)**: Update model parameters to maximize expected log-likelihood.
- **Transition matrix**: re-estimated from expected transition counts (xi)
- **Emission means**: weighted average of observations using gamma
- **Emission variances**: weighted variance of observations using gamma
- **Initial distribution**: set to `gamma[1,:]`

### Initialization

Parameters are initialized via quantile-based partitioning:
1. Sort observations
2. Split into K equal-sized chunks
3. Set each state's emission mean/std to the chunk's sample statistics

This avoids degenerate solutions where all states collapse to the global mean.

### Convergence

The algorithm terminates when:
- Log-likelihood change between iterations falls below `tol = 1e-4`, OR
- Maximum iterations (`max_iter`) is reached

The full log-likelihood history is stored in `model.log_likelihood_history`.

## Discrete HMM (Legacy)

For discrete models, transition and emission matrices are provided directly:

```julia
model = build(MyHiddenMarkovModel, (
    states = collect(1:K),
    T = transition_matrix,  # K x K
    E = emission_matrix     # K x K
))
```

## Bayesian Distribution Fitting

For per-regime distribution fitting beyond Gaussians:

```julia
# Student's t-distribution
chain = learn_distribution_mcmc(StudentTModel(), regime_returns; samples=2000)

# Laplace distribution
chain = learn_distribution_mcmc(LaplaceModel(), regime_returns; samples=2000)
```

Both use Turing.jl's NUTS sampler and return a `Chain` object with posterior samples for the distribution parameters.
