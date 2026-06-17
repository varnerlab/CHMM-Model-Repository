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

## Heavier-Tailed Emission Families

The Gaussian CHMM has a sibling for each heavier-tailed emission family. All share the
same `build` interface and functor; only the per-state emission distribution changes, and
each is fitted by an ECM variant of Baum-Welch:

```julia
# Student-t emissions: per-state location, scale, and degrees-of-freedom ν_k
build(MyStudentTHiddenMarkovModel, (observations=R, number_of_states=K, max_iter=60))

# Laplace emissions: closed-form weighted-median / weighted-MAD M-step
build(MyLaplaceHiddenMarkovModel, (observations=R, number_of_states=K, max_iter=60))

# GED emissions: per-state shape p_k (p=2 → Gaussian, p=1 → Laplace)
build(MyGEDHiddenMarkovModel, (observations=R, number_of_states=K, max_iter=60))
```

The `MyStudentTHiddenMarkovModel` build accepts optional keys `ν_init`, `ν_bounds`, and
`ν_shrink_rate` (a shrinkage prior pulling ν_k toward the Gaussian limit); see the docstring
in `Factory.jl`. The underlying trainers are `baum_welch_student_t`, `baum_welch_laplace`,
and `baum_welch_ged`.

## Semi-Markov CHMM

For regimes with explicit sojourn-duration control (heavy-tailed dwell times on crisis
states), the semi-Markov CHMM carries state-dependent AR(1) emissions plus a per-state
sojourn distribution and is fitted with a dedicated function rather than `build`:

```julia
model = fit_sm_chmm(R, K, :gaussian)   # family ∈ (:gaussian, :student_t, :laplace)
```

## GARCH-Family and Other Baselines

The comparison baselines are fitted through their own entry points:

```julia
# GARCH(1,1) -- classical single-regime benchmark
garch = build(MyGARCHModel, (observations=R,))

# Markov-Switching GARCH (Haas-Mittnik-Paolella)
ms = fit_msgarch_k2(R)        # or fit_msgarch_k3(R)

# Asymmetric / alternative GARCH variants (GARCHFamily.jl)
fit_egarch11(R); fit_gjr11(R); fit_garcht11(R); fit_harrv(R)

# Stochastic-volatility / multifractal / jump-diffusion (SVMSMBaselines.jl)
fit_sv_ar1(R); fit_msm(R); fit_jump_diffusion(R)

# GRU neural generator
gru = build(MyGRUGenerator, (observations=R,))
```

## Bayesian Distribution Fitting

This is distinct from `MyStudentTHiddenMarkovModel` / `MyLaplaceHiddenMarkovModel` above:
those train a full HMM by ECM, whereas this fits a single distribution to one regime's
returns by MCMC. For per-regime distribution fitting beyond Gaussians:

```julia
# Student's t-distribution
chain = learn_distribution_mcmc(StudentTModel(), regime_returns; samples=2000)

# Laplace distribution
chain = learn_distribution_mcmc(LaplaceModel(), regime_returns; samples=2000)
```

Both use Turing.jl's NUTS sampler and return a `Chain` object with posterior samples for the distribution parameters.
