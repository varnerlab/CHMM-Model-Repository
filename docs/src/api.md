# API Reference

This page lists the public types and functions. Detailed docstrings live alongside
the definitions in `src/` (primarily `Types.jl`, `Factory.jl`, and `Compute.jl`).

## Types

### Abstract Types

```julia
abstract type AbstractMarkovModel end
abstract type AbstractDistributionModel end
abstract type AbstractDependenceModel end
```

### Continuous-Emission HMMs (the contribution)

```julia
mutable struct MyContinuousHiddenMarkovModel <: AbstractMarkovModel
```
Continuous HMM trained via Baum-Welch with per-state Gaussian (`Normal`) emissions.

```julia
mutable struct MyStudentTHiddenMarkovModel <: AbstractMarkovModel
```
Per-state Student-t emissions (location μ_k, scale σ_k, degrees-of-freedom ν_k) learned
by ECM. Heavier tails than the Gaussian CHMM at one extra parameter per state.

```julia
mutable struct MyLaplaceHiddenMarkovModel <: AbstractMarkovModel
```
Per-state Laplace emissions (weighted-median location, weighted-MAD scale). Closed-form M-step.

```julia
mutable struct MyGEDHiddenMarkovModel <: AbstractMarkovModel
```
Per-state Generalized Error Distribution emissions with a learned shape `p_k`
(`p_k = 2` → Gaussian, `p_k = 1` → Laplace), letting each regime pick its own kurtosis.

```julia
mutable struct MySemiMarkovContinuousHMM <: AbstractMarkovModel
```
Semi-Markov CHMM with state-dependent AR(1) emissions and explicit sojourn-duration
distributions (negative-binomial, geometric, or truncated discrete Pareto). Between-state
transitions are zero-diagonal (Yu 2010).

### Discrete Models (Baseline)

```julia
mutable struct MyHiddenMarkovModel <: AbstractMarkovModel
```
Discrete HMM with categorical transition and emission distributions.

```julia
mutable struct MyHiddenMarkovModelWithJumps <: AbstractMarkovModel
```
Discrete HMM with a Poisson jump process (regime teleportation). The jump mechanism lives
**only** in this baseline -- the continuous models never teleport.

### GARCH-Family Benchmarks

```julia
mutable struct MyGARCHModel
```
GARCH(1,1): `σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}`. Fitted via MLE (Nelder-Mead).

```julia
mutable struct MyMSGARCHModel
```
Markov-Switching GARCH(1,1) (Haas-Mittnik-Paolella 2004) with K regimes, fitted via the
Hamilton (1989) filter. The asymmetric/alternative variants in `GARCHFamily.jl`
(EGARCH, GJR-GARCH, GARCH-t, HAR-RV) return lightweight fit objects rather than `My...` structs.

### Neural Baseline

```julia
mutable struct MyGRUGenerator
```
Single-hidden-layer GRU + Gaussian head, trained by maximum likelihood. Auto-regressive
RNN synthetic-data baseline (`chain` field is `Any` so the file loads without Flux).

### Distribution Models (Bayesian tags)

```julia
struct StudentTModel <: AbstractDistributionModel end
struct LaplaceModel  <: AbstractDistributionModel end
```
Dispatch tags for Bayesian per-regime distribution fitting via Turing.jl
(see `learn_distribution_mcmc`).

### Cross-Asset Dependence Models

```julia
mutable struct MySingleIndexModel        <: AbstractDependenceModel  # Sharpe (1963) factor model
mutable struct MyGaussianCopulaModel     <: AbstractDependenceModel  # Gaussian copula
mutable struct MyStudentTCopulaModel     <: AbstractDependenceModel  # Student-t copula (tail dependence)
mutable struct MyTruncatedCVineCopulaModel <: AbstractDependenceModel  # truncated level-1 C-vine
```
Multi-asset generators that propagate univariate CHMM marginals to a full asset universe.
See `CrossAsset.jl`.

## Model Construction

All single-asset and dependence models share the `build()` factory (dispatch on type,
data passed as a `NamedTuple`):

```julia
build(::Type{MyHiddenMarkovModel}, data::NamedTuple)
build(::Type{MyHiddenMarkovModelWithJumps}, data::NamedTuple)
build(::Type{MyContinuousHiddenMarkovModel}, data::NamedTuple)
build(::Type{MyStudentTHiddenMarkovModel}, data::NamedTuple)
build(::Type{MyLaplaceHiddenMarkovModel}, data::NamedTuple)
build(::Type{MyGEDHiddenMarkovModel}, data::NamedTuple)
build(::Type{MyGARCHModel}, data::NamedTuple)
build(::Type{MyGRUGenerator}, data::NamedTuple)

# Cross-asset (CrossAsset.jl)
build(::Type{MySingleIndexModel}, data::NamedTuple)
build(::Type{MyGaussianCopulaModel}, data::NamedTuple)
build(::Type{MyStudentTCopulaModel}, data::NamedTuple)
build(::Type{MyTruncatedCVineCopulaModel}, data::NamedTuple)
```

The semi-Markov and MS-GARCH models use dedicated fitters rather than `build`:

```julia
fit_sm_chmm(observations::Vector{Float64}, K::Int, family::Symbol; ...) -> MySemiMarkovContinuousHMM
fit_msgarch_k2(obs::Vector{Float64}; max_iter=2000) -> MyMSGARCHModel
fit_msgarch_k3(obs::Vector{Float64}; max_iter=3000) -> MyMSGARCHModel
```

## Fitting Algorithms

```julia
# Continuous-HMM EM/ECM training (called by the corresponding build methods)
baum_welch(observations::Vector{Float64}, n_states::Int; max_iter=30, tol=1e-4)
baum_welch_student_t(observations::Vector{Float64}, n_states::Int; max_iter=30, ...)
baum_welch_laplace(observations::Vector{Float64}, n_states::Int; max_iter=30, ...)
baum_welch_ged(observations::Vector{Float64}, n_states::Int; max_iter=30, ...)

# Bayesian per-regime distribution fitting (Turing.jl NUTS)
learn_distribution_mcmc(model::AbstractDistributionModel, returns::Vector{Float64}; samples=2000)

# Alternative GARCH-family fits (GARCHFamily.jl)
fit_egarch11(obs); fit_gjr11(obs); fit_garcht11(obs); fit_harrv(obs)

# Stochastic-volatility / multifractal / jump-diffusion baselines (SVMSMBaselines.jl)
fit_sv_ar1(obs); fit_msm(obs; kbar=8); fit_jump_diffusion(obs; dt=1/252, n_terms=20)
```

## Simulation

```julia
# HMM models are callable functors returning a state chain (Vector{Int64})
(m::MyContinuousHiddenMarkovModel)(start::Int64, steps::Int64) -> Vector{Int64}
(m::MyHiddenMarkovModel)(start::Int64, steps::Int64)           -> Vector{Int64}
(m::MyHiddenMarkovModelWithJumps)(start::Int64, steps::Int64)  -> Vector{Int64}
# (the Student-t / Laplace / GED CHMMs share the same functor interface)

# Direct return-path simulators
simulate_garch(model::MyGARCHModel, n_steps::Int64)        -> Vector{Float64}
simulate_msgarch(model::MyMSGARCHModel, n_steps::Int)      -> Vector{Float64}
simulate_sm_chmm(model::MySemiMarkovContinuousHMM, n_steps::Int; ...)
simulate_egarch(fit, n_steps); simulate_gjr(fit, n_steps)
simulate_garcht(fit, n_steps); simulate_harrv(fit, n_steps)
simulate_sv_ar1(params, T; n_paths=1000)
simulate_msm(params, T; n_paths=1000)
simulate_jump_diffusion(params, T; n_paths=1000)

# Multi-asset simulation (returns a T × d × n_paths array)
simulate(model::MySingleIndexModel, market_paths::Matrix{Float64})
simulate(model::MyGaussianCopulaModel, T_sim::Int, n_paths::Int)
simulate(model::MyStudentTCopulaModel, T_sim::Int, n_paths::Int)
simulate(model::MyTruncatedCVineCopulaModel, T_sim::Int, n_paths::Int)
```

## Other Algorithms

```julia
viterbi(observations::Vector{Float64}, model) -> Vector{Int64}   # most-likely state path
walk_forward_regimes(observations, window_size, n_states; max_iter=30) -> Vector{Int64}
```

`viterbi` accepts any of the continuous-emission CHMMs (Gaussian, Student-t, Laplace).

## Validation Metrics (`Metrics.jl`)

```julia
mmd2_rbf(X, Y; γ=nothing)                # RBF maximum mean discrepancy
sig_mmd2(X_windows, Y_windows; depth=3)  # path-signature MMD
discriminator_auc(R_obs, sim_archive)    # learned-discriminator AUC
leverage_effect(r; max_lag=20)
aggregational_kurtosis(r; horizons=[1,5,10,21])
kupiec_lr(breaches, α); christoffersen_lr(breaches); christoffersen_cc(breaches, α)  # VaR backtests
correlation_reproduction(R_obs, sim); per_asset_ks_pass_rates(R_obs, sim)            # cross-asset
```

## Finance

```julia
log_growth_matrix(dataset, firms; Δt, risk_free_rate, keycol) -> Matrix/Vector
vwap(df::DataFrame) -> Vector{Float64}
```

## Data Loading

```julia
MyPortfolioDataSet()              # Training set: 2014-2024
MyOutOfSamplePortfolioDataSet()   # Out-of-sample set: 2025
MyOriginalPortfolioDataSet()      # Original full set: 2014-2024
```

## Source Modules

Loaded in order by `Include.jl`: `Types.jl`, `Files.jl`, `Factory.jl`, `Compute.jl`,
`CrossAsset.jl`, `Visualize.jl`, `Metrics.jl`, `SemiMarkov.jl`, `MSGARCH.jl`,
`GARCHFamily.jl`, `SVMSMBaselines.jl`. The R-backed reference baseline
`MSGARCHReference.jl` is **not** auto-loaded (it requires RCall/R); load it from the
runner that needs it.
