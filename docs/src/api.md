# API Reference

## Types

### Abstract Types

```julia
abstract type AbstractMarkovModel end
abstract type AbstractDistributionModel end
abstract type AbstractPricingModel end
```

### Discrete Models (Baseline)

```julia
mutable struct MyHiddenMarkovModel <: AbstractMarkovModel
```
Discrete HMM with categorical transition and emission distributions.

```julia
mutable struct MyHiddenMarkovModelWithJumps <: AbstractMarkovModel
```
Discrete HMM with Poisson jump process (regime teleportation). Baseline from the discrete paper.

### Continuous HMM (New Contribution)

```julia
mutable struct MyContinuousHiddenMarkovModel <: AbstractMarkovModel
```
Continuous Gaussian HMM trained via Baum-Welch. Emissions are `Normal` distributions per state.

### GARCH Benchmark

```julia
mutable struct MyGARCHModel
```
GARCH(1,1) model: σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}. Fitted via MLE.

### Distribution Models

```julia
struct StudentTModel <: AbstractDistributionModel end
struct LaplaceModel <: AbstractDistributionModel end
```

### Pricing Models

```julia
mutable struct MyEuropeanOptionContract
mutable struct MyCHMMPricingModel <: AbstractPricingModel
mutable struct MyHestonPricingModel <: AbstractPricingModel
struct MyPricingResult
```

## Model Construction

```julia
build(::Type{MyHiddenMarkovModel}, data::NamedTuple)
build(::Type{MyHiddenMarkovModelWithJumps}, data::NamedTuple)
build(::Type{MyContinuousHiddenMarkovModel}, data::NamedTuple)
build(::Type{MyGARCHModel}, data::NamedTuple)
build(::Type{MyEuropeanOptionContract}, data::NamedTuple)
build(::Type{MyCHMMPricingModel}, data::NamedTuple)
build(::Type{MyHestonPricingModel}, data::NamedTuple)
```

## Simulation

```julia
# HMM models are callable functors
(m::MyContinuousHiddenMarkovModel)(start::Int64, steps::Int64) -> Vector{Int64}
(m::MyHiddenMarkovModel)(start::Int64, steps::Int64) -> Vector{Int64}
(m::MyHiddenMarkovModelWithJumps)(start::Int64, steps::Int64) -> Vector{Int64}

# GARCH simulation
simulate_garch(model::MyGARCHModel, n_steps::Int64) -> Vector{Float64}
```

## Algorithms

```julia
baum_welch(observations::Vector{Float64}, n_states::Int; max_iter=30, tol=1e-4)
viterbi(observations::Vector{Float64}, model::MyContinuousHiddenMarkovModel) -> Vector{Int64}
walk_forward_regimes(observations, window_size, n_states; max_iter=30) -> Vector{Int64}
```

## Pricing

```julia
black_scholes(contract, sigma) -> Float64
price(model::MyCHMMPricingModel, contract) -> MyPricingResult
price(model::MyHestonPricingModel, contract) -> MyPricingResult
implied_volatility(contract, market_price) -> Float64
implied_vol_surface(model, S0, r, strikes, expiries) -> Matrix{Float64}
```

## Finance

```julia
log_growth_matrix(dataset, firms; Δt, risk_free_rate, keycol) -> Matrix/Vector
vwap(df::DataFrame) -> Vector{Float64}
```

## Data Loading

```julia
MyPortfolioDataSet()                # Training: 2014-2024
MyOutOfSamplePortfolioDataSet()     # Test: 2025
MyVolatilityDataSet()               # VIX Training
MyOutOfSampleVolatilityDataSet()    # VIX Test
```
