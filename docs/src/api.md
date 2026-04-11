# API Reference

## Types

### Abstract Types

```julia
abstract type AbstractMarkovModel end
abstract type AbstractDistributionModel end
```

### Discrete Models (Legacy)

```julia
mutable struct MyHiddenMarkovModel <: AbstractMarkovModel
```
Basic discrete HMM with categorical transition and emission distributions.

```julia
mutable struct MyHiddenMarkovModelWithJumps <: AbstractMarkovModel
```
Discrete HMM augmented with jump probability `epsilon`, Poisson-distributed jump duration `lambda`, and an inverse transition matrix for jump-state selection.

### Continuous Models

```julia
mutable struct MyContinuousHiddenMarkovModel <: AbstractMarkovModel
```
Continuous Gaussian HMM trained via Baum-Welch. Emissions are `Normal` distributions per state. Stores `log_likelihood_history` for convergence diagnostics.

```julia
mutable struct MyContinuousHiddenMarkovModelWithJumps <: AbstractMarkovModel
```
Continuous Gaussian HMM with Poisson jump process. Wraps a trained base model with jump parameters (`epsilon`, `lambda`, `jump_distribution`).

### Distribution Models

```julia
struct StudentTModel <: AbstractDistributionModel end
struct LaplaceModel <: AbstractDistributionModel end
```
Dispatch tags for Bayesian distribution fitting via Turing.jl.

## Model Construction

```julia
build(::Type{MyHiddenMarkovModel}, data::NamedTuple) -> MyHiddenMarkovModel
build(::Type{MyHiddenMarkovModelWithJumps}, data::NamedTuple) -> MyHiddenMarkovModelWithJumps
build(::Type{MyContinuousHiddenMarkovModel}, data::NamedTuple) -> MyContinuousHiddenMarkovModel
build(::Type{MyContinuousHiddenMarkovModelWithJumps}, data::NamedTuple) -> MyContinuousHiddenMarkovModelWithJumps
```

Factory methods that construct model instances. See [Fitting](@ref) for details on each variant's required `NamedTuple` keys.

## Simulation

```julia
(m::MyContinuousHiddenMarkovModel)(start::Int64, steps::Int64) -> Vector{Int64}
(m::MyContinuousHiddenMarkovModelWithJumps)(start::Int64, steps::Int64) -> Vector{Int64}
(m::MyHiddenMarkovModel)(start::Int64, steps::Int64) -> Vector{Int64}
(m::MyHiddenMarkovModelWithJumps)(start::Int64, steps::Int64) -> Vector{Int64}
```

Functor interface for path simulation. Returns a vector of hidden state indices.

## Algorithms

```julia
baum_welch(observations::Vector{Float64}, n_states::Int; max_iter=30, tol=1e-4)
    -> (T, mu, sigma, pi, ll_history, gamma)
```

Baum-Welch (EM) algorithm for continuous Gaussian HMM parameter estimation. Returns the transition matrix, emission means, emission standard deviations, initial state distribution, log-likelihood history, and posterior state probabilities.

## Finance

```julia
log_growth_matrix(dataset::Dict{String,DataFrame}, firms::Vector{String}; ...) -> Matrix{Float64}
log_growth_matrix(dataset::Dict{String,DataFrame}, firm::String; ...) -> Vector{Float64}
log_growth_matrix(dataset::DataFrame; ...) -> Vector{Float64}
log_growth_matrix(dataset::Vector{Float64}; ...) -> Vector{Float64}
```

Compute annualized excess log returns. Keyword arguments: `Dt` (time step, default `1/252`), `risk_free_rate` (default `0.0`), `keycol` (price column, default `:volume_weighted_average_price`).

```julia
vwap(df::DataFrame) -> Vector{Float64}
```

Cumulative Volume Weighted Average Price.

## Bayesian Inference

```julia
learn_distribution_mcmc(model_type::AbstractDistributionModel, returns::Vector{Float64}; samples=2000) -> Chain
```

MCMC parameter estimation using NUTS. Dispatches to `build_turing_model` for the specified distribution type.

```julia
build_turing_model(::StudentTModel, data) -> Turing.Model
build_turing_model(::LaplaceModel, data) -> Turing.Model
```

Construct Turing probabilistic models with weakly informative priors.

## Visualization

```julia
plot_acf_comparison(observed::Vector, simulated::Vector, title::String, idx::Int;
    is_absolute=false, L=252) -> Plots.Plot
```

Autocorrelation comparison plot with 99% confidence bands. Set `is_absolute=true` to plot ACF of absolute values (volatility clustering).

## Data Loading

```julia
MyPortfolioDataSet() -> Dict{String,Any}           # Training: 2014-2024
MyOutOfSamplePortfolioDataSet() -> Dict{String,Any} # Test: 2025
MyOriginalPortfolioDataSet() -> Dict{String,Any}    # Original dataset
```
