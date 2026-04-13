# API Reference

## Types

### Abstract Types

```julia
abstract type AbstractMarkovModel end
abstract type AbstractDistributionModel end
abstract type AbstractPricingModel end
```

### Continuous HMM

```julia
mutable struct MyContinuousHiddenMarkovModel <: AbstractMarkovModel
```
Continuous Gaussian HMM trained via Baum-Welch. Emissions are `Normal` distributions per state. Stores `log_likelihood_history` for convergence diagnostics.

### Distribution Models

```julia
struct StudentTModel <: AbstractDistributionModel end
struct LaplaceModel <: AbstractDistributionModel end
```
Dispatch tags for Bayesian distribution fitting via Turing.jl.

### Pricing Models

```julia
mutable struct MyEuropeanOptionContract
mutable struct MyCHMMPricingModel <: AbstractPricingModel
mutable struct MyHestonPricingModel <: AbstractPricingModel
struct MyPricingResult
```

## Model Construction

```julia
build(::Type{MyContinuousHiddenMarkovModel}, data::NamedTuple) -> MyContinuousHiddenMarkovModel
build(::Type{MyEuropeanOptionContract}, data::NamedTuple) -> MyEuropeanOptionContract
build(::Type{MyCHMMPricingModel}, data::NamedTuple) -> MyCHMMPricingModel
build(::Type{MyHestonPricingModel}, data::NamedTuple) -> MyHestonPricingModel
```

Factory methods that construct model instances. See [Fitting](@ref) for details on each variant's required `NamedTuple` keys.

## Simulation

```julia
(m::MyContinuousHiddenMarkovModel)(start::Int64, steps::Int64) -> Vector{Int64}
```

Functor interface for path simulation. Returns a vector of hidden state indices.

## Algorithms

```julia
baum_welch(observations::Vector{Float64}, n_states::Int; max_iter=30, tol=1e-4)
    -> (T, mu, sigma, pi, ll_history, gamma)
```

Baum-Welch (EM) algorithm for continuous Gaussian HMM parameter estimation. Returns the transition matrix, emission means, emission standard deviations, initial state distribution, log-likelihood history, and posterior state probabilities.

```julia
viterbi(observations::Vector{Float64}, model::MyContinuousHiddenMarkovModel) -> Vector{Int64}
```

Viterbi decoding of the most likely hidden state sequence.

## Pricing

```julia
black_scholes(contract::MyEuropeanOptionContract, sigma::Float64) -> Float64
price(model::MyCHMMPricingModel, contract::MyEuropeanOptionContract) -> MyPricingResult
price(model::MyHestonPricingModel, contract::MyEuropeanOptionContract) -> MyPricingResult
implied_volatility(contract::MyEuropeanOptionContract, market_price::Float64) -> Float64
implied_vol_surface(model::AbstractPricingModel, S0, r, strikes, expiries) -> Matrix{Float64}
```

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
plot_regime_overlay(dates, prices, states, ticker; title_text="") -> Plots.Plot
plot_emission_pdfs(model::MyContinuousHiddenMarkovModel, ticker; xlabel="Log Return") -> Plots.Plot
plot_implied_vol_surface(strikes, expiries, iv_matrix, title) -> Plots.Plot
plot_pricing_comparison(chmm_result, heston_result, bs_price, title) -> Plots.Plot
plot_mc_convergence(result::MyPricingResult, title) -> Plots.Plot
```

## Data Loading

```julia
MyPortfolioDataSet() -> Dict{String,Any}                # Training: 2014-2024
MyOutOfSamplePortfolioDataSet() -> Dict{String,Any}     # Test: 2025
MyOriginalPortfolioDataSet() -> Dict{String,Any}        # Original dataset
MyVolatilityDataSet() -> Dict{String,Any}               # VIX Training
MyOutOfSampleVolatilityDataSet() -> Dict{String,Any}    # VIX Test
```
