# --- ABSTRACT TYPES ---------------------------------------------------------- #
abstract type AbstractMarkovModel end
abstract type AbstractDistributionModel end
# ----------------------------------------------------------------------------- #


# --- CONTINUOUS MODELS ------------------------------------------------------- #

"""
    mutable struct MyContinuousHiddenMarkovModel <: AbstractMarkovModel

The `MyContinuousHiddenMarkovModel` mutable struct represents a hidden Markov model (HMM) with continuous states and Gaussian emissions.

### Required fields
- `states::Array{Int64,1}`: The states of the model
- `transition::Dict{Int64, Categorical}`: The transition matrix of the model encoded as a dictionary where the `key` is the state and the `value` is a `Categorical` distribution
- `emission::Dict{Int64, Normal}`: The emission matrix of the model encoded as a dictionary where the `key` is the state and the `value` is a `Normal` distribution
- `log_likelihood_history::Array{Float64,1}`: The log likelihood history of the model
### Constructor
- `MyContinuousHiddenMarkovModel()`: Creates a new instance of the `MyContinuousHiddenMarkovModel` struct.
"""
mutable struct MyContinuousHiddenMarkovModel <: AbstractMarkovModel
    
    # data
    states::Array{Int64,1}
    transition::Dict{Int64, Categorical}
    # Emission here is a Normal distribution, not Categorical
    emission::Dict{Int64, Normal} 
    
    # We can store the EM history if we want
    log_likelihood_history::Array{Float64,1}

    # constructor
    MyContinuousHiddenMarkovModel() = new();
end


# --- DISTRIBUTION MODELS (Bayesian Inference) -------------------------------- #

"""
    struct StudentTModel <: AbstractDistributionModel

Dispatch tag for Bayesian Student's t-distribution fitting via Turing.jl.
Used with `build_turing_model` and `learn_distribution_mcmc`.
"""
struct StudentTModel <: AbstractDistributionModel end

"""
    struct LaplaceModel <: AbstractDistributionModel

Dispatch tag for Bayesian Laplace distribution fitting via Turing.jl.
Used with `build_turing_model` and `learn_distribution_mcmc`.
"""
struct LaplaceModel <: AbstractDistributionModel end
# ----------------------------------------------------------------------------- #


# --- PRICING TYPES ----------------------------------------------------------- #
abstract type AbstractPricingModel end

"""
    mutable struct MyEuropeanOptionContract

Represents a European option contract (call or put).

### Required fields
- `S0::Float64`: Current spot price of the underlying
- `K::Float64`: Strike price
- `T::Float64`: Time to expiration in years
- `r::Float64`: Risk-free rate (annualized, continuously compounded)
- `is_call::Bool`: true for call, false for put
"""
mutable struct MyEuropeanOptionContract

    S0::Float64;
    K::Float64;
    T::Float64;
    r::Float64;
    is_call::Bool;

    MyEuropeanOptionContract() = new();
end

"""
    mutable struct MyCHMMPricingModel <: AbstractPricingModel

Monte Carlo option pricer using regime-switching volatility from a trained CHMM.
The volatility_map bridges VIX regime levels to equity volatility: σ_s = median(VIX_level_s) / 100.

### Required fields
- `hmm::AbstractMarkovModel`: Trained CHMM
- `volatility_map::Dict{Int64, Float64}`: state → equity vol σ_s
- `start_distribution::Categorical`: Stationary distribution for initial state sampling
- `n_paths::Int64`: Number of Monte Carlo paths
- `n_steps_per_year::Int64`: Time discretization (252 for daily)
"""
mutable struct MyCHMMPricingModel <: AbstractPricingModel

    hmm::AbstractMarkovModel;
    volatility_map::Dict{Int64, Float64};
    start_distribution::Categorical;
    n_paths::Int64;
    n_steps_per_year::Int64;

    MyCHMMPricingModel() = new();
end

"""
    mutable struct MyHestonPricingModel <: AbstractPricingModel

Monte Carlo option pricer using the Heston stochastic volatility model.
dS = rS dt + √v S dW₁
dv = κ(θ − v)dt + ξ√v dW₂
corr(dW₁, dW₂) = ρ

### Required fields
- `v0::Float64`: Initial variance
- `kappa::Float64`: Mean reversion speed
- `theta::Float64`: Long-run variance
- `xi::Float64`: Vol-of-vol
- `rho::Float64`: Correlation between price and variance Brownians
- `n_paths::Int64`: Number of Monte Carlo paths
- `n_steps_per_year::Int64`: Time discretization
"""
mutable struct MyHestonPricingModel <: AbstractPricingModel

    v0::Float64;
    kappa::Float64;
    theta::Float64;
    xi::Float64;
    rho::Float64;
    n_paths::Int64;
    n_steps_per_year::Int64;

    MyHestonPricingModel() = new();
end

"""
    struct MyPricingResult

Immutable container for Monte Carlo pricing output.

### Fields
- `price::Float64`: Mean discounted payoff (the option price)
- `std_error::Float64`: Standard error of the Monte Carlo estimate
- `n_paths::Int64`: Number of paths used
- `payoffs::Array{Float64,1}`: Individual discounted path payoffs (for diagnostics)
"""
struct MyPricingResult
    price::Float64;
    std_error::Float64;
    n_paths::Int64;
    payoffs::Array{Float64,1};
end
# ----------------------------------------------------------------------------- #