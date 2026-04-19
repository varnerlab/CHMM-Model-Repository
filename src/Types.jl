# --- ABSTRACT TYPES ---------------------------------------------------------- #
abstract type AbstractMarkovModel end
abstract type AbstractDistributionModel end
# ----------------------------------------------------------------------------- #


# --- DISCRETE MODELS (Baseline Comparison) ----------------------------------- #

"""
    mutable struct MyHiddenMarkovModel <: AbstractMarkovModel

Discrete HMM with categorical transition and emission distributions.
Used as the base for the discrete jump model (baseline comparison).

### Required fields
- `states::Array{Int64,1}`: The states of the model
- `transition::Dict{Int64, Categorical}`: Transition distributions per state
- `emission::Dict{Int64, Categorical}`: Emission distributions per state
"""
mutable struct MyHiddenMarkovModel <: AbstractMarkovModel

    # data -
    states::Array{Int64,1}
    transition::Dict{Int64, Categorical}
    emission::Dict{Int64, Categorical}

    # constructor -
    MyHiddenMarkovModel() = new();
end


"""
    mutable struct MyHiddenMarkovModelWithJumps <: AbstractMarkovModel

Discrete HMM augmented with Poisson jump process (regime teleportation).
This is the baseline model from the discrete paper for comparison.

### Required fields
- `states::Array{Int64,1}`: The states of the model
- `transition::Dict{Int64, Categorical}`: Transition distributions per state
- `inverse_transition::Dict{Int64, Categorical}`: Inverse transition (high-low reversed)
- `emission::Dict{Int64, Categorical}`: Emission distributions per state
- `ϵ::Float64`: Jump probability
- `λ::Float64`: Jump duration parameter (Poisson rate)
- `jump_distribution::Poisson`: Jump duration distribution
"""
mutable struct MyHiddenMarkovModelWithJumps <: AbstractMarkovModel

    # data -
    states::Array{Int64,1}
    transition::Dict{Int64, Categorical}
    inverse_transition::Dict{Int64, Categorical}; # high-low probability states reversed
    emission::Dict{Int64, Categorical}
    ϵ::Float64; # jump probability
    λ::Float64; # jump distribution parameter
    jump_distribution::Poisson; # jump distribution

    # constructor -
    MyHiddenMarkovModelWithJumps() = new();
end


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


# --- GARCH MODEL ------------------------------------------------------------- #

"""
    mutable struct MyGARCHModel

GARCH(1,1) model for conditional variance modeling.
σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}

### Required fields
- `ω::Float64`: Constant (intercept), must be > 0
- `α::Float64`: ARCH coefficient (shock impact), must be ≥ 0
- `β::Float64`: GARCH coefficient (persistence), must be ≥ 0
- `μ::Float64`: Mean of the return process
- `σ2_history::Array{Float64,1}`: Fitted conditional variance series
- `log_likelihood::Float64`: Log-likelihood at MLE solution
"""
mutable struct MyGARCHModel

    ω::Float64;
    α::Float64;
    β::Float64;
    μ::Float64;
    σ2_history::Array{Float64,1};
    log_likelihood::Float64;

    MyGARCHModel() = new();
end
# ----------------------------------------------------------------------------- #


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


# --- CROSS-ASSET DEPENDENCE MODELS ------------------------------------------- #
abstract type AbstractDependenceModel end

"""
    mutable struct MySingleIndexModel <: AbstractDependenceModel

Single Index Model (Sharpe 1963). Propagates a univariate market engine path
to a full asset universe via the factor regression

    G_{i,t} = α_i + β_i G_{M,t} + η_{i,t},

where α_i and β_i are OLS estimates on in-sample data and η_{i,t} is resampled
from the empirical residual distribution.

### Fields
- `tickers::Vector{String}`: Asset tickers in column order
- `alphas::Vector{Float64}`: Fitted intercepts
- `betas::Vector{Float64}`: Fitted betas
- `residuals::Matrix{Float64}`: T × d matrix of in-sample residuals
- `r2::Vector{Float64}`: Per-asset R² for the factor regression
"""
mutable struct MySingleIndexModel <: AbstractDependenceModel
    tickers::Vector{String};
    alphas::Vector{Float64};
    betas::Vector{Float64};
    residuals::Matrix{Float64};
    r2::Vector{Float64};
    MySingleIndexModel() = new();
end

"""
    mutable struct MyGaussianCopulaModel <: AbstractDependenceModel

Gaussian copula fitted to pseudo-uniform PIT-transformed observations.
Dependence structure is captured by a correlation matrix Σ estimated from
Kendall's τ via ρ = sin(πτ/2).

### Fields
- `tickers::Vector{String}`: Asset tickers
- `Sigma::Matrix{Float64}`: d × d correlation matrix
- `marginals::Vector{AbstractMarkovModel}`: Per-asset CHMM marginals
"""
mutable struct MyGaussianCopulaModel <: AbstractDependenceModel
    tickers::Vector{String};
    Sigma::Matrix{Float64};
    marginals::Vector{AbstractMarkovModel};
    MyGaussianCopulaModel() = new();
end

"""
    mutable struct MyStudentTCopulaModel <: AbstractDependenceModel

Student-t copula with correlation matrix Σ and ν degrees of freedom.
ν is selected by profile maximum-likelihood over a discrete grid.
Unlike the Gaussian copula, the Student-t copula admits non-zero symmetric
tail dependence λ_U = λ_L > 0.

### Fields
- `tickers::Vector{String}`: Asset tickers
- `Sigma::Matrix{Float64}`: d × d correlation matrix
- `nu::Float64`: Degrees of freedom
- `marginals::Vector{AbstractMarkovModel}`: Per-asset CHMM marginals
"""
mutable struct MyStudentTCopulaModel <: AbstractDependenceModel
    tickers::Vector{String};
    Sigma::Matrix{Float64};
    nu::Float64;
    marginals::Vector{AbstractMarkovModel};
    MyStudentTCopulaModel() = new();
end
# ----------------------------------------------------------------------------- #