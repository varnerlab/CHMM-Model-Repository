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


"""
    mutable struct MyStudentTHiddenMarkovModel <: AbstractMarkovModel

Continuous HMM with per-state Student-t emissions. Each state k has a
location μ_k, a scale σ_k, and degrees-of-freedom ν_k learned by EM (ECM).
Heavier tails than `MyContinuousHiddenMarkovModel` at the cost of one
extra parameter per state.

### Required fields
- `states::Array{Int64,1}`: state labels
- `transition::Dict{Int64, Categorical}`: transition distributions per state
- `emission::Dict{Int64, LocationScale}`: scaled-and-shifted TDist per state (μ + σ * TDist(ν))
- `log_likelihood_history::Array{Float64,1}`: EM log-likelihood trace
"""
mutable struct MyStudentTHiddenMarkovModel <: AbstractMarkovModel

    states::Array{Int64,1}
    transition::Dict{Int64, Categorical}
    emission::Dict{Int64, LocationScale{Float64, Continuous, TDist{Float64}}}
    log_likelihood_history::Array{Float64,1}

    MyStudentTHiddenMarkovModel() = new();
end


"""
    mutable struct MyLaplaceHiddenMarkovModel <: AbstractMarkovModel

Continuous HMM with per-state Laplace emissions. Each state k has a
location μ_k (weighted median) and scale b_k (weighted MAD). M-step is
closed-form once the forward-backward posteriors are available.

### Required fields
- `states::Array{Int64,1}`: state labels
- `transition::Dict{Int64, Categorical}`: transition distributions per state
- `emission::Dict{Int64, Laplace}`: Laplace emissions per state
- `log_likelihood_history::Array{Float64,1}`: EM log-likelihood trace
"""
mutable struct MyLaplaceHiddenMarkovModel <: AbstractMarkovModel

    states::Array{Int64,1}
    transition::Dict{Int64, Categorical}
    emission::Dict{Int64, Laplace}
    log_likelihood_history::Array{Float64,1}

    MyLaplaceHiddenMarkovModel() = new();
end


"""
    mutable struct MySemiMarkovContinuousHMM <: AbstractMarkovModel

Semi-Markov continuous HMM with state-dependent AR(1) emissions (Track C1,
ported from `CHMM-Vol-Model`). Each state k carries an AR(1) conditional
mean plus a residual drawn from the model's chosen family, and an explicit
sojourn-duration distribution (negative-binomial or truncated discrete
Pareto). Between-state transitions have zero diagonal per Yu (2010): the
diagonal is absorbed into the sojourn distribution.

For equity returns the AR(1) φ_k is typically near zero (returns are white
noise) but the per-state sojourn distribution is load-bearing: heavy-tailed
(Pareto) sojourns on the crisis states reproduce clustered-crash
morphology that flat CHMM undershoots.

### Required fields
- `states::Array{Int64,1}`: state labels 1..K
- `transition::Dict{Int64, Categorical}`: zero-diagonal between-state transition
- `emission_family::Symbol`: `:gaussian`, `:student_t`, or `:laplace`
- `emission_mu::Dict{Int64, Float64}`: per-state AR(1) long-run mean
- `emission_phi::Dict{Int64, Float64}`: per-state AR(1) coefficient
- `emission_sigma::Dict{Int64, Float64}`: per-state scale (σ for N/t, b for Laplace)
- `emission_nu::Dict{Int64, Float64}`: per-state Student-t df (Inf for N/L)
- `sojourn::Dict{Int64, DiscreteUnivariateDistribution}`: per-state sojourn pmf
- `sojourn_family::Dict{Int64, Symbol}`: `:nb`, `:pareto`, or `:geometric`
- `log_likelihood_history::Array{Float64,1}`: plug-in init + refit LL
"""
mutable struct MySemiMarkovContinuousHMM <: AbstractMarkovModel

    states::Array{Int64,1}
    transition::Dict{Int64, Categorical}
    emission_family::Symbol
    emission_mu::Dict{Int64, Float64}
    emission_phi::Dict{Int64, Float64}
    emission_sigma::Dict{Int64, Float64}
    emission_nu::Dict{Int64, Float64}
    sojourn::Dict{Int64, DiscreteUnivariateDistribution}
    sojourn_family::Dict{Int64, Symbol}
    log_likelihood_history::Array{Float64,1}

    MySemiMarkovContinuousHMM() = new();
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


# --- GRU GENERATOR (Deep Learning Baseline) ---------------------------------- #

"""
    mutable struct MyGRUGenerator

Recurrent neural-network generator for one-dimensional return series.

A single-hidden-layer GRU encodes a context window of past returns and a
linear head outputs the parameters (μ_t, log σ_t) of a Gaussian
next-step density. Trained by maximum-likelihood (negative log-likelihood)
on the in-sample series; simulation rolls forward by sampling from the
predicted Gaussian and feeding the sample back into the recurrence,
which is the standard auto-regressive RNN-based synthetic-data baseline
used for financial return generation in the prior discrete-HMM paper.

### Fields
- `chain::Any`: trained Flux model (GRU + Dense head); kept as `Any` so
  this file can be loaded without Flux being installed.
- `window::Int64`: context-window length used during training and simulation.
- `μ_x::Float64`, `σ_x::Float64`: pre-fit standardisation moments.
- `loss_history::Array{Float64,1}`: per-epoch training loss.
"""
mutable struct MyGRUGenerator

    chain::Any;
    window::Int64;
    μ_x::Float64;
    σ_x::Float64;
    loss_history::Array{Float64,1};

    MyGRUGenerator() = new();
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