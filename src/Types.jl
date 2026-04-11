# --- ABSTRACT TYPES ---------------------------------------------------------- #
abstract type AbstractMarkovModel end
abstract type AbstractDistributionModel end
# ----------------------------------------------------------------------------- #


# --- DISCRETE MODELS (Legacy) ----------------------------------------------- #

"""
    mutable struct MyHiddenMarkovModel <: AbstractMarkovModel

The `MyHiddenMarkovModel` mutable struct represents a hidden Markov model (HMM) with discrete states.

### Required fields
- `states::Array{Int64,1}`: The states of the model
- `transition::Dict{Int64, Categorical}`: The transition matrix of the model encoded as a dictionary where the `key` is the state and the `value` is a `Categorical` distribution
- `emission::Dict{Int64, Categorical}`: The emission matrix of the model encoded as a dictionary where the `key` is the state and the `value` is a `Categorical` distribution
### Constructor
- `MyHiddenMarkovModel()`: Creates a new instance of the `MyHiddenMarkovModel` struct.
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

The `MyHiddenMarkovModelWithJumps` mutable struct represents a hidden Markov model (HMM) with discrete states and jump probabilities.

### Required fields
- `states::Array{Int64,1}`: The states of the model
- `transition::Dict{Int64, Categorical}`: The transition matrix of the model encoded as a dictionary where the `key` is the state and the `value` is a `Categorical` distribution
- `inverse_transition::Dict{Int64, Categorical}`: The inverse transition matrix of the model encoded as a dictionary where the `key` is the state and the `value` is a `Categorical` distribution
- `emission::Dict{Int64, Categorical}`: The emission matrix of the model encoded as a dictionary where the `key` is the state and the `value` is a `Categorical` distribution
- `ϵ::Float64`: The jump probability
- `λ::Float64`: The jump distribution parameter
- `jump_distribution::Poisson`: The jump distribution
### Constructor
- `MyHiddenMarkovModelWithJumps()`: Creates a new instance of the `MyHiddenMarkovModelWithJumps` struct.
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


# --- CONTINUOUS MODELS (Active) ---------------------------------------------- #

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
    mutable struct MyContinuousHiddenMarkovModelWithJumps <: AbstractMarkovModel

The `MyContinuousHiddenMarkovModelWithJumps` mutable struct represents a hidden Markov model (HMM) with continuous states, Gaussian emissions and jump probabilities.

### Required fields
- `states::Array{Int64,1}`: The states of the model
- `transition::Dict{Int64, Categorical}`: The transition matrix of the model encoded as a dictionary where the `key` is the state and the `value` is a `Categorical` distribution
- `emission::Dict{Int64, Normal}`: The emission matrix of the model encoded as a dictionary where the `key` is the state and the `value` is a `Normal` distribution
- `ϵ::Float64`: The jump probability
- `λ::Float64`: The jump distribution parameter
- `jump_distribution::Poisson`: The jump distribution
### Constructor
- `MyContinuousHiddenMarkovModelWithJumps()`: Creates a new instance of the `MyContinuousHiddenMarkovModelWithJumps` struct.
"""
mutable struct MyContinuousHiddenMarkovModelWithJumps <: AbstractMarkovModel
    
    # Inherited Data (from Base Model)
    states::Array{Int64,1}
    transition::Dict{Int64, Categorical}
    emission::Dict{Int64, Normal}
    
    # Jump Parameters
    ϵ::Float64   # Probability of a jump event starting
    λ::Float64   # Mean duration of the jump (parameter for Poisson)
    jump_distribution::Poisson

    # Constructor
    MyContinuousHiddenMarkovModelWithJumps() = new();
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