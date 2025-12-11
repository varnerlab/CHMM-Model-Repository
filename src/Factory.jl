"""
   function build(model::Type{M}, data::NamedTuple) -> AbstractMarkovModel where {M <: AbstractMarkovModel}

This `build` method constructs a concrete instance of type `M` where `M` is a subtype of `AbstractMarkovModel` type using the data in a [NamedTuple](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple).

### Arguments
- `model::Type{M}`: The type of model to build. This type must be a subtype of `AbstractMarkovModel`.
- `data::NamedTuple`: The data to use to build the model.

The `data::NamedTuple` argument must contain the following `keys`:
- `states::Array{Int64,1}`: The states of the model.
- `T::Array{Float64,2}`: The transition matrix of the model.
- `E::Array{Float64,2}`: The emission matrix of the model.
"""
function build(model::Type{MyHiddenMarkovModel}, data::NamedTuple)::MyHiddenMarkovModel
    
    # initialize -
    m = model(); # build an empty model, add data to it below
    transition = Dict{Int64, Categorical}();
    emission = Dict{Int64, Categorical}();

    # get stuff from the data NamedTuple -
    states = data.states;
    T = data.T; # this is the transition matrix
    E = data.E; # this is the emission matrix

    # build the transition and emission distributions -
    for s ∈ states
        transition[s] = Categorical(T[s,:]);
        emission[s] = Categorical(E[s,:]);
    end

    # add data to the model -
    m.transition = transition;
    m.emission = emission;
    m.states = states;

    # return -
    return m;
end


"""
   function build(model::Type{M}, data::NamedTuple) -> AbstractMarkovModel where {M <: AbstractMarkovModel}

This `build` method constructs a concrete instance of type `M` where `M` is a subtype of `AbstractMarkovModel` type using the data in a [NamedTuple](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple).

### Arguments
- `model::Type{M}`: The type of model to build. This type must be a subtype of `AbstractMarkovModel`.
- `data::NamedTuple`: The data to use to build the model.

The `data::NamedTuple` argument must contain the following `keys`:
- `states::Array{Int64,1}`: The states of the model.
- `T::Array{Float64,2}`: The transition matrix of the model.
- `E::Array{Float64,2}`: The emission matrix of the model.
"""
function build(model::Type{MyHiddenMarkovModelWithJumps}, data::NamedTuple)::MyHiddenMarkovModelWithJumps
    
    # initialize -
    m = model(); # build an empty model, add data to it below
    transition = Dict{Int64, Categorical}();
    inverse_transition = Dict{Int64, Categorical}();
    emission = Dict{Int64, Categorical}();
    ϵ = data.ϵ;
    λ = data.λ;

    # get stuff from the data NamedTuple -
    states = data.states;
    T = data.T; # this is the transition matrix
    E = data.E; # this is the emission matrix

    # build the transition and emission distributions -
    for s ∈ states
        transition[s] = Categorical(T[s,:]);
        emission[s] = Categorical(E[s,:]);
    end

    # build the inverse transition matrix -
    for s ∈ states
        F = sum(1 .- T[s,:]);
        d = (1/F)*(1 .- T[s,:]);
        inverse_transition[s] = Categorical(d);
    end

    # add data to the model -
    m.transition = transition;
    m.inverse_transition = inverse_transition;
    m.emission = emission;
    m.states = states;
    m.ϵ = ϵ;
    m.λ = λ;
    m.jump_distribution = Poisson(λ); # jump distribution

    # return -
    return m;
end


"""
    build(model::Type{MyContinuousHiddenMarkovModel}, data::NamedTuple) -> MyContinuousHiddenMarkovModel

This `build` method constructs and trains a `MyContinuousHiddenMarkovModel` instance using the Baum-Welch algorithm. The model's emission probabilities are modeled by Normal distributions.

### Arguments
- `model::Type{MyContinuousHiddenMarkovModel}`: The type of model to build.
- `data::NamedTuple`: The data for training the model.

The `data` NamedTuple must contain the following keys:
- `observations::Vector{Float64}`: A vector of floating-point observations.
- `number_of_states::Int`: The number of hidden states in the model.

### Returns
- A fully trained `MyContinuousHiddenMarkovModel` instance with transition and emission distributions learned from the data.
"""
function build(model::Type{MyContinuousHiddenMarkovModel}, data::NamedTuple)::MyContinuousHiddenMarkovModel
    
    # Extract training data and model configuration from the data NamedTuple
    obs = data.observations
    n_states = data.number_of_states
    
    # Run the Baum-Welch algorithm to estimate model parameters
    # This is the core training step, yielding the transition matrix, and emission distribution parameters (μ, σ)
    T_matrix, μ_vec, σ_vec, π_vec, ll_hist, γ = baum_welch(obs, n_states)
    
    # Initialize an empty model instance
    m = model()

    # Populate the model with the learned parameters
    m.states = collect(1:n_states)
    m.log_likelihood_history = ll_hist
    
    # Create dictionaries to hold the transition and emission distributions for each state
    transition = Dict{Int64, Categorical}()
    emission = Dict{Int64, Normal}()
    
    # For each state, create a Categorical distribution for transitions and a Normal distribution for emissions
    for s in 1:n_states
        transition[s] = Categorical(T_matrix[s, :])
        emission[s] = Normal(μ_vec[s], σ_vec[s])
    end
    
    # Assign the distribution dictionaries to the model
    m.transition = transition
    m.emission = emission
    
    # Return the trained model
    return m
end


"""
    build(model::Type{MyContinuousHiddenMarkovModelWithJumps}, data::NamedTuple) -> MyContinuousHiddenMarkovModelWithJumps

Builds a `MyContinuousHiddenMarkovModelWithJumps` instance by wrapping a pre-trained continuous HMM with jump parameters.

### Arguments
- `model::Type{MyContinuousHiddenMarkovModelWithJumps}`: The type of model to build.
- `data::NamedTuple`: The data required to build the jump model.

The `data` NamedTuple must contain the following keys:
- `base_model::MyContinuousHiddenMarkovModel`: A trained continuous HMM.
- `epsilon::Float64`: The probability of a jump occurring.
- `lambda::Float64`: The rate parameter for the Poisson distribution that models the number of jumps.

### Returns
- A `MyContinuousHiddenMarkovModelWithJumps` instance that incorporates the behavior of the base model with an added jump process.
"""
function build(model::Type{MyContinuousHiddenMarkovModelWithJumps}, data::NamedTuple)::MyContinuousHiddenMarkovModelWithJumps
    
    # Extract the pre-trained base model and jump parameters from the data
    base_model = data.base_model
    epsilon = data.epsilon
    lambda = data.lambda
    
    # Initialize an empty model instance for the jump-diffusion HMM
    m = model()

    # Copy the parameters from the trained base model
    m.states = base_model.states
    m.transition = base_model.transition
    m.emission = base_model.emission
    
    # Set the jump-specific parameters
    m.ϵ = epsilon
    m.λ = lambda
    m.jump_distribution = Poisson(lambda) # The number of jumps is modeled as a Poisson process
    
    # Return the fully constructed jump-diffusion model
    return m
end


"""
    build_turing_model(::StudentTModel, data::Vector{Float64})

Builds a Turing.jl probabilistic model for data assumed to follow a Student's t-distribution. This is useful for Bayesian inference of the distribution's parameters.

### Arguments
- `::StudentTModel`: A type instance to dispatch to this method.
- `data::Vector{Float64}`: A vector of observations.

### Returns
- A Turing model instance, ready for sampling/inference.

### Model Priors
- `σ`: Scale parameter (standard deviation), drawn from a truncated Cauchy distribution. This is a weakly informative prior.
- `μ`: Location parameter (mean), drawn from a Normal distribution centered at 0.
- `ν`: Degrees of freedom, drawn from an Exponential distribution. This prior favors smaller values of `ν`, accommodating heavy tails.
"""
function build_turing_model(::StudentTModel, data)
    @model function student_t_model(obs)
        # Priors for the distribution parameters
        σ ~ Distributions.Truncated(Distributions.Cauchy(0, 1), 0, Inf) # Scale parameter
        μ ~ Distributions.Normal(0, 0.1)      # Location parameter
        ν ~ Distributions.Exponential(1/30.0) # Degrees of freedom

        # Likelihood: The observations are modeled as a scaled and shifted Student's t-distribution
        obs .~ Distributions.TDist(ν) * σ .+ μ
    end
    return student_t_model(data)
end


"""
    build_turing_model(::LaplaceModel, data::Vector{Float64})

Builds a Turing.jl probabilistic model for data assumed to follow a Laplace (double exponential) distribution.

### Arguments
- `::LaplaceModel`: A type instance to dispatch to this method.
- `data::Vector{Float64}`: A vector of observations.

### Returns
- A Turing model instance, ready for sampling/inference.

### Model Priors
- `μ`: Location parameter (mean), drawn from a Normal distribution centered at 0.
- `b`: Scale parameter, drawn from an Exponential distribution.
"""
function build_turing_model(::LaplaceModel, data)
    @model function laplace_model(obs)
        # Priors for the distribution parameters
        μ ~ Distributions.Normal(0, 0.1)  # Location parameter
        b ~ Distributions.Exponential(1.0) # Scale parameter

        # Likelihood: The observations are modeled as a Laplace distribution
        obs .~ Distributions.Laplace(μ, b)
    end
    return laplace_model(data)
end

# --------------------------------------------------------------------------------------------- #