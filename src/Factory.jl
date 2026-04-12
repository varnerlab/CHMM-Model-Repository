# --- DISCRETE MODEL BUILDERS (Legacy) ---------------------------------------- #

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


# --- CONTINUOUS MODEL BUILDERS ----------------------------------------------- #

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
    
    # Extract training data
    obs = data.observations
    n_states = data.number_of_states
    
    # Check if max_iter is provided in the data, otherwise default to 30
    max_iterations = haskey(data, :max_iter) ? data.max_iter : 30
    
    # Pass max_iterations to the baum_welch function
    T_matrix, μ_vec, σ_vec, π_vec, ll_hist, γ = baum_welch(obs, n_states, max_iter=max_iterations)
    
    # Initialize an empty model instance
    m = model()

    # ... (rest of the function remains exactly the same) ...
    m.states = collect(1:n_states)
    m.log_likelihood_history = ll_hist
    
    transition = Dict{Int64, Categorical}()
    emission = Dict{Int64, Normal}()
    
    for s in 1:n_states
        transition[s] = Categorical(T_matrix[s, :])
        emission[s] = Normal(μ_vec[s], σ_vec[s])
    end
    
    m.transition = transition
    m.emission = emission
    
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


# --- BAYESIAN MODEL BUILDERS ------------------------------------------------- #

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


# --- PRICING MODEL BUILDERS -------------------------------------------------- #

"""
    build(::Type{MyEuropeanOptionContract}, data::NamedTuple) -> MyEuropeanOptionContract

Builds a European option contract.

### NamedTuple keys
- `S0::Float64`: Spot price
- `K::Float64`: Strike price
- `T::Float64`: Time to expiration (years)
- `r::Float64`: Risk-free rate
- `is_call::Bool`: true for call, false for put (default: true)
"""
function build(model::Type{MyEuropeanOptionContract}, data::NamedTuple)::MyEuropeanOptionContract

    m = model();
    m.S0 = data.S0;
    m.K = data.K;
    m.T = data.T;
    m.r = data.r;
    m.is_call = haskey(data, :is_call) ? data.is_call : true;

    return m;
end


"""
    build(::Type{MyCHMMPricingModel}, data::NamedTuple) -> MyCHMMPricingModel

Builds a CHMM-based Monte Carlo pricer. Computes the volatility map by mapping
VIX price levels per regime to equity volatility: σ_s = median(VIX_close | state=s) / 100.

### NamedTuple keys
- `hmm::AbstractMarkovModel`: Trained CHMM (no-jump or with-jump)
- `vix_prices::Vector{Float64}`: VIX close prices aligned with vix_states
- `vix_states::Vector{Int64}`: Viterbi-decoded state sequence for VIX prices
- `n_paths::Int64`: Number of MC paths (default: 10000)
- `n_steps_per_year::Int64`: Discretization (default: 252)
"""
function build(model::Type{MyCHMMPricingModel}, data::NamedTuple)::MyCHMMPricingModel

    m = model();
    hmm = data.hmm;
    m.hmm = hmm;

    # Build volatility map: median VIX level per regime → equity vol
    vix_prices = data.vix_prices;
    vix_states = data.vix_states;
    volatility_map = Dict{Int64, Float64}();
    for s in hmm.states
        mask = findall(x -> x == s, vix_states);
        if length(mask) > 0
            volatility_map[s] = median(vix_prices[mask]) / 100.0;
        else
            volatility_map[s] = 0.20; # fallback
        end
    end
    m.volatility_map = volatility_map;

    # Compute stationary distribution from transition matrix (power method)
    K = length(hmm.states);
    T_mat = zeros(K, K);
    for i in 1:K
        T_mat[i, :] = hmm.transition[i].p;
    end
    π_stat = (T_mat^1000)[1, :];
    m.start_distribution = Categorical(π_stat);

    m.n_paths = haskey(data, :n_paths) ? data.n_paths : 10000;
    m.n_steps_per_year = haskey(data, :n_steps_per_year) ? data.n_steps_per_year : 252;

    return m;
end


"""
    build(::Type{MyHestonPricingModel}, data::NamedTuple) -> MyHestonPricingModel

Builds a Heston stochastic volatility Monte Carlo pricer.

### NamedTuple keys
- `v0::Float64`: Initial variance
- `kappa::Float64`: Mean reversion speed
- `theta::Float64`: Long-run variance
- `xi::Float64`: Vol-of-vol
- `rho::Float64`: Price-vol correlation
- `n_paths::Int64`: Number of MC paths (default: 10000)
- `n_steps_per_year::Int64`: Discretization (default: 252)
"""
function build(model::Type{MyHestonPricingModel}, data::NamedTuple)::MyHestonPricingModel

    m = model();
    m.v0 = data.v0;
    m.kappa = data.kappa;
    m.theta = data.theta;
    m.xi = data.xi;
    m.rho = data.rho;
    m.n_paths = haskey(data, :n_paths) ? data.n_paths : 10000;
    m.n_steps_per_year = haskey(data, :n_steps_per_year) ? data.n_steps_per_year : 252;

    return m;
end

# ----------------------------------------------------------------------------- #