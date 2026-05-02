# --- DISCRETE MODEL BUILDERS (Baseline Comparison) --------------------------- #

"""
    build(model::Type{MyHiddenMarkovModel}, data::NamedTuple) -> MyHiddenMarkovModel

Builds a discrete HMM from provided transition and emission matrices.

### NamedTuple keys
- `states::Array{Int64,1}`: State indices
- `T::Array{Float64,2}`: Transition matrix [K x K]
- `E::Array{Float64,2}`: Emission matrix [K x M]
"""
function build(model::Type{MyHiddenMarkovModel}, data::NamedTuple)::MyHiddenMarkovModel

    m = model();
    transition = Dict{Int64, Categorical}();
    emission = Dict{Int64, Categorical}();

    states = data.states;
    T = data.T;
    E = data.E;

    for s ∈ states
        transition[s] = Categorical(T[s,:]);
        emission[s] = Categorical(E[s,:]);
    end

    m.transition = transition;
    m.emission = emission;
    m.states = states;

    return m;
end


"""
    build(model::Type{MyHiddenMarkovModelWithJumps}, data::NamedTuple) -> MyHiddenMarkovModelWithJumps

Builds a discrete HMM with Poisson jump process (baseline from discrete paper).

### NamedTuple keys
- `states::Array{Int64,1}`: State indices
- `T::Array{Float64,2}`: Transition matrix [K x K]
- `E::Array{Float64,2}`: Emission matrix [K x M]
- `ϵ::Float64`: Jump probability
- `λ::Float64`: Jump duration rate (Poisson parameter)
"""
function build(model::Type{MyHiddenMarkovModelWithJumps}, data::NamedTuple)::MyHiddenMarkovModelWithJumps

    m = model();
    transition = Dict{Int64, Categorical}();
    inverse_transition = Dict{Int64, Categorical}();
    emission = Dict{Int64, Categorical}();
    ϵ = data.ϵ;
    λ = data.λ;

    states = data.states;
    T = data.T;
    E = data.E;

    for s ∈ states
        transition[s] = Categorical(T[s,:]);
        emission[s] = Categorical(E[s,:]);
    end

    for s ∈ states
        F = sum(1 .- T[s,:]);
        d = (1/F)*(1 .- T[s,:]);
        inverse_transition[s] = Categorical(d);
    end

    m.transition = transition;
    m.inverse_transition = inverse_transition;
    m.emission = emission;
    m.states = states;
    m.ϵ = ϵ;
    m.λ = λ;
    m.jump_distribution = Poisson(λ);

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
    T_matrix, μ_vec, σ_vec, _, ll_hist, _ = baum_welch(obs, n_states, max_iter=max_iterations)
    
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
    build(model::Type{MyStudentTHiddenMarkovModel}, data::NamedTuple) -> MyStudentTHiddenMarkovModel

Trains a continuous HMM with per-state Student-t emissions via ECM
(see `baum_welch_student_t`).

### NamedTuple keys
- `observations::Vector{Float64}`: return series
- `number_of_states::Int`: K
- `max_iter::Int=30` (optional): EM iteration cap
- `ν_init::Float64=6.0` (optional): initial degrees of freedom per state
- `ν_bounds::Tuple{Float64,Float64}=(2.1, 50.0)` (optional): bracket for 1D ν search
- `ν_shrink_rate::Float64=0.0` (optional): rate of an exponential shrinkage
  prior on 1/ν_k in the penalised ECM Q-function; larger values pull ν_k
  toward the Gaussian limit ν → ∞. Default 0 recovers the unpenalised ECM.
"""
function build(model::Type{MyStudentTHiddenMarkovModel}, data::NamedTuple)::MyStudentTHiddenMarkovModel

    obs = data.observations;
    n_states = data.number_of_states;
    max_iterations = haskey(data, :max_iter) ? data.max_iter : 30;
    ν_init = haskey(data, :ν_init) ? data.ν_init : 6.0;
    ν_bounds = haskey(data, :ν_bounds) ? data.ν_bounds : (2.1, 50.0);
    ν_shrink_rate = haskey(data, :ν_shrink_rate) ? data.ν_shrink_rate : 0.0;

    T_matrix, μ_vec, σ_vec, ν_vec, _, ll_hist, _ =
        baum_welch_student_t(obs, n_states;
                             max_iter=max_iterations, ν_init=ν_init, ν_bounds=ν_bounds,
                             ν_shrink_rate=ν_shrink_rate);

    m = model();
    m.states = collect(1:n_states);
    m.log_likelihood_history = ll_hist;
    transition = Dict{Int64, Categorical}();
    emission = Dict{Int64, LocationScale{Float64, Continuous, TDist{Float64}}}();
    for s in 1:n_states
        transition[s] = Categorical(T_matrix[s, :]);
        emission[s] = LocationScale(μ_vec[s], σ_vec[s], TDist(ν_vec[s]));
    end
    m.transition = transition;
    m.emission = emission;
    return m;
end


"""
    build(model::Type{MyLaplaceHiddenMarkovModel}, data::NamedTuple) -> MyLaplaceHiddenMarkovModel

Trains a continuous HMM with per-state Laplace emissions via EM
(see `baum_welch_laplace`).

### NamedTuple keys
- `observations::Vector{Float64}`: return series
- `number_of_states::Int`: K
- `max_iter::Int=30` (optional): EM iteration cap
"""
function build(model::Type{MyLaplaceHiddenMarkovModel}, data::NamedTuple)::MyLaplaceHiddenMarkovModel

    obs = data.observations;
    n_states = data.number_of_states;
    max_iterations = haskey(data, :max_iter) ? data.max_iter : 30;

    T_matrix, μ_vec, b_vec, _, ll_hist, _ =
        baum_welch_laplace(obs, n_states; max_iter=max_iterations);

    m = model();
    m.states = collect(1:n_states);
    m.log_likelihood_history = ll_hist;
    transition = Dict{Int64, Categorical}();
    emission = Dict{Int64, Laplace}();
    for s in 1:n_states
        transition[s] = Categorical(T_matrix[s, :]);
        emission[s] = Laplace(μ_vec[s], b_vec[s]);
    end
    m.transition = transition;
    m.emission = emission;
    return m;
end


"""
    build(model::Type{MyGEDHiddenMarkovModel}, data::NamedTuple) -> MyGEDHiddenMarkovModel

Trains a continuous HMM with per-state Generalized Error Distribution
emissions PGeneralizedGaussian(μ_k, α_k, p_k) via ECM (see `baum_welch_ged`).
GED nests Gaussian (p=2) and Laplace (p=1); per-state p_k therefore lets
each regime pick its own kurtosis on the Gaussian-Laplace axis.

### NamedTuple keys
- `observations::Vector{Float64}`: return series
- `number_of_states::Int`: K
- `max_iter::Int=30` (optional): EM iteration cap
- `p_init::Float64=1.5` (optional): initial shape per state
- `p_bounds::Tuple{Float64,Float64}=(0.5, 4.0)` (optional): bracket for p search
"""
function build(model::Type{MyGEDHiddenMarkovModel}, data::NamedTuple)::MyGEDHiddenMarkovModel

    obs = data.observations;
    n_states = data.number_of_states;
    max_iterations = haskey(data, :max_iter) ? data.max_iter : 30;
    p_init = haskey(data, :p_init) ? data.p_init : 1.5;
    p_bounds = haskey(data, :p_bounds) ? data.p_bounds : (0.5, 3.0);

    T_matrix, μ_vec, α_vec, p_vec, _, ll_hist, _ =
        baum_welch_ged(obs, n_states;
                       max_iter=max_iterations, p_init=p_init, p_bounds=p_bounds);

    m = model();
    m.states = collect(1:n_states);
    m.log_likelihood_history = ll_hist;
    transition = Dict{Int64, Categorical}();
    emission = Dict{Int64, PGeneralizedGaussian}();
    for s in 1:n_states
        transition[s] = Categorical(T_matrix[s, :]);
        emission[s] = PGeneralizedGaussian(μ_vec[s], α_vec[s], p_vec[s]);
    end
    m.transition = transition;
    m.emission = emission;
    return m;
end


# --- GARCH MODEL BUILDER ----------------------------------------------------- #

"""
    build(model::Type{MyGARCHModel}, data::NamedTuple) -> MyGARCHModel

Fits a GARCH(1,1) model via maximum likelihood estimation.
σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}

### NamedTuple keys
- `observations::Vector{Float64}`: Return series (same scale as HMM data)
"""
function build(model::Type{MyGARCHModel}, data::NamedTuple)::MyGARCHModel

    obs = data.observations;
    ω, α, β, μ, σ2_hist, ll = _fit_garch11(obs);

    m = model();
    m.ω = ω;
    m.α = α;
    m.β = β;
    m.μ = μ;
    m.σ2_history = σ2_hist;
    m.log_likelihood = ll;

    return m;
end

# ----------------------------------------------------------------------------- #


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


# --- GRU GENERATOR BUILDER --------------------------------------------------- #

"""
    build(model::Type{MyGRUGenerator}, data::NamedTuple) -> MyGRUGenerator

Trains a single-layer GRU + Gaussian-head generator on the provided return
series via maximum-likelihood (negative log-likelihood) and returns the
trained model.

### NamedTuple keys
- `observations::Vector{Float64}` (required): in-sample return series.
- `epochs::Int=20` (optional): number of full training epochs.
- `lr::Float64=1e-3` (optional): Adam learning rate.
- `hidden_dim::Int=32` (optional): GRU hidden-state width.
- `window::Int=20` (optional): context-window length used during training and simulation.
- `seed::Int=20260420` (optional): RNG seed for shuffling and weight init.
- `verbose::Bool=false` (optional): print per-epoch NLL.
"""
function build(model::Type{MyGRUGenerator}, data::NamedTuple)::MyGRUGenerator

    obs = data.observations;
    epochs     = haskey(data, :epochs)     ? data.epochs     : 20;
    lr         = haskey(data, :lr)         ? data.lr         : 1e-3;
    hidden_dim = haskey(data, :hidden_dim) ? data.hidden_dim : 32;
    window     = haskey(data, :window)     ? data.window     : 20;
    seed       = haskey(data, :seed)       ? data.seed       : 20260420;
    verbose    = haskey(data, :verbose)    ? data.verbose    : false;

    m = model();
    train_gru!(m, obs;
        epochs=epochs, lr=lr, hidden_dim=hidden_dim,
        window=window, seed=seed, verbose=verbose);

    return m;
end
