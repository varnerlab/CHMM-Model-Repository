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
    build(model::Type{MyContinuousHiddenMarkovModel}, data::NamedTuple)

Builds and trains a Continuous HMM using the Baum-Welch algorithm on the provided observation data.
data requires keys: `observations` (Vector{Float64}) and `number_of_states` (Int).
"""
function build(model::Type{MyContinuousHiddenMarkovModel}, data::NamedTuple)::MyContinuousHiddenMarkovModel
    
    # 1. Extract settings
    obs = data.observations
    n_states = data.number_of_states
    
    # 2. Run the heavy lifting (Compute.jl)
    T_matrix, μ_vec, σ_vec, π_vec, ll_hist, γ = baum_welch(obs, n_states)
    
    # 3. Initialize the struct
    m = model()
    m.states = collect(1:n_states)
    m.log_likelihood_history = ll_hist
    
    # 4. Construct Dictionaries for Dispatch
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
    build(model::Type{MyContinuousHiddenMarkovModelWithJumps}, data::NamedTuple)

Wraps a trained `MyContinuousHiddenMarkovModel` with jump parameters.
Data requires keys: `base_model`, `epsilon`, `lambda`.
"""
function build(model::Type{MyContinuousHiddenMarkovModelWithJumps}, data::NamedTuple)::MyContinuousHiddenMarkovModelWithJumps
    
    base = data.base_model
    
    m = model()
    # Copy trained parameters
    m.states = base.states
    m.transition = base.transition
    m.emission = base.emission
    
    # Set jump parameters
    m.ϵ = data.epsilon
    m.λ = data.lambda
    m.jump_distribution = Poisson(data.lambda)
    
    return m
end


"""
    build_turing_model(::StudentTModel, data)

Builds the Turing.jl model for a Student's t-distribution.
"""
function build_turing_model(::StudentTModel, data)
    @model function student_t_model(obs)
        σ ~ Distributions.Truncated(Distributions.Cauchy(0, 1), 0, Inf)
        μ ~ Distributions.Normal(0, 0.1)
        ν ~ Distributions.Exponential(1/30.0)
        obs .~ Distributions.TDist(ν) * σ .+ μ
    end
    return student_t_model(data)
end


"""
    build_turing_model(::LaplaceModel, data)

Builds the Turing.jl model for a Laplace distribution.
"""
function build_turing_model(::LaplaceModel, data)
    @model function laplace_model(obs)
        μ ~ Distributions.Normal(0, 0.1)
        b ~ Distributions.Exponential(1.0)
        obs .~ Distributions.Laplace(μ, b)
    end
    return laplace_model(data)
end

# --------------------------------------------------------------------------------------------- #