# --- PRIVATE METHODS --------------------------------------------------------- #

function _logsumexp_vec(x::Array{Float64,1})::Float64
    m = maximum(x);
    return m + log(sum(exp.(x .- m)));
end

# -- Discrete Simulations (Legacy) --

"""
    _simulate(m::MyHiddenMarkovModel, start::Int64, steps::Int64) -> Array{Int64,1}

Private method: Simulates a single path of hidden states for the Discrete HMM.
"""
function _simulate(m::MyHiddenMarkovModel, start::Int64, steps::Int64)::Array{Int64,1}

    # initialize -
    chain = Array{Int64,1}(undef, steps);
    chain[1] = start;

    # main loop -
    for i ∈ 2:steps
        chain[i] = rand(m.transition[chain[i-1]]);
    end

    return chain;
end

"""
    _simulate(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64) -> Array{Int64,1}

Private method: Simulates a single path of hidden states for the Discrete HMM with Jumps.
"""
function _simulate(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64)::Array{Int64,1}

    # initialize -
    chain = Array{Int64,1}(undef, steps);
    tmp_chain = Dict{Int64,Int64}();
    tmp_chain[1] = start;
    counter = 2;

    # main -
    while (counter ≤ steps)
        
        if (rand() < m.ϵ)

            # jump: find the next state.
            number_of_jumps = rand(m.jump_distribution);
            number_of_states = length(m.states);
            bottom_states = [1,2,3]; # super bad
            top_states = [number_of_states-2,number_of_states-1,number_of_states]; # super good
 
            for _ ∈ 1:number_of_jumps
                if (counter ≤ steps)
                    if (rand() < 0.52)
                        tmp_chain[counter] = rand(bottom_states);
                    else
                        tmp_chain[counter] = rand(top_states);
                    end
                    counter += 1;
                end
            end
        else
            # normal transition -
            current_state = tmp_chain[counter-1];
            tmp_chain[counter] = rand(m.transition[current_state]);
            counter += 1;
        end
    end

    # fill the chain -
    for i ∈ 1:steps
        chain[i] = tmp_chain[i];
    end

    # return -
    return chain;
end

# -- Continuous Simulations (New) --

"""
    _simulate(m::MyContinuousHiddenMarkovModel, start::Int64, steps::Int64) -> Array{Int64,1}

Private method: Simulates a path for the Continuous Gaussian HMM.
Uses the transition matrix learned via Baum-Welch.
"""
function _simulate(m::MyContinuousHiddenMarkovModel, start::Int64, steps::Int64)::Array{Int64,1}
    
    # initialize -
    chain = Array{Int64,1}(undef, steps);
    chain[1] = start;

    # main loop -
    for t in 2:steps
        # Transition using the learned transition matrix (stored as Dict of Categoricals)
        chain[t] = rand(m.transition[chain[t-1]]);
    end

    return chain;
end

"""
    _simulate(m::MyContinuousHiddenMarkovModelWithJumps, start::Int64, steps::Int64) -> Array{Int64,1}

Private method: Simulates a path for the Continuous Jump HMM (Regime Teleportation).
Forces the system into extreme tail states (Crash/Boom) when a jump event occurs.
"""
function _simulate(m::MyContinuousHiddenMarkovModelWithJumps, start::Int64, steps::Int64)::Array{Int64,1}

    # initialize -
    chain = Array{Int64,1}(undef, steps);
    chain[1] = start;
    
    n_states = length(m.states);
    # Define Tail States (Assumes states are sorted by return magnitude)
    crash_states = 1:3;
    boom_states = (n_states-2):n_states;

    counter = 2;

    # main loop -
    while (counter <= steps)
        
        # Check for Jump Event
        if (rand() < m.ϵ)
            
            # 1. How long does the jump last?
            duration = rand(m.jump_distribution);
            
            # 2. Teleport loop (Regime Persistence)
            for _ in 1:duration
                if (counter <= steps)
                    
                    # CORRECTION: Flip the coin INSIDE the loop.
                    # This ensures we get volatility (magnitude) without directional bias (trend).
                    target_pool = (rand() < 0.52) ? crash_states : boom_states;

                    # Override normal transition: pick randomly from the selected tail pool
                    chain[counter] = rand(target_pool);
                    counter += 1;
                end
            end
        else
            # Normal Markov Transition
            current_state = chain[counter-1];
            chain[counter] = rand(m.transition[current_state]);
            counter += 1;
        end
    end

    return chain;
end

# ----------------------------------------------------------------------------- #


# --- PUBLIC METHODS ---------------------------------------------------------- #

"""
    viterbi(observations, model::MyContinuousHiddenMarkovModel) -> Vector{Int64}

Decodes the most likely hidden state sequence using the Viterbi algorithm
for a continuous Gaussian HMM.

### Returns
- `states::Vector{Int64}`: Most probable state at each time step.
"""
function viterbi(observations::Vector{Float64}, model::MyContinuousHiddenMarkovModel)::Vector{Int64}

    N = length(observations);
    K = length(model.states);

    # Extract transition matrix
    T_mat = zeros(K, K);
    for i in 1:K
        T_mat[i, :] = model.transition[i].p;
    end

    # log probabilities
    log_delta = zeros(N, K);
    psi = zeros(Int64, N, K);

    # initialization: uniform prior
    for k in 1:K
        log_delta[1, k] = log(1.0 / K) + logpdf(model.emission[k], observations[1]);
    end

    # recursion
    for t in 2:N
        for j in 1:K
            vals = log_delta[t-1, :] .+ log.(T_mat[:, j]);
            log_delta[t, j] = maximum(vals) + logpdf(model.emission[j], observations[t]);
            psi[t, j] = argmax(vals);
        end
    end

    # backtrack
    states = Vector{Int64}(undef, N);
    states[N] = argmax(log_delta[N, :]);
    for t in N-1:-1:1
        states[t] = psi[t+1, states[t+1]];
    end

    return states;
end


"""
    walk_forward_regimes(observations, window_size, n_states; max_iter=30) -> Vector{Int64}

Walk-forward (rolling window) regime classification. At each step, trains a
fresh Baum-Welch model on the preceding `window_size` observations and decodes
the current time step via Viterbi.

### Arguments
- `observations::Vector{Float64}`: Full observation sequence.
- `window_size::Int`: Training window length (e.g., 252 for 1 year).
- `n_states::Int`: Number of hidden states.
- `max_iter::Int=30`: Max EM iterations per window.

### Returns
- `regimes::Vector{Int64}`: Decoded regime for each out-of-sample time step
  (length = `length(observations) - window_size`).
"""
function walk_forward_regimes(observations::Vector{Float64}, window_size::Int, n_states::Int; max_iter::Int=30)::Vector{Int64}

    N = length(observations);
    regimes = Vector{Int64}(undef, N - window_size);

    p = Progress(N - window_size, desc="Walk-forward: ", showspeed=true);

    for i in (window_size+1):N
        window = observations[(i - window_size):(i-1)];

        model = build(MyContinuousHiddenMarkovModel,
            (observations=window, number_of_states=n_states, max_iter=max_iter));

        decoded = viterbi(window, model);
        current_state = decoded[end];

        # Canonical ordering: state 1 = lowest variance (calm)
        variances = [std(model.emission[s]) for s in model.states];
        sorted_idx = sortperm(variances);
        rank_map = Dict(sorted_idx[r] => r for r in 1:n_states);
        regimes[i - window_size] = rank_map[current_state];

        next!(p);
    end

    return regimes;
end


"""
    vwap(df::DataFrame) -> Array{Float64,1}

Calculates the Volume Weighted Average Price (VWAP) for each row in the DataFrame.
Requires columns: `high`, `low`, `close`, `volume`.
"""
function vwap(df::DataFrame)::Array{Float64,1}

    # Get the number of rows in the DataFrame
    n = nrow(df)
    
    # Initialize an array to store the VWAP values
    vwap_array = Array{Float64,1}(undef, n)
    
    # Initialize cumulative price and volume
    cumulative_pv = 0.0  # sum of price * volume
    cumulative_volume = 0.0

    # Calculate VWAP for each row
    for i in 1:n
        typical_price = (df.high[i] + df.low[i] + df.close[i]) / 3
        volume = df.volume[i]

        cumulative_pv += typical_price * volume
        cumulative_volume += volume

        vwap_array[i] = cumulative_pv / cumulative_volume
    end

    # Return the VWAP array
    return vwap_array
end

"""
    learn_distribution_mcmc(model_type::AbstractDistributionModel, returns::Vector{Float64}; samples::Int = 2000)

Uses a Bayesian MCMC approach (NUTS sampler) to learn the parameters of the specified
probability distribution model given the return data.

Returns a Turing.jl `Chain` object containing posterior samples.
"""
function learn_distribution_mcmc(model_type::AbstractDistributionModel, returns::Vector{Float64}; samples::Int = 2000)
    
    # 1. Build the correct model based on the input type
    #    (Dispatched via Factory.jl)
    turing_model = build_turing_model(model_type, returns);

    # 2. Sample from the posterior using NUTS
    chain = sample(turing_model, NUTS(), samples);

    return chain
end


"""
    baum_welch(observations::Array{Float64,1}, number_of_states::Int64; 
        max_iter::Int64=20, tol::Float64=1e-4) -> Tuple

Estimates the parameters of a Continuous Gaussian Hidden Markov Model using 
the Baum-Welch (Expectation-Maximization) algorithm.

### Arguments
- `observations`: Vector of continuous observations (e.g., daily returns).
- `number_of_states`: Number of hidden regimes to model.
- `max_iter`: Maximum number of EM iterations (default: 20).
- `tol`: Convergence tolerance for Log-Likelihood (default: 1e-4).

### Returns
A tuple containing:
1. `T`: Transition Matrix [K x K]
2. `μ`: Vector of Mean values for each state [K]
3. `σ`: Vector of Std Dev values for each state [K]
4. `π`: Initial Probability Vector [K]
5. `ll_history`: Vector of Log-Likelihood values per iteration
6. `gamma`: Matrix of posterior state probabilities [N x K]
"""
function baum_welch(observations::Array{Float64,1}, number_of_states::Int64; 
    max_iter::Int64=30, tol::Float64=1e-4)::Tuple{Array{Float64,2}, Array{Float64,1}, Array{Float64,1}, Array{Float64,1}, Array{Float64,1}, Array{Float64,2}}
    
    # initialize -
    N = length(observations);
    K = number_of_states;
    
    # 1. ROBUST INITIALIZATION (Quantile Based) ------------------------------- #
    # We split sorted data into K chunks to initialize means/stds
    sorted_data = sort(observations);
    chunk_size = floor(Int, N / K);
    
    curr_μ = zeros(K);
    curr_σ = zeros(K);
    
    for s in 1:K
        start_idx = (s - 1) * chunk_size + 1;
        end_idx = (s == K) ? N : (s * chunk_size);
        data_subset = sorted_data[start_idx:end_idx];
        
        curr_μ[s] = mean(data_subset);
        curr_σ[s] = std(data_subset);
        if (curr_σ[s] < 1e-6)
            curr_σ[s] = 1e-6; # Prevent collapse
        end
    end

    # Initialize T and π uniformly (can be improved with diagonal dominance)
    curr_T = ones(K, K) ./ K;
    curr_π = ones(K) ./ K;
    
    # Storage for history
    ll_history = Float64[];
    final_gamma = zeros(N, K);
    
    # 2. EM LOOP -------------------------------------------------------------- #
    prev_ll = -Inf;
    
    for iter in 1:max_iter
        
        # --- E-STEP: Compute Forward-Backward Probabilities ---
        log_B = zeros(N, K);
        for t in 1:N
            for k in 1:K
                d = Normal(curr_μ[k], curr_σ[k]);
                log_B[t, k] = logpdf(d, observations[t]);
            end
        end
        
        # Forward (Alpha)
        log_alpha = zeros(N, K);
        log_alpha[1, :] = log.(curr_π) .+ log_B[1, :];
        for t in 2:N
            for j in 1:K
                 log_alpha[t, j] = _logsumexp_vec(log_alpha[t-1, :] .+ log.(curr_T[:, j])) + log_B[t, j];
            end
        end
        
        # Backward (Beta)
        log_beta = zeros(N, K);
        # log_beta[N, :] is implicitly 0.0 (log(1))
        for t in N-1:-1:1
            for i in 1:K
                log_terms = log.(curr_T[i, :]) .+ log_B[t+1, :] .+ log_beta[t+1, :];
                log_beta[t, i] = _logsumexp_vec(log_terms);
            end
        end
        
        # Gamma (Posterior State Probability)
        log_gamma = log_alpha .+ log_beta;
        γ = zeros(N, K);
        for t in 1:N
            γ[t, :] = exp.(log_gamma[t, :] .- _logsumexp_vec(log_gamma[t, :]));
        end
        
        # Xi (Posterior Transition Probability)
        expected_transitions = zeros(K, K);
        for t in 1:N-1
            log_denom = _logsumexp_vec(log_alpha[t, :] .+ log_beta[t, :]);
            for i in 1:K
                for j in 1:K
                    log_xi = log_alpha[t, i] + log(curr_T[i, j]) + log_B[t+1, j] + log_beta[t+1, j] - log_denom;
                    expected_transitions[i, j] += exp(log_xi);
                end
            end
        end
        
        # --- M-STEP: Update Parameters ---
        new_π = γ[1, :];
        
        # Update Means and Variances
        for k in 1:K
            w_sum = sum(γ[:, k]);
            if (w_sum > 0)
                curr_μ[k] = sum(γ[:, k] .* observations) / w_sum;
                curr_σ[k] = sqrt(sum(γ[:, k] .* (observations .- curr_μ[k]).^2) / w_sum);
                if (curr_σ[k] < 1e-6)
                     curr_σ[k] = 1e-6; 
                end
            end
        end
        
        # Update Transition Matrix
        for i in 1:K
            r_sum = sum(expected_transitions[i, :]);
            if (r_sum > 0)
                curr_T[i, :] = expected_transitions[i, :] ./ r_sum;
            end
        end
        
        # Check Convergence
        current_ll = _logsumexp_vec(log_alpha[N, :]);
        push!(ll_history, current_ll);
        
        if (abs(current_ll - prev_ll) < tol)
            final_gamma = γ;
            break;
        end
        prev_ll = current_ll;
        final_gamma = γ;
    end
    
    # return -
    return (curr_T, curr_μ, curr_σ, curr_π, ll_history, final_gamma);
end


# --- FUNCTORS (Simulation Interface) ----------------------------------------- #

# Discrete Models
"""
    (m::MyHiddenMarkovModel)(start::Int64, steps::Int64) -> Array{Int64,1}

Functor call to simulate a path for the Discrete HMM.
"""
(m::MyHiddenMarkovModel)(start::Int64, steps::Int64) = _simulate(m, start, steps);

"""
    (m::MyHiddenMarkovModelWithJumps)(start::Int64, steps::Int64) -> Array{Int64,1}

Functor call to simulate a path for the Discrete Jump HMM.
"""
(m::MyHiddenMarkovModelWithJumps)(start::Int64, steps::Int64) = _simulate(m, start, steps);

# Continuous Models
"""
    (m::MyContinuousHiddenMarkovModel)(start::Int64, steps::Int64) -> Array{Int64,1}

Functor call to simulate a path for the Continuous Gaussian HMM.
"""
(m::MyContinuousHiddenMarkovModel)(start::Int64, steps::Int64) = _simulate(m, start, steps);

"""
    (m::MyContinuousHiddenMarkovModelWithJumps)(start::Int64, steps::Int64) -> Array{Int64,1}

Functor call to simulate a path for the Continuous Jump HMM (Teleportation).
"""
(m::MyContinuousHiddenMarkovModelWithJumps)(start::Int64, steps::Int64) = _simulate(m, start, steps);




# ========================================================================================= #
# Growth Calculation Functions
# ========================================================================================= #

"""
    log_growth_matrix(dataset, firms; ...)

Computes the excess log returns for **multiple firms** provided in a Dictionary.
Result is a Matrix (Time x Firms).
"""
function log_growth_matrix(dataset::Dict{String, DataFrame}, 
    firms::Array{String,1}; Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0, 
    testfirm="AAPL", keycol::Symbol = :volume_weighted_average_price)::Array{Float64,2}

    # initialize -
    number_of_firms = length(firms);
    number_of_trading_days = nrow(dataset[testfirm]);
    return_matrix = Array{Float64,2}(undef, number_of_trading_days-1, number_of_firms);

    # main loop -
    for i ∈ eachindex(firms) 
        # get the firm data -
        firm_index = firms[i];
        firm_data = dataset[firm_index];

        # compute the log returns -
        for j ∈ 2:number_of_trading_days
            S₁ = firm_data[j-1, keycol];
            S₂ = firm_data[j, keycol];
            return_matrix[j-1, i] = (1/Δt)*(log(S₂/S₁)) - risk_free_rate;
        end
    end

    # return -
    return return_matrix;
end

"""
    log_growth_matrix(dataset, firm; ...)

Computes the excess log returns for a **single firm** (by ticker string) from a Dictionary.
Result is a Vector.
"""
function log_growth_matrix(dataset::Dict{String, DataFrame}, 
    firm::String; Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0, 
    keycol::Symbol = :volume_weighted_average_price)::Array{Float64,1}

    # initialize -
    number_of_trading_days = nrow(dataset[firm]);
    return_matrix = Array{Float64,1}(undef, number_of_trading_days-1);

    # get the firm data -
    firm_data = dataset[firm];

    # compute the log returns -
    for j ∈ 2:number_of_trading_days
        S₁ = firm_data[j-1, keycol];
        S₂ = firm_data[j, keycol];
        return_matrix[j-1] = (1/Δt)*log(S₂/S₁) - risk_free_rate;
    end

    # return -
    return return_matrix;
end

"""
    log_growth_matrix(dataset::DataFrame; ...)

Computes the excess log returns for a **single DataFrame**.
Useful when the data is already extracted from the dictionary.
"""
function log_growth_matrix(dataset::DataFrame; 
    Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0,
    keycol::Symbol = :volume_weighted_average_price)::Array{Float64,1}

    # initialize -
    firm_data = dropmissing(dataset, disallowmissing=true);
    number_of_trading_periods = nrow(firm_data);
    return_matrix = Array{Float64,1}(undef, number_of_trading_periods - 1);

    # compute the log returns -
    for j ∈ 2:number_of_trading_periods
        S₁ = firm_data[j-1, keycol];
        S₂ = firm_data[j, keycol];
        return_matrix[j-1] = (1/Δt)*log(S₂/S₁) - risk_free_rate;
    end

    # return -
    return return_matrix;
end

"""
    log_growth_matrix(dataset::Array{Float64,1}; ...)

Computes the excess log returns for a **raw array of prices**.
Useful for quick calculations on raw vectors.
"""
function log_growth_matrix(dataset::Array{Float64,1}; 
    Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0)::Array{Float64,1}

    # initialize -
    number_of_trading_periods = length(dataset);
    return_matrix = Array{Float64,1}(undef, number_of_trading_periods-1);

    # compute the log returns -
    for j ∈ 2:number_of_trading_periods
        S₁ = dataset[j-1];
        S₂ = dataset[j];
        return_matrix[j-1] = (1/Δt)*log(S₂/S₁) - risk_free_rate;
    end

    # return -
    return return_matrix;
end