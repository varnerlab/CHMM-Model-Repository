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

function _simulate(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64)::Array{Int64,1}

    # initialize -
    chain = Array{Int64,1}(undef, steps);
    tmp_chain = Dict{Int64,Int64}();
    tmp_chain[1] = start;
    counter = 2;

    # main -
    jump_state = start;
    while (counter ≤ steps)
        
        if (rand() < m.ϵ)

            # # jump: find the next state. It is lowest probability state from here
            number_of_jumps = rand(m.jump_distribution);
            number_of_states = length(m.states);
            bottom_states = [1,2,3]; # super bad
            top_states = [number_of_states-2,number_of_states-1,number_of_states]; # super good
 
            @show number_of_jumps

            for _ ∈ 1:number_of_jumps
                if (rand() < 0.52)
                    tmp_chain[counter] = rand(bottom_states) # a jump transition to bottom states
                else
                    tmp_chain[counter] = rand(top_states) # a jump transition to top states
                end
                counter += 1;
            end
        else
            tmp_chain[counter] = rand(m.transition[jump_state]); # a normal transition
            counter += 1; # increment counter
        end

        jump_state = tmp_chain[counter-1]; # get the last state
    end

    # populate the chain from tmp_chain -
    for i ∈ 1:steps
        chain[i] = tmp_chain[i];
    end

    # return -
    return chain;
end

(m::MyHiddenMarkovModel)(start::Int64, steps::Int64) = _simulate(m, start, steps); 
(m::MyHiddenMarkovModelWithJumps)(start::Int64, steps::Int64) = _simulate(m, start, steps); 

# function log_growth_matrix(dataset::Dict{String, DataFrame}, 
#     firms::Array{String,1}; Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0, 
#     testfirm="AAPL", keycol::Symbol = :volume_weighted_average_price)::Array{Float64,2}

#     # initialize -
#     number_of_firms = length(firms);
#     number_of_trading_days = nrow(dataset[testfirm]);
#     return_matrix = Array{Float64,2}(undef, number_of_trading_days-1, number_of_firms);

#     # main loop -
#     for i ∈ eachindex(firms) 

#         # get the firm data -
#         firm_index = firms[i];
#         firm_data = dataset[firm_index];

#         # compute the log returns -
#         for j ∈ 2:number_of_trading_days
#             S₁ = firm_data[j-1, keycol];
#             S₂ = firm_data[j, keycol];
#             return_matrix[j-1, i] = (1/Δt)*(log(S₂/S₁)) - risk_free_rate;
#         end
#     end

#     # return -
#     return return_matrix;
# end

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
    learn_return_distribution_mcmc(returns::Vector{Float64}; samples::Int = 2000)

Uses a Bayesian MCMC approach to learn the parameters of a Student's t-distribution
fitted to the equity returns data.

Returns a Turing.jl `Chain` object containing the posterior distributions of the parameters.
"""
function learn_distribution_mcmc(model_type::AbstractDistributionModel, returns::Vector{Float64}; samples::Int = 2000)
    
    # 1. Build the correct model based on the input type (e.g., StudentTModel())
    #    Julia's multiple dispatch calls the correct function from Factory.jl
    model_instance = build_turing_model(model_type, returns)

    # 2. Run the MCMC sampler
    chain = Turing.sample(model_instance, NUTS(), samples)

    # 3. Return the resulting chain
    return chain
end


# --- Helper for Log Sum Exp ---
function _logsumexp_vec(x)
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

"""
    baum_welch(observations::Vector{Float64}, n_states::Int, max_iter::Int=20, tol::Float64=1e-4)

Runs the Expectation-Maximization algorithm to learn parameters for a Gaussian HMM.
Returns: (TransitionMatrix, Means, Stds, InitProbs, LogLikelihoodHistory, Gamma)
"""
function baum_welch(obs::Vector{Float64}, n_states::Int; max_iter::Int=20, tol::Float64=1e-4)
    
    N = length(obs)
    K = n_states
    
    # --- 1. ROBUST INITIALIZATION (Quantile Based) ---
    # This logic was previously in your notebook
    sorted_data = sort(obs)
    chunk_size = floor(Int, N / K)
    
    curr_μ = zeros(K)
    curr_σ = zeros(K)
    
    for s in 1:K
        start_idx = (s - 1) * chunk_size + 1
        end_idx = (s == K) ? N : (s * chunk_size)
        data_subset = sorted_data[start_idx:end_idx]
        
        curr_μ[s] = mean(data_subset)
        curr_σ[s] = std(data_subset)
        if curr_σ[s] < 1e-6; curr_σ[s] = 1e-6; end
    end

    # Initialize T (Transition) and π (Start) uniformly or with diagonal dominance
    curr_T = ones(K, K) ./ K
    curr_π = ones(K) ./ K
    
    # Storage for history
    ll_history = Float64[]
    final_gamma = zeros(N, K)
    
    # --- 2. EM LOOP ---
    prev_ll = -Inf
    
    for iter in 1:max_iter
        # --- E-STEP ---
        log_B = zeros(N, K)
        for t in 1:N
            for k in 1:K
                d = Normal(curr_μ[k], curr_σ[k])
                log_B[t, k] = logpdf(d, obs[t])
            end
        end
        
        # Forward (Alpha)
        log_alpha = zeros(N, K)
        log_alpha[1, :] = log.(curr_π) .+ log_B[1, :]
        for t in 2:N
            for j in 1:K
                 log_alpha[t, j] = _logsumexp_vec(log_alpha[t-1, :] .+ log.(curr_T[:, j])) + log_B[t, j]
            end
        end
        
        # Backward (Beta)
        log_beta = zeros(N, K)
        # log_beta[N, :] is already 0.0 (log(1))
        for t in N-1:-1:1
            for i in 1:K
                log_terms = log.(curr_T[i, :]) .+ log_B[t+1, :] .+ log_beta[t+1, :]
                log_beta[t, i] = _logsumexp_vec(log_terms)
            end
        end
        
        # Gamma
        log_gamma = log_alpha .+ log_beta
        γ = zeros(N, K)
        for t in 1:N
            γ[t, :] = exp.(log_gamma[t, :] .- _logsumexp_vec(log_gamma[t, :]))
        end
        
        # Xi (Transitions)
        expected_transitions = zeros(K, K)
        for t in 1:N-1
            log_denom = _logsumexp_vec(log_alpha[t, :] .+ log_beta[t, :])
            for i in 1:K
                for j in 1:K
                    log_xi = log_alpha[t, i] + log(curr_T[i, j]) + log_B[t+1, j] + log_beta[t+1, j] - log_denom
                    expected_transitions[i, j] += exp(log_xi)
                end
            end
        end
        
        # --- M-STEP ---
        new_π = γ[1, :]
        
        for k in 1:K
            w_sum = sum(γ[:, k])
            if w_sum > 0
                curr_μ[k] = sum(γ[:, k] .* obs) / w_sum
                curr_σ[k] = sqrt(sum(γ[:, k] .* (obs .- curr_μ[k]).^2) / w_sum)
                if curr_σ[k] < 1e-6; curr_σ[k] = 1e-6; end
            end
        end
        
        for i in 1:K
            r_sum = sum(expected_transitions[i, :])
            if r_sum > 0
                curr_T[i, :] = expected_transitions[i, :] ./ r_sum
            end
        end
        
        # Check Convergence
        current_ll = _logsumexp_vec(log_alpha[N, :])
        push!(ll_history, current_ll)
        
        if abs(current_ll - prev_ll) < tol
            final_gamma = γ
            break
        end
        prev_ll = current_ll
        final_gamma = γ
    end
    
    return curr_T, curr_μ, curr_σ, curr_π, ll_history, final_gamma
end


"""
    simulate_jumps(m::MyContinuousHiddenMarkovModelWithJumps, steps::Int, n_paths::Int)

Simulates paths where the system can 'teleport' to tail states based on a Poisson process.
"""
function simulate_jumps_stationary(m::MyContinuousHiddenMarkovModelWithJumps, steps::Int, n_paths::Int, π_dist::Vector{Float64})
    
    archive = Array{Int64, 2}(undef, steps, n_paths)
    n_states = length(m.states)
    
    # Define Tail States
    crash_states = 1:3
    boom_states = (n_states-2):n_states
    
    for i in 1:n_paths
        # --- CHANGE IS HERE ---
        # Instead of rand(m.states), we sample from the Stationary Distribution
        current_state = sample(m.states, Weights(π_dist)) 
        
        t = 1
        while t <= steps
            archive[t, i] = current_state
            
            # (Jump Logic remains the same...)
            if rand() < m.ϵ
                duration = rand(m.jump_distribution)
                target_pool = (rand() < 0.5) ? crash_states : boom_states
                
                for _ in 1:duration
                    if t < steps
                        t += 1
                        current_state = rand(target_pool)
                        archive[t, i] = current_state
                    end
                end
            else
                if t < steps
                    t += 1
                    current_state = rand(m.transition[current_state])
                else
                    break
                end
            end
        end
    end
    return archive
end