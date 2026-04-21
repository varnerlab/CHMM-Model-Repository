# --- PRIVATE METHODS --------------------------------------------------------- #

function _logsumexp_vec(x::Array{Float64,1})::Float64
    m = maximum(x);
    return m + log(sum(exp.(x .- m)));
end

# -- Discrete Simulations (Baseline Comparison) --

"""
    _simulate(m::MyHiddenMarkovModel, start::Int64, steps::Int64) -> Array{Int64,1}

Simulates a single path of hidden states for the Discrete HMM.
"""
function _simulate(m::MyHiddenMarkovModel, start::Int64, steps::Int64)::Array{Int64,1}

    chain = Array{Int64,1}(undef, steps);
    chain[1] = start;

    for i ∈ 2:steps
        chain[i] = rand(m.transition[chain[i-1]]);
    end

    return chain;
end

"""
    _simulate(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64) -> Array{Int64,1}

Simulates a single path of hidden states for the Discrete HMM with Poisson jumps
(regime teleportation). Baseline model from the discrete paper.
"""
function _simulate(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64)::Array{Int64,1}

    chain = Array{Int64,1}(undef, steps);
    tmp_chain = Dict{Int64,Int64}();
    tmp_chain[1] = start;
    counter = 2;

    while (counter ≤ steps)

        if (rand() < m.ϵ)

            number_of_jumps = rand(m.jump_distribution);
            number_of_states = length(m.states);
            bottom_states = [1,2,3];
            top_states = [number_of_states-2,number_of_states-1,number_of_states];

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
            current_state = tmp_chain[counter-1];
            tmp_chain[counter] = rand(m.transition[current_state]);
            counter += 1;
        end
    end

    for i ∈ 1:steps
        chain[i] = tmp_chain[i];
    end

    return chain;
end

# -- Continuous Simulations --

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


"""
    baum_welch_student_t(observations, number_of_states; max_iter=30, tol=1e-4,
                         ν_init=6.0, ν_bounds=(2.1, 50.0)) -> Tuple

ECM (Expectation-Conditional-Maximization) estimation for a continuous HMM
with per-state Student-t emissions t_ν_k(μ_k, σ_k). The E-step augments the
standard forward-backward with the latent precision
u_{t,k} = (ν_k + 1) / (ν_k + ((o_t - μ_k)/σ_k)^2)
and the M-step updates (μ_k, σ_k) in closed form given u_{t,k}; ν_k is
updated by a one-dimensional golden-section search on the Q-function over
ν_bounds, following Peel & McLachlan (2000).

Returns (T, μ, σ, ν, π, ll_history, gamma).
"""
function baum_welch_student_t(observations::Array{Float64,1}, number_of_states::Int64;
    max_iter::Int64=30, tol::Float64=1e-4,
    ν_init::Float64=6.0, ν_bounds::Tuple{Float64,Float64}=(2.1, 50.0))

    N = length(observations);
    K = number_of_states;

    # Quantile-based init on μ, σ; uniform init on T, π; shared ν_init per state.
    sorted_data = sort(observations);
    chunk_size = floor(Int, N / K);
    curr_μ = zeros(K); curr_σ = zeros(K); curr_ν = fill(ν_init, K);
    for s in 1:K
        start_idx = (s - 1) * chunk_size + 1;
        end_idx = (s == K) ? N : (s * chunk_size);
        data_subset = sorted_data[start_idx:end_idx];
        curr_μ[s] = mean(data_subset);
        curr_σ[s] = max(std(data_subset), 1e-6);
    end
    curr_T = ones(K, K) ./ K;
    curr_π = ones(K) ./ K;

    ll_history = Float64[];
    final_gamma = zeros(N, K);
    prev_ll = -Inf;

    # Helper: Student-t log-density.
    _logpdf_t(x, μ, σ, ν) = logpdf(LocationScale(μ, σ, TDist(ν)), x);

    # Helper: Q-function of ν_k (up to constants independent of ν_k).
    # Q(ν) = Σ_t γ_t(k) * [logpdf_t(o_t; μ_k, σ_k, ν)]
    function _q_of_nu(ν, γk, o, μ, σ)
        acc = 0.0; n = length(o);
        d = LocationScale(μ, σ, TDist(ν));
        @inbounds for t in 1:n
            acc += γk[t] * logpdf(d, o[t]);
        end
        return acc;
    end

    # Golden-section search over ν ∈ ν_bounds (maximize Q).
    function _gss_nu(γk, o, μ, σ, lo, hi; iters=40)
        φ = (sqrt(5.0) - 1.0) / 2.0;
        a = lo; b = hi;
        c = b - φ*(b - a); d = a + φ*(b - a);
        fc = _q_of_nu(c, γk, o, μ, σ); fd = _q_of_nu(d, γk, o, μ, σ);
        for _ in 1:iters
            if fc > fd
                b = d; d = c; fd = fc;
                c = b - φ*(b - a); fc = _q_of_nu(c, γk, o, μ, σ);
            else
                a = c; c = d; fc = fd;
                d = a + φ*(b - a); fd = _q_of_nu(d, γk, o, μ, σ);
            end
        end
        return 0.5*(a + b);
    end

    for iter in 1:max_iter

        # E-STEP: emission log-likelihoods + forward-backward.
        log_B = zeros(N, K);
        for t in 1:N, k in 1:K
            log_B[t, k] = _logpdf_t(observations[t], curr_μ[k], curr_σ[k], curr_ν[k]);
        end

        log_alpha = zeros(N, K);
        log_alpha[1, :] = log.(curr_π) .+ log_B[1, :];
        for t in 2:N, j in 1:K
            log_alpha[t, j] = _logsumexp_vec(log_alpha[t-1, :] .+ log.(curr_T[:, j])) + log_B[t, j];
        end

        log_beta = zeros(N, K);
        for t in N-1:-1:1, i in 1:K
            log_terms = log.(curr_T[i, :]) .+ log_B[t+1, :] .+ log_beta[t+1, :];
            log_beta[t, i] = _logsumexp_vec(log_terms);
        end

        γ = zeros(N, K);
        for t in 1:N
            γ[t, :] = exp.((log_alpha[t, :] .+ log_beta[t, :]) .-
                           _logsumexp_vec(log_alpha[t, :] .+ log_beta[t, :]));
        end

        expected_transitions = zeros(K, K);
        for t in 1:N-1
            log_denom = _logsumexp_vec(log_alpha[t, :] .+ log_beta[t, :]);
            for i in 1:K, j in 1:K
                log_xi = log_alpha[t, i] + log(curr_T[i, j]) + log_B[t+1, j] + log_beta[t+1, j] - log_denom;
                expected_transitions[i, j] += exp(log_xi);
            end
        end

        # Latent precisions: u_{t,k} = (ν_k + 1) / (ν_k + ((o_t - μ_k)/σ_k)^2)
        u = zeros(N, K);
        for t in 1:N, k in 1:K
            δ2 = ((observations[t] - curr_μ[k]) / curr_σ[k])^2;
            u[t, k] = (curr_ν[k] + 1.0) / (curr_ν[k] + δ2);
        end

        # M-STEP (CM): μ_k, σ_k first given u, then ν_k via GSS.
        new_π = γ[1, :]; curr_π = new_π;

        for k in 1:K
            wu = γ[:, k] .* u[:, k];
            Σwu = sum(wu);
            Σγ = sum(γ[:, k]);
            if Σwu > 0
                curr_μ[k] = sum(wu .* observations) / Σwu;
            end
            if Σγ > 0
                σ2 = sum(wu .* (observations .- curr_μ[k]).^2) / Σγ;
                curr_σ[k] = max(sqrt(max(σ2, 1e-12)), 1e-6);
            end
            γk = γ[:, k];
            if Σγ > 0
                curr_ν[k] = _gss_nu(γk, observations, curr_μ[k], curr_σ[k],
                                    ν_bounds[1], ν_bounds[2]);
            end
        end

        for i in 1:K
            r_sum = sum(expected_transitions[i, :]);
            if r_sum > 0
                curr_T[i, :] = expected_transitions[i, :] ./ r_sum;
            end
        end

        current_ll = _logsumexp_vec(log_alpha[N, :]);
        push!(ll_history, current_ll);
        final_gamma = γ;
        if abs(current_ll - prev_ll) < tol
            break;
        end
        prev_ll = current_ll;
    end

    return (curr_T, curr_μ, curr_σ, curr_ν, curr_π, ll_history, final_gamma);
end


"""
    _weighted_median(x, w) -> Float64

Weighted median of observations `x` with weights `w ≥ 0`. Returns the value
at which the cumulative weight first crosses half of the total. Ties are
broken by linear interpolation between adjacent order statistics.
"""
function _weighted_median(x::AbstractVector{<:Real}, w::AbstractVector{<:Real})::Float64
    n = length(x);
    if n == 0; return 0.0; end
    total = sum(w);
    if total <= 0; return median(x); end
    idx = sortperm(x);
    cum = 0.0; half = total / 2.0;
    for i in 1:n
        cum += w[idx[i]];
        if cum >= half
            if i == 1; return Float64(x[idx[1]]); end
            # linear interpolation between i-1 and i
            prev_cum = cum - w[idx[i]];
            frac = (half - prev_cum) / max(w[idx[i]], 1e-12);
            return Float64(x[idx[i-1]]) + frac * Float64(x[idx[i]] - x[idx[i-1]]);
        end
    end
    return Float64(x[idx[end]]);
end


"""
    baum_welch_laplace(observations, number_of_states; max_iter=30, tol=1e-4) -> Tuple

EM estimation for a continuous HMM with per-state Laplace emissions
Laplace(μ_k, b_k). The E-step is the standard log-space forward-backward
with Laplace log-densities; the M-step uses weighted-median location and
weighted mean-absolute-deviation scale, which are the weighted-MLE
estimators of the Laplace parameters.

Returns (T, μ, b, π, ll_history, gamma).
"""
function baum_welch_laplace(observations::Array{Float64,1}, number_of_states::Int64;
    max_iter::Int64=30, tol::Float64=1e-4)

    N = length(observations);
    K = number_of_states;

    sorted_data = sort(observations);
    chunk_size = floor(Int, N / K);
    curr_μ = zeros(K); curr_b = zeros(K);
    for s in 1:K
        start_idx = (s - 1) * chunk_size + 1;
        end_idx = (s == K) ? N : (s * chunk_size);
        data_subset = sorted_data[start_idx:end_idx];
        curr_μ[s] = median(data_subset);
        curr_b[s] = max(mean(abs.(data_subset .- curr_μ[s])), 1e-6);
    end
    curr_T = ones(K, K) ./ K;
    curr_π = ones(K) ./ K;

    ll_history = Float64[];
    final_gamma = zeros(N, K);
    prev_ll = -Inf;

    for iter in 1:max_iter

        log_B = zeros(N, K);
        for t in 1:N, k in 1:K
            log_B[t, k] = logpdf(Laplace(curr_μ[k], curr_b[k]), observations[t]);
        end

        log_alpha = zeros(N, K);
        log_alpha[1, :] = log.(curr_π) .+ log_B[1, :];
        for t in 2:N, j in 1:K
            log_alpha[t, j] = _logsumexp_vec(log_alpha[t-1, :] .+ log.(curr_T[:, j])) + log_B[t, j];
        end

        log_beta = zeros(N, K);
        for t in N-1:-1:1, i in 1:K
            log_terms = log.(curr_T[i, :]) .+ log_B[t+1, :] .+ log_beta[t+1, :];
            log_beta[t, i] = _logsumexp_vec(log_terms);
        end

        γ = zeros(N, K);
        for t in 1:N
            γ[t, :] = exp.((log_alpha[t, :] .+ log_beta[t, :]) .-
                           _logsumexp_vec(log_alpha[t, :] .+ log_beta[t, :]));
        end

        expected_transitions = zeros(K, K);
        for t in 1:N-1
            log_denom = _logsumexp_vec(log_alpha[t, :] .+ log_beta[t, :]);
            for i in 1:K, j in 1:K
                log_xi = log_alpha[t, i] + log(curr_T[i, j]) + log_B[t+1, j] + log_beta[t+1, j] - log_denom;
                expected_transitions[i, j] += exp(log_xi);
            end
        end

        curr_π = γ[1, :];
        for k in 1:K
            wk = γ[:, k];
            Σw = sum(wk);
            if Σw > 0
                curr_μ[k] = _weighted_median(observations, wk);
                curr_b[k] = max(sum(wk .* abs.(observations .- curr_μ[k])) / Σw, 1e-6);
            end
        end
        for i in 1:K
            r_sum = sum(expected_transitions[i, :]);
            if r_sum > 0
                curr_T[i, :] = expected_transitions[i, :] ./ r_sum;
            end
        end

        current_ll = _logsumexp_vec(log_alpha[N, :]);
        push!(ll_history, current_ll);
        final_gamma = γ;
        if abs(current_ll - prev_ll) < tol
            break;
        end
        prev_ll = current_ll;
    end

    return (curr_T, curr_μ, curr_b, curr_π, ll_history, final_gamma);
end


# --- FUNCTORS (Simulation Interface) ----------------------------------------- #

"""
    (m::MyContinuousHiddenMarkovModel)(start::Int64, steps::Int64) -> Array{Int64,1}

Functor call to simulate a path for the Continuous Gaussian HMM.
"""
(m::MyContinuousHiddenMarkovModel)(start::Int64, steps::Int64) = _simulate(m, start, steps);

function _simulate(m::MyStudentTHiddenMarkovModel, start::Int64, steps::Int64)::Array{Int64,1}
    chain = Array{Int64,1}(undef, steps);
    chain[1] = start;
    for t in 2:steps
        chain[t] = rand(m.transition[chain[t-1]]);
    end
    return chain;
end

function _simulate(m::MyLaplaceHiddenMarkovModel, start::Int64, steps::Int64)::Array{Int64,1}
    chain = Array{Int64,1}(undef, steps);
    chain[1] = start;
    for t in 2:steps
        chain[t] = rand(m.transition[chain[t-1]]);
    end
    return chain;
end

(m::MyStudentTHiddenMarkovModel)(start::Int64, steps::Int64) = _simulate(m, start, steps);
(m::MyLaplaceHiddenMarkovModel)(start::Int64, steps::Int64) = _simulate(m, start, steps);

# Discrete Models (Baseline)
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


# ========================================================================================= #
# Continuous HMM — Return and Price Path Simulation
# ========================================================================================= #

# Union alias for the three continuous CHMM families that all share the
# Dict{Int64, UnivariateDistribution} emission interface used below.
const _ContinuousCHMM = Union{MyContinuousHiddenMarkovModel,
                              MyStudentTHiddenMarkovModel,
                              MyLaplaceHiddenMarkovModel};

"""
    _stationary_distribution(chmm) -> Categorical

Power-iterate the CHMM transition matrix to obtain a stable initial-state
distribution.
"""
function _stationary_distribution(chmm::_ContinuousCHMM)::Categorical
    K = length(chmm.states);
    T_mat = zeros(K, K);
    for i in 1:K
        T_mat[i, :] = probs(chmm.transition[i]);
    end
    π_stat = (T_mat ^ 1000)[1, :];
    π_stat .= max.(π_stat, 1e-12);
    π_stat ./= sum(π_stat);
    return Categorical(π_stat);
end

"""
    simulate_returns(chmm, n_steps; start=nothing, n_paths=1) -> Vector or Matrix

Simulates synthetic return paths from a trained continuous HMM
(Gaussian, Student-t, or Laplace emissions). Returns are on the same
scale the model was trained on, i.e. annualized excess log returns
`G_t = (1/Δt) * ln(P_t / P_{t-1}) - r_f` when trained via
`log_growth_matrix`.

### Arguments
- `chmm`: trained `MyContinuousHiddenMarkovModel`, `MyStudentTHiddenMarkovModel`,
  or `MyLaplaceHiddenMarkovModel`.
- `n_steps::Int`: path length.

### Keyword arguments
- `start::Union{Nothing,Int,Categorical}=nothing`: initial state. If `nothing`,
  draws from the stationary distribution; `Int` forces a specific state;
  a `Categorical` is sampled once per path.
- `n_paths::Int=1`: number of independent paths. With `n_paths == 1` a
  `Vector{Float64}` is returned; otherwise a `n_steps × n_paths` `Matrix`.
"""
function simulate_returns(chmm::_ContinuousCHMM, n_steps::Int;
    start::Union{Nothing,Int,Categorical}=nothing,
    n_paths::Int=1)

    start_dist = start === nothing ? _stationary_distribution(chmm) : start;

    function _one_path()
        s0 = start_dist isa Int ? start_dist : rand(start_dist);
        states = chmm(s0, n_steps);
        out = Vector{Float64}(undef, n_steps);
        @inbounds for t in 1:n_steps
            out[t] = rand(chmm.emission[states[t]]);
        end
        return out;
    end

    if n_paths == 1
        return _one_path();
    end

    paths = Matrix{Float64}(undef, n_steps, n_paths);
    for p in 1:n_paths
        paths[:, p] = _one_path();
    end
    return paths;
end

"""
    simulate_prices(chmm, S0, n_steps; Δt=1/252, risk_free_rate=0.0,
                    start=nothing, n_paths=1) -> Vector or Matrix

Simulates equity price paths from a trained continuous HMM by converting
simulated returns back through the project's log-return convention:

    ln(P_t / P_{t-1}) = (G_t + r_f) * Δt
    P_t = P_{t-1} * exp((G_t + r_f) * Δt)

Output path length is `n_steps + 1` (includes `P_0 = S0`). With
`n_paths == 1` returns a `Vector`; otherwise an `(n_steps+1) × n_paths`
`Matrix`.

### Arguments
- `chmm`: trained continuous CHMM.
- `S0::Real`: initial spot price.
- `n_steps::Int`: number of return steps to roll forward.

### Keyword arguments
- `Δt::Float64=1/252`: time step (annualized-returns convention).
- `risk_free_rate::Float64=0.0`: `r_f` used when the model was trained.
- `start`, `n_paths`: same semantics as `simulate_returns`.
"""
function simulate_prices(chmm::_ContinuousCHMM, S0::Real, n_steps::Int;
    Δt::Float64=1.0/252.0, risk_free_rate::Float64=0.0,
    start::Union{Nothing,Int,Categorical}=nothing,
    n_paths::Int=1)

    R = simulate_returns(chmm, n_steps; start=start, n_paths=n_paths);

    if n_paths == 1
        P = Vector{Float64}(undef, n_steps + 1);
        P[1] = Float64(S0);
        @inbounds for t in 1:n_steps
            P[t + 1] = P[t] * exp((R[t] + risk_free_rate) * Δt);
        end
        return P;
    end

    P = Matrix{Float64}(undef, n_steps + 1, n_paths);
    P[1, :] .= Float64(S0);
    @inbounds for p in 1:n_paths
        for t in 1:n_steps
            P[t + 1, p] = P[t, p] * exp((R[t, p] + risk_free_rate) * Δt);
        end
    end
    return P;
end


# ========================================================================================= #
# GARCH(1,1) — Fitting and Simulation
# ========================================================================================= #

"""
    _garch11_loglikelihood(params, obs) -> Float64

Negative log-likelihood for GARCH(1,1). Used internally by the MLE optimizer.
σ²_t = ω + α * (r_{t-1} - μ)² + β * σ²_{t-1}
"""
function _garch11_loglikelihood(params::Vector{Float64}, obs::Vector{Float64})::Float64

    ω = params[1]; α = params[2]; β = params[3]; μ = params[4];
    N = length(obs);

    # Stationarity and positivity constraints — return large penalty if violated
    if ω ≤ 0 || α < 0 || β < 0 || (α + β) ≥ 1.0
        return 1e10;
    end

    σ2 = ω / (1.0 - α - β); # unconditional variance as initial value
    ll = 0.0;

    for t in 1:N
        r = obs[t] - μ;
        ll += -0.5 * (log(2π) + log(σ2) + r^2 / σ2);
        if t < N
            σ2 = ω + α * r^2 + β * σ2;
            σ2 = max(σ2, 1e-12); # floor
        end
    end

    return -ll; # negative because we minimize
end

"""
    _fit_garch11(obs::Vector{Float64}) -> Tuple

Fits GARCH(1,1) via grid-initialized Nelder-Mead optimization.
Returns (ω, α, β, μ, σ2_history, log_likelihood).
"""
function _fit_garch11(obs::Vector{Float64})

    N = length(obs);
    μ_init = mean(obs);
    var_init = var(obs);

    # Grid search for good initial parameters
    best_nll = Inf;
    best_params = [var_init * 0.05, 0.05, 0.90, μ_init];

    for α_try in [0.02, 0.05, 0.10, 0.15]
        for β_try in [0.70, 0.80, 0.85, 0.90]
            if α_try + β_try < 0.999
                ω_try = var_init * (1.0 - α_try - β_try);
                p = [ω_try, α_try, β_try, μ_init];
                nll = _garch11_loglikelihood(p, obs);
                if nll < best_nll
                    best_nll = nll;
                    best_params = copy(p);
                end
            end
        end
    end

    # Nelder-Mead optimization (simplex method — no gradient needed)
    params = copy(best_params);
    simplex = [copy(params) for _ in 1:(length(params)+1)];
    for i in 2:length(simplex)
        simplex[i][i-1] *= 1.2; # perturb each dimension
    end

    for _ in 1:2000
        # Evaluate
        vals = [_garch11_loglikelihood(s, obs) for s in simplex];
        order = sortperm(vals);
        simplex = simplex[order];
        vals = vals[order];

        # Check convergence
        if abs(vals[end] - vals[1]) < 1e-8
            break;
        end

        n = length(params);
        # Centroid (excluding worst)
        centroid = sum(simplex[1:n]) ./ n;

        # Reflection
        reflected = centroid .+ (centroid .- simplex[end]);
        f_r = _garch11_loglikelihood(reflected, obs);

        if f_r < vals[1]
            # Expansion
            expanded = centroid .+ 2.0 .* (reflected .- centroid);
            f_e = _garch11_loglikelihood(expanded, obs);
            simplex[end] = f_e < f_r ? expanded : reflected;
        elseif f_r < vals[n]
            simplex[end] = reflected;
        else
            # Contraction
            contracted = centroid .+ 0.5 .* (simplex[end] .- centroid);
            f_c = _garch11_loglikelihood(contracted, obs);
            if f_c < vals[end]
                simplex[end] = contracted;
            else
                # Shrink
                for i in 2:length(simplex)
                    simplex[i] = simplex[1] .+ 0.5 .* (simplex[i] .- simplex[1]);
                end
            end
        end
    end

    # Best result
    vals = [_garch11_loglikelihood(s, obs) for s in simplex];
    best = simplex[argmin(vals)];
    ω, α, β, μ = best[1], best[2], best[3], best[4];

    # Reconstruct σ² history
    σ2_hist = zeros(N);
    σ2_hist[1] = ω / max(1.0 - α - β, 1e-6);
    for t in 2:N
        r = obs[t-1] - μ;
        σ2_hist[t] = ω + α * r^2 + β * σ2_hist[t-1];
        σ2_hist[t] = max(σ2_hist[t], 1e-12);
    end

    ll = -_garch11_loglikelihood(best, obs);

    return (ω, α, β, μ, σ2_hist, ll);
end


"""
    simulate_garch(model::MyGARCHModel, n_steps::Int64) -> Vector{Float64}

Simulates a return series from a fitted GARCH(1,1) model.
"""
function simulate_garch(model::MyGARCHModel, n_steps::Int64)::Vector{Float64}

    returns = zeros(n_steps);
    σ2 = model.ω / max(1.0 - model.α - model.β, 1e-6); # start at unconditional variance

    for t in 1:n_steps
        returns[t] = model.μ + sqrt(σ2) * randn();
        σ2 = model.ω + model.α * (returns[t] - model.μ)^2 + model.β * σ2;
        σ2 = max(σ2, 1e-12);
    end

    return returns;
end



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

# --- GRU GENERATOR (Deep Learning Baseline) ---------------------------------- #
#
# Auto-regressive GRU generator for one-dimensional return series.
# Predicts the (μ_t, log σ_t) parameters of a Gaussian next-step density and
# is trained by negative log-likelihood. Style follows the build/simulate
# factory pattern used elsewhere in this package and in JumpHMM.jl.

"""
    _gru_make_chain(input_dim::Int, hidden_dim::Int) -> Flux model

Construct a single-layer GRU encoder + linear (μ, log σ) head suitable for
auto-regressive return-series generation. Output is a 2-vector (μ, log σ).
"""
function _gru_make_chain(input_dim::Int, hidden_dim::Int)
    # Process a (features, seq_len) sequence through a GRU encoder, take the
    # final hidden state, and project to 2 outputs (μ, log σ) of the next-step
    # Gaussian density. Flux's modern GRU returns (hidden, seq_len), so we
    # slice the final time-step before the Dense head.
    return Flux.Chain(
        Flux.GRU(input_dim => hidden_dim),
        x -> x[:, end],
        Flux.Dense(hidden_dim => 2)
    );
end


"""
    _gru_step_chain(input_dim::Int, hidden_dim::Int, gru_weights, dense_weights) -> Stateful step closure

Build a stateful single-step closure used by `simulate_gru`. The GRU cell is
unrolled manually so we can stream one observation at a time and condition on
the previous hidden state, mirroring auto-regressive sampling.
"""
function _gru_step_chain(chain)
    # Extract the trained GRU and Dense layers, plus the cell for stepwise eval.
    gru_layer  = chain[1];
    dense_head = chain[3];
    return gru_layer, dense_head;
end


"""
    _gru_window_pairs(z::Vector{Float64}, w::Int) -> (Xs, ys)

Build training pairs from a standardised return series `z`. For each anchor
`t = w+1, ..., T`, the input is the contiguous window `z[t-w:t-1]` and the
target is `z[t]`. Returns a vector of (1, w) Float32 windows and a vector
of Float32 scalar targets, suitable for streaming through a `Flux.GRU`
column-by-column.
"""
function _gru_window_pairs(z::Vector{Float64}, w::Int)
    n = length(z);
    n_pairs = n - w;
    Xs = Vector{Matrix{Float32}}(undef, n_pairs);
    ys = Vector{Float32}(undef, n_pairs);
    for i in 1:n_pairs
        win = z[i:(i + w - 1)];
        Xs[i] = reshape(Float32.(win), 1, w);
        ys[i] = Float32(z[i + w]);
    end
    return Xs, ys;
end


"""
    train_gru!(model::MyGRUGenerator, observations::Vector{Float64};
               epochs=20, lr=1e-3, hidden_dim=32, window=20, seed=20260420,
               verbose=false) -> MyGRUGenerator

Train the GRU generator by negative log-likelihood on the standardised
in-sample series. Mutates `model.chain`, `model.μ_x`, `model.σ_x`,
`model.window`, and `model.loss_history`. Returns the same model for
chaining.

The Gaussian NLL loss for one (μ, logσ, y) triple is
    ℓ = logσ + ½ ((y − μ) / exp(logσ))²
plus an additive constant that is dropped during optimisation.
"""
function train_gru!(model::MyGRUGenerator, observations::Vector{Float64};
                    epochs::Int=20, lr::Float64=1e-3,
                    hidden_dim::Int=32, window::Int=20,
                    seed::Int=20260420, verbose::Bool=false)::MyGRUGenerator

    Random.seed!(seed);

    # Standardise input to keep gradients well-scaled.
    μ_x = mean(observations); σ_x = std(observations);
    z = (observations .- μ_x) ./ σ_x;

    # Build network and unroll training pairs.
    chain = _gru_make_chain(1, hidden_dim);
    Xs, ys = _gru_window_pairs(z, window);
    n_pairs = length(Xs);

    # Negative log-likelihood loss for a single (window, target) pair.
    # X has shape (features=1, seq_len=window); chain returns 2-vector (μ, logσ).
    function _nll(chain, X, y)
        out = chain(X);
        μ = out[1]; logσ = out[2];
        logσ = clamp(logσ, -6.0f0, 4.0f0);
        return logσ + 0.5f0 * ((y - μ) / exp(logσ))^2;
    end

    opt_state = Flux.setup(Flux.Optimisers.Adam(lr), chain);
    loss_history = Float64[];

    for epoch in 1:epochs
        order = Random.shuffle(1:n_pairs);
        epoch_loss = 0.0;
        for i in order
            X = Xs[i]; y = ys[i];
            grads = Flux.gradient(c -> _nll(c, X, y), chain);
            Flux.update!(opt_state, chain, grads[1]);
            epoch_loss += Float64(_nll(chain, X, y));
        end
        epoch_loss /= n_pairs;
        push!(loss_history, epoch_loss);
        if verbose
            println("  GRU epoch $epoch: NLL = $(round(epoch_loss, digits=4))");
        end
    end

    model.chain = chain;
    model.window = window;
    model.μ_x = μ_x;
    model.σ_x = σ_x;
    model.loss_history = loss_history;
    return model;
end


"""
    simulate_gru(model::MyGRUGenerator, n_steps::Int;
                 seed_window::Vector{Float64}=Float64[],
                 burn_in::Int=64) -> Vector{Float64}

Generate a synthetic return path of length `n_steps` from the trained
generator. The recurrence is initialised by streaming `seed_window` through
the GRU; if no seed is provided the model's training-distribution mean
(zero in standardised space) is used. After `burn_in` samples the warm-up
prefix is discarded.
"""
function simulate_gru(model::MyGRUGenerator, n_steps::Int;
                      seed_window::Vector{Float64}=Float64[],
                      burn_in::Int=64)::Vector{Float64}

    chain = model.chain;
    w = model.window;

    # Standardise seed_window (or fall back to zeros).
    if isempty(seed_window)
        z_seed = zeros(Float64, w);
    else
        if length(seed_window) < w
            pad = zeros(Float64, w - length(seed_window));
            sw  = vcat(pad, seed_window);
        else
            sw  = seed_window[(end - w + 1):end];
        end
        z_seed = (sw .- model.μ_x) ./ model.σ_x;
    end

    # Auto-regressive rollout: maintain a sliding context of length `w`.
    # At each step, encode the window through the GRU + Dense head, sample
    # the next standardised return, append it, and slide the window forward.
    total   = burn_in + n_steps;
    sampled = zeros(Float64, total);
    context = copy(z_seed);

    @inbounds for t in 1:total
        X   = reshape(Float32.(context), 1, w);   # (features=1, seq_len=w)
        out = chain(X);                            # 2-vector: (μ, log σ)
        μ_t  = Float64(out[1]);
        ls_t = clamp(Float64(out[2]), -6.0, 4.0);
        z_next = μ_t + exp(ls_t) * randn();
        sampled[t] = z_next;
        # Slide context window forward by one step (append z_next, drop oldest).
        context = vcat(context[2:end], z_next);
    end

    # Drop burn-in and unstandardise back to the return scale.
    z_path = sampled[(burn_in + 1):end];
    return model.μ_x .+ model.σ_x .* z_path;
end
