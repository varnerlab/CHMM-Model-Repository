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
    viterbi(observations, model) -> Vector{Int64}

Decodes the most likely hidden state sequence using the Viterbi algorithm for
a continuous HMM with Gaussian, Student-t, or Laplace emissions. All three
share the same state-space + transition-matrix structure and only differ in
the univariate emission density, so the recursion is identical modulo the
`logpdf(model.emission[k], ·)` call.

### Returns
- `states::Vector{Int64}`: Most probable state at each time step.
"""
function viterbi(observations::Vector{Float64},
    model::Union{MyContinuousHiddenMarkovModel,
                 MyStudentTHiddenMarkovModel,
                 MyLaplaceHiddenMarkovModel,
                 MyGEDHiddenMarkovModel})::Vector{Int64}

    N = length(observations);
    K = length(model.states);

    # Extract transition matrix
    T_mat = zeros(K, K);
    for i in 1:K
        T_mat[i, :] = probs(model.transition[i]);
    end

    # log probabilities
    log_delta = zeros(N, K);
    psi = zeros(Int64, N, K);

    # initialization: fitted initial-state distribution π when the model
    # carries one; uniform prior as the fallback (e.g. models deserialized
    # from before the `initial` field existed).
    π0 = isdefined(model, :initial) ? probs(model.initial) : fill(1.0 / K, K);
    for k in 1:K
        log_delta[1, k] = log(π0[k]) + logpdf(model.emission[k], observations[1]);
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
4. `π`: Initial Probability Vector [K] (held at its uniform initial value for this family, by documented convention)
5. `ll_history`: Vector of Log-Likelihood values per iteration
6. `gamma`: Matrix of posterior state probabilities [N x K]

The returned parameters are the iterate whose observed-data log-likelihood is
`ll_history[end]` (each iterate is evaluated before the next M-step can mutate
it), and `gamma` is the smoothed posterior computed under exactly those
parameters.
"""
function baum_welch(observations::Array{Float64,1}, number_of_states::Int64;
    max_iter::Int64=30, tol::Float64=1e-4,
    init::Union{Nothing,NamedTuple}=nothing)::Tuple{Array{Float64,2}, Array{Float64,1}, Array{Float64,1}, Array{Float64,1}, Array{Float64,1}, Array{Float64,2}}

    # initialize -
    N = length(observations);
    K = number_of_states;

    # 1. ROBUST INITIALIZATION (Quantile Based) ------------------------------- #
    # We split sorted data into K chunks to initialize means/stds. An explicit
    # `init = (μ = ..., σ = ..., T = ..., π = ...)` overrides the quantile
    # start (used by baum_welch_multistart for optimizer-sensitivity runs).
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

    if init !== nothing
        curr_μ = copy(collect(Float64, init.μ));
        curr_σ = max.(copy(collect(Float64, init.σ)), 1e-6);
        curr_T = copy(Matrix{Float64}(init.T));
        curr_π = copy(collect(Float64, init.π));
    end
    
    # Storage for history
    ll_history = Float64[];
    final_gamma = zeros(N, K);
    
    # 2. EM LOOP -------------------------------------------------------------- #
    # Evaluate-then-update ordering (third-review item 4): every iteration first
    # evaluates the observed-data likelihood of the CURRENT parameters, then
    # tests convergence, and only then performs an M-step. The loop runs one
    # extra evaluation pass so that the final M-step update is itself evaluated:
    # the returned parameters are always the ones whose likelihood is
    # ll_history[end], and final_gamma is the posterior computed under exactly
    # those parameters (EM on the Gaussian family is monotone, so the last
    # evaluated iterate is also the best).
    prev_ll = -Inf;

    for iter in 1:(max_iter + 1)

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

        # Likelihood of the CURRENT (pre-M-step) parameters.
        current_ll = _logsumexp_vec(log_alpha[N, :]);
        push!(ll_history, current_ll);

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
        final_gamma = γ;

        # Convergence / iteration-cap check BEFORE mutating the parameters, so
        # the returned iterate is the one just evaluated.
        if (abs(current_ll - prev_ll) < tol) || (iter == max_iter + 1)
            break;
        end
        prev_ll = current_ll;

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
        # NOTE: π intentionally stays at its uniform initial value for the
        # Gaussian family (documented convention; the paper's CHMM-N parameter
        # counts and BIC penalties exclude π). The t/L/GED fitters apply the
        # γ₁ update.

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
    end

    # return -
    return (curr_T, curr_μ, curr_σ, curr_π, ll_history, final_gamma);
end


"""
    baum_welch_multistart(observations, number_of_states; n_starts=5,
                          max_iter=2000, tol=1e-4, seed=0, jitter=0.25) -> Tuple

Multistart Gaussian Baum-Welch with per-start optimizer diagnostics
(2026-07-16 sixth review, findings 1-2: the deterministic quantile start plus
a fixed 60-iteration cap left high-state fits unconverged with no retained
optimizer evidence).

Start 1 is the canonical quantile initialization; starts 2..n_starts jitter it
(μ shifted by `jitter`·σ·N(0,1), σ scaled by exp(`jitter`·N(0,1)), transition
rows renormalized from |1/K + (jitter/K)·N(0,1)|), each run under the same
`max_iter`/`tol` contract as `baum_welch`. Returns

    (T, μ, σ, π, ll_history, gamma, diagnostics)

for the start with the highest final evaluated log-likelihood, where
`diagnostics` is a Vector of per-start NamedTuples
`(start, ll, n_evals, final_increment, converged)` and `converged` is
`final_increment < tol` (i.e. the run stopped on the tolerance rule rather
than the iteration cap).
"""
function baum_welch_multistart(observations::Array{Float64,1}, number_of_states::Int64;
    n_starts::Int64=5, max_iter::Int64=2000, tol::Float64=1e-4,
    seed::Int64=0, jitter::Float64=0.25)

    K = number_of_states;
    N = length(observations);

    # canonical quantile start parameters (mirrors baum_welch's own init)
    sorted_data = sort(observations);
    chunk_size = floor(Int, N / K);
    μq = zeros(K); σq = zeros(K);
    for s in 1:K
        lo = (s - 1) * chunk_size + 1;
        hi = (s == K) ? N : (s * chunk_size);
        μq[s] = mean(sorted_data[lo:hi]);
        σq[s] = max(std(sorted_data[lo:hi]), 1e-6);
    end

    best = nothing;
    best_ll = -Inf;
    diagnostics = NamedTuple[];
    for s in 1:n_starts
        init = nothing;
        if s > 1
            rng = Random.MersenneTwister(seed + 7919 * s);
            μ0 = μq .+ jitter .* σq .* randn(rng, K);
            σ0 = max.(σq .* exp.(jitter .* randn(rng, K)), 1e-6);
            T0 = abs.(1.0 / K .+ (jitter / K) .* randn(rng, K, K)) .+ 1e-8;
            T0 = T0 ./ sum(T0, dims=2);
            init = (μ = μ0, σ = σ0, T = T0, π = ones(K) ./ K);
        end
        out = baum_welch(observations, K; max_iter=max_iter, tol=tol, init=init);
        llh = out[5];
        inc = length(llh) >= 2 ? llh[end] - llh[end-1] : 0.0;
        push!(diagnostics, (start = s, ll = llh[end], n_evals = length(llh),
                            final_increment = inc, converged = abs(inc) < tol));
        if llh[end] > best_ll
            best_ll = llh[end];
            best = out;
        end
    end

    return (best[1], best[2], best[3], best[4], best[5], best[6], diagnostics);
end


"""
    truncated_pareto_logpmf(α, D) -> Vector{Float64}

Log-pmf of the truncated discrete Pareto duration law
p_α(d) = d^{-(α+1)} / Z_D(α) on d = 1..D, with Z_D(α) = Σ_{d=1}^{D} d^{-(α+1)}.
"""
function truncated_pareto_logpmf(α::Float64, D::Int)
    w = [-(α + 1.0) * log(Float64(d)) for d in 1:D];
    mw = maximum(w);
    Z = mw + log(sum(exp.(w .- mw)));
    return w .- Z;
end

"""
    truncated_pareto_logsf(α, D) -> Vector{Float64}

Log survival function of the truncated discrete Pareto duration law:
log S_α(d) = log P(D ≥ d) = log Σ_{d'=d}^{D} p_α(d'), for d = 1..D, computed by
reverse pairwise log-sum-exp accumulation of `truncated_pareto_logpmf`.
S_α(1) = 1 exactly; S_α(D) = p_α(D).
"""
function truncated_pareto_logsf(α::Float64, D::Int)
    lp = truncated_pareto_logpmf(α, D);
    ls = similar(lp);
    ls[D] = lp[D];
    for d in (D-1):-1:1
        a, b = lp[d], ls[d+1];
        ls[d] = a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b));
    end
    ls[1] = min(ls[1], 0.0);   # clamp the exact-1 head against roundoff
    return ls;
end

"""
    fit_truncated_pareto_alpha(expected_counts, D; censored_counts=nothing,
                               bounds=(0.05, 8.0)) -> Float64

Exact M-step for the truncated discrete Pareto duration parameter from expected
COMPLETED duration counts w_d and (optionally) expected right-CENSORED terminal
counts c_d: maximizes the expected complete-data log-likelihood

    ℓ(α) = Σ_d w_d · log p_α(d) + Σ_d c_d · log S_α(d),

with S_α(d) = P(D ≥ d) the truncated-law survival function. Without censoring
the objective is concave in α (one-parameter exponential family with natural
parameter -(α+1) and sufficient statistic log d), so a golden-section search
over `bounds` finds the maximizer. (2026-07-16 sixth review, finding 4: the
previous update α = 1 / E_w[log d] is the continuous UNTRUNCATED Pareto
formula and does not optimize the declared truncated discrete likelihood, so
the EM carrying it was not an exact ML update.)

With `censored_counts` the survival term is a difference of convex functions,
so concavity is no longer guaranteed; the search then runs a coarse 64-point
log-spaced grid scan over `bounds` first and refines by golden section on the
bracket around the grid argmax (seventh review, finding 3: the HSMM terminal
segment is right-censored, so the duration M-step must include the censored
survival term).
"""
function fit_truncated_pareto_alpha(expected_counts::Vector{Float64}, D::Int;
                                    censored_counts::Union{Nothing,Vector{Float64}}=nothing,
                                    bounds::Tuple{Float64,Float64}=(0.05, 8.0))
    s = sum(expected_counts) + (censored_counts === nothing ? 0.0 : sum(censored_counts));
    if s <= 1e-9; return 1.5; end
    ℓ = if censored_counts === nothing
        α -> sum(expected_counts[d] * truncated_pareto_logpmf(α, D)[d] for d in 1:D);
    else
        function (α)
            lp = truncated_pareto_logpmf(α, D);
            ls = truncated_pareto_logsf(α, D);
            return sum(expected_counts[d] * lp[d] + censored_counts[d] * ls[d] for d in 1:D);
        end
    end
    lo, hi = bounds;
    if censored_counts !== nothing
        # Non-concave case: bracket the golden section around a coarse grid argmax.
        grid = exp.(range(log(lo), log(hi), length=64));
        vals = ℓ.(grid);
        i★ = argmax(vals);
        lo = grid[max(i★ - 1, 1)];
        hi = grid[min(i★ + 1, length(grid))];
    end
    gr = (sqrt(5.0) - 1.0) / 2.0;
    a, b = lo, hi;
    c = b - gr * (b - a); d = a + gr * (b - a);
    fc = ℓ(c); fd = ℓ(d);
    for _ in 1:200
        if fc > fd
            b, d, fd = d, c, fc;
            c = b - gr * (b - a); fc = ℓ(c);
        else
            a, c, fc = c, d, fd;
            d = a + gr * (b - a); fd = ℓ(d);
        end
        if (b - a) < 1e-8; break; end
    end
    return (a + b) / 2;
end


"""
    baum_welch_student_t(observations, number_of_states; max_iter=30, tol=1e-4,
                         ν_init=6.0, ν_bounds=(2.1, 50.0), ν_shrink_rate=0.0) -> Tuple

ECM (Expectation-Conditional-Maximization) estimation with a hybrid surrogate
ν block for a continuous HMM with per-state Student-t emissions
t_ν_k(μ_k, σ_k). The E-step augments the
standard forward-backward with the latent precision
u_{t,k} = (ν_k + 1) / (ν_k + ((o_t - μ_k)/σ_k)^2)
and the M-step updates (μ_k, σ_k) in closed form given u_{t,k}; ν_k is
updated by a one-dimensional golden-section search on the penalised
Q-function over ν_bounds.

The penalised objective is

    Q_pen(ν) = Q(ν) - ν_shrink_rate / ν

which corresponds to an exponential prior on 1/ν (equivalently a Pareto-like
shrinkage of ν toward the Gaussian limit ν → ∞). Setting ν_shrink_rate = 0
recovers the unpenalised update. The ν block is a hybrid generalised
block-coordinate (surrogate) step: it maximizes the posterior-weighted
marginal Student-t log-likelihood sum_t γ_t(k) log t_ν(o_t; μ_k, σ_k) with
the smoothed posteriors γ held fixed. This follows the spirit of the
Liu & Rubin (1995) ECME ν-block, which is an observed-data-likelihood step
only in their single-component i.i.d. setting (all γ_t(k) = 1); for mixtures
the posterior-weighted objective differs from the observed-data likelihood
(Peel & McLachlan 2000 note ECME does not extend straightforwardly), and in
the HMM the observed-data likelihood also depends on ν_k through the forward
recursion, so observed-data ascent is not guaranteed and convergence is
diagnosed from the log-likelihood trace (see the paper's algorithms
appendix). Setting ν_shrink_rate > 0 pulls heavy-tailed states back
toward moderate tail weight and reduces the CHMM-t IS kurtosis overshoot.

Returns (T, μ, σ, ν, π, ll_history, gamma). The returned parameters are the
best evaluated iterate: their observed-data log-likelihood equals
maximum(ll_history), and `gamma` is the smoothed posterior computed under
exactly those parameters.
"""
function baum_welch_student_t(observations::Array{Float64,1}, number_of_states::Int64;
    max_iter::Int64=30, tol::Float64=1e-4,
    ν_init::Float64=6.0, ν_bounds::Tuple{Float64,Float64}=(2.1, 50.0),
    ν_shrink_rate::Float64=0.0)

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

    # Helper: Q-function of ν_k (up to constants independent of ν_k), with
    # optional 1/ν shrinkage penalty.
    # Q_pen(ν) = Σ_t γ_t(k) * [logpdf_t(o_t; μ_k, σ_k, ν)] - ν_shrink_rate / ν
    function _q_of_nu(ν, γk, o, μ, σ)
        acc = 0.0; n = length(o);
        d = LocationScale(μ, σ, TDist(ν));
        @inbounds for t in 1:n
            acc += γk[t] * logpdf(d, o[t]);
        end
        if ν_shrink_rate > 0.0
            acc -= ν_shrink_rate / ν;
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

    # Best-evaluated checkpoint (third-review item 4). The hybrid surrogate ν
    # block does not guarantee observed-data ascent, so every iterate is scored
    # BEFORE it can be mutated and the best finite iterate (parameters, LL, γ)
    # is what the routine returns: the returned parameters always correspond to
    # maximum(ll_history). A non-finite forward pass (Student-t EM at large K
    # can drive a state's σ_k to the 1e-6 floor) also restores this checkpoint.
    best_ll = -Inf;
    best_μ = copy(curr_μ); best_σ = copy(curr_σ); best_ν = copy(curr_ν);
    best_T = copy(curr_T); best_π = copy(curr_π);
    best_gamma = zeros(N, K);

    for iter in 1:(max_iter + 1)

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

        # Likelihood of the CURRENT (pre-M-step) parameters. If the forward
        # pass went non-finite the incoming params are degenerate: fall back to
        # the best evaluated checkpoint and stop.
        current_ll = _logsumexp_vec(log_alpha[N, :]);
        if !isfinite(current_ll)
            break;
        end
        push!(ll_history, current_ll);

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
        final_gamma = γ;

        if current_ll > best_ll
            best_ll = current_ll;
            best_μ = copy(curr_μ); best_σ = copy(curr_σ); best_ν = copy(curr_ν);
            best_T = copy(curr_T); best_π = copy(curr_π);
            best_gamma = copy(γ);
        end

        # Convergence / iteration-cap check BEFORE mutating the parameters.
        if (abs(current_ll - prev_ll) < tol) || (iter == max_iter + 1)
            break;
        end
        prev_ll = current_ll;

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
    end

    # Return the best evaluated finite iterate (ascent is not guaranteed for
    # the hybrid ν block, and a non-finite forward pass lands here too).
    if isfinite(best_ll)
        curr_μ = best_μ; curr_σ = best_σ; curr_ν = best_ν;
        curr_T = best_T; curr_π = best_π;
        final_gamma = best_gamma;
    end

    return (curr_T, curr_μ, curr_σ, curr_ν, curr_π, ll_history, final_gamma);
end


"""
    baum_welch_student_t_shared_nu(observations, number_of_states; max_iter=30, tol=1e-4,
                                   ν_init=6.0, ν_bounds=(2.1, 50.0)) -> Tuple

ECM estimation for a continuous HMM with Student-t emissions t_ν(μ_k, σ_k)
sharing a SINGLE degrees-of-freedom parameter ν across all K states (the
standard Student-t HMM of the time-series literature, as opposed to the
per-state ν_k mixture of `baum_welch_student_t`). The E-step and the
(μ_k, σ_k) CM updates are identical to the per-state fitter; the ν block is a
golden-section search on the AGGREGATE Q-function

    Q(ν) = Σ_t Σ_k γ_t(k) log t_ν(o_t; μ_k, σ_k)

with the smoothed posteriors γ held fixed. As with the per-state ν block this
is a hybrid surrogate step, so observed-data ascent is not guaranteed and the
routine keeps a best-evaluated checkpoint: every iterate is scored by a
forward pass BEFORE it can be mutated, and the returned parameters are the
best finite evaluated iterate.

Returns (T, μ, σ, ν, π, ll_history, gamma) with ν::Float64 the shared
degrees of freedom. The returned parameters' observed-data log-likelihood
equals maximum(ll_history), and `gamma` is the smoothed posterior computed
under exactly those parameters.
"""
function baum_welch_student_t_shared_nu(observations::Array{Float64,1}, number_of_states::Int64;
    max_iter::Int64=30, tol::Float64=1e-4,
    ν_init::Float64=6.0, ν_bounds::Tuple{Float64,Float64}=(2.1, 50.0))

    N = length(observations);
    K = number_of_states;

    # Quantile-based init on μ, σ; uniform init on T, π; shared ν_init.
    sorted_data = sort(observations);
    chunk_size = floor(Int, N / K);
    curr_μ = zeros(K); curr_σ = zeros(K);
    for s in 1:K
        start_idx = (s - 1) * chunk_size + 1;
        end_idx = (s == K) ? N : (s * chunk_size);
        data_subset = sorted_data[start_idx:end_idx];
        curr_μ[s] = mean(data_subset);
        curr_σ[s] = max(std(data_subset), 1e-6);
    end
    curr_ν = ν_init;
    curr_T = ones(K, K) ./ K;
    curr_π = ones(K) ./ K;

    ll_history = Float64[];
    final_gamma = zeros(N, K);
    prev_ll = -Inf;

    _logpdf_t(x, μ, σ, ν) = logpdf(LocationScale(μ, σ, TDist(ν)), x);

    # Aggregate Q-function of the shared ν (up to constants independent of ν):
    # Q(ν) = Σ_k Σ_t γ_t(k) logpdf_t(o_t; μ_k, σ_k, ν)
    function _q_shared_nu(ν, γ, μ, σ, o)
        acc = 0.0; n = length(o);
        @inbounds for k in 1:length(μ)
            d = LocationScale(μ[k], σ[k], TDist(ν));
            for t in 1:n
                acc += γ[t, k] * logpdf(d, o[t]);
            end
        end
        return acc;
    end

    # Golden-section search over the shared ν ∈ ν_bounds (maximize Q).
    function _gss_shared_nu(γ, μ, σ, o, lo, hi; iters=40)
        φ = (sqrt(5.0) - 1.0) / 2.0;
        a = lo; b = hi;
        c = b - φ*(b - a); d = a + φ*(b - a);
        fc = _q_shared_nu(c, γ, μ, σ, o); fd = _q_shared_nu(d, γ, μ, σ, o);
        for _ in 1:iters
            if fc > fd
                b = d; d = c; fd = fc;
                c = b - φ*(b - a); fc = _q_shared_nu(c, γ, μ, σ, o);
            else
                a = c; c = d; fc = fd;
                d = a + φ*(b - a); fd = _q_shared_nu(d, γ, μ, σ, o);
            end
        end
        return 0.5*(a + b);
    end

    # Best-evaluated checkpoint: same contract as the per-state fitter. The
    # shared-ν surrogate block does not guarantee observed-data ascent, so
    # every iterate is scored BEFORE mutation and the best finite iterate
    # (parameters, LL, γ) is what the routine returns.
    best_ll = -Inf;
    best_μ = copy(curr_μ); best_σ = copy(curr_σ); best_ν = curr_ν;
    best_T = copy(curr_T); best_π = copy(curr_π);
    best_gamma = zeros(N, K);

    for iter in 1:(max_iter + 1)

        # E-STEP: emission log-likelihoods + forward-backward.
        log_B = zeros(N, K);
        for t in 1:N, k in 1:K
            log_B[t, k] = _logpdf_t(observations[t], curr_μ[k], curr_σ[k], curr_ν);
        end

        log_alpha = zeros(N, K);
        log_alpha[1, :] = log.(curr_π) .+ log_B[1, :];
        for t in 2:N, j in 1:K
            log_alpha[t, j] = _logsumexp_vec(log_alpha[t-1, :] .+ log.(curr_T[:, j])) + log_B[t, j];
        end

        # Likelihood of the CURRENT (pre-M-step) parameters; a non-finite
        # forward pass falls back to the best evaluated checkpoint.
        current_ll = _logsumexp_vec(log_alpha[N, :]);
        if !isfinite(current_ll)
            break;
        end
        push!(ll_history, current_ll);

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
        final_gamma = γ;

        if current_ll > best_ll
            best_ll = current_ll;
            best_μ = copy(curr_μ); best_σ = copy(curr_σ); best_ν = curr_ν;
            best_T = copy(curr_T); best_π = copy(curr_π);
            best_gamma = copy(γ);
        end

        # Convergence / iteration-cap check BEFORE mutating the parameters.
        if (abs(current_ll - prev_ll) < tol) || (iter == max_iter + 1)
            break;
        end
        prev_ll = current_ll;

        expected_transitions = zeros(K, K);
        for t in 1:N-1
            log_denom = _logsumexp_vec(log_alpha[t, :] .+ log_beta[t, :]);
            for i in 1:K, j in 1:K
                log_xi = log_alpha[t, i] + log(curr_T[i, j]) + log_B[t+1, j] + log_beta[t+1, j] - log_denom;
                expected_transitions[i, j] += exp(log_xi);
            end
        end

        # Latent precisions with the shared ν.
        u = zeros(N, K);
        for t in 1:N, k in 1:K
            δ2 = ((observations[t] - curr_μ[k]) / curr_σ[k])^2;
            u[t, k] = (curr_ν + 1.0) / (curr_ν + δ2);
        end

        # M-STEP (CM): μ_k, σ_k given u, then the shared ν via GSS on the
        # aggregate Q.
        curr_π = γ[1, :];

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
        end
        curr_ν = _gss_shared_nu(γ, curr_μ, curr_σ, observations,
                                ν_bounds[1], ν_bounds[2]);

        for i in 1:K
            r_sum = sum(expected_transitions[i, :]);
            if r_sum > 0
                curr_T[i, :] = expected_transitions[i, :] ./ r_sum;
            end
        end
    end

    # Return the best evaluated finite iterate.
    if isfinite(best_ll)
        curr_μ = best_μ; curr_σ = best_σ; curr_ν = best_ν;
        curr_T = best_T; curr_π = best_π;
        final_gamma = best_gamma;
    end

    return (curr_T, curr_μ, curr_σ, curr_ν, curr_π, ll_history, final_gamma);
end


"""
    _weighted_median(x, w) -> Float64

Weighted median of observations `x` with weights `w ≥ 0`. Returns the first
order statistic whose cumulative weight reaches half of the total, i.e. a
minimiser of the weighted L1 objective `Σ w_i |x_i - μ|`. When the half-weight
falls exactly on an order statistic's cumulative boundary any point in the
closed interval up to the next order statistic is optimal; the `cum >= half`
condition returns the lower endpoint (e.g. the smaller of two equal-weight
points), which is a valid minimiser.
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
            return Float64(x[idx[i]]);
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

Returns (T, μ, b, π, ll_history, gamma). The returned parameters are the
iterate whose observed-data log-likelihood is `ll_history[end]` (each iterate
is evaluated before the next M-step can mutate it), and `gamma` is the
smoothed posterior computed under exactly those parameters.
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

    # Evaluate-then-update ordering (third-review item 4): see baum_welch. The
    # returned parameters are always the iterate whose likelihood is
    # ll_history[end], with final_gamma computed under exactly those parameters.
    for iter in 1:(max_iter + 1)

        log_B = zeros(N, K);
        for t in 1:N, k in 1:K
            log_B[t, k] = logpdf(Laplace(curr_μ[k], curr_b[k]), observations[t]);
        end

        log_alpha = zeros(N, K);
        log_alpha[1, :] = log.(curr_π) .+ log_B[1, :];
        for t in 2:N, j in 1:K
            log_alpha[t, j] = _logsumexp_vec(log_alpha[t-1, :] .+ log.(curr_T[:, j])) + log_B[t, j];
        end

        # Likelihood of the CURRENT (pre-M-step) parameters.
        current_ll = _logsumexp_vec(log_alpha[N, :]);
        push!(ll_history, current_ll);

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
        final_gamma = γ;

        # Convergence / iteration-cap check BEFORE mutating the parameters.
        if (abs(current_ll - prev_ll) < tol) || (iter == max_iter + 1)
            break;
        end
        prev_ll = current_ll;

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
    end

    return (curr_T, curr_μ, curr_b, curr_π, ll_history, final_gamma);
end


"""
    baum_welch_ged(observations, number_of_states; max_iter=30, tol=1e-4,
                   p_init=1.5, p_bounds=(0.5, 4.0)) -> Tuple

ECM estimation for a continuous HMM with per-state Generalized Error
Distribution (GED) emissions b_k(x) = PGeneralizedGaussian(μ_k, α_k, p_k).

The GED log-density is

    log f(x; μ, α, p) = log p − log(2α) − log Γ(1/p) − (|x − μ|/α)^p

which contains Gaussian (p=2, σ = α/√2) and Laplace (p=1, b = α) as
special cases. Per-state p_k therefore lets each regime pick its own
kurtosis on the Gaussian-Laplace axis.

Each EM iteration runs the standard log-space forward-backward E-step
with `logpdf.(PGeneralizedGaussian, ·)` log-densities, then a three-stage
ECM M-step:

    CM1: μ_k ← argmin_μ Σ_t γ_t(k) |o_t − μ|^{p_k}     (golden-section, bracketed)
    CM2: α_k ← [(p_k / W_k) Σ_t γ_t(k) |o_t − μ_k|^{p_k}]^{1/p_k}   (closed form)
    CM3: p_k ← argmax_p Q_k(μ_k, α_k, p)                (golden-section on p_bounds)

where W_k = Σ_t γ_t(k). CM2 is the unique zero of ∂Q_k/∂α_k. CM1 and CM3
are golden-section searches, which assume a unimodal objective on the
bracket; the L^p location objective of CM1 is non-convex for p_k < 1, so
the searched point need not be the CM-block maximiser there and the
monotone-ascent guarantee of Meng & Rubin (1993) ECM does not apply in
general. The routine therefore scores every parameter iterate on the
observed-data likelihood before it can be mutated and returns the best
finite iterate; convergence is diagnosed from the log-likelihood trace.

Returns (T, μ, α, p, π, ll_history, gamma). The returned parameters are the
best evaluated iterate: their observed-data log-likelihood equals
maximum(ll_history), and `gamma` is the smoothed posterior computed under
exactly those parameters.
"""
function baum_welch_ged(observations::Array{Float64,1}, number_of_states::Int64;
    max_iter::Int64=30, tol::Float64=1e-4,
    p_init::Float64=1.5, p_bounds::Tuple{Float64,Float64}=(0.5, 3.0))

    N = length(observations);
    K = number_of_states;

    # Quantile-based init for (μ, α); shared p_init per state. α init is the
    # chunk std (correct order of magnitude across the (Gaussian, Laplace)
    # boundary; ECM corrects within a few iterations).
    sorted_data = sort(observations);
    chunk_size = floor(Int, N / K);
    curr_μ = zeros(K); curr_α = zeros(K); curr_p = fill(p_init, K);
    for s in 1:K
        start_idx = (s - 1) * chunk_size + 1;
        end_idx = (s == K) ? N : (s * chunk_size);
        data_subset = sorted_data[start_idx:end_idx];
        curr_μ[s] = mean(data_subset);
        curr_α[s] = max(std(data_subset), 1e-6);
    end
    curr_T = ones(K, K) ./ K;
    curr_π = ones(K) ./ K;

    ll_history = Float64[];
    final_gamma = zeros(N, K);
    prev_ll = -Inf;

    # Helpers.
    _logpdf_ged(x, μ, α, p) = logpdf(PGeneralizedGaussian(μ, α, p), x);

    # Weighted L^p loss for the location update (we minimize this).
    function _loc_loss(μ, p, w, x)
        acc = 0.0;
        @inbounds for t in eachindex(x)
            acc += w[t] * abs(x[t] - μ)^p;
        end
        return acc;
    end

    # Golden-section MIN over [a, b]. Used for the location update.
    function _gss_min(f, a, b; iters=40)
        φ = (sqrt(5.0) - 1.0) / 2.0;
        c = b - φ*(b - a); d = a + φ*(b - a);
        fc = f(c); fd = f(d);
        for _ in 1:iters
            if fc < fd
                b = d; d = c; fd = fc;
                c = b - φ*(b - a); fc = f(c);
            else
                a = c; c = d; fc = fd;
                d = a + φ*(b - a); fd = f(d);
            end
        end
        return 0.5*(a + b);
    end

    # Golden-section MAX of the per-state Q over p ∈ p_bounds.
    function _gss_max_p(γk, o, μ, α, lo, hi; iters=40)
        # Q_k(p) up to constants in p, given fixed (μ, α):
        #   Q_k(p) = Σ_t γk[t] [log p − log(2α) − log Γ(1/p) − (|o_t − μ|/α)^p]
        function _q(p)
            acc = 0.0;
            dist = PGeneralizedGaussian(μ, α, p);
            @inbounds for t in eachindex(o)
                acc += γk[t] * logpdf(dist, o[t]);
            end
            return acc;
        end
        φ = (sqrt(5.0) - 1.0) / 2.0;
        a = lo; b = hi;
        c = b - φ*(b - a); d = a + φ*(b - a);
        fc = _q(c); fd = _q(d);
        for _ in 1:iters
            if fc > fd
                b = d; d = c; fd = fc;
                c = b - φ*(b - a); fc = _q(c);
            else
                a = c; c = d; fc = fd;
                d = a + φ*(b - a); fd = _q(d);
            end
        end
        return 0.5*(a + b);
    end

    # Best-evaluated checkpoint, mirroring the Student-t routine (the L^p
    # location objective is non-convex for p < 1, so observed-data ascent is
    # not guaranteed): every iterate is scored before mutation and the best
    # finite iterate is returned, i.e. the returned parameters correspond to
    # maximum(ll_history). A non-finite forward pass restores it too.
    best_ll = -Inf;
    best_μ = copy(curr_μ); best_α = copy(curr_α); best_p = copy(curr_p);
    best_T = copy(curr_T); best_π = copy(curr_π);
    best_gamma = zeros(N, K);

    for iter in 1:(max_iter + 1)

        # E-STEP: emission log-likelihoods + forward-backward.
        log_B = zeros(N, K);
        for t in 1:N, k in 1:K
            log_B[t, k] = _logpdf_ged(observations[t], curr_μ[k], curr_α[k], curr_p[k]);
        end

        log_alpha = zeros(N, K);
        log_alpha[1, :] = log.(curr_π) .+ log_B[1, :];
        for t in 2:N, j in 1:K
            log_alpha[t, j] = _logsumexp_vec(log_alpha[t-1, :] .+ log.(curr_T[:, j])) + log_B[t, j];
        end

        # Likelihood of the CURRENT (pre-M-step) parameters.
        current_ll = _logsumexp_vec(log_alpha[N, :]);
        if !isfinite(current_ll)
            break;
        end
        push!(ll_history, current_ll);

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
        final_gamma = γ;

        if current_ll > best_ll
            best_ll = current_ll;
            best_μ = copy(curr_μ); best_α = copy(curr_α); best_p = copy(curr_p);
            best_T = copy(curr_T); best_π = copy(curr_π);
            best_gamma = copy(γ);
        end

        # Convergence / iteration-cap check BEFORE mutating the parameters.
        if (abs(current_ll - prev_ll) < tol) || (iter == max_iter + 1)
            break;
        end
        prev_ll = current_ll;

        expected_transitions = zeros(K, K);
        for t in 1:N-1
            log_denom = _logsumexp_vec(log_alpha[t, :] .+ log_beta[t, :]);
            for i in 1:K, j in 1:K
                log_xi = log_alpha[t, i] + log(curr_T[i, j]) + log_B[t+1, j] + log_beta[t+1, j] - log_denom;
                expected_transitions[i, j] += exp(log_xi);
            end
        end

        # M-STEP (CM): μ_k → α_k → p_k.
        curr_π = γ[1, :];

        for k in 1:K
            wk = γ[:, k];
            Wk = sum(wk);
            if Wk <= 0
                continue;
            end

            # CM1: μ_k bracketed L^{p_k} location estimator. Bracket scales
            # with the current state's α_k so it always brackets the optimum
            # for any reasonable data range.
            pk = curr_p[k]; αk = curr_α[k];
            μlo = curr_μ[k] - 5.0 * αk; μhi = curr_μ[k] + 5.0 * αk;
            μlo = min(μlo, minimum(observations) - 1e-6);
            μhi = max(μhi, maximum(observations) + 1e-6);
            curr_μ[k] = _gss_min(μ -> _loc_loss(μ, pk, wk, observations), μlo, μhi);

            # CM2: α_k closed-form given (μ_k, p_k).
            ssum = 0.0;
            @inbounds for t in 1:N
                ssum += wk[t] * abs(observations[t] - curr_μ[k])^pk;
            end
            α_new = (pk * ssum / Wk)^(1.0 / pk);
            curr_α[k] = max(α_new, 1e-6);

            # CM3: p_k bracketed maximization given (μ_k, α_k).
            curr_p[k] = _gss_max_p(wk, observations, curr_μ[k], curr_α[k],
                                   p_bounds[1], p_bounds[2]);
        end

        for i in 1:K
            r_sum = sum(expected_transitions[i, :]);
            if r_sum > 0
                curr_T[i, :] = expected_transitions[i, :] ./ r_sum;
            end
        end
    end

    # Return the best evaluated finite iterate (ascent is not guaranteed for
    # the non-convex p < 1 location block, and a non-finite forward pass lands
    # here too).
    if isfinite(best_ll)
        curr_μ = best_μ; curr_α = best_α; curr_p = best_p;
        curr_T = best_T; curr_π = best_π;
        final_gamma = best_gamma;
    end

    return (curr_T, curr_μ, curr_α, curr_p, curr_π, ll_history, final_gamma);
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

function _simulate(m::MyGEDHiddenMarkovModel, start::Int64, steps::Int64)::Array{Int64,1}
    chain = Array{Int64,1}(undef, steps);
    chain[1] = start;
    for t in 2:steps
        chain[t] = rand(m.transition[chain[t-1]]);
    end
    return chain;
end

(m::MyStudentTHiddenMarkovModel)(start::Int64, steps::Int64) = _simulate(m, start, steps);
(m::MyLaplaceHiddenMarkovModel)(start::Int64, steps::Int64) = _simulate(m, start, steps);
(m::MyGEDHiddenMarkovModel)(start::Int64, steps::Int64) = _simulate(m, start, steps);

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
                              MyLaplaceHiddenMarkovModel,
                              MyGEDHiddenMarkovModel};

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
    for i in 2:lastindex(simplex)
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
                for i in 2:lastindex(simplex)
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
