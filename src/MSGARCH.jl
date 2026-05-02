# ========================================================================================= #
# MSGARCH.jl
#
# Markov-Switching GARCH(1,1) per Haas, Mittnik, Paolella (2004).
#
# Each regime k has an independent GARCH(1,1) recursion, updated at every t regardless of
# the active regime (path-independent per HMP). Observation r_t is drawn from regime s_t:
#   σ²_{k,t} = ω_k + α_k (r_{t-1} - μ)² + β_k σ²_{k,t-1}   for every k at every t
#   r_t = μ + sqrt(σ²_{s_t,t}) * ε_t,   ε_t ~ N(0,1)
#
# Estimation: Hamilton (1989) filter for the forward log-likelihood, optimized via
# Nelder-Mead over the joint param vector. For K=2, 8 parameters: (ω_1,α_1,β_1),
# (ω_2,α_2,β_2), p_11, p_22. μ is pegged to the sample mean (state-invariant per HMP).
#
# Scope: fixed K=2 (parsimony). Extend via a `_fit_msgarch_K(obs, K)` wrapper if needed.
# ========================================================================================= #

using LinearAlgebra;
using Statistics;

# ------------------------------------------------------------------------------------------- #
# Hamilton filter: forward log-likelihood
# ------------------------------------------------------------------------------------------- #

"""
    _hamilton_filter_msgarch(obs, ω, α, β, μ, T) -> (ll, σ2_hist, γ)

Returns the forward log-likelihood, the K-regime σ² histories (N × K), and the filtered
regime posteriors γ[t,k] = P(s_t = k | y_1:t).
"""
function _hamilton_filter_msgarch(obs::Vector{Float64},
    ω::Vector{Float64}, α::Vector{Float64}, β::Vector{Float64},
    μ::Float64, T::Matrix{Float64})

    N = length(obs);
    K = length(ω);

    # Stationary start distribution (eigenvector of T^T at eigenvalue 1)
    # Quick approach: power iteration
    π_stat = ones(K) ./ K;
    for _ in 1:50; π_stat = T' * π_stat; π_stat = max.(π_stat, 1e-12); π_stat ./= sum(π_stat); end

    # Initial σ² at regime-conditional unconditional variance ω_k / (1 - α_k - β_k)
    σ2 = Matrix{Float64}(undef, N, K);
    for k in 1:K
        denom = max(1.0 - α[k] - β[k], 1e-6);
        σ2[1, k] = ω[k] / denom;
    end

    γ = Matrix{Float64}(undef, N, K);
    ll = 0.0;

    # Predict-update loop
    log2π_half = 0.5 * log(2π);
    log_emit = zeros(K);
    for t in 1:N
        # Predicted state distribution π_t = T' γ_{t-1}
        if t == 1
            π_pred = π_stat;
        else
            π_pred = T' * γ[t-1, :];
            π_pred ./= sum(π_pred);
        end
        # Emission likelihood at t, given σ²[t, k]
        for k in 1:K
            r = obs[t] - μ;
            s2 = σ2[t, k];
            if s2 <= 0 || !isfinite(s2)
                log_emit[k] = -1e10;
            else
                log_emit[k] = -log2π_half - 0.5 * log(s2) - 0.5 * r^2 / s2;
            end
        end
        # Filter update
        raw = π_pred .* exp.(log_emit .- maximum(log_emit));
        rsum = sum(raw);
        if !isfinite(rsum) || rsum <= 0
            γ[t, :] = π_pred;
            ll += maximum(log_emit) + log(max(1e-12, sum(π_pred .* exp.(log_emit .- maximum(log_emit)))));
        else
            γ[t, :] = raw ./ rsum;
            ll += maximum(log_emit) + log(rsum);
        end

        # Update σ² for next step (per-regime recursion)
        if t < N
            r = obs[t] - μ;
            for k in 1:K
                σ2[t+1, k] = ω[k] + α[k] * r^2 + β[k] * σ2[t, k];
                σ2[t+1, k] = max(σ2[t+1, k], 1e-12);
            end
        end
    end

    return ll, σ2, γ;
end

# ------------------------------------------------------------------------------------------- #
# Negative log-likelihood wrapper for Nelder-Mead (params -> nll)
# ------------------------------------------------------------------------------------------- #

"""
    _msgarch_nll_k2(params::Vector{Float64}, obs::Vector{Float64}, μ::Float64) -> Float64

Parameter layout for K=2: [ω_1, α_1, β_1, ω_2, α_2, β_2, logit_p11, logit_p22].
Returns negative log-likelihood; +1e10 penalty on constraint violation.
"""
function _msgarch_nll_k2(params::Vector{Float64}, obs::Vector{Float64}, μ::Float64)
    ω = [params[1], params[4]];
    α = [params[2], params[5]];
    β = [params[3], params[6]];
    # constraints
    for k in 1:2
        if ω[k] <= 0 || α[k] < 0 || β[k] < 0 || (α[k] + β[k]) >= 1.0
            return 1e10;
        end
    end
    p11 = 1.0 / (1.0 + exp(-params[7]));
    p22 = 1.0 / (1.0 + exp(-params[8]));
    T = [p11 (1-p11); (1-p22) p22];
    ll, _, _ = _hamilton_filter_msgarch(obs, ω, α, β, μ, T);
    if !isfinite(ll); return 1e10; end
    return -ll;
end

# ------------------------------------------------------------------------------------------- #
# Fit + simulate
# ------------------------------------------------------------------------------------------- #

"""
    fit_msgarch_k2(obs; max_iter=2000) -> MyMSGARCHModel

Fit MS-GARCH(1,1) with K=2 regimes. Grid-initialize over (ω, α, β) and transition stickiness,
then Nelder-Mead. Regime 1 is canonicalized as the calm regime (lower unconditional variance).
"""
function fit_msgarch_k2(obs::Vector{Float64}; max_iter::Int=2000)::MyMSGARCHModel
    μ = mean(obs);
    var_obs = var(obs);

    # Grid init
    best_nll = Inf; best_params = zeros(8);
    for α_try in [0.05, 0.10]
        for β_try in [0.80, 0.90]
            for vol_ratio in [0.5, 1.5, 3.0]
                ω1 = var_obs * (1 - α_try - β_try) / vol_ratio;
                ω2 = var_obs * (1 - α_try - β_try) * vol_ratio;
                for p_stick in [0.9, 0.98]
                    logit = log(p_stick / (1 - p_stick));
                    p = [ω1, α_try, β_try, ω2, α_try, β_try, logit, logit];
                    nll = _msgarch_nll_k2(p, obs, μ);
                    if nll < best_nll
                        best_nll = nll; best_params = copy(p);
                    end
                end
            end
        end
    end

    # Nelder-Mead refinement
    params = copy(best_params);
    d = length(params);
    simplex = [copy(params) for _ in 1:(d + 1)];
    for i in 2:(d + 1); simplex[i][i - 1] *= 1.15; end

    for _ in 1:max_iter
        vals = [_msgarch_nll_k2(s, obs, μ) for s in simplex];
        order = sortperm(vals);
        simplex = simplex[order];
        vals = vals[order];
        if abs(vals[end] - vals[1]) < 1e-6; break; end
        centroid = sum(simplex[1:d]) ./ d;
        reflected = centroid .+ (centroid .- simplex[end]);
        f_r = _msgarch_nll_k2(reflected, obs, μ);
        if f_r < vals[1]
            expanded = centroid .+ 2.0 .* (reflected .- centroid);
            f_e = _msgarch_nll_k2(expanded, obs, μ);
            simplex[end] = f_e < f_r ? expanded : reflected;
        elseif f_r < vals[d]
            simplex[end] = reflected;
        else
            contracted = centroid .+ 0.5 .* (simplex[end] .- centroid);
            f_c = _msgarch_nll_k2(contracted, obs, μ);
            if f_c < vals[end]
                simplex[end] = contracted;
            else
                for i in 2:(d + 1)
                    simplex[i] = simplex[1] .+ 0.5 .* (simplex[i] .- simplex[1]);
                end
            end
        end
    end

    vals = [_msgarch_nll_k2(s, obs, μ) for s in simplex];
    best = simplex[argmin(vals)];
    ω = [best[1], best[4]];
    α = [best[2], best[5]];
    β = [best[3], best[6]];
    p11 = 1.0 / (1.0 + exp(-best[7]));
    p22 = 1.0 / (1.0 + exp(-best[8]));
    T = [p11 (1-p11); (1-p22) p22];

    # Canonicalize: regime 1 is the calm regime (lower unconditional variance)
    uv = [ω[k] / max(1 - α[k] - β[k], 1e-6) for k in 1:2];
    if uv[1] > uv[2]
        ω = reverse(ω); α = reverse(α); β = reverse(β);
        T = [T[2, 2] T[2, 1]; T[1, 2] T[1, 1]];
    end

    ll, σ2_hist, γ = _hamilton_filter_msgarch(obs, ω, α, β, μ, T);

    m = MyMSGARCHModel();
    m.K = 2;
    m.ω = ω; m.α = α; m.β = β; m.μ = μ; m.T = T;
    m.σ2_histories = σ2_hist;
    m.gamma = γ;
    m.log_likelihood = ll;
    return m;
end

"""
    simulate_msgarch(m::MyMSGARCHModel, n_steps::Int) -> Vector{Float64}

Generate a return series from a fitted MS-GARCH model. Each regime's σ² is rolled
forward at each step; at each t we sample s_t from the Markov chain and draw r_t from the
active regime's conditional Normal.
"""
function simulate_msgarch(m::MyMSGARCHModel, n_steps::Int)::Vector{Float64}
    K = m.K;
    returns = zeros(n_steps);
    # Stationary start
    π_stat = ones(K) ./ K;
    for _ in 1:50; π_stat = m.T' * π_stat; π_stat = max.(π_stat, 1e-12); π_stat ./= sum(π_stat); end
    s = rand(Categorical(π_stat));
    # Initial σ² at regime-conditional unconditional variance
    σ2 = Vector{Float64}(undef, K);
    for k in 1:K; σ2[k] = m.ω[k] / max(1 - m.α[k] - m.β[k], 1e-6); end

    for t in 1:n_steps
        returns[t] = m.μ + sqrt(σ2[s]) * randn();
        r = returns[t] - m.μ;
        # Update σ² for all regimes
        for k in 1:K
            σ2[k] = m.ω[k] + m.α[k] * r^2 + m.β[k] * σ2[k];
            σ2[k] = max(σ2[k], 1e-12);
        end
        # Sample next regime
        s = rand(Categorical(m.T[s, :]));
    end
    return returns;
end

# ========================================================================================= #
# K = 3 extension (revision response to referee M7; "a single two-regime specification is
# not sufficient to represent that family"). Parameterises the K x K transition matrix by
# the K(K-1) off-diagonal logits of a softmax-normalised row, following the same
# unconstrained optimisation style as the K=2 logit-stick representation.
# ========================================================================================= #

# Transition-matrix packing for K = 3: 6 free parameters representing the three rows.
# We use diagonal-dominant parameterisation: each row k is (p_{k,1}, p_{k,2}, p_{k,3})
# where the two off-diagonal entries are exp(η_{k,j}) / (1 + Σ exp(η_{k,l})) and the
# diagonal is the residual, enforcing p_{k,k} > 0.5 tendency via the init.
function _unpack_T3(logits::Vector{Float64})::Matrix{Float64}
    # logits layout: 2 per row, total 6. Row k has logits (η_{k,j1}, η_{k,j2}) for the
    # two non-diagonal columns; the diagonal entry is 1 / (1 + Σ exp(η)).
    T = zeros(3, 3);
    for k in 1:3
        η1 = logits[2*(k-1) + 1];
        η2 = logits[2*(k-1) + 2];
        e1 = exp(η1); e2 = exp(η2);
        denom = 1.0 + e1 + e2;
        # Non-diagonal columns j1, j2 are (k mod 3)+1 and ((k+1) mod 3)+1.
        j1 = (k % 3) + 1;
        j2 = ((k + 1) % 3) + 1;
        T[k, k] = 1.0 / denom;
        T[k, j1] = e1 / denom;
        T[k, j2] = e2 / denom;
    end
    return T;
end

function _msgarch_nll_k3(params::Vector{Float64}, obs::Vector{Float64}, μ::Float64)
    ω = [params[1], params[4], params[7]];
    α = [params[2], params[5], params[8]];
    β = [params[3], params[6], params[9]];
    for k in 1:3
        if ω[k] <= 0 || α[k] < 0 || β[k] < 0 || (α[k] + β[k]) >= 0.999
            return 1e10;
        end
    end
    T = _unpack_T3(params[10:15]);
    ll, _, _ = _hamilton_filter_msgarch(obs, ω, α, β, μ, T);
    if !isfinite(ll); return 1e10; end
    return -ll;
end

"""
    fit_msgarch_k3(obs; max_iter=2000) -> MyMSGARCHModel

Three-regime MS-GARCH(1,1), fit by Nelder-Mead over 15 parameters: three regime
triples (ω, α, β) plus six transition-matrix logits. Grid-initialised over (calm /
medium / stress) volatility ratios. Regime canonicalisation sorts states by
unconditional variance ascending (state 1 = calm, state 3 = stress).
"""
function fit_msgarch_k3(obs::Vector{Float64}; max_iter::Int=3000)::MyMSGARCHModel
    μ = mean(obs);
    var_obs = var(obs);

    best_nll = Inf; best_params = zeros(15);
    for α_try in [0.05, 0.10]
        for β_try in [0.85, 0.90]
            denom = (1 - α_try - β_try);
            if denom < 0.05; continue; end
            # Three regimes at 0.5x, 1.0x, 2.5x the unconditional variance
            for (r1, r2, r3) in [(0.4, 1.0, 3.0), (0.5, 1.2, 2.5), (0.3, 1.0, 4.0)]
                ω1 = var_obs * denom * r1; ω2 = var_obs * denom * r2; ω3 = var_obs * denom * r3;
                # Diagonal-dominant transition init: diagonal ~0.9, each off-diagonal ~0.05.
                # Off-diagonal logit so that off-diag prob ~0.05 -> logit = log(0.05 / 0.90)
                η_init = log(0.05 / 0.90);
                logits = fill(η_init, 6);
                p = [ω1, α_try, β_try, ω2, α_try, β_try, ω3, α_try, β_try,
                     logits[1], logits[2], logits[3], logits[4], logits[5], logits[6]];
                nll = _msgarch_nll_k3(p, obs, μ);
                if nll < best_nll; best_nll = nll; best_params = copy(p); end
            end
        end
    end

    best, _ = begin
        d = length(best_params);
        simplex = [copy(best_params) for _ in 1:(d + 1)];
        for i in 2:(d + 1)
            step = abs(simplex[i][i-1]) > 1e-12 ? simplex[i][i-1] * 0.12 : 0.05;
            simplex[i][i-1] += step;
        end
        for _ in 1:max_iter
            vals = [_msgarch_nll_k3(s, obs, μ) for s in simplex];
            order = sortperm(vals);
            simplex = simplex[order];
            vals = vals[order];
            if abs(vals[end] - vals[1]) < 1e-6; break; end
            centroid = sum(simplex[1:d]) ./ d;
            reflected = centroid .+ (centroid .- simplex[end]);
            f_r = _msgarch_nll_k3(reflected, obs, μ);
            if f_r < vals[1]
                expanded = centroid .+ 2.0 .* (reflected .- centroid);
                f_e = _msgarch_nll_k3(expanded, obs, μ);
                simplex[end] = f_e < f_r ? expanded : reflected;
            elseif f_r < vals[d]
                simplex[end] = reflected;
            else
                contracted = centroid .+ 0.5 .* (simplex[end] .- centroid);
                f_c = _msgarch_nll_k3(contracted, obs, μ);
                if f_c < vals[end]
                    simplex[end] = contracted;
                else
                    for i in 2:(d + 1)
                        simplex[i] = simplex[1] .+ 0.5 .* (simplex[i] .- simplex[1]);
                    end
                end
            end
        end
        vals = [_msgarch_nll_k3(s, obs, μ) for s in simplex];
        simplex[argmin(vals)], minimum(vals);
    end

    ω = [best[1], best[4], best[7]];
    α = [best[2], best[5], best[8]];
    β = [best[3], best[6], best[9]];
    T = _unpack_T3(best[10:15]);

    # Canonicalise states by ascending unconditional variance.
    uv = [ω[k] / max(1 - α[k] - β[k], 1e-6) for k in 1:3];
    order = sortperm(uv);
    ω = ω[order]; α = α[order]; β = β[order];
    P = zeros(3, 3);
    for i in 1:3, j in 1:3
        P[i, j] = T[order[i], order[j]];
    end
    T = P;

    ll, σ2_hist, γ = _hamilton_filter_msgarch(obs, ω, α, β, μ, T);

    m = MyMSGARCHModel();
    m.K = 3;
    m.ω = ω; m.α = α; m.β = β; m.μ = μ; m.T = T;
    m.σ2_histories = σ2_hist;
    m.gamma = γ;
    m.log_likelihood = ll;
    return m;
end
