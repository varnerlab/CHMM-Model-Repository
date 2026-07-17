# =========================================================================== #
# hsmm_core.jl — explicit-duration HSMM core (Yu 2010 forward-backward over
# (state, duration) pairs), Gaussian emissions, off-diagonal transitions,
# truncated discrete Pareto sojourns.
#
# Finite-window convention (2026-07-16 seventh review, finding 3): the terminal
# segment is RIGHT-CENSORED. The likelihood marginalizes over an ongoing final
# sojourn via the duration survival function S_α(d) = P(D ≥ d),
#
#   L = Σ_s Σ_{dc} P(O_{1:T-dc}, sojourn of s starts at T-dc+1) · S_s(dc)
#         · Π_{τ=T-dc+1..T} b_s(O_τ),
#
# replacing the earlier exact-boundary convention ll = logsumexp(logf[:, T]),
# which conditioned every segmentation on a sojourn ending exactly at T and
# therefore did not match the simulator (which draws an ordinary duration and
# right-truncates at T). Under this censored convention the estimator and the
# simulator describe the same finite-window model. The duration M-step gains
# the corresponding expected censored counts (library
# fit_truncated_pareto_alpha with censored_counts).
#
# Initial-segment convention: the first sojourn starts at t = 1 (no equilibrium
# left-censoring); this is the standard simplification and is stated in the
# artifact headers.
#
# This file is include-able with no script side effects (all sizes and
# tolerances are explicit arguments) so tests can enumerate tiny problems
# against the recursions directly (test/test_hsmm_core.jl).
# =========================================================================== #

using Random, Statistics, LinearAlgebra, Printf

# --------------------------------------------------------------------------- #
# Gaussian log-density helper
# --------------------------------------------------------------------------- #
@inline function _logpdf_gauss(x, μ, σ)
    z = (x - μ) / σ;
    return -0.5 * z * z - log(σ) - 0.5 * log(2π);
end

function _logsumexp(x::AbstractVector{Float64})
    isempty(x) && return -Inf;
    m = maximum(x); m == -Inf && return -Inf;
    return m + log(sum(exp.(x .- m)));
end

function _logsumexp_pair(a::Float64, b::Float64)
    a == -Inf && return b;
    b == -Inf && return a;
    if a > b
        return a + log1p(exp(b - a));
    else
        return b + log1p(exp(a - b));
    end
end

# --------------------------------------------------------------------------- #
# HSMM model container
# --------------------------------------------------------------------------- #
mutable struct MLHSMM
    K::Int
    π::Vector{Float64}                      # initial state distribution
    A::Matrix{Float64}                      # off-diagonal transition matrix (rows sum to 1, A[k,k] = 0)
    μ::Vector{Float64}                      # state means
    σ::Vector{Float64}                      # state stds
    α::Vector{Float64}                      # per-state Pareto α
    log_p::Matrix{Float64}                  # log p_s(d), size K × D
    ll_history::Vector{Float64}
end

function _init_hsmm(R::Vector{Float64}, K::Int, D::Int;
                    init::Union{Nothing,NamedTuple}=nothing,
                    log_p0::Union{Nothing,Matrix{Float64}}=nothing)
    if init !== nothing
        log_p = zeros(K, D);
        @inbounds for s in 1:K
            log_p[s, :] = truncated_pareto_logpmf(init.α[s], D);
        end
        return MLHSMM(K, copy(init.π), copy(init.A), copy(init.μ), copy(init.σ),
                      copy(init.α), log_p, Float64[]);
    end
    # Quantile-based init: sort R into K equal chunks, each chunk gives state μ, σ
    perm = sortperm(R);
    n = length(R);
    chunks = [perm[(i-1)*n÷K + 1 : i*n÷K] for i in 1:K];
    μ = [mean(R[c]) for c in chunks];
    σ = [max(std(R[c]), 0.5 * std(R)) for c in chunks];
    π = fill(1.0 / K, K);
    # Off-diagonal init: uniform over j ≠ i
    A = fill(1.0 / (K - 1), K, K);
    @inbounds for i in 1:K; A[i, i] = 0.0; end
    α = fill(1.5, K);                       # Pareto α init
    log_p = zeros(K, D);
    if log_p0 !== nothing
        log_p .= log_p0;                    # caller-supplied duration-law init
    else
        @inbounds for s in 1:K
            log_p[s, :] = truncated_pareto_logpmf(α[s], D);
        end
    end
    return MLHSMM(K, π, A, μ, σ, α, log_p, Float64[]);
end

# --------------------------------------------------------------------------- #
# E-step: forward-backward over (state, duration), right-censored terminal
# segment.
#
# Conventions (log-space):
#   logb[s, t]        = log b_s(O_t)
#   seg_b(s, t, d)    = Σ_{τ=t-d+1..t} logb[s, τ]
#   log_fstart[s, t0] = log P(O_{1:t0-1}, sojourn of s starts at t0)
#   logf[s, t]        = log P(O_{1:t}, sojourn of s ENDS at t)   (t = 1..T-1)
#   logg[s, t+1]      = log P(O_{t+1:T} | sojourn of s starts at t+1)
#   log_gnext[s, u]   = logsumexp_{s'≠s}( log_A[s, s'] + logg[s', u] )
#   log_S[s, d]       = log P(D_s ≥ d)  (survival of the truncated law, 1..D)
# --------------------------------------------------------------------------- #
function _e_step(m::MLHSMM, R::Vector{Float64}, D::Int)
    K = m.K; T = length(R);
    Dt = min(D, T);
    logb = Matrix{Float64}(undef, K, T);
    @inbounds for s in 1:K, t in 1:T
        logb[s, t] = _logpdf_gauss(R[t], m.μ[s], m.σ[s]);
    end
    cum_b = zeros(K, T + 1);
    @inbounds for s in 1:K, t in 1:T
        cum_b[s, t + 1] = cum_b[s, t] + logb[s, t];
    end
    seg_b = (s, t, d) -> cum_b[s, t + 1] - cum_b[s, t - d + 1];

    # Per-state log-survival of the truncated duration law (full 1..D support).
    log_S = Matrix{Float64}(undef, K, D);
    @inbounds for s in 1:K
        log_S[s, D] = m.log_p[s, D];
        for d in (D-1):-1:1
            log_S[s, d] = _logsumexp_pair(m.log_p[s, d], log_S[s, d + 1]);
        end
        log_S[s, 1] = min(log_S[s, 1], 0.0);
    end

    log_π = log.(m.π);
    log_A = log.(max.(m.A, 1e-300));

    # Forward pass: log_fstart interleaved with logf (logf needed through T-1;
    # log_fstart needed through T for the censored likelihood).
    logf       = fill(-Inf, K, T);
    log_fstart = fill(-Inf, K, T);
    @inbounds for t in 1:T
        for s in 1:K
            if t == 1
                log_fstart[s, t] = log_π[s];
            else
                acc = -Inf;
                for sp in 1:K
                    sp == s && continue;
                    acc = _logsumexp_pair(acc, logf[sp, t - 1] + log_A[sp, s]);
                end
                log_fstart[s, t] = acc;
            end
        end
        t == T && break;   # logf[:, T] is not part of the censored likelihood
        for s in 1:K
            acc = -Inf;
            for d in 1:min(t, Dt)
                acc = _logsumexp_pair(acc,
                        log_fstart[s, t - d + 1] + m.log_p[s, d] + seg_b(s, t, d));
            end
            logf[s, t] = acc;
        end
    end

    # Censored likelihood: terminal segment of observed length dc, survival-weighted.
    ll = -Inf;
    @inbounds for s in 1:K, dc in 1:Dt
        ll = _logsumexp_pair(ll,
                log_fstart[s, T - dc + 1] + log_S[s, dc] + seg_b(s, T, dc));
    end

    # Backward pass with censored terminal branch.
    logg      = fill(-Inf, K, T + 1);
    log_gnext = fill(-Inf, K, T + 1);
    @inbounds for t in (T-1):-1:0
        for s in 1:K
            acc = -Inf;
            for d in 1:min(Dt, T - t - 1)
                acc = _logsumexp_pair(acc,
                        m.log_p[s, d] + seg_b(s, t + d, d) + log_gnext[s, t + d + 1]);
            end
            if T - t <= Dt
                acc = _logsumexp_pair(acc, log_S[s, T - t] + seg_b(s, T, T - t));
            end
            logg[s, t + 1] = acc;
        end
        for s in 1:K
            acc = -Inf;
            for sp in 1:K
                sp == s && continue;
                acc = _logsumexp_pair(acc, log_A[s, sp] + logg[sp, t + 1]);
            end
            log_gnext[s, t + 1] = acc;
        end
    end

    # Completed-sojourn posterior η_t(s, d) for sojourns ending at t ≤ T-1.
    log_eta = fill(-Inf, K, T, Dt);
    @inbounds for t in 1:(T-1)
        for s in 1:K
            gp = log_gnext[s, t + 1];
            for d in 1:min(t, Dt)
                log_eta[s, t, d] = log_fstart[s, t - d + 1] + m.log_p[s, d] +
                                   seg_b(s, t, d) + gp;
            end
        end
    end
    log_eta .-= ll;

    # Censored terminal posterior η_c(s, dc); Σ exp = 1 (every path has exactly
    # one terminal segment).
    log_eta_c = fill(-Inf, K, Dt);
    @inbounds for s in 1:K, dc in 1:Dt
        log_eta_c[s, dc] = log_fstart[s, T - dc + 1] + log_S[s, dc] +
                           seg_b(s, T, dc) - ll;
    end

    # State occupancy γ: completed sojourns spread over their spans, censored
    # terminal segments over T-dc+1..T. Σ_s γ[s, t] = 1 for every t.
    log_γ = fill(-Inf, K, T);
    @inbounds for s in 1:K, t_end in 1:(T-1)
        for d in 1:min(t_end, Dt)
            le = log_eta[s, t_end, d];
            le == -Inf && continue;
            for τ in (t_end - d + 1):t_end
                log_γ[s, τ] = _logsumexp_pair(log_γ[s, τ], le);
            end
        end
    end
    @inbounds for s in 1:K, dc in 1:Dt
        le = log_eta_c[s, dc];
        le == -Inf && continue;
        for τ in (T - dc + 1):T
            log_γ[s, τ] = _logsumexp_pair(log_γ[s, τ], le);
        end
    end

    # Boundary transition posterior ξ_t(s', s) — inherits censoring through logg.
    log_ξ = fill(-Inf, K, K, T);
    @inbounds for t in 1:(T-1)
        for sp in 1:K, s in 1:K
            sp == s && continue;
            log_ξ[sp, s, t] = logf[sp, t] + log_A[sp, s] + logg[s, t + 1];
        end
    end
    log_ξ .-= ll;

    return (ll=ll, log_γ=log_γ, log_eta=log_eta, log_eta_c=log_eta_c,
            log_ξ=log_ξ, logg=logg, log_fstart=log_fstart);
end

# --------------------------------------------------------------------------- #
# M-step
# --------------------------------------------------------------------------- #
function _m_step!(m::MLHSMM, R::Vector{Float64}, post, D::Int;
                  alpha_bounds::Tuple{Float64,Float64}=(0.05, 8.0),
                  fit_duration!::Union{Nothing,Function}=nothing)
    K = m.K; T = length(R); Dt = min(D, T);
    γ = exp.(post.log_γ);
    # Emissions: weighted Gaussian MLE on γ
    @inbounds for s in 1:K
        ws = γ[s, :]; sw = sum(ws);
        if sw < 1e-9
            continue;
        end
        m.μ[s] = sum(ws .* R) / sw;
        m.σ[s] = sqrt(max(sum(ws .* (R .- m.μ[s]).^2) / sw, 1e-10));
    end

    # Initial: mass that the first sojourn is in state s — completed first
    # sojourns η(s, d, d) plus the whole-series censored term η_c(s, T).
    log_π_new = fill(-Inf, K);
    @inbounds for s in 1:K
        for d in 1:min(Dt, T - 1)
            log_π_new[s] = _logsumexp_pair(log_π_new[s], post.log_eta[s, d, d]);
        end
        if T <= Dt
            log_π_new[s] = _logsumexp_pair(log_π_new[s], post.log_eta_c[s, T]);
        end
    end
    π_new = exp.(log_π_new); π_new ./= max(sum(π_new), 1e-300);
    m.π .= π_new;

    # Transitions: A[s', s] ∝ sum_t exp(log_ξ[s', s, t])
    A_new = zeros(K, K);
    @inbounds for sp in 1:K, s in 1:K
        sp == s && continue;
        s_total = -Inf;
        for t in 1:(T-1)
            s_total = _logsumexp_pair(s_total, post.log_ξ[sp, s, t]);
        end
        A_new[sp, s] = exp(s_total);
    end
    @inbounds for sp in 1:K
        rs = sum(A_new[sp, :]);
        if rs > 1e-12
            A_new[sp, :] ./= rs;
        else
            A_new[sp, :] .= 1.0 / (K - 1);
            A_new[sp, sp] = 0.0;
        end
        A_new[sp, sp] = 0.0;
    end
    m.A .= A_new;

    # Sojourn distribution: censored Pareto MLE on expected completed counts w_d
    # plus expected censored terminal counts c_d (padded to the full law support D).
    w = zeros(K, D);
    c = zeros(K, D);
    @inbounds for s in 1:K, t in 1:(T-1)
        for d in 1:min(t, Dt)
            w[s, d] += exp(post.log_eta[s, t, d]);
        end
    end
    @inbounds for s in 1:K, dc in 1:Dt
        c[s, dc] = exp(post.log_eta_c[s, dc]);
    end
    if fit_duration! !== nothing
        # Pluggable duration M-step (e.g. discrete-Gamma method-of-moments in
        # run_hsmm_gamma.jl): updates the model's duration parameters and log_p
        # from the expected completed counts w and censored counts c.
        fit_duration!(m, w, c, D);
    else
        @inbounds for s in 1:K
            if sum(w[s, :]) + sum(c[s, :]) > 1e-9
                m.α[s] = fit_truncated_pareto_alpha(w[s, :], D;
                            censored_counts=c[s, :], bounds=alpha_bounds);
                m.log_p[s, :] = truncated_pareto_logpmf(m.α[s], D);
            end
        end
    end
    return nothing;
end

# --------------------------------------------------------------------------- #
# EM driver (evaluate-before-update contract: returned parameters are always
# the last EVALUATED iterate, whose likelihood is ll_history[end]).
# --------------------------------------------------------------------------- #
function fit_hsmm_ml(R::Vector{Float64}, K::Int; D::Int, max_iter::Int, tol::Float64,
                     alpha_bounds::Tuple{Float64,Float64}=(0.05, 8.0),
                     init::Union{Nothing,NamedTuple}=nothing,
                     log_p0::Union{Nothing,Matrix{Float64}}=nothing,
                     fit_duration!::Union{Nothing,Function}=nothing,
                     verbose::Bool=true)
    verbose && println("[hsmm-ml] K=$K, T=$(length(R)), D=$D ...");
    m = _init_hsmm(R, K, D; init=init, log_p0=log_p0);
    last_ll = -Inf;
    for it in 1:(max_iter + 1)
        post = _e_step(m, R, D);
        ll = post.ll;
        push!(m.ll_history, ll);
        verbose && @printf("  [%2d] log-lik = %.4f  (per-obs %.5f)\n", it, ll, ll / length(R));
        if (abs(ll - last_ll) / max(length(R), 1) < tol && it > 4) || (it == max_iter + 1)
            verbose && println(it == max_iter + 1 ? "  → iteration cap" : "  → converged at iter $it");
            break;
        end
        last_ll = ll;
        _m_step!(m, R, post, D; alpha_bounds=alpha_bounds, fit_duration! = fit_duration!);
    end
    return m;
end

"""
    fit_hsmm_ml_multistart(R, K; n_starts, D, max_iter, tol, seed=0, jitter=0.25,
                           alpha_bounds=(0.05, 8.0), verbose=true)
        -> (model, diagnostics)

Multistart wrapper: start 1 is the canonical quantile init; starts ≥ 2 jitter the
quantile emissions, the off-diagonal transition rows, and the duration exponents
(rng = MersenneTwister(seed + 7919·start), the baum_welch_multistart pattern).
Returns the best-likelihood model (last-evaluated-iterate contract preserved) and
per-start diagnostics (start, ll, n_evals, final_increment, converged) with
`converged` under the per-observation tolerance rule.
"""
function fit_hsmm_ml_multistart(R::Vector{Float64}, K::Int; n_starts::Int, D::Int,
                                max_iter::Int, tol::Float64, seed::Int=0,
                                jitter::Float64=0.25,
                                alpha_bounds::Tuple{Float64,Float64}=(0.05, 8.0),
                                verbose::Bool=true)
    n = length(R); perm = sortperm(R);
    chunks = [perm[(i-1)*n÷K + 1 : i*n÷K] for i in 1:K];
    μq = [mean(R[c]) for c in chunks];
    σq = [max(std(R[c]), 0.5 * std(R)) for c in chunks];

    best = nothing;
    diagnostics = NamedTuple[];
    for s_idx in 1:n_starts
        init = if s_idx == 1
            nothing;
        else
            rng = MersenneTwister(seed + 7919 * s_idx);
            μ0 = μq .+ jitter .* σq .* randn(rng, K);
            σ0 = max.(σq .* exp.(jitter .* randn(rng, K)), 1e-6);
            A0 = abs.(1.0 / (K - 1) .+ (jitter / (K - 1)) .* randn(rng, K, K));
            @inbounds for i in 1:K; A0[i, i] = 0.0; end
            A0 ./= sum(A0, dims=2);
            α0 = clamp.(1.5 .* exp.(0.5 .* randn(rng, K)), alpha_bounds[1], alpha_bounds[2]);
            (π = fill(1.0 / K, K), A = A0, μ = μ0, σ = σ0, α = α0);
        end
        verbose && println("[hsmm-ml multistart] start $s_idx / $n_starts");
        mdl = fit_hsmm_ml(R, K; D=D, max_iter=max_iter, tol=tol,
                          alpha_bounds=alpha_bounds, init=init, verbose=verbose);
        h = mdl.ll_history;
        inc = length(h) >= 2 ? h[end] - h[end-1] : 0.0;
        push!(diagnostics, (start=s_idx, ll=h[end], n_evals=length(h),
                            final_increment=inc,
                            converged=abs(inc) / max(length(R), 1) < tol));
        if best === nothing || h[end] > best.ll_history[end]
            best = mdl;
        end
    end
    return best, diagnostics;
end

# --------------------------------------------------------------------------- #
# Simulation. Draws ordinary durations and right-truncates the final sojourn at
# T — consistent with the right-censored likelihood convention above.
# --------------------------------------------------------------------------- #
function simulate_hsmm(m::MLHSMM; T::Int, D::Int, n_paths::Int)
    K = m.K; out = Matrix{Float64}(undef, T, n_paths);
    cum_p = Matrix{Float64}(undef, K, D);
    @inbounds for s in 1:K
        ps = exp.(m.log_p[s, :]); ps ./= sum(ps);
        cum_p[s, :] = cumsum(ps);
    end
    cum_A = Matrix{Float64}(undef, K, K);
    @inbounds for s in 1:K
        cum_A[s, :] = cumsum(m.A[s, :]);
    end
    cum_π = cumsum(m.π);
    for p in 1:n_paths
        path = Vector{Float64}(undef, T);
        s = searchsortedfirst(cum_π, rand());
        s = clamp(s, 1, K);
        t = 1;
        while t <= T
            d = searchsortedfirst(cum_p[s, :], rand());
            d = clamp(d, 1, D);
            t_end = min(t + d - 1, T);
            for τ in t:t_end
                path[τ] = m.μ[s] + m.σ[s] * randn();
            end
            t = t_end + 1;
            if t <= T
                u = rand();
                # row of A may sum to 1 with diagonal zero; cum_A handles it
                s_new = searchsortedfirst(cum_A[s, :], u);
                s_new = clamp(s_new, 1, K);
                if s_new == s
                    # rare guard against numerical edge: pick a uniform alternative
                    s_new = mod(s, K) + 1;
                end
                s = s_new;
            end
        end
        out[:, p] = path;
    end
    return out;
end
