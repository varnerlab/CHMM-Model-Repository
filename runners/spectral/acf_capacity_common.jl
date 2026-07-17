# ========================================================================================= #
# acf_capacity_common.jl — realizable ACF-targeted HMM optimization core.
#
# Directly optimizes a VALID stationary Gaussian-emission K-state HMM so that its
# population |G| ACF (the paper's spectral identity, computed exactly from (T, μ, σ))
# best matches an observed sample ACF. Unlike a free-exponential curve fit, every
# candidate here is a realizable HMM: T is row-stochastic with strictly positive
# entries (row-wise softmax of logits), σ > 0 (log-parametrized), and the stationary
# law is the exact left eigenvector of T. What this optimizer attains is therefore a
# certificate of ATTAINABILITY for the K-state class on the target curve — no claim
# about global optimality is made (multistart, finite-difference Adam), so the
# attained error is an upper bound on the class's best error, which is the direction
# needed for "a K-state HMM can track this curve at least this well".
#
# Objective: SSE between the model population ACF and the sample ACF over lags
# 1..maxlag (smooth, band-agnostic; near/far-band MAEs are reported, not optimized).
# The marginal is left unconstrained and disclosed by the caller. Note the |G| ACF is
# invariant to a common positive scaling of all (μ_k, σ_k) — a harmless flat
# direction for the optimizer; disclosure statistics must be scale-free.
#
# No new dependencies: hand-rolled Adam on central finite-difference gradients
# (the objective is smooth: softmax, exp, linear solve, polynomial recursion; one
# evaluation is O(K^3 + maxlag·K^2), ~microseconds at K = 18). Optim.jl/ForwardDiff
# are deliberately not used — they are not direct project dependencies.
# ========================================================================================= #

using Random, LinearAlgebra, Statistics

"""
    _unpack_acf_params(θ, K) -> (T, μ, σ)

θ layout: K·(K-1) transition logits (row-major; the logit of column K is pinned
to 0 per row, removing the softmax null direction), then μ (K), then log σ (K).
"""
function _unpack_acf_params(θ::Vector{Float64}, K::Int)
    nT = K * (K - 1);
    T = Matrix{Float64}(undef, K, K);
    logits = Vector{Float64}(undef, K);
    @inbounds for i in 1:K
        for j in 1:(K-1)
            logits[j] = θ[(i-1)*(K-1) + j];
        end
        logits[K] = 0.0;
        mx = maximum(logits);
        s = 0.0;
        for j in 1:K
            T[i, j] = exp(logits[j] - mx);
            s += T[i, j];
        end
        for j in 1:K
            T[i, j] /= s;
        end
    end
    μ = θ[nT+1 : nT+K];
    σ = exp.(θ[nT+K+1 : nT+2K]);
    return T, μ, σ;
end

"""
    _pack_acf_params(T, μ, σ) -> θ

Inverse of `_unpack_acf_params` up to the softmax row-shift gauge (row logits are
log(T[i,j]/T[i,K])); entries are clamped at 1e-12 before the ratio so seeding from
fitted transition matrices with (near-)zero entries is safe.
"""
function _pack_acf_params(T::AbstractMatrix{Float64}, μ::AbstractVector{Float64},
                          σ::AbstractVector{Float64})
    K = length(μ);
    θ = Vector{Float64}(undef, K*(K-1) + 2K);
    @inbounds for i in 1:K, j in 1:(K-1)
        θ[(i-1)*(K-1) + j] = log(max(T[i, j], 1e-12) / max(T[i, K], 1e-12));
    end
    θ[K*(K-1)+1 : K*(K-1)+K] .= μ;
    θ[K*(K-1)+K+1 : end] .= log.(max.(σ, 1e-12));
    return θ;
end

"""
    _stationary_pi_unchecked(T) -> π̄

Constrained left-eigenvector solve as in `_stationary_pi` (spectral_common.jl) but
without the eigvals uniqueness check and asserts, for use inside the optimization
loop: softmax rows are strictly positive, so the chain is irreducible and aperiodic
and the stationary law is unique. Call the checked `_stationary_pi` once on the
reported optimum.
"""
function _stationary_pi_unchecked(T::AbstractMatrix{Float64})
    K = size(T, 1);
    A = vcat(transpose(T) - I, ones(1, K));
    b = vcat(zeros(K), 1.0);
    π̄ = A \ b;
    π̄ = max.(π̄, 0.0);
    π̄ ./= sum(π̄);
    return π̄;
end

"""
    _folded_moments(μ, σ) -> (m, M)

Analytic per-state moments of a Gaussian emission: m_k = E[|G| | s = k]
(folded-normal mean) and M_k = E[G² | s = k] = μ_k² + σ_k². Same formulas as
`_T_pibar_m` (spectral_common.jl).
"""
function _folded_moments(μ::AbstractVector{Float64}, σ::AbstractVector{Float64})
    K = length(μ);
    m = zeros(K); M = zeros(K);
    @inbounds for k in 1:K
        z = μ[k] / σ[k];
        m[k] = σ[k] * sqrt(2 / π) * exp(-z^2 / 2) + μ[k] * (1 - 2 * cdf(Normal(), -z));
        M[k] = μ[k]^2 + σ[k]^2;
    end
    return m, M;
end

"""
    _population_acf(T, π̄, m, M, maxlag) -> Vector{Float64}

Population |G| ACF ρ(τ) = (m' diag(π̄) T^τ m − (π̄'m)²) / (π̄'M − (π̄'m)²) by the
vector recursion u ← T u (O(maxlag·K²); no matrix powers, no eigendecomposition).
Agrees with `_theoretical_acf` (spectral_common.jl) to machine precision.
"""
function _population_acf(T::AbstractMatrix{Float64}, π̄::AbstractVector{Float64},
                         m::AbstractVector{Float64}, M::AbstractVector{Float64},
                         maxlag::Int)
    μG = dot(π̄, m);
    σ²G = dot(π̄, M) - μG^2;
    w = π̄ .* m;
    u = copy(m);
    ρ = zeros(maxlag);
    @inbounds for τ in 1:maxlag
        u = T * u;
        ρ[τ] = (dot(w, u) - μG^2) / σ²G;
    end
    return ρ;
end

"""
    _acf_objective(θ, ρ̂, K) -> Float64

SSE of the model population ACF against the target sample ACF over 1..length(ρ̂).
Degenerate parameter points (non-finite θ or vanishing |G| variance) return a
finite penalty (1e6) so finite-difference steps move away without NaN propagation.
"""
function _acf_objective(θ::Vector{Float64}, ρ̂::Vector{Float64}, K::Int)
    all(isfinite, θ) || return 1e6;
    T, μ, σ = _unpack_acf_params(θ, K);
    all(isfinite, T) || return 1e6;
    π̄ = _stationary_pi_unchecked(T);
    all(isfinite, π̄) || return 1e6;
    m, M = _folded_moments(μ, σ);
    μG = dot(π̄, m);
    σ²G = dot(π̄, M) - μG^2;
    (isfinite(σ²G) && σ²G > 1e-14 * max(dot(π̄, M), eps())) || return 1e6;
    ρ = _population_acf(T, π̄, m, M, length(ρ̂));
    sse = sum(abs2, ρ .- ρ̂);
    return isfinite(sse) ? sse : 1e6;
end

"""
    _fd_gradient!(g, f, θ; h=1e-6)

Central finite-difference gradient of f at θ, written into g (θ is restored).
"""
function _fd_gradient!(g::Vector{Float64}, f, θ::Vector{Float64}; h::Float64=1e-6)
    @inbounds for i in eachindex(θ)
        old = θ[i];
        θ[i] = old + h; fp = f(θ);
        θ[i] = old - h; fm = f(θ);
        θ[i] = old;
        g[i] = (fp - fm) / (2h);
    end
    return g;
end

"""
    _adam_minimize(f, θ0; lr=0.05, n_iter=4000, patience=300, rel_tol=1e-9, h=1e-6)
        -> (θ_best, f_best, n_used, converged)

Adam (β₁ = 0.9, β₂ = 0.999) on finite-difference gradients, learning rate halved
every 1,000 iterations; returns the best-evaluated iterate (never the raw last
step). `converged` is true when the best value has not improved by a relative
`rel_tol` for `patience` consecutive iterations (early stop); false means the
iteration cap was reached.
"""
function _adam_minimize(f, θ0::Vector{Float64}; lr::Float64=0.05, n_iter::Int=4000,
                        patience::Int=300, rel_tol::Float64=1e-9, h::Float64=1e-6)
    θ = copy(θ0);
    n = length(θ);
    mvec = zeros(n); vvec = zeros(n); g = zeros(n);
    β1, β2, ϵ = 0.9, 0.999, 1e-8;
    f_best = f(θ); θ_best = copy(θ);
    stall = 0; it_used = 0; converged = false;
    for it in 1:n_iter
        it_used = it;
        _fd_gradient!(g, f, θ; h=h);
        lr_t = lr * 0.5^div(it - 1, 1000);
        @inbounds for i in 1:n
            mvec[i] = β1 * mvec[i] + (1 - β1) * g[i];
            vvec[i] = β2 * vvec[i] + (1 - β2) * g[i]^2;
            mh = mvec[i] / (1 - β1^it);
            vh = vvec[i] / (1 - β2^it);
            θ[i] -= lr_t * mh / (sqrt(vh) + ϵ);
        end
        fv = f(θ);
        if fv < f_best - rel_tol * max(abs(f_best), 1e-12)
            f_best = fv; copyto!(θ_best, θ); stall = 0;
        else
            if fv < f_best
                f_best = fv; copyto!(θ_best, θ);
            end
            stall += 1;
            if stall >= patience
                converged = true;
                break;
            end
        end
    end
    return θ_best, f_best, it_used, converged;
end

"""
    fit_acf_hmm(ρ̂, K; n_starts, seed, R=nothing, ml_seed=nothing, n_iter=4000)
        -> NamedTuple

Multistart ACF-targeted fit of a valid K-state Gaussian-emission HMM to the target
sample ACF ρ̂. Starts:
  1. deterministic sticky quantile start (diag 0.95; μ, σ from K sorted chunks of
     R when provided, a fixed spread otherwise);
  2. optional `ml_seed = (T = ..., μ = ..., σ = ...)` from a likelihood fit —
     starting Adam there guarantees the attained SSE is no worse than the
     likelihood fit's own population-ACF SSE on this target;
  3. random persistence-diverse starts: row-softmax of δ·I + N(0,1) logits with
     per-start δ ~ U(0, 6) (seeds both fast- and slow-mixing chains), jittered
     quantile emissions; rng = MersenneTwister(seed + 7919·start).

Returns the best start's realizable HMM with band MAEs, the optimized population
ACF, eigenvalue magnitudes of T (unit eigenvalue excluded), and per-start
diagnostics (start, sse_init, sse_final, n_iter, converged).
"""
function fit_acf_hmm(ρ̂::Vector{Float64}, K::Int; n_starts::Int, seed::Int,
                     R::Union{Nothing,Vector{Float64}}=nothing,
                     ml_seed::Union{Nothing,NamedTuple}=nothing, n_iter::Int=4000)
    f = θ -> _acf_objective(θ, ρ̂, K);

    if R !== nothing
        n = length(R); perm = sortperm(R);
        chunks = [perm[(i-1)*n÷K + 1 : i*n÷K] for i in 1:K];
        μq = [mean(R[c]) for c in chunks];
        σq = [max(std(R[c]), 1e-4) for c in chunks];
    else
        μq = collect(range(-1.0, 1.0, length=K));
        σq = collect(range(0.5, 2.0, length=K));
    end

    starts = Vector{Vector{Float64}}();
    T_sticky = fill(0.05 / (K - 1), K, K);
    @inbounds for i in 1:K; T_sticky[i, i] = 0.95; end
    push!(starts, _pack_acf_params(T_sticky, μq, σq));
    if ml_seed !== nothing
        push!(starts, _pack_acf_params(ml_seed.T, ml_seed.μ, ml_seed.σ));
    end
    s_idx = length(starts);
    while length(starts) < n_starts
        s_idx += 1;
        rng = MersenneTwister(seed + 7919 * s_idx);
        δ = 6.0 * rand(rng);
        Λ = δ .* Matrix{Float64}(I, K, K) .+ randn(rng, K, K);
        Trand = exp.(Λ .- maximum(Λ, dims=2));
        Trand ./= sum(Trand, dims=2);
        μ0 = μq .+ 0.5 .* σq .* randn(rng, K);
        σ0 = σq .* exp.(0.5 .* randn(rng, K));
        push!(starts, _pack_acf_params(Trand, μ0, σ0));
    end

    diags = NamedTuple[];
    best_θ = nothing; best_sse = Inf; best_start = 0;
    for (si, θ0) in enumerate(starts)
        sse0 = f(θ0);
        θb, fb, nit, conv = _adam_minimize(f, θ0; n_iter=n_iter);
        push!(diags, (start=si, sse_init=sse0, sse_final=fb, n_iter=nit, converged=conv));
        if fb < best_sse
            best_sse = fb; best_θ = θb; best_start = si;
        end
    end

    T, μ, σ = _unpack_acf_params(best_θ, K);
    π̄ = _stationary_pi_unchecked(T);
    m, M = _folded_moments(μ, σ);
    ρ = _population_acf(T, π̄, m, M, length(ρ̂));
    nb = min(63, length(ρ̂));
    near = mean(abs.(ρ[1:nb] .- ρ̂[1:nb]));
    far = length(ρ̂) > 63 ? mean(abs.(ρ[64:end] .- ρ̂[64:end])) : NaN;
    lam = sort(abs.(eigvals(T)); rev=true)[2:end];
    return (T=T, π̄=π̄, μ=μ, σ=σ, sse=best_sse, near_mae=near, far_mae=far,
            best_start=best_start, n_starts=length(starts),
            converged=diags[best_start].converged, n_iter=diags[best_start].n_iter,
            diagnostics=diags, abs_lams=lam, rho=ρ);
end
