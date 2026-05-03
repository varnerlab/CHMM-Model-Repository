# =========================================================================== #
# run_hsmm_ml_intermediate_K.jl
#
# Peer-review item P3.3 (R1.W4, R3.W6). Same ML HSMM as run_hsmm_ml.jl but at
# intermediate K ∈ {6, 9, 12} to address the comparison gap: the body Table 1
# reports HSMM at K = 3 (clean OoS KS = 91%) and the appendix notes that the
# K = 18 HSMM "collapses to a near-degenerate optimum." This script fills the
# K = 6, 9, 12 rows so the CHMM-vs-HSMM comparison is at like-for-like K.
#
# Same protocol as run_hsmm_ml.jl: Yu (2010) forward-backward over (state,
# duration) pairs, Gaussian emissions, off-diagonal transitions, truncated
# discrete Pareto sojourns with D_max = 200, Pareto α clamped to [0.3, 5.0]
# (already a regularizer; if the K = 9, 12 fits also collapse we report that
# explicitly rather than stretching the comparison).
#
# Outputs:
#   results/hsmm_ml/hsmm_ml_K6.jld2, K9.jld2, K12.jld2
#   results/hsmm_ml/hsmm_ml_intermediate_K_metrics.csv
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random
using Statistics
using StatsBase
using LinearAlgebra
using Printf
using JLD2
using DelimitedFiles
using Distributions

const SEED        = 20260420;
const TICKER      = "SPY";
const RISK_FREE   = 0.0;
const DT          = 1/252;
const N_PATHS     = 1000;
const L_LAGS      = 252;
const D_MAX       = 200;       # max sojourn length; well above empirical max
const MAX_ITER    = 40;        # EM max iterations
const TOL         = 1e-3;      # log-likelihood tolerance per observation
const ALPHA_KS    = 0.05;
const KS          = [6, 9, 12];   # K values to fit (intermediate panel)

const OUT_DIR     = joinpath(_ROOT, "results", "hsmm_ml");
mkpath(OUT_DIR);

Random.seed!(SEED);

println("="^70)
println("  ML HSMM (Yu 2010 explicit-duration EM) on $TICKER")
println("  K values:  $KS")
println("  D_max:     $D_MAX")
println("  Max iter:  $MAX_ITER (tol $TOL per obs)")
println("  Seed:      $SEED")
println("="^70)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_is          = log_growth_matrix(train_dataset, TICKER; Δt=DT, risk_free_rate=RISK_FREE);
R_oos         = log_growth_matrix(oos_dataset,  TICKER; Δt=DT, risk_free_rate=RISK_FREE);
n_is          = length(R_is);
n_oos         = length(R_oos);
println("[setup] IS = $n_is, OoS = $n_oos")

# --------------------------------------------------------------------------- #
# Truncated discrete Pareto: pmf p(d) ∝ d^(-α-1) on {1, ..., D_max}
# --------------------------------------------------------------------------- #
function _pareto_logpmf(α::Float64, D::Int)
    w = [-(α + 1.0) * log(Float64(d)) for d in 1:D];
    mw = maximum(w);
    Z = mw + log(sum(exp.(w .- mw)));
    return w .- Z;
end

# Hill estimator MLE for truncated discrete Pareto from expected duration counts
function _fit_pareto_alpha(expected_counts::Vector{Float64}, D::Int)
    # weighted log mean: sum_d w_d * log d / sum w_d, MLE α ≈ 1 / E[log d]
    s = sum(expected_counts);
    if s <= 1e-9; return 1.5; end
    log_d_mean = sum(expected_counts[d] * log(Float64(d)) for d in 1:D) / s;
    α = 1.0 / max(log_d_mean, 1e-3);
    return clamp(α, 0.3, 5.0);
end

# --------------------------------------------------------------------------- #
# Gaussian log-density helper
# --------------------------------------------------------------------------- #
@inline function _logpdf_gauss(x, μ, σ)
    z = (x - μ) / σ;
    return -0.5 * z * z - log(σ) - 0.5 * log(2π);
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
    log_p::Matrix{Float64}                  # log p_s(d), size K × D_max
    ll_history::Vector{Float64}
end

function _init_hsmm(R::Vector{Float64}, K::Int, D::Int)
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
    @inbounds for s in 1:K
        log_p[s, :] = _pareto_logpmf(α[s], D);
    end
    return MLHSMM(K, π, A, μ, σ, α, log_p, Float64[]);
end

# --------------------------------------------------------------------------- #
# E-step: forward-backward over (state, duration)
#
# Conventions (log-space):
#   logb[s, t]           = log b_s(O_t)
#   logA_seg[s, t, d]    = sum of logb[s, t-d+1 ... t]   (segment log-emission for state s ending at t with duration d)
#   logf[s, t]           = log P(O_{1:t}, sojourn in state s ENDS at time t)
#   logg[s, t]           = log P(O_{t+1:T} | sojourn in state s STARTS at time t+1)
#                          (g[s, T] = 0)
# --------------------------------------------------------------------------- #
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

function _e_step(m::MLHSMM, R::Vector{Float64}; D::Int=D_MAX)
    K = m.K; T = length(R);
    Dt = min(D, T);
    # logb[s, t]
    logb = Matrix{Float64}(undef, K, T);
    @inbounds for s in 1:K, t in 1:T
        logb[s, t] = _logpdf_gauss(R[t], m.μ[s], m.σ[s]);
    end
    # cumulative segment emission: logA_seg[s, t, d] = sum_{τ=t-d+1..t} logb[s, τ]
    # implement via cumulative sums
    cum_b = zeros(K, T + 1);
    @inbounds for s in 1:K, t in 1:T
        cum_b[s, t + 1] = cum_b[s, t] + logb[s, t];
    end
    seg_b = (s, t, d) -> cum_b[s, t + 1] - cum_b[s, t - d + 1];

    # Forward: logf[s, t] = log P(O_{1:t}, sojourn s ends at t)
    # f_t(s) = sum_{d=1..min(t, D)} log_p[s, d] * (I[t==d] * π_s + sum_{s'≠s} A[s', s] * f_{t-d}(s')) * seg_b(s, t, d)
    logf = fill(-Inf, K, T);
    log_π = log.(m.π);
    log_A = log.(max.(m.A, 1e-300));
    @inbounds for t in 1:T
        d_max = min(t, Dt);
        for s in 1:K
            terms = Float64[];
            for d in 1:d_max
                pre = m.log_p[s, d] + seg_b(s, t, d);
                if t - d == 0
                    # initial segment
                    push!(terms, log_π[s] + pre);
                else
                    # transition from any s' ≠ s
                    parts = Float64[];
                    for sp in 1:K
                        sp == s && continue;
                        push!(parts, logf[sp, t - d] + log_A[sp, s] + pre);
                    end
                    push!(terms, _logsumexp(parts));
                end
            end
            logf[s, t] = _logsumexp(terms);
        end
    end

    # Backward: logg[s, t] = log P(O_{t+1:T} | sojourn s STARTS at t+1)
    # g_t(s) = sum_{d=1..min(D, T-t)} log_p[s, d] * seg_b(s, t+d, d) * (T-t-d == 0 ? 1 : sum_{s'≠s} A[s, s'] * g_{t+d}(s'))
    logg = fill(-Inf, K, T + 1);
    @inbounds for s in 1:K; logg[s, T + 1] = 0.0; end  # boundary at t = T
    @inbounds for t in (T-1):-1:0
        for s in 1:K
            terms = Float64[];
            d_max = min(Dt, T - t);
            for d in 1:d_max
                pre = m.log_p[s, d] + seg_b(s, t + d, d);
                if t + d == T
                    push!(terms, pre);   # last segment, no further continuation needed
                else
                    parts = Float64[];
                    for sp in 1:K
                        sp == s && continue;
                        push!(parts, log_A[s, sp] + logg[sp, t + d + 1]);
                    end
                    push!(terms, pre + _logsumexp(parts));
                end
            end
            logg[s, t + 1] = _logsumexp(terms);
        end
    end

    # log-likelihood: log P(O_{1:T}) = log sum_s logf[s, T]
    ll = _logsumexp(logf[:, T]);

    # Posteriors:
    # γ_t(s) = P(s_t = s | O)  — accumulated by integrating sojourn-end posteriors
    # We need three quantities:
    #   - η_t(s, d): P(sojourn s ending at t with duration d | O)  — for sojourn distribution update
    #   - ξ_t(s', s): P(transition s' -> s at boundary at time t | O)  — for transition update
    #   - state-occupancy: P(s_t = s | O), aggregated for emission update
    # All in log-space, normalised by ll.

    # log_eta[s, t, d] = logf[s, t] - log_p[s, d] - seg_b(s, t, d) + (initialised) + transition + (final)
    # Cleaner: redo the forward terms but record per-(s,t,d) contribution and add backward weight.
    #   eta_t(s, d) ∝  Σ_{s' ≠ s} f_{t-d}(s') A[s', s] * p_s(d) * seg_b(s, t, d) * g_t(s)   (for t-d > 0)
    #              ∝  π_s p_s(d) seg_b(s, d, d) g_d(s)                                          (for t = d)
    log_eta = fill(-Inf, K, T, Dt);
    @inbounds for t in 1:T
        d_max = min(t, Dt);
        # backward weight factor: log P(O_{t+1:T} | sojourn s ended at t)
        # = log sum_{s' != s} A[s, s'] * g_t(s')  (next sojourn starts at t+1 in s' ≠ s)
        # For t = T: terminal, weight = 0
        for s in 1:K
            if t == T
                log_g_post = 0.0;
            else
                parts = Float64[];
                for sp in 1:K
                    sp == s && continue;
                    push!(parts, log_A[s, sp] + logg[sp, t + 1]);
                end
                log_g_post = _logsumexp(parts);
            end
            for d in 1:d_max
                pre = m.log_p[s, d] + seg_b(s, t, d) + log_g_post;
                if t - d == 0
                    log_eta[s, t, d] = log_π[s] + pre;
                else
                    parts = Float64[];
                    for sp in 1:K
                        sp == s && continue;
                        push!(parts, logf[sp, t - d] + log_A[sp, s] + pre);
                    end
                    log_eta[s, t, d] = _logsumexp(parts);
                end
            end
        end
    end
    log_eta .-= ll;  # normalise

    # γ_t(s): aggregate probability of being in state s at time t
    # γ_t(s) = Σ_{d ≥ 1} Σ_{end ≥ t, end - d + 1 ≤ t} η_end(s, d)
    # Equivalent: every (state, sojourn-end time, duration) triple with start ≤ t ≤ end places mass on γ_t(s)
    log_γ = fill(-Inf, K, T);
    @inbounds for s in 1:K, t_end in 1:T
        d_max = min(t_end, Dt);
        for d in 1:d_max
            t_start = t_end - d + 1;
            for τ in t_start:t_end
                log_γ[s, τ] = _logsumexp_pair(log_γ[s, τ], log_eta[s, t_end, d]);
            end
        end
    end

    # ξ_t(s', s): boundary transition mass from sojourn ending at t in s' into new sojourn starting at t+1 in s
    # ξ_t(s', s) ∝ f_t(s') A[s', s] g_t(s), per pair (s', s) with s' ≠ s
    log_ξ = fill(-Inf, K, K, T);
    @inbounds for t in 1:(T-1)
        for sp in 1:K, s in 1:K
            sp == s && continue;
            log_ξ[sp, s, t] = logf[sp, t] + log_A[sp, s] + logg[s, t + 1];
        end
    end
    log_ξ .-= ll;

    return (ll=ll, log_γ=log_γ, log_eta=log_eta, log_ξ=log_ξ);
end

# --------------------------------------------------------------------------- #
# M-step
# --------------------------------------------------------------------------- #
function _m_step!(m::MLHSMM, R::Vector{Float64}, post; D::Int=D_MAX)
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

    # Initial: π_s = sum_d η_d(s, d) (mass that the first sojourn is in state s)
    log_π_new = fill(-Inf, K);
    @inbounds for s in 1:K
        for d in 1:Dt
            log_π_new[s] = _logsumexp_pair(log_π_new[s], post.log_eta[s, d, d]);
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

    # Sojourn distribution: per-state Pareto MLE on expected duration counts
    eta_d = zeros(K, Dt);
    @inbounds for s in 1:K, t in 1:T
        d_max = min(t, Dt);
        for d in 1:d_max
            eta_d[s, d] += exp(post.log_eta[s, t, d]);
        end
    end
    @inbounds for s in 1:K
        if sum(eta_d[s, :]) > 1e-9
            m.α[s] = _fit_pareto_alpha(eta_d[s, :], Dt);
            m.log_p[s, :] = _pareto_logpmf(m.α[s], Dt);
        end
    end
    return nothing;
end

# --------------------------------------------------------------------------- #
# EM driver
# --------------------------------------------------------------------------- #
function fit_hsmm_ml(R::Vector{Float64}, K::Int; D::Int=D_MAX, max_iter::Int=MAX_ITER, tol::Float64=TOL)
    println("[hsmm-ml] K=$K, T=$(length(R)), D=$D ...");
    m = _init_hsmm(R, K, D);
    last_ll = -Inf;
    for it in 1:max_iter
        post = _e_step(m, R; D=D);
        ll = post.ll;
        push!(m.ll_history, ll);
        @printf("  [%2d] log-lik = %.4f  (per-obs %.5f)\n", it, ll, ll / length(R));
        if abs(ll - last_ll) / max(length(R), 1) < tol && it > 4
            println("  → converged at iter $it");
            break;
        end
        last_ll = ll;
        _m_step!(m, R, post; D=D);
    end
    return m;
end

# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
function simulate_hsmm(m::MLHSMM; T::Int, D::Int=D_MAX, n_paths::Int=N_PATHS)
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

# --------------------------------------------------------------------------- #
# Metrics (parallel to run_diagnostics.jl eval_full)
# --------------------------------------------------------------------------- #
function eval_panel(observed::Vector{Float64}, sim::Matrix{Float64}; L::Int=L_LAGS, α::Float64=ALPHA_KS)
    np = size(sim, 2); n_o = length(observed);
    L_use = min(L, n_o - 1);
    acf_o     = autocor(abs.(observed), 1:L_use);
    acf_o_raw = autocor(observed, 1:L_use);
    ks_pass = 0; kurt_s = 0.0;
    acf_mae = 0.0; acf_mae_raw = 0.0;
    @inbounds for i in 1:np
        s = sim[:, i];
        ks_p = pvalue(ApproximateTwoSampleKSTest(s, observed));
        ks_pass += (ks_p ≥ α) ? 1 : 0;
        ms = mean(s); ss = std(s);
        kurt_s += sum(((s .- ms) ./ ss).^4) / length(s) - 3.0;
        acf_mae     += mean(abs.(autocor(abs.(s), 1:L_use) .- acf_o));
        acf_mae_raw += mean(abs.(autocor(s, 1:L_use) .- acf_o_raw));
    end
    return (
        ks_rate = ks_pass / np,
        kurt    = kurt_s / np,
        acf_mae = acf_mae / np,
        acf_mae_raw = acf_mae_raw / np,
    );
end

# --------------------------------------------------------------------------- #
# Run pipeline at K = 3 and K = 18
# --------------------------------------------------------------------------- #
using HypothesisTests
results = Dict{Int,Any}();
for K in KS
    println("\n" * "="^70)
    println("  Fitting ML HSMM at K = $K")
    println("="^70)
    Random.seed!(SEED + K);
    @time m = fit_hsmm_ml(R_is, K; D=D_MAX, max_iter=MAX_ITER, tol=TOL);
    sim_is  = simulate_hsmm(m; T=n_is,  n_paths=N_PATHS);
    sim_oos = simulate_hsmm(m; T=n_oos, n_paths=N_PATHS);
    metr_is  = eval_panel(R_is,  sim_is);
    metr_oos = eval_panel(R_oos, sim_oos);
    @printf("[K=%d] IS  KS = %.1f%%  kurt = %.3f  ACF-MAE |G| = %.4f  raw = %.4f\n",
        K, 100 * metr_is.ks_rate, metr_is.kurt, metr_is.acf_mae, metr_is.acf_mae_raw);
    @printf("[K=%d] OoS KS = %.1f%%  kurt = %.3f  ACF-MAE |G| = %.4f  raw = %.4f\n",
        K, 100 * metr_oos.ks_rate, metr_oos.kurt, metr_oos.acf_mae, metr_oos.acf_mae_raw);
    results[K] = (model=m, metr_is=metr_is, metr_oos=metr_oos);
    save(joinpath(OUT_DIR, "hsmm_ml_K$K.jld2"), Dict(
        "K" => K,
        "alpha" => m.α,
        "mu" => m.μ,
        "sigma" => m.σ,
        "A" => m.A,
        "pi" => m.π,
        "log_p" => m.log_p,
        "ll_history" => m.ll_history,
        "metr_is" => metr_is,
        "metr_oos" => metr_oos,
    ));
end

# Write CSV summary
open(joinpath(OUT_DIR, "hsmm_ml_intermediate_K_metrics.csv"), "w") do io
    write(io, "K,IS_KS,OoS_KS,IS_kurt,OoS_kurt,IS_ACF_MAE_abs,OoS_ACF_MAE_abs,IS_ACF_MAE_raw,OoS_ACF_MAE_raw,n_iter,final_ll\n")
    for K in KS
        r = results[K];
        write(io, @sprintf("%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%d,%.4f\n",
            K, r.metr_is.ks_rate, r.metr_oos.ks_rate,
            r.metr_is.kurt, r.metr_oos.kurt,
            r.metr_is.acf_mae, r.metr_oos.acf_mae,
            r.metr_is.acf_mae_raw, r.metr_oos.acf_mae_raw,
            length(r.model.ll_history), last(r.model.ll_history)));
    end
end

println("\n[done] ML HSMM fits saved to $OUT_DIR")
