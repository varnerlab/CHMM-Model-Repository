# ========================================================================================= #
# SemiMarkov.jl
#
# Track C1 port from CHMM-Vol-Model/src/Compute.jl:1300..1755.
# Semi-Markov continuous HMM with state-dependent AR(1) emissions (Gaussian / Student-t /
# Laplace residuals) and explicit per-state sojourn distributions (NB or truncated Pareto).
#
# Plug-in estimator per Yu (2010) §IV plus the vol-paper plan Section 5:
#   1. Fit matching flat CHMM family via `build` (Baum-Welch initialization).
#   2. Viterbi-decode the in-sample state sequence.
#   3. Reorder states by emission location so state 1 is calm and state K is crisis.
#   4. Per-state AR(1) residual fit on within-state lag pairs.
#   5. Per-state sojourn-family pick (NB vs truncated Pareto by marginal log-likelihood).
#   6. Off-diagonal between-state transition matrix from observed regime switches.
#
# Simulation draws an explicit-duration state sequence (sojourn from m.sojourn[s], then
# transition from m.transition[s]) and rolls AR(1) emissions within each sojourn.
# ========================================================================================= #

using Distributions;
using LinearAlgebra;
using Statistics;

# ------------------------------------------------------------------------------------------- #
# Truncated discrete Pareto sojourn distribution
# ------------------------------------------------------------------------------------------- #

"""
    TruncatedDiscretePareto

Discrete power-law distribution on {d_min, ..., d_max} with pmf p(d) ∝ d^(-α - 1).
Heavy-tailed sojourn family for crisis / long-lived regimes.
"""
struct TruncatedDiscretePareto <: DiscreteUnivariateDistribution
    α::Float64
    d_min::Int
    d_max::Int
    _pmf::Vector{Float64}
    _cdf::Vector{Float64}
end

function _build_tdp(α::Float64, d_min::Int, d_max::Int)::TruncatedDiscretePareto
    support = d_min:d_max;
    w = [Float64(d)^(-α - 1.0) for d in support];
    Z = sum(w);
    pmf = w ./ Z;
    cdf = cumsum(pmf);
    return TruncatedDiscretePareto(α, d_min, d_max, pmf, cdf);
end

Base.rand(d::TruncatedDiscretePareto) = begin
    u = rand();
    idx = searchsortedfirst(d._cdf, u);
    idx = min(idx, length(d._pmf));
    return d.d_min + idx - 1;
end

Distributions.logpdf(d::TruncatedDiscretePareto, x::Integer) = begin
    if x < d.d_min || x > d.d_max; return -Inf; end
    return log(d._pmf[x - d.d_min + 1]);
end

Distributions.logpdf(d::TruncatedDiscretePareto, x::Real) = logpdf(d, Int(round(x)));
Distributions.mean(d::TruncatedDiscretePareto) = sum((d.d_min:d.d_max) .* d._pmf);

# ------------------------------------------------------------------------------------------- #
# Sojourn-family helpers
# ------------------------------------------------------------------------------------------- #

function _fit_nb_mom(durs::Vector{Int})::DiscreteUnivariateDistribution
    n = length(durs);
    if n < 3; return Geometric(1.0 / max(mean(durs), 1.0)); end
    μ = mean(durs); v = var(durs);
    if v <= μ || μ <= 0
        return Geometric(1.0 / max(μ, 1.0));
    end
    p = μ / v; r = μ^2 / (v - μ);
    p = clamp(p, 1e-6, 1.0 - 1e-6);
    r = max(r, 0.1);
    return NegativeBinomial(r, p);
end

function _fit_pareto_mle(durs::Vector{Int}; d_max_override::Int=0)::TruncatedDiscretePareto
    d_min = max(1, minimum(durs));
    safe_durs = [d for d in durs if d >= d_min];
    N = length(safe_durs);
    denom = sum(log(Float64(d) / Float64(d_min)) for d in safe_durs);
    α = N > 0 && denom > 0 ? N / denom : 1.5;
    α = clamp(α, 0.3, 5.0);
    d_max = d_max_override > 0 ? d_max_override : max(maximum(durs) * 2, 100);
    return _build_tdp(α, d_min, d_max);
end

function _pick_sojourn_family(durs::Vector{Int})
    if length(durs) < 3
        return (:nb, _fit_nb_mom(durs));
    end
    nb = _fit_nb_mom(durs);
    pa = _fit_pareto_mle(durs);
    ll_nb = sum(logpdf(nb, max(d, 1)) for d in durs);
    ll_pa = sum(logpdf(pa, d) for d in durs);
    return ll_pa > ll_nb ? (:pareto, pa) : (:nb, nb);
end

function _runs(states::Vector{Int})
    out = Tuple{Int,Int}[];
    if isempty(states); return out; end
    curr = states[1]; len = 1;
    for t in 2:length(states)
        if states[t] == curr
            len += 1;
        else
            push!(out, (curr, len));
            curr = states[t]; len = 1;
        end
    end
    push!(out, (curr, len));
    return out;
end

# ------------------------------------------------------------------------------------------- #
# Per-state AR(1) residual fit
# ------------------------------------------------------------------------------------------- #

function _fit_ar1_residual_state(y_prev::Vector{Float64}, y_curr::Vector{Float64},
                                 family::Symbol)
    n = length(y_prev);
    if n < 3
        μ0 = n > 0 ? mean(y_curr) : 0.0;
        σ0 = n > 1 ? max(std(y_curr), 1e-4) : 1e-4;
        ν0 = family == :student_t ? 30.0 : Inf;
        return (μ0, 0.0, σ0, ν0);
    end
    X = hcat(ones(n), y_prev);
    β = X \ y_curr;
    α_ols, φ_ols = β[1], β[2];
    ε = y_curr .- X * β;
    φ = clamp(φ_ols, -0.999, 0.999);
    μ = abs(1 - φ) > 1e-6 ? α_ols / (1 - φ) : mean(y_curr);

    if family == :gaussian
        σ = max(std(ε), 1e-4);
        return (μ, φ, σ, Inf);
    elseif family == :laplace
        b = max(mean(abs.(ε)), 1e-4);
        return (μ, φ, b, Inf);
    elseif family == :student_t
        best_ν = 30.0; best_σ = max(std(ε), 1e-4); best_nll = Inf;
        for ν in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)
            for σ_try in range(max(std(ε)*0.4, 1e-4), max(std(ε)*3.0, 0.2); length=30)
                nll = -sum(logpdf(LocationScale(0.0, σ_try, TDist(ν)), e) for e in ε);
                if nll < best_nll
                    best_nll = nll; best_σ = σ_try; best_ν = ν;
                end
            end
        end
        return (μ, φ, best_σ, best_ν);
    else
        error("Unknown residual family: $family (expected :gaussian, :student_t, or :laplace)");
    end
end

# ------------------------------------------------------------------------------------------- #
# Plug-in estimator
# ------------------------------------------------------------------------------------------- #

"""
    fit_sm_chmm(observations, K, family; max_iter, min_sojourn_samples)
        -> MySemiMarkovContinuousHMM

Plug-in estimator for the semi-Markov CHMM. `family` in
{`:gaussian`, `:student_t`, `:laplace`}.
"""
function fit_sm_chmm(observations::Vector{Float64}, K::Int, family::Symbol;
    max_iter::Int=60, min_sojourn_samples::Int=5)::MySemiMarkovContinuousHMM

    init_model = if family == :gaussian
        build(MyContinuousHiddenMarkovModel,
              (observations=observations, number_of_states=K, max_iter=max_iter));
    elseif family == :student_t
        build(MyStudentTHiddenMarkovModel,
              (observations=observations, number_of_states=K, max_iter=max_iter));
    elseif family == :laplace
        build(MyLaplaceHiddenMarkovModel,
              (observations=observations, number_of_states=K, max_iter=max_iter));
    else
        error("Unknown residual family: $family (expected :gaussian, :student_t, or :laplace)");
    end
    init_ll = isempty(init_model.log_likelihood_history) ? -Inf : last(init_model.log_likelihood_history);

    decoded = viterbi(observations, init_model);

    K_actual = length(init_model.states);
    state_locs = if family == :laplace
        [init_model.emission[s].μ for s in init_model.states];
    else
        [mean(init_model.emission[s]) for s in init_model.states];
    end
    perm = sortperm(state_locs);
    rank_map = Dict(perm[r] => r for r in 1:K_actual);
    decoded_reord = [rank_map[s] for s in decoded];

    em_μ = Dict{Int64,Float64}();
    em_φ = Dict{Int64,Float64}();
    em_σ = Dict{Int64,Float64}();
    em_ν = Dict{Int64,Float64}();
    N = length(observations);
    for k in 1:K_actual
        ys_prev = Float64[]; ys_curr = Float64[];
        for t in 2:N
            if decoded_reord[t] == k && decoded_reord[t-1] == k
                push!(ys_prev, observations[t-1]);
                push!(ys_curr, observations[t]);
            end
        end
        μ_k, φ_k, σ_k, ν_k = _fit_ar1_residual_state(ys_prev, ys_curr, family);
        em_μ[k] = μ_k; em_φ[k] = φ_k; em_σ[k] = σ_k; em_ν[k] = ν_k;
    end

    runs = _runs(decoded_reord);
    sojourn = Dict{Int64, DiscreteUnivariateDistribution}();
    sojourn_fam = Dict{Int64, Symbol}();
    for k in 1:K_actual
        durs = [d for (s, d) in runs if s == k];
        if length(durs) < min_sojourn_samples
            μd = isempty(durs) ? 5.0 : mean(durs);
            sojourn[k] = Geometric(1.0 / max(μd, 1.0));
            sojourn_fam[k] = :geometric;
        else
            fam, dist = _pick_sojourn_family(durs);
            sojourn[k] = dist;
            sojourn_fam[k] = fam;
        end
    end

    trans_counts = zeros(K_actual, K_actual);
    for i in 1:(length(runs) - 1)
        s_from = runs[i][1]; s_to = runs[i + 1][1];
        if s_from != s_to
            trans_counts[s_from, s_to] += 1.0;
        end
    end
    transition = Dict{Int64, Categorical}();
    for i in 1:K_actual
        row = trans_counts[i, :];
        if sum(row) <= 0
            row = ones(K_actual); row[i] = 0.0;
        end
        row ./= sum(row);
        transition[i] = Categorical(row);
    end

    m = MySemiMarkovContinuousHMM();
    m.states = collect(1:K_actual);
    m.transition = transition;
    m.emission_family = family;
    m.emission_mu = em_μ;
    m.emission_phi = em_φ;
    m.emission_sigma = em_σ;
    m.emission_nu = em_ν;
    m.sojourn = sojourn;
    m.sojourn_family = sojourn_fam;
    m.log_likelihood_history = [init_ll];
    return m;
end

# ------------------------------------------------------------------------------------------- #
# Simulation
# ------------------------------------------------------------------------------------------- #

function _simulate(m::MySemiMarkovContinuousHMM, start::Int64, steps::Int64)::Array{Int64,1}
    chain = Array{Int64,1}(undef, steps);
    if steps <= 0; return chain; end
    t = 1;
    s = start;
    while t <= steps
        d = max(1, Int(rand(m.sojourn[s])));
        t_end = min(t + d - 1, steps);
        for τ in t:t_end; chain[τ] = s; end
        t = t_end + 1;
        if t <= steps
            s = rand(m.transition[s]);
        end
    end
    return chain;
end
(m::MySemiMarkovContinuousHMM)(start::Int64, steps::Int64) = _simulate(m, start, steps);

@inline function _draw_residual(family::Symbol, σ_or_b::Float64, ν::Float64)
    if family == :gaussian
        return σ_or_b * randn();
    elseif family == :student_t
        return σ_or_b * rand(TDist(ν));
    elseif family == :laplace
        return rand(Laplace(0.0, σ_or_b));
    else
        error("Unknown residual family: $family");
    end
end

"""
    simulate_sm_chmm(m, n_steps; y0=NaN, n_paths=1)

Return a single path (Vector) when `n_paths==1`, else an (n_steps × n_paths) matrix.
"""
function simulate_sm_chmm(m::MySemiMarkovContinuousHMM, n_steps::Int;
    y0::Float64=NaN, n_paths::Int=1)

    K = length(m.states);
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(m.transition[i]); end
    π_stat = (T_mat ^ 1000)[1, :];
    π_stat .= max.(π_stat, 1e-12); π_stat ./= sum(π_stat);
    start_dist = Categorical(π_stat);
    family = m.emission_family;

    function _one_path()
        s0 = rand(start_dist);
        states = _simulate(m, s0, n_steps);
        out = Vector{Float64}(undef, n_steps);
        k0 = states[1];
        y_prev = isnan(y0) ?
            (m.emission_mu[k0] + _draw_residual(family, m.emission_sigma[k0], m.emission_nu[k0])) :
            y0;
        @inbounds for t in 1:n_steps
            k = states[t];
            μ = m.emission_mu[k]; φ = m.emission_phi[k];
            ε = _draw_residual(family, m.emission_sigma[k], m.emission_nu[k]);
            y_t = μ + φ * (y_prev - μ) + ε;
            out[t] = y_t; y_prev = y_t;
        end
        return out;
    end

    if n_paths == 1; return _one_path(); end
    paths = Matrix{Float64}(undef, n_steps, n_paths);
    for p in 1:n_paths
        paths[:, p] = _one_path();
    end
    return paths;
end
