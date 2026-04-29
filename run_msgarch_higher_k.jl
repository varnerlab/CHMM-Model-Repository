# ========================================================================================= #
# run_msgarch_higher_k.jl
#
# MS-GARCH(1,1) fitting at higher state counts (K = 4, K = 6) on SPY IS, addressing
# peer-review item R2-W3: the body's MS-GARCH baseline is at K = 2 (Haas-Mittnik-Paolella
# 2004) which is unfair against a K = 18 CHMM. The standard MSGARCH literature ceiling is
# typically K = 3 or K = 4; K = 6 is non-standard but tractable.
#
# This script generalises the existing fit_msgarch_k3 architecture to general K via a
# softmax-from-logits transition-matrix parameterisation and Nelder-Mead minimisation of
# the negative Hamilton-filter log-likelihood. We refit at K = 4 and K = 6, simulate
# 1000 IS / OoS paths each, and write a panel that augments the appendix
# tab:extended_baselines.
#
# Output: results/msgarch_baselines/MSGARCH_higher_K.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using LinearAlgebra
using HypothesisTests
using StatsBase
using Printf

const SEED      = 20260422;
const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const N_PATHS   = 1000;
const L_LAGS    = 252;
const K_GRID    = [4, 6];

const OUT_DIR = joinpath(_ROOT, "results", "msgarch_baselines");
mkpath(OUT_DIR);

println("="^80);
println("  MS-GARCH at higher K  (R2-W3)  Seed $SEED  K = $K_GRID");
println("="^80);

Random.seed!(SEED);

# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY IS / OoS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
all_tickers = keys(filtered) |> collect |> sort;
all_R = log_growth_matrix(filtered, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
n_is = length(R_is);
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
n_oos = length(R_oos);
println("  IS = $n_is days  OoS = $n_oos days");

# --------------------------------------------------------------------------------------- #
# General-K transition unpacking (softmax over off-diagonals; diagonal = residual)
# --------------------------------------------------------------------------------------- #
"""
For each row k, the K-1 off-diagonal logits are decoded via the softmax
    p_{k,j} = exp(η_{k,j}) / (1 + sum_l exp(η_{k,l}))      for j != k
and the diagonal is p_{k,k} = 1 / (1 + sum_l exp(η_{k,l})). The non-diagonal columns
are taken in cyclic order (k+1, k+2, ..., k-1) mod K (so the parameterisation is
invariant under index relabelling up to a fixed convention).
"""
function unpack_T(logits::Vector{Float64}, K::Int)::Matrix{Float64}
    @assert length(logits) == K * (K - 1);
    T = zeros(K, K);
    for k in 1:K
        es = zeros(K - 1);
        for l in 1:(K - 1)
            es[l] = exp(logits[(k - 1) * (K - 1) + l]);
        end
        denom = 1.0 + sum(es);
        T[k, k] = 1.0 / denom;
        for l in 1:(K - 1)
            j = ((k + l - 1) % K) + 1;
            T[k, j] = es[l] / denom;
        end
    end
    return T;
end

# --------------------------------------------------------------------------------------- #
# Hamilton filter (re-implementation, matching src/MSGARCH.jl _hamilton_filter_msgarch)
# --------------------------------------------------------------------------------------- #
function hamilton_filter(obs::Vector{Float64},
    ω::Vector{Float64}, α::Vector{Float64}, β::Vector{Float64},
    μ::Float64, T::Matrix{Float64})
    N = length(obs); K = length(ω);
    π_stat = ones(K) / K;
    for _ in 1:50
        π_stat = T' * π_stat
        π_stat = max.(π_stat, 1e-12)
        π_stat ./= sum(π_stat)
    end
    σ2 = Matrix{Float64}(undef, N, K);
    for k in 1:K
        denom = max(1.0 - α[k] - β[k], 1e-6);
        σ2[1, k] = ω[k] / denom;
    end
    γ = Matrix{Float64}(undef, N, K);
    ll = 0.0;
    log2π_half = 0.5 * log(2π);
    log_emit = zeros(K);
    for t in 1:N
        π_pred = (t == 1) ? π_stat : (T' * γ[t-1, :]);
        if t > 1
            π_pred ./= sum(π_pred);
        end
        for k in 1:K
            r = obs[t] - μ;
            s2 = σ2[t, k];
            if s2 <= 0 || !isfinite(s2)
                log_emit[k] = -1e10;
            else
                log_emit[k] = -log2π_half - 0.5 * log(s2) - 0.5 * r^2 / s2;
            end
        end
        m = maximum(log_emit);
        raw = π_pred .* exp.(log_emit .- m);
        rsum = sum(raw);
        if !isfinite(rsum) || rsum <= 0
            γ[t, :] = π_pred;
            ll += m + log(max(1e-12, sum(π_pred .* exp.(log_emit .- m))));
        else
            γ[t, :] = raw / rsum;
            ll += m + log(rsum);
        end
        if t < N
            r = obs[t] - μ;
            for k in 1:K
                σ2[t+1, k] = max(ω[k] + α[k] * r^2 + β[k] * σ2[t, k], 1e-12);
            end
        end
    end
    return ll, σ2, γ;
end

function nll_msgarch(params::Vector{Float64}, obs::Vector{Float64}, μ::Float64, K::Int)
    @assert length(params) == 3 * K + K * (K - 1);
    ω = params[1:3:(3 * K)];
    α = params[2:3:(3 * K)];
    β = params[3:3:(3 * K)];
    for k in 1:K
        if ω[k] <= 0 || α[k] < 0 || β[k] < 0 || (α[k] + β[k]) >= 0.999
            return 1e10;
        end
    end
    T = unpack_T(params[(3 * K + 1):end], K);
    ll, _, _ = hamilton_filter(obs, ω, α, β, μ, T);
    return isfinite(ll) ? -ll : 1e10;
end

# --------------------------------------------------------------------------------------- #
# Nelder-Mead with multistart (adapted from src/MSGARCH.jl)
# --------------------------------------------------------------------------------------- #
function nelder_mead(fn, x0::Vector{Float64}, max_iter::Int, tol::Float64)
    d = length(x0);
    simplex = [copy(x0) for _ in 1:(d + 1)];
    for i in 2:(d + 1)
        step = abs(simplex[i][i-1]) > 1e-12 ? simplex[i][i-1] * 0.12 : 0.05;
        simplex[i][i-1] += step;
    end
    for iter in 1:max_iter
        vals = [fn(s) for s in simplex];
        order = sortperm(vals);
        simplex = simplex[order];
        vals = vals[order];
        if abs(vals[end] - vals[1]) < tol; break; end
        centroid = sum(simplex[1:d]) ./ d;
        reflected = centroid .+ (centroid .- simplex[end]);
        f_r = fn(reflected);
        if f_r < vals[1]
            expanded = centroid .+ 2.0 .* (reflected .- centroid);
            f_e = fn(expanded);
            simplex[end] = f_e < f_r ? expanded : reflected;
        elseif f_r < vals[d]
            simplex[end] = reflected;
        else
            contracted = centroid .+ 0.5 .* (simplex[end] .- centroid);
            f_c = fn(contracted);
            if f_c < vals[end]
                simplex[end] = contracted;
            else
                for i in 2:(d + 1)
                    simplex[i] = simplex[1] .+ 0.5 .* (simplex[i] .- simplex[1]);
                end
            end
        end
    end
    vals = [fn(s) for s in simplex];
    return simplex[argmin(vals)], minimum(vals);
end

function fit_msgarch_kg(obs::Vector{Float64}, K::Int; max_iter::Int=5000)
    μ_obs = mean(obs); var_obs = var(obs);
    nparams = 3 * K + K * (K - 1);
    fn(p) = nll_msgarch(p, obs, μ_obs, K);

    # Multistart initialisations: vary (α, β) base and variance-ratio spread
    best_nll = Inf; best_params = zeros(nparams);
    for α_try in [0.05, 0.10]
        for β_try in [0.85, 0.90]
            denom = (1 - α_try - β_try);
            if denom < 0.05; continue; end
            # Variance ratios spread across K regimes (geometric spread)
            for spread in [2.5, 4.0, 6.0]
                ratios = [spread^((k - (K + 1) / 2) / max(K / 2, 1)) for k in 1:K];
                # Normalise so geometric mean is 1
                g = exp(mean(log.(ratios))); ratios ./= g;
                params = zeros(nparams);
                for k in 1:K
                    params[3 * (k - 1) + 1] = var_obs * denom * ratios[k];
                    params[3 * (k - 1) + 2] = α_try;
                    params[3 * (k - 1) + 3] = β_try;
                end
                # Diagonal-dominant transition init: each off-diagonal logit ≈ log(0.05 / 0.90)
                logit0 = log(0.05 / 0.90);
                for j in 1:(K * (K - 1))
                    params[3 * K + j] = logit0;
                end
                v = fn(params);
                if v < best_nll
                    best_nll = v;
                    best_params = copy(params);
                end
            end
        end
    end

    # Polish from the best init via Nelder-Mead
    best, best_v = nelder_mead(fn, best_params, max_iter, 1e-6);

    ω = best[1:3:(3 * K)];
    α = best[2:3:(3 * K)];
    β = best[3:3:(3 * K)];
    T = unpack_T(best[(3 * K + 1):end], K);
    # Canonicalise by ascending unconditional variance
    uv = [ω[k] / max(1 - α[k] - β[k], 1e-6) for k in 1:K];
    order = sortperm(uv);
    ω = ω[order]; α = α[order]; β = β[order];
    P = zeros(K, K);
    for i in 1:K, j in 1:K; P[i, j] = T[order[i], order[j]]; end
    T = P;
    ll, σ2, γ = hamilton_filter(obs, ω, α, β, μ_obs, T);
    return (K=K, ω=ω, α=α, β=β, μ=μ_obs, T=T, ll=ll, nll=best_v);
end

# --------------------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------------------- #
function simulate_msgarch_kg(model, n_steps::Int)::Vector{Float64}
    K = model.K;
    π_stat = ones(K) / K;
    for _ in 1:50
        π_stat = model.T' * π_stat
        π_stat = max.(π_stat, 1e-12)
        π_stat ./= sum(π_stat)
    end
    s = sample(1:K, Weights(π_stat));
    σ2 = [model.ω[k] / max(1 - model.α[k] - model.β[k], 1e-6) for k in 1:K];
    out = Vector{Float64}(undef, n_steps);
    for t in 1:n_steps
        ε = sqrt(σ2[s]) * randn();
        out[t] = model.μ + ε;
        for k in 1:K
            σ2[k] = max(model.ω[k] + model.α[k] * (out[t] - model.μ)^2 + model.β[k] * σ2[k], 1e-12);
        end
        # Transition
        s = sample(1:K, Weights(model.T[s, :]));
    end
    return out;
end

# --------------------------------------------------------------------------------------- #
# Headline panel evaluation
# --------------------------------------------------------------------------------------- #
function eval_panel(R_obs::AbstractVector, sim_archive::AbstractMatrix; L::Int=L_LAGS)
    np = size(sim_archive, 2);
    n_o = length(R_obs);
    μ_o = mean(R_obs); σ_o = std(R_obs);
    kurt_o = sum(((R_obs .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L, n_o - 1);
    acf_o     = autocor(abs.(R_obs), 1:L_use);
    acf_o_raw = autocor(R_obs,        1:L_use);

    ks_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0; acf_mae_raw_s = 0.0;
    for i in 1:np
        s = sim_archive[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= 0.05
            ks_pass += 1;
        end
        μ_s = mean(s); σ_s = std(s);
        kurt_s += sum(((s .- μ_s) ./ σ_s).^4) / length(s) - 3.0;
        acf_s     = autocor(abs.(s), 1:L_use);
        acf_s_raw = autocor(s,       1:L_use);
        acf_mae_s     += mean(abs.(acf_o     .- acf_s));
        acf_mae_raw_s += mean(abs.(acf_o_raw .- acf_s_raw));
    end
    return (
        ks_pct      = round(100 * ks_pass / np, digits=1),
        sim_kurt    = round(kurt_s / np, digits=2),
        obs_kurt    = round(kurt_o, digits=2),
        acf_mae     = round(acf_mae_s / np, digits=4),
        acf_mae_raw = round(acf_mae_raw_s / np, digits=4),
    );
end

# --------------------------------------------------------------------------------------- #
# Run the K-grid
# --------------------------------------------------------------------------------------- #
results = Vector{Any}();
for K in K_GRID
    println("\n[fit] MS-GARCH at K = $K  (params = $(3 * K + K * (K - 1)))");
    Random.seed!(SEED + K);
    @time model = fit_msgarch_kg(R_is, K; max_iter=5000);
    println("  log-lik = $(round(model.ll, digits=2))");
    println("  unconditional std per regime: $([round(sqrt(model.ω[k] / max(1 - model.α[k] - model.β[k], 1e-6)), digits=3) for k in 1:K])");
    println("  diagonal of T: $([round(model.T[k, k], digits=3) for k in 1:K])");

    println("[sim] $N_PATHS IS + OoS paths...");
    Random.seed!(SEED + 100 + K);
    sim_is  = Matrix{Float64}(undef, n_is,  N_PATHS);
    sim_oos = Matrix{Float64}(undef, n_oos, N_PATHS);
    for p in 1:N_PATHS
        sim_is[:,  p] = simulate_msgarch_kg(model, n_is);
        sim_oos[:, p] = simulate_msgarch_kg(model, n_oos);
    end

    is_panel  = eval_panel(R_is,  sim_is);
    oos_panel = eval_panel(R_oos, sim_oos);

    println("  IS  KS = $(is_panel.ks_pct)%  sim kurt = $(is_panel.sim_kurt)  |G| ACF-MAE = $(is_panel.acf_mae)");
    println("  OoS KS = $(oos_panel.ks_pct)%  sim kurt = $(oos_panel.sim_kurt)");

    push!(results, (K=K, model=model, is=is_panel, oos=oos_panel));
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "MSGARCH_higher_K.txt");
open(out_path, "w") do io
    println(io, "="^110);
    println(io, "MS-GARCH at higher K  (R2-W3)");
    println(io, "="^110);
    println(io, "Setup: SPY IS = $n_is days, OoS = $n_oos days, paths = $N_PATHS, seed = $SEED.");
    println(io, "Generalised Haas-Mittnik-Paolella 2004 specification: regime-specific GARCH(1,1) with");
    println(io, "  σ²_t(k) = ω_k + α_k * (G_{t-1} - μ)² + β_k * σ²_{t-1}(k)");
    println(io, "and a hidden Markov chain T (transition matrix) modulating regimes. Fit by Nelder-Mead");
    println(io, "with diagonal-dominant T initialisation, geometric variance-ratio spread across regimes,");
    println(io, "and ascending-variance regime canonicalisation.");
    println(io);
    @printf(io, "%-3s | %-7s | %-9s | %-7s | %-7s | %-9s | %-9s | %-9s\n",
            "K","#params","log-lik","KS IS%","KS OoS%","kurt sim","kurt obs","|G| ACF-MAE");
    println(io, "-"^110);
    for r in results
        np = 3 * r.K + r.K * (r.K - 1);
        @printf(io, "%-3d | %-7d | %-9.2f | %-7.1f | %-7.1f | %-9.2f | %-9.2f | %-9.4f\n",
                r.K, np, r.model.ll, r.is.ks_pct, r.oos.ks_pct,
                r.is.sim_kurt, r.is.obs_kurt, r.is.acf_mae);
    end
    println(io);
    println(io, "Comparison points (from existing tables):");
    println(io, "  MS-GARCH K=2 (Haas):  KS IS = 27.7%   KS OoS = 38.7%   kurt = 4.7   |G| ACF-MAE = 0.0367");
    println(io, "  MS-GARCH K=3:          KS IS = 36.1%   KS OoS = 33.1%   kurt = 4.1   |G| ACF-MAE = 0.0284");
    println(io, "  CHMM-N K=18:           KS IS = 94.1%   KS OoS = 81.8%   kurt = 5.0   |G| ACF-MAE = 0.0509");
    println(io);
    println(io, "Per-regime parameter detail:");
    for r in results
        println(io);
        println(io, "K = $(r.K):");
        for k in 1:r.K
            uv = sqrt(r.model.ω[k] / max(1 - r.model.α[k] - r.model.β[k], 1e-6));
            @printf(io, "  regime %d: ω=%.4f  α=%.4f  β=%.4f  uncond σ=%.3f  T_kk=%.3f\n",
                    k, r.model.ω[k], r.model.α[k], r.model.β[k], uv, r.model.T[k, k]);
        end
    end
end

println("\n[done] Wrote $out_path");
