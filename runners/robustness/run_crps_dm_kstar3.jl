# ========================================================================================= #
# run_crps_dm_kstar3.jl
#
# HAC Diebold-Mariano inference for the Table 2 K* = 3 CHMM CRPS column.
#
# The paper claims the displayed K* = 3 CHMM variants are statistically indistinguishable
# on OoS sample-CRPS. This runner backs that claim on the CURRENT headline protocol
# (the archived _attic/runners/run_crps_dm.jl covered only the legacy CHMM-N/t/L rows
# against a cache that no longer exists). It:
#
#   1. Fits all FIVE displayed K* = 3 CHMM rows on the SPY in-sample window:
#      CHMM-N, penalised CHMM-t (lambda = 20), CHMM-L, CHMM-GED (per-cell reseed:
#      SEED for the fit, SEED + 1 for the simulation, exactly as in
#      runners/headline/run_kstar3_headline.jl), and the shared-nu CHMM-t (the
#      ecm_shared_nu code path of runners/headline/run_chmm_t_shared_nu.jl, with that
#      runner's own seed convention: Random.seed!(SEED), deterministic ECM fit, then
#      IS-length + OoS-length simulation in that order so the OoS bundle reproduces
#      results/chmm_t_shared_nu/sims_K3.jld2).
#   2. From ONE 1000-path unconditional simulation bundle per model, computes the
#      per-day OoS sample-CRPS loss series l_m(t), t = 1..T_OoS, with the unbiased
#      sorted-ensemble estimator of run_kstar3_headline.jl / sec:crps_methods:
#        CRPS_t = (1/N) sum_i |x_{t,i} - y_t| - (1/(N(N-1))) sum_{i<j} |x_{t,(i)} - x_{t,(j)}|
#      where the second term uses sum_{i<j}(x_(j) - x_(i)) = sum_i x_(i) (2i - N - 1).
#   3. Runs Diebold-Mariano tests with a Newey-West HAC variance (Bartlett kernel,
#      bandwidth h = floor(T_OoS^(1/3)) = 8, the _attic/runners/run_crps_dm.jl
#      convention) on all 10 pairwise loss differentials among the five rows, and
#      applies a Holm step-down correction over the 10 comparisons.
#
# Output:
#   results/crps_dm_kstar3/crps_dm_kstar3.txt   (setup, mean CRPS, DM panel, conclusion)
#   results/crps_dm_kstar3/per_day_losses.csv   (t, chmm_n, chmm_t_pen, chmm_l, chmm_ged,
#                                                chmm_t_shared_nu)
#   ../CHMM-Paper-Repository/results/robustness/crps_dm_kstar3.csv
#                                               (pair, mean_diff, dm_t, p_raw, p_holm)
#
# Usage: julia --project=. runners/robustness/run_crps_dm_kstar3.jl
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
using Random, Statistics, StatsBase, Printf;

const TICKER          = "SPY";
const RISK_FREE_RATE  = 0.0;
const ΔT              = 1/252;
const MAX_ITER        = 60;
const N_PATHS         = 1000;
const K_STAR          = 3;
const SEED            = 20260420;        # paper-canonical global seed
const LAMBDA_T        = 20.0;            # penalised CHMM-t 1/ν_k shrinkage rate
const NU_BOUNDS       = (2.1, 50.0);     # shared-ν ECM GSS bracket (run_chmm_t_shared_nu.jl)
const NU_INIT         = 6.0;
const ALPHA           = 0.05;            # Holm family-wise level for the conclusion line
const OUT_DIR         = joinpath(_ROOT, "results", "crps_dm_kstar3");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-Paper-Repository", "results", "robustness"));
mkpath(OUT_DIR);

println("="^88)
println("  K* = 3 CHMM CRPS Diebold-Mariano panel (Table 2 CRPS inference)")
println("  Rows: CHMM-N, CHMM-t (penalised λ = $LAMBDA_T), CHMM-L, CHMM-GED, shared-ν CHMM-t")
println("  Seed: $SEED   Paths: $N_PATHS   K*: $K_STAR")
println("="^88)

# ========================================================================================= #
# Helpers reproduced verbatim from run_kstar3_headline.jl (fit / stationary / simulate),
# so the per-model OoS simulation bundles match the headline panel to the draw.
# ========================================================================================= #
function _train_headline(family::Symbol, obs::Vector{Float64}, K::Int, max_iter::Int)
    if family == :gaussian
        return build(MyContinuousHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    elseif family == :student_t_pen
        return build(MyStudentTHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter,
             ν_shrink_rate=LAMBDA_T));
    elseif family == :laplace
        return build(MyLaplaceHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    elseif family == :ged
        return build(MyGEDHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    else
        error("Unknown emission family: $family")
    end
end

function _stationary(model, K::Int)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    π_stat = (T_mat^1000)[1, :];
    return T_mat, Categorical(π_stat);
end

function _simulate_paths(model, start_dist, n_is::Int, n_oos::Int, n_paths::Int)
    sim_is = Array{Float64,2}(undef, n_is, n_paths);
    sim_oos = Array{Float64,2}(undef, n_oos, n_paths);
    for i in 1:n_paths
        s0 = rand(start_dist);
        st = model(s0, n_is);
        for j in 1:n_is; sim_is[j,i] = rand(model.emission[st[j]]); end
        s0 = rand(start_dist);
        st = model(s0, n_oos);
        for j in 1:n_oos; sim_oos[j,i] = rand(model.emission[st[j]]); end
    end
    return sim_is, sim_oos;
end

# ========================================================================================= #
# Shared-ν Student-t ECM, reproduced verbatim from runners/headline/run_chmm_t_shared_nu.jl
# (single ν across all K states, GSS on the aggregate Q-function).
# ========================================================================================= #
function _logsumexp(v::AbstractVector)
    m = maximum(v)
    isfinite(m) || return m
    return m + log(sum(exp.(v .- m)))
end

function ecm_shared_nu(obs::AbstractVector, K::Int;
        max_iter::Int=60, tol::Float64=1e-4,
        ν_init::Float64=NU_INIT, ν_bounds::Tuple{Float64,Float64}=NU_BOUNDS)
    N = length(obs)
    sorted_data = sort(obs)
    chunk = floor(Int, N/K)
    μ = zeros(K); σ = zeros(K)
    for s in 1:K
        a = (s-1)*chunk + 1
        b = (s == K) ? N : s*chunk
        μ[s] = mean(sorted_data[a:b])
        σ[s] = max(std(sorted_data[a:b]), 1e-6)
    end
    ν = ν_init
    T = ones(K, K) ./ K
    π = ones(K) ./ K

    function logpdf_t(x, μk, σk, νk)
        return logpdf(LocationScale(μk, σk, TDist(νk)), x)
    end

    # Aggregate Q-function over shared ν: Σ_t Σ_k γ_t(k) logpdf_t(o_t; μ_k, σ_k, ν)
    function Q_shared(νv, γ, μ, σ, obs)
        acc = 0.0; n = length(obs); kk = size(γ, 2)
        for k in 1:kk
            d = LocationScale(μ[k], σ[k], TDist(νv))
            for t in 1:n
                acc += γ[t, k] * logpdf(d, obs[t])
            end
        end
        return acc
    end

    function gss_nu(γ, μ, σ, obs, lo, hi; iters=40)
        φ = (sqrt(5)-1)/2
        a = lo; b = hi
        c = b - φ*(b-a); d = a + φ*(b-a)
        fc = Q_shared(c, γ, μ, σ, obs); fd = Q_shared(d, γ, μ, σ, obs)
        for _ in 1:iters
            if fc > fd
                b = d; d = c; fd = fc
                c = b - φ*(b-a); fc = Q_shared(c, γ, μ, σ, obs)
            else
                a = c; c = d; fc = fd
                d = a + φ*(b-a); fd = Q_shared(d, γ, μ, σ, obs)
            end
        end
        return 0.5*(a+b)
    end

    last_μ = copy(μ); last_σ = copy(σ); last_ν = ν
    last_T = copy(T); last_π = copy(π)

    prev_ll = -Inf
    for _ in 1:max_iter
        # E-STEP
        log_B = zeros(N, K)
        for t in 1:N, k in 1:K
            log_B[t, k] = logpdf_t(obs[t], μ[k], σ[k], ν)
        end
        log_α = zeros(N, K)
        log_α[1, :] = log.(π) .+ log_B[1, :]
        for t in 2:N, j in 1:K
            log_α[t, j] = _logsumexp(log_α[t-1, :] .+ log.(T[:, j])) + log_B[t, j]
        end
        ll_now = _logsumexp(log_α[N, :])
        if !isfinite(ll_now)
            μ = last_μ; σ = last_σ; ν = last_ν; T = last_T; π = last_π
            break
        end
        last_μ = copy(μ); last_σ = copy(σ); last_ν = ν
        last_T = copy(T); last_π = copy(π)

        log_β = zeros(N, K)
        for t in N-1:-1:1, i in 1:K
            log_β[t, i] = _logsumexp(log.(T[i, :]) .+ log_B[t+1, :] .+ log_β[t+1, :])
        end

        γ = zeros(N, K)
        for t in 1:N
            denom = _logsumexp(log_α[t, :] .+ log_β[t, :])
            γ[t, :] = exp.((log_α[t, :] .+ log_β[t, :]) .- denom)
        end

        ξ_sum = zeros(K, K)
        for t in 1:N-1
            denom = _logsumexp(log_α[t, :] .+ log_β[t, :])
            for i in 1:K, j in 1:K
                ξ_sum[i, j] += exp(log_α[t, i] + log(T[i, j]) + log_B[t+1, j] + log_β[t+1, j] - denom)
            end
        end

        # Latent precisions with shared ν
        u = zeros(N, K)
        for t in 1:N, k in 1:K
            δ2 = ((obs[t] - μ[k]) / σ[k])^2
            u[t, k] = (ν + 1.0) / (ν + δ2)
        end

        # M-STEP
        π = γ[1, :]
        for k in 1:K
            wu = γ[:, k] .* u[:, k]
            Σwu = sum(wu); Σγ = sum(γ[:, k])
            if Σwu > 0
                μ[k] = sum(wu .* obs) / Σwu
            end
            if Σγ > 0
                σ2 = sum(wu .* (obs .- μ[k]).^2) / Σγ
                σ[k] = max(sqrt(max(σ2, 1e-12)), 1e-6)
            end
        end
        # Shared ν: GSS on aggregate Q
        ν = gss_nu(γ, μ, σ, obs, ν_bounds[1], ν_bounds[2])

        for i in 1:K
            r = sum(ξ_sum[i, :])
            if r > 0
                T[i, :] = ξ_sum[i, :] ./ r
            end
        end

        if abs(ll_now - prev_ll) < tol; break; end
        prev_ll = ll_now
    end

    return (T=T, μ=μ, σ=σ, ν=ν, π=π)
end

function _stationary_T(T)
    π = (T^1000)[1, :]
    return Categorical(π)
end

function _walk_states(T, s0, n)
    out = zeros(Int, n)
    out[1] = s0
    for j in 2:n
        out[j] = rand(Categorical(T[out[j-1], :]))
    end
    return out
end

function _sim_shared_nu(fit, n::Int, np::Int)
    sd = _stationary_T(fit.T)
    sim = Matrix{Float64}(undef, n, np)
    for p in 1:np
        s0 = rand(sd)
        st = _walk_states(fit.T, s0, n)
        for j in 1:n
            sim[j, p] = rand(LocationScale(fit.μ[st[j]], fit.σ[st[j]], TDist(fit.ν)))
        end
    end
    return sim
end

# ========================================================================================= #
# Sample CRPS via the unbiased sorted-ensemble identity (run_kstar3_headline.jl definition)
# ========================================================================================= #
function _sample_crps(y::Real, x::AbstractVector)
    N = length(x);
    s1 = mean(abs.(x .- y));
    xs = sort(x);
    s2_terms = sum(xs[i] * (2i - N - 1) for i in 1:N);
    s2 = s2_terms / (N * (N - 1));
    return s1 - s2;
end

function _crps_series(R::AbstractVector, sim::AbstractMatrix)
    n = length(R);
    size(sim, 1) == n || error("Ensemble T=$(size(sim,1)) mismatch with observed length=$n");
    losses = Array{Float64}(undef, n);
    @inbounds for t in 1:n
        losses[t] = _sample_crps(R[t], view(sim, t, :));
    end
    return losses;
end

# ========================================================================================= #
# Diebold-Mariano with Newey-West HAC variance (Bartlett kernel), from
# _attic/runners/run_crps_dm.jl: bandwidth h = floor(T^(1/3)).
# ========================================================================================= #
function cdf_std_normal(x::Real)
    return 0.5 * (1 + erf(x / sqrt(2)));
end

function dm_statistic(loss_a::AbstractVector, loss_b::AbstractVector;
                      bandwidth::Union{Nothing,Int}=nothing)
    T = length(loss_a);
    length(loss_b) == T || error("DM requires equal-length loss series");
    d = loss_a .- loss_b;
    dbar = mean(d);
    h = bandwidth === nothing ? max(1, floor(Int, T^(1/3))) : bandwidth;
    # Newey-West (Bartlett kernel) long-run variance estimator
    γ0 = sum((d .- dbar) .^ 2) / T;
    γ_sum = γ0;
    for k in 1:h
        γk = sum((d[1:end-k] .- dbar) .* (d[1+k:end] .- dbar)) / T;
        w = 1.0 - k / (h + 1);
        γ_sum += 2 * w * γk;
    end
    σ_lr = sqrt(max(γ_sum, eps()));
    dm = dbar / (σ_lr / sqrt(T));
    pval = 2 * (1 - cdf_std_normal(abs(dm)));
    return (dm=dm, dbar=dbar, σ_lr=σ_lr, h=h, pval=pval, T=T);
end

# Holm step-down adjustment over a family of m raw p-values.
function holm_adjust(p_raw::Vector{Float64})
    m = length(p_raw);
    order = sortperm(p_raw);
    adj = similar(p_raw);
    running_max = 0.0;
    for (rank, idx) in enumerate(order)
        val = min(1.0, (m - rank + 1) * p_raw[idx]);
        running_max = max(running_max, val);
        adj[idx] = running_max;
    end
    return adj;
end

# ========================================================================================= #
# Load SPY data (identical to run_kstar3_headline.jl)
# ========================================================================================= #
println("\n[1/4] Loading SPY data...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) ∈ train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
list_of_all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, list_of_all_tickers; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
idx_spy = findfirst(x -> x == TICKER, list_of_all_tickers);
R_is = all_R[:, idx_spy];
n_steps = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset, TICKER; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
n_steps_oos = length(R_oos);
println("  IS: $n_steps obs | OoS: $n_steps_oos obs")

# ========================================================================================= #
# Fit each row, simulate one bundle, score the per-day OoS CRPS series
# ========================================================================================= #
println("\n[2/4] Fitting the five K* = $K_STAR CHMM rows and scoring per-day OoS CRPS...")

const MODEL_KEYS   = ["chmm_n", "chmm_t_pen", "chmm_l", "chmm_ged", "chmm_t_shared_nu"];
const MODEL_LABELS = Dict(
    "chmm_n"           => "CHMM-N",
    "chmm_t_pen"       => "CHMM-t (penalised, lambda = $(Int(LAMBDA_T)))",
    "chmm_l"           => "CHMM-L",
    "chmm_ged"         => "CHMM-GED",
    "chmm_t_shared_nu" => "CHMM-t (shared nu)");
const HEADLINE_FAMILY = Dict(
    "chmm_n"     => :gaussian,
    "chmm_t_pen" => :student_t_pen,
    "chmm_l"     => :laplace,
    "chmm_ged"   => :ged);

loss_series = Dict{String,Vector{Float64}}();
mean_crps   = Dict{String,Float64}();

for key in MODEL_KEYS
    label = MODEL_LABELS[key];
    println("\n  $label | K = $K_STAR")
    local sim_oos;
    if key == "chmm_t_shared_nu"
        # Shared-ν runner convention: seed once, deterministic ECM fit, then IS-length
        # simulation followed by OoS-length simulation (RNG stream order preserved so the
        # OoS bundle matches results/chmm_t_shared_nu/sims_K3.jld2).
        Random.seed!(SEED);
        fit_time = @elapsed shared_fit = ecm_shared_nu(R_is, K_STAR; max_iter=MAX_ITER);
        @printf("    fit %.2f s | nu_shared = %.3f\n", fit_time, shared_fit.ν);
        _ = _sim_shared_nu(shared_fit, n_steps, N_PATHS);       # IS bundle (RNG order only)
        sim_oos = _sim_shared_nu(shared_fit, n_steps_oos, N_PATHS);
    else
        # Headline per-cell reseed convention: SEED for the fit, SEED + 1 for the bundle.
        family = HEADLINE_FAMILY[key];
        Random.seed!(SEED);
        fit_time = @elapsed model = _train_headline(family, R_is, K_STAR, MAX_ITER);
        @printf("    fit %.2f s\n", fit_time);
        _, start_dist = _stationary(model, K_STAR);
        Random.seed!(SEED + 1);
        _, sim_oos = _simulate_paths(model, start_dist, n_steps, n_steps_oos, N_PATHS);
    end
    losses = _crps_series(R_oos, sim_oos);
    loss_series[key] = losses;
    mean_crps[key] = mean(losses);
    @printf("    OoS mean CRPS = %.4f   median = %.4f   (T = %d, N = %d)\n",
            mean(losses), median(losses), n_steps_oos, N_PATHS);
end

# ========================================================================================= #
# Pairwise HAC DM tests + Holm correction over the 10 comparisons
# ========================================================================================= #
println("\n[3/4] Pairwise HAC Diebold-Mariano tests (10 pairs, Holm-corrected)...")

pairs_list = Tuple{String,String}[];
for i in 1:length(MODEL_KEYS), j in (i+1):length(MODEL_KEYS)
    push!(pairs_list, (MODEL_KEYS[i], MODEL_KEYS[j]));
end

dm_rows = NamedTuple[];
for (a, b) in pairs_list
    out = dm_statistic(loss_series[a], loss_series[b]);
    push!(dm_rows, (a=a, b=b, mean_diff=out.dbar, dm=out.dm, p_raw=out.pval, h=out.h));
end
p_holm = holm_adjust([r.p_raw for r in dm_rows]);
dm_rows = [(; r..., p_holm=p_holm[i]) for (i, r) in enumerate(dm_rows)];
bw = dm_rows[1].h;

n_reject_raw  = count(r -> r.p_raw  < ALPHA, dm_rows);
n_reject_holm = count(r -> r.p_holm < ALPHA, dm_rows);
for r in dm_rows
    @printf("  %-16s vs %-16s  dbar %+.6f  DM %+7.4f  p %.4f  p_Holm %.4f\n",
            r.a, r.b, r.mean_diff, r.dm, r.p_raw, r.p_holm);
end
println("  Rejections at $(ALPHA): raw = $n_reject_raw / 10, Holm = $n_reject_holm / 10");

# ========================================================================================= #
# Write artefacts
# ========================================================================================= #
println("\n[4/4] Writing artefacts...")

# --- per-day loss series ------------------------------------------------------------------ #
per_day_path = joinpath(OUT_DIR, "per_day_losses.csv");
open(per_day_path, "w") do io
    println(io, "t," * join(MODEL_KEYS, ","));
    for t in 1:n_steps_oos
        @printf(io, "%d", t);
        for key in MODEL_KEYS
            @printf(io, ",%.6f", loss_series[key][t]);
        end
        println(io);
    end
end
println("  $per_day_path");

# --- human-readable panel ----------------------------------------------------------------- #
txt_path = joinpath(OUT_DIR, "crps_dm_kstar3.txt");
open(txt_path, "w") do io
    println(io, "="^110);
    println(io, "K* = 3 CHMM CRPS Diebold-Mariano panel (Table 2 tab:model_comparison CRPS inference)");
    println(io, "="^110);
    println(io);
    println(io, "Setup    : SPY daily log excess growth; IS n = $n_steps, OoS T = $n_steps_oos; seed = $SEED; K* = $K_STAR.");
    println(io, "Rows     : CHMM-N, CHMM-t (penalised ECM, lambda = $LAMBDA_T), CHMM-L, CHMM-GED (per-cell reseed:");
    println(io, "           SEED fit / SEED+1 simulation, as run_kstar3_headline.jl), and shared-nu CHMM-t");
    println(io, "           (ecm_shared_nu of run_chmm_t_shared_nu.jl, nu bounds $NU_BOUNDS, own seed convention).");
    println(io, "Ensemble : one $N_PATHS-path unconditional (marginal-predictive) simulation bundle per model.");
    println(io, "CRPS     : unbiased sample CRPS via the sorted-ensemble identity (sec:crps_methods),");
    println(io, "           CRPS_t = (1/N) sum_i |x_i - y_t| - (1/(N(N-1))) sum_{i<j} |x_i - x_j|.");
    println(io, "DM       : Newey-West HAC variance, Bartlett kernel, bandwidth h = floor(T^(1/3)) = $bw;");
    println(io, "           two-sided p-values under the standard-normal null; Holm step-down over the 10 pairs.");
    println(io);
    println(io, "-"^110);
    println(io, "Per-model mean OoS sample-CRPS (lower is better):");
    println(io);
    @printf(io, "  %-38s %12s %12s\n", "Model", "mean CRPS", "median CRPS");
    println(io, "  ", "-"^64);
    for key in MODEL_KEYS
        @printf(io, "  %-38s %12.4f %12.4f\n",
                MODEL_LABELS[key], mean_crps[key], median(loss_series[key]));
    end
    println(io);
    println(io, "-"^110);
    println(io, "Pairwise Diebold-Mariano tests on the per-day OoS CRPS loss differential d_t = l_A(t) - l_B(t).");
    println(io, "Positive DM means row A's CRPS exceeds row B's (A worse). p-values are two-sided.");
    println(io);
    @printf(io, "  %-16s %-16s %12s %10s %10s %10s\n",
            "A", "B", "mean d_t", "DM t", "p raw", "p Holm");
    println(io, "  ", "-"^78);
    for r in dm_rows
        @printf(io, "  %-16s %-16s %+12.6f %+10.4f %10.4f %10.4f\n",
                r.a, r.b, r.mean_diff, r.dm, r.p_raw, r.p_holm);
    end
    println(io);
    println(io, "-"^110);
    if n_reject_holm == 0
        println(io, "Conclusion: none of the 10 pairwise DM tests rejects at the $(ALPHA) level after Holm");
        println(io, "correction (raw rejections: $n_reject_raw/10). The five displayed K* = 3 CHMM rows are");
        println(io, "statistically indistinguishable on mean OoS sample-CRPS under the HAC DM test.");
    else
        println(io, "Conclusion: $n_reject_holm of the 10 pairwise DM tests reject at the $(ALPHA) level after");
        println(io, "Holm correction (raw rejections: $n_reject_raw/10). The displayed K* = 3 CHMM rows are");
        println(io, "NOT all statistically indistinguishable on mean OoS sample-CRPS; see the panel above.");
    end
    println(io);
    println(io, "Source: runners/robustness/run_crps_dm_kstar3.jl (per-day losses in per_day_losses.csv).");
end
println("  $txt_path");

# --- paper-repo summary CSV --------------------------------------------------------------- #
mkpath(PAPER_ROBUSTNESS_DIR);
paper_csv_path = joinpath(PAPER_ROBUSTNESS_DIR, "crps_dm_kstar3.csv");
open(paper_csv_path, "w") do io
    println(io, "pair,mean_diff,dm_t,p_raw,p_holm");
    for r in dm_rows
        @printf(io, "%s_vs_%s,%.6f,%.4f,%.4f,%.4f\n",
                r.a, r.b, r.mean_diff, r.dm, r.p_raw, r.p_holm);
    end
end
println("  $paper_csv_path");

println("\nDone. Mean OoS CRPS by row:");
for key in MODEL_KEYS
    @printf("  %-38s %.4f\n", MODEL_LABELS[key], mean_crps[key]);
end
@printf("DM: %d/10 raw rejections, %d/10 Holm rejections at alpha = %.2f (h = %d, T = %d).\n",
        n_reject_raw, n_reject_holm, ALPHA, bw, n_steps_oos);
