# ========================================================================================= #
# run_kstar3_headline.jl
#
# Regenerates the K* = 3 four-emission headline panel consumed by body Table 2
# (tab:model_comparison) CHMM rows. Fits CHMM-N, CHMM-t (penalised ECM at λ = 20),
# CHMM-L, and CHMM-GED on the SPY in-sample window at K* = 3, simulates 1000 IS- and
# OoS-length paths, and scores the seven-metric distributional panel plus the OoS
# sample-CRPS.
#
# This driver supersedes the earlier one-shot script that produced
# results/kstar3_headline/metrics.csv but was never committed. It reuses the exact
# protocol of run_multi_emission_analysis.jl (per-cell reseed: SEED for the fit,
# SEED + 1 for the 1000-path simulation), so the CHMM-N / -L / -GED rows reproduce
# the K = 3 cells of results/SPY/multi_emission/ to the decimal, and the penalised
# CHMM-t row is computed with the same harness rather than carried as a fixed value.
#
# Output:
#   results/kstar3_headline/metrics.csv    (schema consumed by the paper)
#   results/kstar3_headline/summary.txt    (human-readable panel)
#
# Usage: julia --project=. runners/headline/run_kstar3_headline.jl
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
using Random, Statistics, StatsBase, Printf;

const TICKER          = "SPY";
const RISK_FREE_RATE  = 0.0;
const ΔT              = 1/252;
const MAX_ITER        = 60;
const N_PATHS         = 1000;
const L               = 252;
const K_STAR          = 3;
const SEED            = 20260420;        # paper-canonical global seed
const LAMBDA_T        = 20.0;            # penalised CHMM-t 1/ν_k shrinkage rate
const RESULTS_DIR     = joinpath(_ROOT, "results");
const OUT_DIR         = joinpath(RESULTS_DIR, "kstar3_headline");
mkpath(OUT_DIR);

println("="^70)
println("  K* = 3 four-emission headline (Table 2 CHMM rows)")
println("  Families: CHMM-N, CHMM-t (penalised λ = $LAMBDA_T), CHMM-L, CHMM-GED")
println("  Seed: $SEED   Paths: $N_PATHS   K*: $K_STAR")
println("="^70)

# ========================================================================================= #
# Helpers (reproduced verbatim from run_multi_emission_analysis.jl so the metric
# definitions and seeding are identical across the two runners)
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

function eval_metrics(observed, sim_archive; L_val=L)
    np = size(sim_archive, 2); n_o = length(observed);
    μ_o = mean(observed); σ_o = std(observed);
    kurt_o = sum(((observed .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L_val, n_o - 1);
    acf_o = autocor(abs.(observed), 1:L_use);
    acf_o_raw = autocor(observed, 1:L_use);

    ks_pass = 0; ad_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0; acf_mae_raw_s = 0.0;
    w1_s = 0.0; hell_s = 0.0;
    ks_pvals = Float64[];

    obs_qprobs = range(0.01, 0.99, length=99);
    obs_quantiles = quantile(observed, obs_qprobs);
    sim_qmatrix = zeros(99, np);

    for i in 1:np
        sim = sim_archive[:, i];

        pval_ks = pvalue(ApproximateTwoSampleKSTest(observed, sim));
        push!(ks_pvals, pval_ks);
        if pval_ks > 0.05; ks_pass += 1; end

        pval_ad = pvalue(KSampleADTest(observed, sim));
        if pval_ad > 0.05; ad_pass += 1; end

        μ_s = mean(sim); σ_s = std(sim);
        kurt_s += sum(((sim .- μ_s) ./ σ_s).^4) / length(sim) - 3.0;

        acf_sim = autocor(abs.(sim), 1:L_use);
        acf_mae_s += mean(abs.(acf_o .- acf_sim));

        acf_sim_raw = autocor(sim, 1:L_use);
        acf_mae_raw_s += mean(abs.(acf_o_raw .- acf_sim_raw));

        obs_sorted = sort(observed); sim_sorted = sort(sim);
        n_min = min(length(obs_sorted), length(sim_sorted));
        obs_q = [obs_sorted[max(1, round(Int, k*length(obs_sorted)/n_min))] for k in 1:n_min];
        sim_q = [sim_sorted[max(1, round(Int, k*length(sim_sorted)/n_min))] for k in 1:n_min];
        w1_s += mean(abs.(obs_q .- sim_q));

        lo = min(minimum(observed), minimum(sim)) - 10;
        hi = max(maximum(observed), maximum(sim)) + 10;
        edges = range(lo, hi, length=101);
        h_o = fit(Histogram, observed, edges).weights ./ n_o;
        h_s = fit(Histogram, sim, edges).weights ./ length(sim);
        hell_s += sqrt(sum((sqrt.(h_o) .- sqrt.(h_s)).^2)) / sqrt(2);

        sim_qmatrix[:, i] = quantile(sim, obs_qprobs);
    end

    return (ks=round(100*ks_pass/np, digits=1),
            ad=round(100*ad_pass/np, digits=1),
            kurt=round(kurt_s/np, digits=2), kurt_obs=round(kurt_o, digits=2),
            acf_mae=round(acf_mae_s/np, digits=4),
            acf_mae_raw=round(acf_mae_raw_s/np, digits=4),
            w1=round(w1_s/np, digits=3), hell=round(hell_s/np, digits=4));
end

# Sample CRPS via the unbiased sorted-ensemble identity (from run_crps_extra_rows.jl)
function _sample_crps(y::Real, x::AbstractVector)
    N = length(x);
    s1 = mean(abs.(x .- y));
    xs = sort(x);
    s2_terms = sum(xs[i] * (2i - N - 1) for i in 1:N);
    s2 = s2_terms / (N * (N - 1));
    return s1 - s2;
end

function _crps_path_mean(R::AbstractVector, sim::AbstractMatrix)
    n = length(R);
    mean_crps = 0.0;
    @inbounds for t in 1:n
        x = view(sim, t, :);
        mean_crps += _sample_crps(R[t], x);
    end
    return mean_crps / n;
end

# ========================================================================================= #
# Load SPY data (identical to run_multi_emission_analysis.jl)
# ========================================================================================= #
println("\n[1/3] Loading SPY data...")
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
# Fit + evaluate the four headline families at K* = 3
# ========================================================================================= #
println("\n[2/3] Fitting + evaluating the four CHMM families at K* = $K_STAR...")

const FAMILIES = [:gaussian, :student_t_pen, :laplace, :ged];
const FAMILY_LABEL = Dict(
    :gaussian       => "CHMM-N",
    :student_t_pen  => "CHMM-t (penalised, lambda = $(Int(LAMBDA_T)))",
    :laplace        => "CHMM-L",
    :ged            => "CHMM-GED");

rows = Vector{NamedTuple}();
for family in FAMILIES
    label = FAMILY_LABEL[family];
    println("\n  $label | K = $K_STAR")

    Random.seed!(SEED);
    fit_time = @elapsed model = _train_headline(family, R_is, K_STAR, MAX_ITER);
    _, start_dist = _stationary(model, K_STAR);

    Random.seed!(SEED + 1);
    sim_is, sim_oos = _simulate_paths(model, start_dist, n_steps, n_steps_oos, N_PATHS);
    m_is  = eval_metrics(R_is,  sim_is);
    m_oos = eval_metrics(R_oos, sim_oos);
    crps_oos = round(_crps_path_mean(R_oos, sim_oos), digits=4);

    push!(rows, (
        family=family, label=label, K=K_STAR,
        is_ks=m_is.ks, is_kurt=m_is.kurt, is_acf=m_is.acf_mae, is_acf_raw=m_is.acf_mae_raw,
        oos_ks=m_oos.ks, oos_kurt=m_oos.kurt, oos_acf=m_oos.acf_mae, oos_acf_raw=m_oos.acf_mae_raw,
        crps=crps_oos, fit_time=round(fit_time, digits=2)));

    @printf("    IS KS %.1f%%  kurt %.2f | OoS KS %.1f%%  kurt %.2f | CRPS %.4f\n",
            m_is.ks, m_is.kurt, m_oos.ks, m_oos.kurt, crps_oos);
end

# ========================================================================================= #
# Write artefacts
# ========================================================================================= #
println("\n[3/3] Writing results/kstar3_headline/{metrics.csv,summary.txt}...")

open(joinpath(OUT_DIR, "metrics.csv"), "w") do io
    println(io, "family,label,K,is_ks_pct,is_sim_kurt,is_acf_mae,is_acf_mae_raw,oos_ks_pct,oos_sim_kurt,oos_acf_mae,oos_acf_mae_raw,oos_crps,fit_time_s");
    for r in rows
        @printf(io, "%s,\"%s\",%d,%.1f,%.2f,%.4f,%.4f,%.1f,%.2f,%.4f,%.4f,%.4f,%.2f\n",
                r.family, r.label, r.K,
                r.is_ks, r.is_kurt, r.is_acf, r.is_acf_raw,
                r.oos_ks, r.oos_kurt, r.oos_acf, r.oos_acf_raw,
                r.crps, r.fit_time);
    end
end

open(joinpath(OUT_DIR, "summary.txt"), "w") do io
    println(io, "="^110);
    println(io, "K* = 3 four-emission headline (Table 2 tab:model_comparison CHMM rows)");
    println(io, "="^110);
    println(io);
    _exkurt(x) = sum(((x .- mean(x)) ./ std(x)).^4) / length(x) - 3.0;
    println(io, "Setup: SPY IS = $n_steps, OoS = $n_steps_oos, paths = $N_PATHS, seed = $SEED, K* = $K_STAR");
    println(io, "Penalised CHMM-t: 1/ν_k shrinkage at λ = $LAMBDA_T on the per-state ECM Q-function.");
    println(io, "Observed excess kurtosis: IS = $(round(_exkurt(R_is), digits=2)), OoS = $(round(_exkurt(R_oos), digits=2))");
    println(io);
    @printf(io, "%-34s %7s %8s %8s %9s %10s %9s %8s\n",
            "Model", "IS KS%", "OoS KS%", "Kurt IS", "Kurt OoS", "|G_t| ACF", "G_t ACF", "CRPS OoS");
    println(io, "-"^110);
    for r in rows
        @printf(io, "%-34s %7.1f %8.1f %8.2f %9.2f %10.4f %9.4f %8.4f\n",
                r.label, r.is_ks, r.oos_ks, r.is_kurt, r.oos_kurt, r.is_acf, r.is_acf_raw, r.crps);
    end
    println(io);
    println(io, "Source: results/kstar3_headline/metrics.csv (run_kstar3_headline.jl)");
end

println("\nDone. Headline panel:");
for r in rows
    @printf("  %-34s IS KS %.1f  OoS KS %.1f  kurt %.2f/%.2f  CRPS %.4f\n",
            r.label, r.is_ks, r.oos_ks, r.is_kurt, r.oos_kurt, r.crps);
end
