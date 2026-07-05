# ========================================================================================= #
# run_table2_full.jl
#
# Post-processor that assembles ONE authoritative Table 2 with the FULL seven-metric panel
# (KS, AD, excess kurtosis, |G| ACF-MAE, raw ACF-MAE, Wasserstein-1, Hellinger, Coverage)
# for BOTH the in-sample and out-of-sample windows, for every generator on the K*=3
# headline slide. Reuses the validated eval_full metric function; loads saved sim archives
# where available (baselines, shared-nu, MS-GARCH) and refits the K=3 CHMM family cheaply.
#
# Output: results/SPY/Table-2-Full.{csv,txt}
# Usage:  julia --project=. runners/headline/run_table2_full.jl
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
using Random, Statistics, HypothesisTests, StatsBase, Printf, Distributions, JLD2, DataFrames;

const TICKER = "SPY";
const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const SEED = 20260420;
const N_PATHS = 1000;
const MAX_ITER = 60;
const L = 252;
const RESULTS_DIR = joinpath(_ROOT, "results");

# ----------------------------------------------------------------------------------------- #
# Observed IS / OoS series (same convention as the other headline runners)
# ----------------------------------------------------------------------------------------- #
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
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
println("Observed: IS T=$n_steps, OoS T=$n_steps_oos");

# ----------------------------------------------------------------------------------------- #
# Full seven-metric panel (verbatim from run_baselines_and_cross_asset.jl::eval_full)
# ----------------------------------------------------------------------------------------- #
function eval_full(observed, sim_archive; L_val=L)
    np = size(sim_archive, 2); n_o = length(observed);
    μ_o = mean(observed); σ_o = std(observed);
    kurt_o = sum(((observed .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L_val, n_o - 1);
    acf_o = autocor(abs.(observed), 1:L_use);
    acf_o_raw = autocor(observed, 1:L_use);
    ks_pass = 0; ad_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0; acf_mae_raw_s = 0.0;
    w1_s = 0.0; hell_s = 0.0;
    obs_qprobs = range(0.01, 0.99, length=99);
    obs_quantiles = quantile(observed, obs_qprobs);
    sim_qmatrix = zeros(99, np);
    for i in 1:np
        sim = sim_archive[:, i];
        pval_ks = pvalue(ApproximateTwoSampleKSTest(observed, sim));
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
    cov_count = 0;
    for q in 1:99
        lo_env = quantile(sim_qmatrix[q, :], 0.05);
        hi_env = quantile(sim_qmatrix[q, :], 0.95);
        if obs_quantiles[q] >= lo_env && obs_quantiles[q] <= hi_env; cov_count += 1; end
    end
    return (ks=round(100*ks_pass/np, digits=1), ad=round(100*ad_pass/np, digits=1),
            kurt=round(kurt_s/np, digits=2), kurt_obs=round(kurt_o, digits=2),
            acf_mae=round(acf_mae_s/np, digits=4), acf_mae_raw=round(acf_mae_raw_s/np, digits=4),
            w1=round(w1_s/np, digits=3), hell=round(hell_s/np, digits=4),
            cov=round(100.0*cov_count/99, digits=1));
end

# ----------------------------------------------------------------------------------------- #
# CHMM fit + simulate helpers (verbatim from run_multi_emission_analysis.jl)
# ----------------------------------------------------------------------------------------- #
function _train_family(family::Symbol, obs::Vector{Float64}, K::Int, max_iter::Int)
    if family == :gaussian
        return build(MyContinuousHiddenMarkovModel, (observations=obs, number_of_states=K, max_iter=max_iter));
    elseif family == :student_t
        return build(MyStudentTHiddenMarkovModel, (observations=obs, number_of_states=K, max_iter=max_iter));
    elseif family == :laplace
        return build(MyLaplaceHiddenMarkovModel, (observations=obs, number_of_states=K, max_iter=max_iter));
    elseif family == :ged
        return build(MyGEDHiddenMarkovModel, (observations=obs, number_of_states=K, max_iter=max_iter));
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
        s0 = rand(start_dist); st = model(s0, n_is);
        for j in 1:n_is; sim_is[j,i] = rand(model.emission[st[j]]); end
        s0 = rand(start_dist); st = model(s0, n_oos);
        for j in 1:n_oos; sim_oos[j,i] = rand(model.emission[st[j]]); end
    end
    return sim_is, sim_oos;
end

# ----------------------------------------------------------------------------------------- #
# Assemble every generator's IS/OoS metric rows
# ----------------------------------------------------------------------------------------- #
rows = Vector{NamedTuple}();
addrow!(name, mi, mo) = push!(rows, (name=name, mi=mi, mo=mo));

# (1) Baselines from the saved archive (Bootstrap, Gaussian, Laplace, Discrete NJ/WJ, GARCH)
arch = load(joinpath(RESULTS_DIR, "baselines_archive", "sim_archive_cache.jld2"), "archive");
for name in ["Bootstrap", "Gaussian", "Laplace", "DiscreteNJ", "DiscreteWJ", "GARCH"]
    println("[eval] baseline $name ...");
    addrow!(name, eval_full(R_is, arch[name].is), eval_full(R_oos, arch[name].oos));
end

# (2) CHMM family at K*=3 (refit; deterministic seed matches multi_emission)
for (family, label) in [(:gaussian, "CHMM-N"), (:student_t, "CHMM-t (per-state nu)"),
                        (:laplace, "CHMM-L"), (:ged, "CHMM-GED")]
    println("[fit]  $label at K=3 ...");
    Random.seed!(SEED); model = _train_family(family, R_is, 3, MAX_ITER);
    _, start_dist = _stationary(model, 3);
    Random.seed!(SEED + 1); sim_is, sim_oos = _simulate_paths(model, start_dist, n_steps, n_steps_oos, N_PATHS);
    addrow!(label, eval_full(R_is, sim_is), eval_full(R_oos, sim_oos));
end

# (3) Shared-nu CHMM-t at K*=3 (from the sim archive saved by run_chmm_t_shared_nu.jl)
snu = joinpath(RESULTS_DIR, "chmm_t_shared_nu", "sims_K3.jld2");
if isfile(snu)
    println("[eval] CHMM-t (shared nu) at K=3 ...");
    d = load(snu); addrow!("CHMM-t (shared nu)", eval_full(R_is, d["is"]), eval_full(R_oos, d["oos"]));
else
    println("[skip] shared-nu sims not found ($snu); run run_chmm_t_shared_nu.jl first.");
end

# (4) MS-GARCH K=3 (bonus; sims saved by the MS-GARCH reference runner)
msg = joinpath(RESULTS_DIR, "msgarch_reference", "sims_K3.jld2");
if isfile(msg)
    println("[eval] MS-GARCH (K=3) ...");
    d = load(msg); addrow!("MS-GARCH (K=3)", eval_full(R_is, d["is"]), eval_full(R_oos, d["oos"]));
end

# ----------------------------------------------------------------------------------------- #
# Emit Table-2-Full.{csv,txt}
# ----------------------------------------------------------------------------------------- #
outdir = joinpath(RESULTS_DIR, "SPY"); mkpath(outdir);
csv_path = joinpath(outdir, "Table-2-Full.csv");
open(csv_path, "w") do io
    println(io, "generator,ks_is,ad_is,kurt_is,acf_is,acf_raw_is,w1_is,hell_is,cov_is," *
                "ks_oos,ad_oos,kurt_oos,acf_oos,acf_raw_oos,w1_oos,hell_oos,cov_oos,kurt_obs_is,kurt_obs_oos");
    for r in rows
        @printf(io, "%s,%.1f,%.1f,%.2f,%.4f,%.4f,%.3f,%.4f,%.1f,%.1f,%.1f,%.2f,%.4f,%.4f,%.3f,%.4f,%.1f,%.2f,%.2f\n",
                r.name, r.mi.ks, r.mi.ad, r.mi.kurt, r.mi.acf_mae, r.mi.acf_mae_raw, r.mi.w1, r.mi.hell, r.mi.cov,
                r.mo.ks, r.mo.ad, r.mo.kurt, r.mo.acf_mae, r.mo.acf_mae_raw, r.mo.w1, r.mo.hell, r.mo.cov,
                r.mi.kurt_obs, r.mo.kurt_obs);
    end
end
txt_path = joinpath(outdir, "Table-2-Full.txt");
open(txt_path, "w") do io
    println(io, "TABLE 2 (FULL) — SPY, K*=3, $N_PATHS paths, seed $SEED. Observed excess kurtosis: IS $(rows[1].mi.kurt_obs), OoS $(rows[1].mo.kurt_obs).");
    println(io, "="^160);
    @printf(io, "%-22s | %-31s | %-31s\n", "", "IN-SAMPLE (T=$n_steps)", "OUT-OF-SAMPLE (T=$n_steps_oos)");
    @printf(io, "%-22s | %5s %5s %6s %7s %6s %6s %5s | %5s %5s %6s %7s %6s %6s %5s\n",
            "Generator", "KS", "AD", "kurt", "ACF|G|", "W1", "Hell", "Cov", "KS", "AD", "kurt", "ACF|G|", "W1", "Hell", "Cov");
    println(io, "-"^160);
    for r in rows
        @printf(io, "%-22s | %5.1f %5.1f %6.2f %7.4f %6.3f %6.4f %5.1f | %5.1f %5.1f %6.2f %7.4f %6.3f %6.4f %5.1f\n",
                r.name, r.mi.ks, r.mi.ad, r.mi.kurt, r.mi.acf_mae, r.mi.w1, r.mi.hell, r.mi.cov,
                r.mo.ks, r.mo.ad, r.mo.kurt, r.mo.acf_mae, r.mo.w1, r.mo.hell, r.mo.cov);
    end
end
println("\nWrote $csv_path\nWrote $txt_path");
