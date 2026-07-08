# ========================================================================================= #
# run_garcht_full_metrics.jl
#
# Scores the GARCH(1,1) Student-t baseline on the SAME full seven-metric panel used by
# run_table2_full.jl (eval_full): KS, AD, excess kurtosis, |G| ACF-MAE, raw ACF-MAE,
# Wasserstein-1, Hellinger, Coverage, for IS and OoS. The garch suite
# (run_garch_suite.jl) only reports KS/AD/kurt/ACF/VaR for GARCH-t; this adds the
# Coverage / Wasserstein-1 / Hellinger cells needed for the deck's slide-13 table so the
# GARCH-t column is apples-to-apples with the Bootstrap and CHMM columns of Table-2-Full.
#
# The GARCH-t fit and simulated paths reproduce run_garch_suite.jl's seed policy exactly,
# so KS/AD/kurt/ACF here match GARCH_Suite.txt; the three new metrics come from eval_full.
#
# Output: results/garch_suite/GARCHt_Full_Metrics.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
using Random, Statistics, HypothesisTests, StatsBase, Printf, Distributions, DataFrames;

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const N_PATHS   = 1000;
const L         = 252;
const SEED      = 20260422;   # identical to run_garch_suite.jl

# ---- data (same as run_garch_suite.jl / run_table2_full.jl) ------------------------------ #
println("[data] Loading SPY IS + OoS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]); n_is = length(R_is);
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE)); n_oos = length(R_oos);
println("  IS $n_is, OoS $n_oos");

# ---- eval_full: verbatim from run_table2_full.jl ---------------------------------------- #
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

# ---- GARCH-t fit + sim (reproduce run_garch_suite.jl seed policy) ------------------------ #
function _simulate_paths(sim_fn, T::Int, n_paths::Int)::Matrix{Float64}
    paths = Matrix{Float64}(undef, T, n_paths);
    for p in 1:n_paths; paths[:, p] = sim_fn(); end
    return paths;
end

println("[fit] GARCH(1,1) Student-t...");
Random.seed!(SEED + 4);
fit_gt = fit_garcht11(R_is);
println("  GARCH-t: omega=$(round(fit_gt.ω, digits=4)) alpha=$(round(fit_gt.α, digits=3)) beta=$(round(fit_gt.β, digits=3)) nu=$(round(fit_gt.ν, digits=2))");

Random.seed!(SEED + 100);
sim_is = _simulate_paths(() -> simulate_garcht(fit_gt, n_is), n_is, N_PATHS);
Random.seed!(SEED + 101);
sim_oos = _simulate_paths(() -> simulate_garcht(fit_gt, n_oos), n_oos, N_PATHS);

println("[eval] GARCH-t full metrics...");
mi = eval_full(R_is, sim_is);
mo = eval_full(R_oos, sim_oos);

open(joinpath(_ROOT, "results", "garch_suite", "GARCHt_Full_Metrics.txt"), "w") do io
    for (tag, m) in (("IS", mi), ("OOS", mo))
        println(io, "GARCHT_FULL $tag ks=$(m.ks) ad=$(m.ad) kurt=$(m.kurt) kurt_obs=$(m.kurt_obs) " *
                    "acf=$(m.acf_mae) acf_raw=$(m.acf_mae_raw) w1=$(m.w1) hell=$(m.hell) cov=$(m.cov)");
    end
end
println("="^72)
println("GARCHT_FULL IS  ks=$(mi.ks) ad=$(mi.ad) kurt=$(mi.kurt) kurt_obs=$(mi.kurt_obs) acf=$(mi.acf_mae) w1=$(mi.w1) hell=$(mi.hell) cov=$(mi.cov)")
println("GARCHT_FULL OOS ks=$(mo.ks) ad=$(mo.ad) kurt=$(mo.kurt) kurt_obs=$(mo.kurt_obs) acf=$(mo.acf_mae) w1=$(mo.w1) hell=$(mo.hell) cov=$(mo.cov)")
println("="^72)
@info "Done."
