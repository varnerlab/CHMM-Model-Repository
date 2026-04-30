# ========================================================================================= #
# run_kstar3_headline.jl
#
# Produces the headline four-emission rows at K* = 3 (CHMM-N, CHMM-t penalised at lambda
# = 20, CHMM-L, CHMM-GED) on the SPY IS / OoS windows. The K* = 3 block becomes the body
# headline in the resubmission per peer-review item R1 W1 (the four-fold and six-fold
# rolling-origin CV on the pre-2020 slice cannot distinguish K = 6 from K = 3 outside
# sampling error; see results/k_selection_validation/K_Selection_Kfold*Pre2020.txt).
#
# Outputs:
#   results/kstar3_headline/metrics.csv        -- one row per family with the Table 3 columns
#   results/kstar3_headline/summary.txt        -- human-readable summary
#   ../CHMM-paper/results/robustness/kstar3_headline.csv -- paper-side import
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
using StatsBase
using Printf
using LinearAlgebra

const SEED      = 20260420;        # match the K*=6 headline seed convention
const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const N_PATHS   = 1_000;
const L_LAGS    = 252;
const K_STAR    = 3;
const MAX_ITER  = 60;
const LAMBDA    = 20.0;            # body penalty rate for CHMM-t

const OUT_DIR              = joinpath(_ROOT, "results", "kstar3_headline");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^80);
println("  K*=3 four-emission headline (R1 W1 contingency)");
println("="^80);
Random.seed!(SEED);

# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY IS + OoS...");
train_dataset = MyPortfolioDataSet()["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet()["dataset"];
R_is  = Vector{Float64}(log_growth_matrix(train_dataset, TICKER; Δt=DT, risk_free_rate=RISK_FREE));
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset,   TICKER; Δt=DT, risk_free_rate=RISK_FREE));
n_is  = length(R_is); n_oos = length(R_oos);
@printf("  T_IS = %d   T_OoS = %d\n", n_is, n_oos);

obs_kurt_is  = sum(((R_is  .- mean(R_is))  ./ std(R_is))  .^ 4) / n_is  - 3.0;
obs_kurt_oos = sum(((R_oos .- mean(R_oos)) ./ std(R_oos)) .^ 4) / n_oos - 3.0;
@printf("  observed excess kurtosis: IS = %.2f, OoS = %.2f\n", obs_kurt_is, obs_kurt_oos);

# --------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------- #
function _stationary(model, K::Int)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    π_stat = (T_mat^1000)[1, :];
    π_stat = max.(π_stat, 1e-12); π_stat ./= sum(π_stat);
    return T_mat, Categorical(π_stat);
end

function _simulate_paths(model, start_dist, n_is::Int, n_oos::Int, n_paths::Int; seed::Int)
    Random.seed!(seed);
    sim_is  = Array{Float64,2}(undef, n_is,  n_paths);
    sim_oos = Array{Float64,2}(undef, n_oos, n_paths);
    for i in 1:n_paths
        s0 = rand(start_dist); st = model(s0, n_is);
        for j in 1:n_is;  sim_is[j, i]  = rand(model.emission[st[j]]); end
        s0 = rand(start_dist); st = model(s0, n_oos);
        for j in 1:n_oos; sim_oos[j, i] = rand(model.emission[st[j]]); end
    end
    return sim_is, sim_oos;
end

# Sample CRPS via the unbiased sorted-ensemble identity (matches run_crps_extra_rows.jl)
function _sample_crps(y::Real, x::AbstractVector)
    N = length(x);
    s1 = mean(abs.(x .- y));
    xs = sort(collect(x));
    s2 = sum(xs[i] * (2i - N - 1) for i in 1:N) / (N * (N - 1));
    return s1 - s2;
end

function _crps_path_mean(R::AbstractVector, sim::AbstractMatrix)
    n = length(R); mean_crps = 0.0;
    @inbounds for t in 1:n
        mean_crps += _sample_crps(R[t], view(sim, t, :));
    end
    return mean_crps / n;
end

function _eval(R_obs, sim_archive)
    np = size(sim_archive, 2);
    n_o = length(R_obs);
    μ_o = mean(R_obs); σ_o = std(R_obs);
    L_use = min(L_LAGS, n_o - 1);
    acf_o     = autocor(abs.(R_obs), 1:L_use);
    acf_o_raw = autocor(R_obs,        1:L_use);

    ks_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0; acf_mae_raw_s = 0.0;
    for i in 1:np
        s = sim_archive[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= 0.05; ks_pass += 1; end
        μ_s = mean(s); σ_s = std(s);
        kurt_s += sum(((s .- μ_s) ./ σ_s) .^ 4) / length(s) - 3.0;
        acf_mae_s     += mean(abs.(acf_o     .- autocor(abs.(s), 1:L_use)));
        acf_mae_raw_s += mean(abs.(acf_o_raw .- autocor(s,        1:L_use)));
    end
    return (
        ks_pct      = round(100 * ks_pass / np, digits=1),
        sim_kurt    = round(kurt_s / np, digits=2),
        acf_mae     = round(acf_mae_s / np, digits=4),
        acf_mae_raw = round(acf_mae_raw_s / np, digits=4),
    );
end

# --------------------------------------------------------------------------------------- #
# Build + simulate per family
# --------------------------------------------------------------------------------------- #
function _fit_family(family::Symbol, K::Int)
    if family == :gaussian
        return build(MyContinuousHiddenMarkovModel,
            (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    elseif family == :student_t_pen
        # Penalised CHMM-t at λ = 20 (exponential 1/ν_k shrinkage rate;
        # body operating point in Table 3, see Factory.jl `ν_shrink_rate` kwarg)
        return build(MyStudentTHiddenMarkovModel,
            (observations=R_is, number_of_states=K, max_iter=MAX_ITER,
             ν_shrink_rate=LAMBDA));
    elseif family == :laplace
        return build(MyLaplaceHiddenMarkovModel,
            (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    elseif family == :ged
        return build(MyGEDHiddenMarkovModel,
            (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    else
        error("Unknown family: $family");
    end
end

const FAMILIES = [
    (:gaussian,      "CHMM-N",                          SEED + 11),
    (:student_t_pen, "CHMM-t (penalised, lambda = 20)", SEED + 12),
    (:laplace,       "CHMM-L",                          SEED + 13),
    (:ged,           "CHMM-GED",                        SEED + 14),
];

results = Vector{NamedTuple}();

for (family, label, fam_seed) in FAMILIES
    println("\n[$label] Fitting at K = $K_STAR ...");
    Random.seed!(fam_seed);
    t0 = time();
    m = _fit_family(family, K_STAR);
    dt_fit = time() - t0;
    @printf("  fit in %.1f s\n", dt_fit);

    _, start_dist = _stationary(m, K_STAR);
    sim_is, sim_oos = _simulate_paths(m, start_dist, n_is, n_oos, N_PATHS; seed=fam_seed + 1000);

    is_panel  = _eval(R_is,  sim_is);
    oos_panel = _eval(R_oos, sim_oos);
    crps_oos  = round(_crps_path_mean(R_oos, sim_oos), digits=4);

    @printf("  IS  KS = %4.1f%%  sim kurt = %5.2f  |G_t| ACF-MAE = %.4f  G_t ACF-MAE = %.4f\n",
            is_panel.ks_pct,  is_panel.sim_kurt,  is_panel.acf_mae,  is_panel.acf_mae_raw);
    @printf("  OoS KS = %4.1f%%  sim kurt = %5.2f  |G_t| ACF-MAE = %.4f  G_t ACF-MAE = %.4f  CRPS = %.4f\n",
            oos_panel.ks_pct, oos_panel.sim_kurt, oos_panel.acf_mae, oos_panel.acf_mae_raw, crps_oos);

    push!(results, (
        family=family, label=label, K=K_STAR,
        is=is_panel, oos=oos_panel, crps_oos=crps_oos, fit_time_s=dt_fit,
    ));
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "metrics.csv");
open(csv_path, "w") do io
    println(io, "family,label,K,is_ks_pct,is_sim_kurt,is_acf_mae,is_acf_mae_raw,oos_ks_pct,oos_sim_kurt,oos_acf_mae,oos_acf_mae_raw,oos_crps,fit_time_s");
    for r in results
        @printf(io, "%s,\"%s\",%d,%.1f,%.2f,%.4f,%.4f,%.1f,%.2f,%.4f,%.4f,%.4f,%.2f\n",
                String(r.family), r.label, r.K,
                r.is.ks_pct, r.is.sim_kurt, r.is.acf_mae, r.is.acf_mae_raw,
                r.oos.ks_pct, r.oos.sim_kurt, r.oos.acf_mae, r.oos.acf_mae_raw,
                r.crps_oos, r.fit_time_s);
    end
end
# Mirror to paper-side robustness dir
cp(csv_path, joinpath(PAPER_ROBUSTNESS_DIR, "kstar3_headline.csv"); force=true);

summary_path = joinpath(OUT_DIR, "summary.txt");
open(summary_path, "w") do io
    println(io, "="^110);
    println(io, "K* = 3 four-emission headline (R1 W1 contingency)");
    println(io, "="^110);
    println(io);
    @printf(io, "Setup: SPY IS = %d, OoS = %d, paths = %d, seed = %d, K* = %d\n",
            n_is, n_oos, N_PATHS, SEED, K_STAR);
    @printf(io, "Observed excess kurtosis: IS = %.2f, OoS = %.2f\n", obs_kurt_is, obs_kurt_oos);
    println(io);
    @printf(io, "%-32s %-7s %-7s %-8s %-8s %-9s %-9s %-7s\n",
            "Model", "IS KS%", "OoS KS%", "Kurt IS", "Kurt OoS", "|G_t| ACF", "G_t ACF", "CRPS OoS");
    println(io, "-"^110);
    for r in results
        @printf(io, "%-32s %-7.1f %-7.1f %-8.2f %-8.2f %-9.4f %-9.4f %-7.4f\n",
                r.label, r.is.ks_pct, r.oos.ks_pct,
                r.is.sim_kurt, r.oos.sim_kurt,
                r.is.acf_mae, r.is.acf_mae_raw,
                r.crps_oos);
    end
    println(io);
    println(io, "Source: results/kstar3_headline/metrics.csv");
end

println("\n" * "="^80);
println("  K* = 3 headline complete.");
@printf("  Metrics: %s\n", csv_path);
@printf("  Paper:   %s\n", joinpath(PAPER_ROBUSTNESS_DIR, "kstar3_headline.csv"));
println("="^80);
