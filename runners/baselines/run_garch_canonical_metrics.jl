# ========================================================================================= #
# run_garch_canonical_metrics.jl
#
# ONE canonical GARCH(1,1) row for Table 2 (third-review finding 9).
#
# Background: the paper's main generator-comparison table carried a GARCH(1,1) row
# (IS KS 27.4, OoS KS 59.6, |G| ACF-MAE 0.0490) produced by an OLD headline-pipeline fit,
# while the appendix's grid-initialised maximum-likelihood re-fit of the SAME specification
# (runners/baselines/run_garch_suite.jl) attains |G| ACF-MAE 0.0309. The review requires a
# single canonical multi-start fit with ALL Table-2 columns regenerated from it.
#
# This runner:
#   1. Fits GARCH(1,1) on SPY IS with the SAME grid-initialised Nelder-Mead maximum-
#      likelihood procedure as run_garch_suite.jl -- both call the deterministic
#      `build(MyGARCHModel, (observations=R_is,))` -> `_fit_garch11` (src/Compute.jl),
#      which grid-searches (alpha, beta) initial pairs before the simplex refinement.
#      The fit takes no random numbers, so the canonical parameters are identical under
#      any seed; the seed below governs the simulation draws only.
#   2. Simulates 1000 IS-length + 1000 OoS-length paths under the HEADLINE seed
#      convention of runners/headline/run_baselines_and_cross_asset.jl
#      (SEED = 20260420, Random.seed!(SEED + 6) before the fit, then per-path
#      interleaved IS/OoS simulation), so the row is draw-comparable with the other
#      rows of Table 2.
#   3. Scores the FULL Table-2 metric panel with `eval_full` reproduced VERBATIM from
#      run_baselines_and_cross_asset.jl (KS/AD pass rates at alpha = 0.05, population
#      excess kurtosis, |G_t| and raw-G_t ACF-MAE over 252 lags, Wasserstein-1,
#      Hellinger, quantile coverage) on BOTH the IS and OoS windows, plus the OoS
#      mean sample-CRPS with the unbiased sorted-ensemble estimator reproduced
#      VERBATIM from runners/headline/run_kstar3_headline.jl.
#
# Outputs:
#   results/garch_canonical/garch_canonical_metrics.txt
#   results/garch_canonical/garch_canonical_metrics.csv
#   ../CHMM-Paper-Repository/results/robustness/garch_canonical_metrics.csv
#
# Usage: julia --project=. runners/baselines/run_garch_canonical_metrics.jl
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
using Random, Printf;

const TICKER          = "SPY";
const RISK_FREE_RATE  = 0.0;
const ΔT              = 1/252;
const N_PATHS         = 1000;
const L               = 252;
const SEED            = 20260420;   # paper-canonical global seed (headline convention)

const OUT_DIR = joinpath(_ROOT, "results", "garch_canonical");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-Paper-Repository", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^80)
println("  Canonical GARCH(1,1) Table-2 row (grid-initialised ML fit, full metric panel)")
println("  Seed $SEED (headline convention), $N_PATHS paths")
println("="^80)

# ---- data (verbatim from run_baselines_and_cross_asset.jl) ------------------------------- #
println("\n[data] Loading SPY IS + OoS...");
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
n_is = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset, TICKER; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
n_oos = length(R_oos);
println("  IS $n_is, OoS $n_oos");

# ---- eval_full: verbatim from run_baselines_and_cross_asset.jl --------------------------- #
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

# ---- sample CRPS: verbatim from run_kstar3_headline.jl (sorted-ensemble identity) -------- #
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
    for t in 1:n
        x = view(sim, t, :);
        mean_crps += _sample_crps(R[t], x);
    end
    return mean_crps / n;
end

# ---- canonical fit + headline-convention simulation --------------------------------------- #
# Fit: grid-initialised Nelder-Mead ML (`_fit_garch11`), identical procedure to the
# GARCH(1,1) reference row of run_garch_suite.jl. The fit is deterministic; the seed
# governs the simulation draws, placed exactly as in run_baselines_and_cross_asset.jl.
println("\n[fit] GARCH(1,1) canonical grid-initialised ML fit...");
Random.seed!(SEED + 6);
garch_model = build(MyGARCHModel, (observations=R_is,));
persistence = garch_model.α + garch_model.β;
σ_uncond = sqrt(garch_model.ω / max(1.0 - persistence, 1e-6));
@printf("  omega = %.6f  alpha = %.4f  beta = %.4f  mu = %.6f\n",
        garch_model.ω, garch_model.α, garch_model.β, garch_model.μ);
@printf("  persistence alpha+beta = %.4f  unconditional sigma = %.4f  logL = %.2f\n",
        persistence, σ_uncond, garch_model.log_likelihood);

println("[sim] $N_PATHS IS-length + $N_PATHS OoS-length paths (headline interleaved order)...");
garch_is = Array{Float64,2}(undef, n_is, N_PATHS);
garch_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    garch_is[:, i] = simulate_garch(garch_model, n_is);
    garch_oos[:, i] = simulate_garch(garch_model, n_oos);
end

println("[eval] Full Table-2 metric panel (IS + OoS) + OoS sample-CRPS...");
m_is = eval_full(R_is, garch_is);
m_oos = eval_full(R_oos, garch_oos);
crps_oos = _crps_path_mean(R_oos, garch_oos);

# ---- report -------------------------------------------------------------------------------- #
println("\n" * "="^80);
println("  Canonical GARCH(1,1) row:");
@printf("  IS  KS %5.1f%%  AD %5.1f%%  kurt %5.2f (obs %5.2f)  ACF|G| %.4f  ACFraw %.4f  W1 %.3f  Hell %.4f  Cov %5.1f%%\n",
        m_is.ks, m_is.ad, m_is.kurt, m_is.kurt_obs, m_is.acf_mae, m_is.acf_mae_raw, m_is.w1, m_is.hell, m_is.cov);
@printf("  OoS KS %5.1f%%  AD %5.1f%%  kurt %5.2f (obs %5.2f)  ACF|G| %.4f  ACFraw %.4f  W1 %.3f  Hell %.4f  Cov %5.1f%%\n",
        m_oos.ks, m_oos.ad, m_oos.kurt, m_oos.kurt_obs, m_oos.acf_mae, m_oos.acf_mae_raw, m_oos.w1, m_oos.hell, m_oos.cov);
@printf("  OoS mean sample-CRPS = %.4f\n", crps_oos);
println("="^80);

# ---- artefacts ------------------------------------------------------------------------------ #
txt_path = joinpath(OUT_DIR, "garch_canonical_metrics.txt");
open(txt_path, "w") do io
    println(io, "="^100);
    println(io, "Canonical GARCH(1,1) Table-2 row -- grid-initialised maximum-likelihood fit, full metric panel.");
    println(io, "="^100);
    println(io, "");
    println(io, "Fit        : grid-initialised Nelder-Mead ML (`_fit_garch11` via build(MyGARCHModel, ...)),");
    println(io, "             the SAME deterministic procedure as the GARCH(1,1) reference row of");
    println(io, "             runners/baselines/run_garch_suite.jl. Supersedes the old headline-pipeline row");
    println(io, "             (IS KS 27.4 / OoS KS 59.6 / |G| ACF-MAE 0.0490) retained in earlier drafts.");
    println(io, "Data       : $TICKER, IS $n_is obs, OoS $n_oos obs.");
    println(io, "Simulation : $N_PATHS paths per window; seed $SEED (headline convention: Random.seed!(SEED+6)");
    println(io, "             before the fit, per-path interleaved IS/OoS draws, as run_baselines_and_cross_asset.jl).");
    println(io, "Metrics    : eval_full verbatim from run_baselines_and_cross_asset.jl (KS/AD pass rate at");
    println(io, "             alpha = 0.05, population excess kurtosis, ACF-MAE over $L lags on |G_t| and raw G_t,");
    println(io, "             Wasserstein-1, Hellinger, 99-quantile coverage); OoS mean sample-CRPS via the");
    println(io, "             unbiased sorted-ensemble estimator of run_kstar3_headline.jl.");
    @printf(io, "Parameters : omega = %.6f, alpha = %.4f, beta = %.4f, mu = %.6f, alpha+beta = %.4f, logL = %.2f\n",
            garch_model.ω, garch_model.α, garch_model.β, garch_model.μ, persistence, garch_model.log_likelihood);
    println(io, "");
    println(io, "-"^100);
    @printf(io, "%-8s | %6s | %6s | %8s | %8s | %10s | %10s | %6s | %6s | %6s\n",
            "Window", "KS%", "AD%", "Kurt", "KurtObs", "ACF-MAE|G|", "ACF-MAEraw", "W1", "Hell", "Cov%");
    println(io, "-"^100);
    for (tag, m) in (("IS", m_is), ("OoS", m_oos))
        @printf(io, "%-8s | %6.1f | %6.1f | %8.2f | %8.2f | %10.4f | %10.4f | %6.3f | %6.4f | %6.1f\n",
                tag, m.ks, m.ad, m.kurt, m.kurt_obs, m.acf_mae, m.acf_mae_raw, m.w1, m.hell, m.cov);
    end
    println(io, "-"^100);
    @printf(io, "OoS mean sample-CRPS = %.4f\n", crps_oos);
    println(io, "="^100);
    println(io, "Source: runners/baselines/run_garch_canonical_metrics.jl");
end
println("\n  Text report : $txt_path");

csv_header = "model,IS_KS_pct,OoS_KS_pct,IS_AD_pct,OoS_AD_pct,IS_sim_kurt,OoS_sim_kurt," *
             "IS_kurt_obs,OoS_kurt_obs,IS_ACF_MAE_absG,OoS_ACF_MAE_absG,IS_ACF_MAE_raw," *
             "OoS_ACF_MAE_raw,OoS_CRPS,IS_W1,IS_Hellinger,IS_cov_pct,OoS_cov_pct," *
             "omega,alpha,beta,mu,loglik,seed,n_paths";
csv_row = @sprintf("GARCH(1,1) canonical,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.4f,%.1f,%.1f,%.6f,%.4f,%.4f,%.6f,%.2f,%d,%d",
        m_is.ks, m_oos.ks, m_is.ad, m_oos.ad, m_is.kurt, m_oos.kurt,
        m_is.kurt_obs, m_oos.kurt_obs, m_is.acf_mae, m_oos.acf_mae, m_is.acf_mae_raw,
        m_oos.acf_mae_raw, crps_oos, m_is.w1, m_is.hell, m_is.cov, m_oos.cov,
        garch_model.ω, garch_model.α, garch_model.β, garch_model.μ, garch_model.log_likelihood,
        SEED, N_PATHS);

csv_path = joinpath(OUT_DIR, "garch_canonical_metrics.csv");
open(csv_path, "w") do io
    println(io, csv_header);
    println(io, csv_row);
end
println("  Local CSV   : $csv_path");

paper_csv_path = joinpath(PAPER_ROBUSTNESS_DIR, "garch_canonical_metrics.csv");
cp(csv_path, paper_csv_path; force=true);
println("  Paper CSV   : $paper_csv_path");

println("\nDone.");
