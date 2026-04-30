# ========================================================================== #
# run_msgarch_reference.jl
#
# Reference MS-GARCH baseline (Ardia et al. 2019, JSS, CRAN MSGARCH package)
# driven from Julia via RCall.jl. Produces the Table 2 row that replaces the
# in-house Nelder-Mead self-fit and addresses peer-review item R2 W1
# (also R1 W1, R3 RE1).
#
# Prerequisites (one-time):
#   1. R >= 4.2 on PATH.
#   2. cd r_msgarch && Rscript setup.R
#      (or, if renv.lock is committed: Rscript -e 'renv::restore(prompt = FALSE)')
# Run from the CHMM-Model project root:
#   julia --project=. run_msgarch_reference.jl
#
# Outputs (under results/msgarch_reference/):
#   models_K{K}.jld2          fitted MyMSGARCHReferenceModel per K
#   sims_K{K}.jld2            (n_steps x n_paths) IS / OoS path matrices
#   metrics.csv               Table 2-friendly per-K metrics row
#   summary.txt               human-readable summary, with R + MSGARCH versions
#   fit_log.txt               per-K fit timing + parameter posterior summary
# ========================================================================== #

using Pkg; Pkg.activate(".");
include("Include.jl");

# MSGARCHReference is opt-in (see Include.jl): load it here, where R is
# required. Errors out with a setup hint if R or RCall.jl are missing.
include(joinpath(_PATH_TO_SRC, "MSGARCHReference.jl"));

using Random;
using Statistics;
using HypothesisTests;
using StatsBase;
using Printf;
using CSV;

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
const SEED            = 20260422;
const TICKER          = "SPY";
const RISK_FREE       = 0.0;
const DT              = 1 / 252;
const N_PATHS         = 1_000;
const L_LAGS          = 252;

# Peer review asks for K in {2, 3, 4} (Ardia 2019 conventional ceiling).
# Add 6 if you want parity with the in-house run_msgarch_higher_k.jl panel.
const K_GRID          = [2, 3, 4];

# MCMC budget — Ardia 2019 default ratios. Total retained per chain after
# thinning = (n_mcmc - n_burnin) / n_thin. The reference fit is single-chain
# DEMC; multi-chain is a thinning convention not a pooling convention here.
const FIT_METHOD      = "MCMC";       # "MCMC" (reference) or "ML" (sanity)
const N_MCMC          = 12_500;
const N_BURNIN        = 2_500;
const N_THIN          = 10;

const VARIANCE_SPEC   = "sGARCH";     # vanilla GARCH(1,1), HMP 2004 baseline
const DISTRIBUTION    = "norm";       # Gaussian innovations, Table 2 baseline

const OUT_DIR = joinpath(_ROOT, "results", "msgarch_reference");
mkpath(OUT_DIR);

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
println("="^80);
println("  MS-GARCH reference baseline (CRAN MSGARCH via RCall)");
println("  Peer-review items addressed: R1 W1, R2 W1, R3 RE1");
println("="^80);

versions = msgarch_reference_versions();
println("R version       : $(versions.r_version)");
println("MSGARCH version : $(versions.msgarch_version)");
println("Platform        : $(versions.platform)  ($(versions.os))");
println("Timestamp (UTC) : $(versions.timestamp_utc)");
println();

# --------------------------------------------------------------------------- #
# Data — match the existing run_msgarch_baselines.jl data path exactly
# --------------------------------------------------------------------------- #
println("[data] Loading SPY IS + OoS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String, DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
all_tickers = keys(filtered) |> collect |> sort;
all_R = log_growth_matrix(filtered, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==(TICKER), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
n_is = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, TICKER; Δt=DT, risk_free_rate=RISK_FREE));
n_oos = length(R_oos);

println("  IS days = $n_is, OoS days = $n_oos");
println();

# --------------------------------------------------------------------------- #
# Metric helper — same shape as run_msgarch_higher_k.jl::eval_panel
# --------------------------------------------------------------------------- #
function _eval_panel(R_obs::AbstractVector, sim_archive::AbstractMatrix; L::Int=L_LAGS)
    np   = size(sim_archive, 2);
    n_o  = length(R_obs);
    μ_o  = mean(R_obs); σ_o = std(R_obs);
    kurt_o = sum(((R_obs .- μ_o) ./ σ_o) .^ 4) / n_o - 3.0;
    L_use  = min(L, n_o - 1);
    acf_o     = autocor(abs.(R_obs), 1:L_use);
    acf_o_raw = autocor(R_obs,        1:L_use);

    ks_pass = 0;
    kurt_s = 0.0; acf_mae_s = 0.0; acf_mae_raw_s = 0.0;
    for i in 1:np
        s = sim_archive[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= 0.05; ks_pass += 1; end
        μ_s = mean(s); σ_s = std(s);
        kurt_s += sum(((s .- μ_s) ./ σ_s) .^ 4) / length(s) - 3.0;
        acf_mae_s     += mean(abs.(acf_o     .- autocor(abs.(s), 1:L_use)));
        acf_mae_raw_s += mean(abs.(acf_o_raw .- autocor(s,       1:L_use)));
    end
    return (
        ks_pct      = round(100 * ks_pass / np, digits=1),
        sim_kurt    = round(kurt_s / np, digits=2),
        obs_kurt    = round(kurt_o, digits=2),
        acf_mae     = round(acf_mae_s / np, digits=4),
        acf_mae_raw = round(acf_mae_raw_s / np, digits=4),
    );
end

# --------------------------------------------------------------------------- #
# K grid
# --------------------------------------------------------------------------- #
panel = Vector{NamedTuple}();
fit_log_lines = String[];

for K in K_GRID
    println("[fit] K = $K  ($(FIT_METHOD), MSGARCH $(versions.msgarch_version))");
    Random.seed!(SEED + K);
    t0 = time();
    model = fit_msgarch_reference(
        R_is, K;
        variance_spec = VARIANCE_SPEC,
        distribution  = DISTRIBUTION,
        fit_method    = FIT_METHOD,
        n_mcmc        = N_MCMC,
        n_burnin      = N_BURNIN,
        n_thin        = N_THIN,
        seed          = SEED + K,
    );
    dt_fit = time() - t0;
    println("  fit complete in $(round(dt_fit, digits=1)) s, log-lik = $(round(model.loglik, digits=2))");

    # JLD2 cannot save the live R-side fit object (Rcpp pointers don't
    # roundtrip). Persist a portable summary dict instead.
    save(joinpath(OUT_DIR, "models_K$(K).jld2"),
         "K",             model.K,
         "variance_spec", model.variance_spec,
         "distribution",  model.distribution,
         "fit_method",    model.fit_method,
         "loglik",        model.loglik,
         "par_names",     model.par_names,
         "par_post_mean", model.par_post_mean,
         "par_post_sd",   model.par_post_sd,
         "transition",    model.transition);

    println("[sim] $N_PATHS IS + OoS paths...");
    sim_is  = simulate_msgarch_reference(model, n_is;  n_paths=N_PATHS, seed=SEED + 100 + K);
    sim_oos = simulate_msgarch_reference(model, n_oos; n_paths=N_PATHS, seed=SEED + 200 + K);
    save(joinpath(OUT_DIR, "sims_K$(K).jld2"), Dict("is" => sim_is, "oos" => sim_oos));

    is_panel  = _eval_panel(R_is,  sim_is);
    oos_panel = _eval_panel(R_oos, sim_oos);

    println("  IS  KS = $(is_panel.ks_pct)%  sim kurt = $(is_panel.sim_kurt)  |G| ACF-MAE = $(is_panel.acf_mae)");
    println("  OoS KS = $(oos_panel.ks_pct)%  sim kurt = $(oos_panel.sim_kurt)  |G| ACF-MAE = $(oos_panel.acf_mae)");
    println();

    push!(panel, (
        K=K, fit_time_s=dt_fit, loglik=model.loglik,
        is=is_panel, oos=oos_panel, model=model,
    ));

    push!(fit_log_lines, "="^72);
    push!(fit_log_lines, "K = $K  ($(FIT_METHOD))");
    push!(fit_log_lines, "  fit time : $(round(dt_fit, digits=1)) s");
    push!(fit_log_lines, "  log-lik  : $(round(model.loglik, digits=2))");
    vspec_str = join(model.variance_spec, ", ");
    dspec_str = join(model.distribution,  ", ");
    push!(fit_log_lines, "  variance : $vspec_str");
    push!(fit_log_lines, "  distrib. : $dspec_str");
    push!(fit_log_lines, "  parameters (post-mean / post-sd):");
    for j in 1:length(model.par_names)
        push!(fit_log_lines, @sprintf("    %-20s  %+.5f  %+.5f",
              model.par_names[j], model.par_post_mean[j], model.par_post_sd[j]));
    end
    push!(fit_log_lines, "  transition (posterior mean):");
    for i in 1:K
        push!(fit_log_lines, "    " * join([@sprintf("%.4f", model.transition[i,j]) for j in 1:K], "  "));
    end
end

# --------------------------------------------------------------------------- #
# Output: machine-readable CSV
# --------------------------------------------------------------------------- #
metrics_path = joinpath(OUT_DIR, "metrics.csv");
open(metrics_path, "w") do io
    println(io, "K,fit_method,n_paths,n_is,n_oos,loglik,fit_time_s," *
                "is_ks_pct,is_sim_kurt,is_obs_kurt,is_acf_mae,is_acf_mae_raw," *
                "oos_ks_pct,oos_sim_kurt,oos_obs_kurt,oos_acf_mae,oos_acf_mae_raw");
    for r in panel
        @printf(io, "%d,%s,%d,%d,%d,%.4f,%.2f,%.1f,%.2f,%.2f,%.4f,%.4f,%.1f,%.2f,%.2f,%.4f,%.4f\n",
                r.K, FIT_METHOD, N_PATHS, n_is, n_oos, r.loglik, r.fit_time_s,
                r.is.ks_pct,  r.is.sim_kurt,  r.is.obs_kurt,  r.is.acf_mae,  r.is.acf_mae_raw,
                r.oos.ks_pct, r.oos.sim_kurt, r.oos.obs_kurt, r.oos.acf_mae, r.oos.acf_mae_raw);
    end
end

# --------------------------------------------------------------------------- #
# Output: human-readable summary
# --------------------------------------------------------------------------- #
summary_path = joinpath(OUT_DIR, "summary.txt");
open(summary_path, "w") do io
    println(io, "="^110);
    println(io, "MS-GARCH reference baseline  (CRAN MSGARCH $(versions.msgarch_version) via RCall)");
    println(io, "Peer-review items addressed: R1 W1, R2 W1, R3 RE1.");
    println(io, "="^110);
    println(io);
    println(io, "Setup");
    println(io, "  Ticker          : $TICKER");
    println(io, "  IS / OoS days   : $n_is / $n_oos");
    println(io, "  Paths per K     : $N_PATHS");
    println(io, "  Variance spec   : $VARIANCE_SPEC");
    println(io, "  Distribution    : $DISTRIBUTION");
    println(io, "  Fit method      : $FIT_METHOD");
    if FIT_METHOD == "MCMC"
        println(io, "  MCMC budget     : $N_MCMC draws, $N_BURNIN burn-in, thin $N_THIN");
    end
    println(io, "  Seed            : $SEED");
    println(io, "  R version       : $(versions.r_version)");
    println(io, "  MSGARCH version : $(versions.msgarch_version)");
    println(io, "  Platform / OS   : $(versions.platform) / $(versions.os)");
    println(io, "  Generated (UTC) : $(versions.timestamp_utc)");
    println(io);
    println(io, "Headline panel (KS pass-rate at α = 0.05, asymptotic-iid null):");
    @printf(io, "  %-3s  %-8s  %-8s  %-8s  %-8s  %-9s  %-9s\n",
            "K", "KS IS%", "KS OoS%", "kurt IS", "kurt OoS", "|G| ACF IS", "|G| ACF OoS");
    println(io, "  ", "-"^78);
    for r in panel
        @printf(io, "  %-3d  %-8.1f  %-8.1f  %-8.2f  %-8.2f  %-9.4f  %-9.4f\n",
                r.K, r.is.ks_pct, r.oos.ks_pct,
                r.is.sim_kurt, r.oos.sim_kurt,
                r.is.acf_mae, r.oos.acf_mae);
    end
    println(io);
    println(io, "Reference comparison points (from run_msgarch_higher_k.jl, in-house Nelder-Mead):");
    println(io, "  In-house K=2 :  KS IS = 27.7%   KS OoS = 38.7%   |G| ACF-MAE = 0.0367");
    println(io, "  In-house K=3 :  KS IS = 36.1%   KS OoS = 33.1%   |G| ACF-MAE = 0.0284");
    println(io, "  CHMM-N K=18  :  KS IS = 94.1%   KS OoS = 81.8%   |G| ACF-MAE = 0.0509");
    println(io);
    println(io, "Files");
    println(io, "  models_K{K}.jld2   fitted MyMSGARCHReferenceModel per K");
    println(io, "  sims_K{K}.jld2     IS / OoS path matrices ($N_PATHS paths each)");
    println(io, "  metrics.csv        machine-readable per-K metrics");
    println(io, "  fit_log.txt        per-K fit timing + parameter posterior detail");
end

# --------------------------------------------------------------------------- #
# Output: per-K fit log
# --------------------------------------------------------------------------- #
open(joinpath(OUT_DIR, "fit_log.txt"), "w") do io
    println(io, "MS-GARCH reference fit log");
    println(io, "R $(versions.r_version) / MSGARCH $(versions.msgarch_version) / $(versions.timestamp_utc)");
    println(io);
    for line in fit_log_lines; println(io, line); end
end

println("="^80);
println("  Reference MS-GARCH baseline complete.");
println("  Outputs: $OUT_DIR");
println("  CSV row for Table 2: $metrics_path");
println("="^80);
