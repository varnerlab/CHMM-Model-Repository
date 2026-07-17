# =========================================================================== #
# run_hsmm_ml.jl
#
# Explicit-duration HSMM (local EM maximum-likelihood fits) at K in {3, 18} on
# SPY using Yu (2010) forward-backward over augmented (state, duration) pairs.
# Gaussian emissions, off-diagonal transitions (no self-loops), truncated
# discrete Pareto sojourns with D_max = 200. Model core: hsmm_core.jl.
#
# This is the ML counterpart to the plug-in SemiMarkov.jl (which Viterbi-decodes
# from a flat CHMM and fits sojourns post-hoc); here the entire (transition,
# emission, sojourn) tuple is updated jointly via EM.
#
# Seventh-review corrections (2026-07-16, findings 2-4):
#   1. RIGHT-CENSORED terminal segment: the likelihood marginalizes an ongoing
#      final sojourn through the duration survival function, matching the
#      simulator's right-truncation; the earlier convention conditioned every
#      segmentation on a sojourn boundary exactly at T (estimator/generator
#      mismatch). Duration M-step gains the expected censored counts.
#   2. MULTISTART: the headline fits are multistart local-EM estimates
#      (5 starts at K = 3, 3 at K = 18) with per-start diagnostics retained;
#      they are constrained local maximum-likelihood fits, not certified
#      global optima.
#   3. SENSITIVITY: a K = 3 grid over D_max in {100, 200, 400} x alpha lower
#      bound in {0.02, 0.05, 0.2} (single canonical start each) quantifies the
#      dependence of the fitted duration law and downstream metrics on the
#      truncation point and the exponent search floor — the fitted persistent
#      state's alpha sits at the search floor, so this dependence is part of
#      the result, not a nuisance. K = 18 is not gridded (compute scoping;
#      the headline K = 18 row carries multistart evidence only).
#
# Initial-segment convention: first sojourn starts at t = 1 (no equilibrium
# left-censoring). Likelihood values are NOT comparable to artifacts produced
# before the censoring correction.
#
# Outputs:
#   results/hsmm_ml/hsmm_ml_K3.jld2
#   results/hsmm_ml/hsmm_ml_K18.jld2
#   results/hsmm_ml/hsmm_ml_metrics.csv
#   results/hsmm_ml/hsmm_ml_sensitivity.csv
# =========================================================================== #

const _PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."));
using Pkg; Pkg.activate(_PROJECT_ROOT);
include(joinpath(_PROJECT_ROOT, "Include.jl"));

using Random
using Statistics
using StatsBase
using LinearAlgebra
using Printf
using JLD2
using DelimitedFiles
using Distributions
using HypothesisTests

include(joinpath(@__DIR__, "hsmm_core.jl"));

const SEED        = 20260420;
const TICKER      = "SPY";
const RISK_FREE   = 0.0;
const DT          = 1/252;
const N_PATHS     = 1000;
const L_LAGS      = 252;
const D_MAX       = 200;       # headline truncation (grid: 100/200/400 at K = 3)
const MAX_ITER    = 400;       # EM max iterations
const TOL         = 1e-5;      # log-likelihood tolerance per observation
const ALPHA_KS    = 0.05;
const KS          = [3, 18];   # K values to fit
const N_STARTS    = Dict(3 => 5, 18 => 3);
const ALPHA_LB    = 0.05;      # headline exponent search floor (grid: 0.02/0.05/0.2)
const ALPHA_UB    = 8.0;
const SENS_DMAX   = [100, 200, 400];
const SENS_ALB    = [0.02, 0.05, 0.2];

const OUT_DIR     = joinpath(_ROOT, "results", "hsmm_ml");
mkpath(OUT_DIR);

Random.seed!(SEED);

println("="^70)
println("  ML HSMM (Yu 2010 explicit-duration EM, right-censored terminal segment) on $TICKER")
println("  K values:  $KS  (starts: $N_STARTS)")
println("  D_max:     $D_MAX (sensitivity: $SENS_DMAX x alpha_lb $SENS_ALB at K = 3)")
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
# Headline multistart fits at K = 3 and K = 18
# --------------------------------------------------------------------------- #
results = Dict{Int,Any}();
for K in KS
    println("\n" * "="^70)
    println("  Fitting ML HSMM at K = $K ($(N_STARTS[K]) starts)")
    println("="^70)
    Random.seed!(SEED + K);
    @time m, diags = fit_hsmm_ml_multistart(R_is, K; n_starts=N_STARTS[K], D=D_MAX,
                                            max_iter=MAX_ITER, tol=TOL,
                                            seed=SEED + K,
                                            alpha_bounds=(ALPHA_LB, ALPHA_UB));
    for d in diags
        @printf("  start %d: ll = %.4f, evals = %d, final inc/obs = %.2e, converged = %s\n",
                d.start, d.ll, d.n_evals, d.final_increment / n_is, d.converged);
    end
    lls = [d.ll for d in diags];
    @printf("  cross-start ll spread = %.4f nats\n", maximum(lls) - minimum(lls));
    sim_is  = simulate_hsmm(m; T=n_is,  D=D_MAX, n_paths=N_PATHS);
    sim_oos = simulate_hsmm(m; T=n_oos, D=D_MAX, n_paths=N_PATHS);
    metr_is  = eval_panel(R_is,  sim_is);
    metr_oos = eval_panel(R_oos, sim_oos);
    @printf("[K=%d] IS  KS = %.1f%%  kurt = %.3f  ACF-MAE |G| = %.4f  raw = %.4f\n",
        K, 100 * metr_is.ks_rate, metr_is.kurt, metr_is.acf_mae, metr_is.acf_mae_raw);
    @printf("[K=%d] OoS KS = %.1f%%  kurt = %.3f  ACF-MAE |G| = %.4f  raw = %.4f\n",
        K, 100 * metr_oos.ks_rate, metr_oos.kurt, metr_oos.acf_mae, metr_oos.acf_mae_raw);
    results[K] = (model=m, metr_is=metr_is, metr_oos=metr_oos, diags=diags);
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
        "diagnostics" => diags,
        "D_max" => D_MAX,
        "alpha_bounds" => (ALPHA_LB, ALPHA_UB),
        "convention" => "right_censored",
    ));
end

# Write CSV summary (final_ll is the right-censored likelihood; not comparable
# to pre-censoring artifacts).
open(joinpath(OUT_DIR, "hsmm_ml_metrics.csv"), "w") do io
    write(io, "K,IS_KS,OoS_KS,IS_kurt,OoS_kurt,IS_ACF_MAE_abs,OoS_ACF_MAE_abs,IS_ACF_MAE_raw,OoS_ACF_MAE_raw,n_iter,final_ll,final_inc_perobs,n_starts,best_start,ll_spread,n_converged_starts,D_max,alpha_lb\n")
    for K in KS
        r = results[K];
        h = r.model.ll_history;
        inc = length(h) >= 2 ? (h[end] - h[end-1]) / max(n_is, 1) : 0.0;
        lls = [d.ll for d in r.diags];
        best_start = argmax(lls);
        write(io, @sprintf("%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%d,%.4f,%.2e,%d,%d,%.4f,%d,%d,%.2f\n",
            K, r.metr_is.ks_rate, r.metr_oos.ks_rate,
            r.metr_is.kurt, r.metr_oos.kurt,
            r.metr_is.acf_mae, r.metr_oos.acf_mae,
            r.metr_is.acf_mae_raw, r.metr_oos.acf_mae_raw,
            length(h), last(h), inc,
            length(r.diags), best_start, maximum(lls) - minimum(lls),
            count(d -> d.converged, r.diags), D_MAX, ALPHA_LB));
    end
end

# --------------------------------------------------------------------------- #
# Sensitivity grid at K = 3: D_max x alpha lower bound, single canonical start
# --------------------------------------------------------------------------- #
println("\n" * "="^70)
println("  Sensitivity grid at K = 3: D_max in $SENS_DMAX x alpha_lb in $SENS_ALB")
println("="^70)
open(joinpath(OUT_DIR, "hsmm_ml_sensitivity.csv"), "w") do io
    write(io, "K,D_max,alpha_lb,ll,ll_perobs,n_iter,alphas,IS_KS,IS_kurt,IS_ACF_MAE_abs\n")
    for Dm in SENS_DMAX, alb in SENS_ALB
        Random.seed!(SEED + 3);
        m = fit_hsmm_ml(R_is, 3; D=Dm, max_iter=MAX_ITER, tol=TOL,
                        alpha_bounds=(alb, ALPHA_UB), verbose=false);
        sim_is = simulate_hsmm(m; T=n_is, D=Dm, n_paths=N_PATHS);
        metr   = eval_panel(R_is, sim_is);
        h = m.ll_history;
        @printf("  D_max=%3d alpha_lb=%.2f: ll/obs = %.5f, iters = %3d, alphas = %s, IS KS = %.1f%%, ACF = %.4f\n",
                Dm, alb, h[end] / n_is, length(h),
                join([@sprintf("%.3f", a) for a in sort(m.α)], "/"),
                100 * metr.ks_rate, metr.acf_mae);
        write(io, @sprintf("%d,%d,%.2f,%.4f,%.6f,%d,%s,%.4f,%.4f,%.6f\n",
              3, Dm, alb, h[end], h[end] / n_is, length(h),
              join([@sprintf("%.4f", a) for a in sort(m.α)], ";"),
              metr.ks_rate, metr.kurt, metr.acf_mae));
    end
end

println("\n[done] ML HSMM fits saved to $OUT_DIR")
