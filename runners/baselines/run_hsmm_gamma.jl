# =========================================================================== #
# run_hsmm_gamma.jl
#
# Closes peer-review item P3.14 / R1 RE3: Gamma-sojourn HSMM as a co-headline
# foil to the truncated-Pareto HSMM. Same Yu (2010) forward-backward EM core
# as run_hsmm_ml.jl (hsmm_core.jl; right-censored terminal segment, seventh
# review finding 3), but with discrete-Gamma sojourns (continuous Gamma(α, β)
# discretized to integer durations 1..D_max via P(D = d) = F_Γ(d) - F_Γ(d-1))
# supplied through the core's pluggable duration M-step.
#
# The duration update is METHOD-OF-MOMENTS on the expected COMPLETED duration
# counts (censored terminal counts are excluded from the moment update — a
# disclosed approximation; the likelihood and E-step are censoring-correct).
# Because the duration block is a moment update rather than an exact
# conditional maximizer, EM monotonicity is not guaranteed for this runner;
# it is a sensitivity row, labeled "moment-updated" throughout.
#
# R1 RE3 framing: a Gamma sojourn (closer to Bulla & Berzel 2008) produces
# lighter-tailed sojourns and more frequent regime transitions than the
# truncated Pareto, trading the clustering axis against the marginal axis.
#
# Outputs:
#   results/hsmm_gamma/hsmm_gamma_K3.jld2
#   results/hsmm_gamma/hsmm_gamma_K18.jld2
#   results/hsmm_gamma/hsmm_gamma_metrics.csv
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
const D_MAX       = 200;       # max sojourn length
const MAX_ITER    = 40;        # EM max iterations (moment-updated sensitivity row)
const TOL         = 1e-3;      # log-likelihood tolerance per observation
const ALPHA_KS    = 0.05;
const KS          = [3, 18];   # K values to fit

const OUT_DIR     = joinpath(_ROOT, "results", "hsmm_gamma");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-Paper-Repository", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

Random.seed!(SEED);

println("="^70)
println("  Moment-updated Gamma-sojourn HSMM (censored-likelihood EM core; MoM duration block) on $TICKER")
println("  K values:  $KS")
println("  D_max:     $D_MAX")
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
# Truncated discrete Gamma: pmf p(d) = F_Γ(d; α, β) - F_Γ(d-1; α, β) on
# {1, ..., D_max}, then renormalised. α = shape, β = scale; mean αβ, variance αβ².
# --------------------------------------------------------------------------- #
function _gamma_logpmf(α::Float64, β::Float64, D::Int)
    α = clamp(α, 0.3, 50.0); β = clamp(β, 0.5, 200.0);
    G = Distributions.Gamma(α, β);
    p = Vector{Float64}(undef, D);
    @inbounds for d in 1:D
        p[d] = max(cdf(G, Float64(d)) - cdf(G, Float64(d - 1)), 1e-300);
    end
    p ./= sum(p);
    return log.(p);
end

# Method-of-moments fit on expected duration counts: α = m1^2 / Var, β = Var / m1.
# Robust fallbacks for degenerate counts (variance ≈ 0 or near-zero total mass).
function _fit_gamma_ab(expected_counts::Vector{Float64}, D::Int)
    s = sum(expected_counts);
    if s <= 1e-9; return (2.0, 5.0); end
    m1 = sum(expected_counts[d] * Float64(d) for d in 1:D) / s;
    m2 = sum(expected_counts[d] * Float64(d)^2 for d in 1:D) / s;
    var_d = max(m2 - m1^2, 1e-3);
    α = clamp(m1^2 / var_d, 0.3, 50.0);
    β = clamp(var_d / m1,    0.5, 200.0);
    return (α, β);
end

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
# Run pipeline at K = 3 and K = 18
# --------------------------------------------------------------------------- #
results = Dict{Int,Any}();
for K in KS
    println("\n" * "="^70)
    println("  Fitting moment-updated Gamma-sojourn HSMM at K = $K")
    println("="^70)
    Random.seed!(SEED + K);
    # Per-state Gamma scale lives beside the core model (core stores shape in m.α).
    βs = fill(5.0, K);
    log_p0 = vcat([reshape(_gamma_logpmf(2.0, 5.0, D_MAX), 1, D_MAX) for _ in 1:K]...);
    fit_gamma_duration! = function (m::MLHSMM, w::Matrix{Float64}, c::Matrix{Float64}, D::Int)
        # MoM on expected COMPLETED durations only (censored counts excluded).
        @inbounds for s in 1:m.K
            if sum(w[s, :]) > 1e-9
                a, b = _fit_gamma_ab(w[s, :], D);
                m.α[s] = a; βs[s] = b;
                m.log_p[s, :] = _gamma_logpmf(a, b, D);
            end
        end
        return nothing;
    end
    @time m = fit_hsmm_ml(R_is, K; D=D_MAX, max_iter=MAX_ITER, tol=TOL,
                          log_p0=log_p0, fit_duration! = fit_gamma_duration!);
    sim_is  = simulate_hsmm(m; T=n_is,  D=D_MAX, n_paths=N_PATHS);
    sim_oos = simulate_hsmm(m; T=n_oos, D=D_MAX, n_paths=N_PATHS);
    metr_is  = eval_panel(R_is,  sim_is);
    metr_oos = eval_panel(R_oos, sim_oos);
    @printf("[K=%d] IS  KS = %.1f%%  kurt = %.3f  ACF-MAE |G| = %.4f  raw = %.4f\n",
        K, 100 * metr_is.ks_rate, metr_is.kurt, metr_is.acf_mae, metr_is.acf_mae_raw);
    @printf("[K=%d] OoS KS = %.1f%%  kurt = %.3f  ACF-MAE |G| = %.4f  raw = %.4f\n",
        K, 100 * metr_oos.ks_rate, metr_oos.kurt, metr_oos.acf_mae, metr_oos.acf_mae_raw);
    results[K] = (model=m, beta=copy(βs), metr_is=metr_is, metr_oos=metr_oos);
    save(joinpath(OUT_DIR, "hsmm_gamma_K$K.jld2"), Dict(
        "K" => K,
        "alpha" => m.α,
        "beta" => βs,
        "mu" => m.μ,
        "sigma" => m.σ,
        "A" => m.A,
        "pi" => m.π,
        "log_p" => m.log_p,
        "ll_history" => m.ll_history,
        "metr_is" => metr_is,
        "metr_oos" => metr_oos,
        "convention" => "right_censored",
        "duration_update" => "method_of_moments_completed_only",
    ));
end

# Write CSV summary (mirror to paper-side robustness/)
open(joinpath(OUT_DIR, "hsmm_gamma_metrics.csv"), "w") do io
    write(io, "K,IS_KS,OoS_KS,IS_kurt,OoS_kurt,IS_ACF_MAE_abs,OoS_ACF_MAE_abs,IS_ACF_MAE_raw,OoS_ACF_MAE_raw,n_iter,final_ll\n")
    for K in KS
        r = results[K];
        write(io, @sprintf("%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%d,%.4f\n",
            K, r.metr_is.ks_rate, r.metr_oos.ks_rate,
            r.metr_is.kurt, r.metr_oos.kurt,
            r.metr_is.acf_mae, r.metr_oos.acf_mae,
            r.metr_is.acf_mae_raw, r.metr_oos.acf_mae_raw,
            length(r.model.ll_history), last(r.model.ll_history)));
    end
end

# Mirror CSV to paper-side robustness/
cp(joinpath(OUT_DIR, "hsmm_gamma_metrics.csv"),
   joinpath(PAPER_ROBUSTNESS_DIR, "hsmm_gamma_metrics.csv"); force=true);

println("\n[done] Moment-updated Gamma-sojourn HSMM fits saved to $OUT_DIR");
println("Paper CSV: $(joinpath(PAPER_ROBUSTNESS_DIR, "hsmm_gamma_metrics.csv"))");
println();
println("Comparison reference (read live from results/hsmm_ml/hsmm_ml_metrics.csv, truncated Pareto sojourn):");
let ml_csv = joinpath(_ROOT, "results", "hsmm_ml", "hsmm_ml_metrics.csv")
    if isfile(ml_csv)
        for line in readlines(ml_csv)[2:end]
            f = split(line, ",");
            println("  K=$(f[1]) (Pareto):  IS KS = $(round(100*parse(Float64,f[2]),digits=1))%, OoS KS = $(round(100*parse(Float64,f[3]),digits=1))%, |G_t| ACF-MAE = $(f[6])");
        end
    else
        println("  (hsmm_ml_metrics.csv not found; run runners/baselines/run_hsmm_ml.jl)");
    end
end
