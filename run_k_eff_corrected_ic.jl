# ========================================================================================= #
# run_k_eff_corrected_ic.jl
#
# K_eff-corrected information criterion re-rank.  Addresses peer-review item P4.7 (R2 Tier 2):
# "Re-compute AIC, BIC, HQC, CAIC at every K in the sweep using K_eff from the
#  standardized-distance single-linkage diagnostic in place of K_nom in the parameter-count
#  term.  Report which K* each criterion selects under this correction."
#
# Method:
#   1. Refit CHMM-N at each K in {3, 6, 9, 12, 15, 18, 21} on the pre-2020 estimation slice
#      (same slice as run_k_selection_validation_pre2020.jl, so K* is comparable to that
#      table).
#   2. Apply the standardized-distance single-linkage diagnostic of run_state_distinctness.jl
#      (tau = 0.20 in the standardized (mu, sigma) plane) to count K_eff at each K.
#   3. Recompute AIC/BIC/HQC/CAIC with p_eff = K_eff * (K_eff - 1) + 2 * K_eff in place of
#      p_nom = K * (K - 1) + 2 * K in the penalty term.  Log-likelihood is unchanged.
#   4. Report K* under each criterion under K_nom (existing) and under K_eff (corrected).
#
# Outputs:
#   results/k_eff_corrected_ic/K_Eff_Corrected_IC.txt
#   ../CHMM-paper/results/robustness/k_eff_corrected_ic.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using LinearAlgebra
using Dates
using Printf

const SEED      = 20260422;
const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const MAX_ITER  = 60;
const K_GRID    = [3, 6, 9, 12, 15, 18, 21];
const TAU       = 0.20;

const OUT_DIR              = joinpath(_ROOT, "results", "k_eff_corrected_ic");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^72);
println("  K_eff-corrected IC re-rank  (peer-review P4.7 / R2 Tier 2)");
println("  Seed $SEED, K grid $K_GRID, single-linkage tau = $TAU");
println("="^72);

Random.seed!(SEED);

# --------------------------------------------------------------------------------------- #
println("\n[data] Loading and slicing SPY into pre-2020 estimation...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
spy_is = train_dataset["SPY"];
dates_is = Date.(spy_is.timestamp);
closes_is = Vector{Float64}(spy_is.close);
order = sortperm(dates_is);
dates_is  = dates_is[order];
closes_is = closes_is[order];

function _log_growth_series(closes::Vector{Float64}; Δt::Float64=DT, rf::Float64=RISK_FREE)
    N = length(closes);
    r = Vector{Float64}(undef, N - 1);
    for t in 2:N
        r[t-1] = (1/Δt) * log(closes[t] / closes[t-1]) - rf;
    end
    return r;
end

function _slice_between(dates::Vector{Date}, closes::Vector{Float64}, t0::Date, t1::Date)
    idx0 = findfirst(d -> d >= t0, dates);
    idx1 = findlast(d -> d <= t1, dates);
    return _log_growth_series(closes[idx0:idx1]);
end

R_est = _slice_between(dates_is, closes_is, Date(2014,1,3),  Date(2018,6,29));
n_est = length(R_est);
println("  estimation $n_est days  (2014-01-03 → 2018-06-29)");

# --------------------------------------------------------------------------------------- #
# Single-linkage clustering at standardized-distance threshold tau (replicates
# run_state_distinctness.jl exactly).
# --------------------------------------------------------------------------------------- #
function _single_linkage(D::AbstractMatrix, tau::Float64)
    n = size(D, 1);
    labels = collect(1:n);
    for i in 1:n, j in (i+1):n
        if D[i, j] < tau
            old = labels[j]; new = labels[i];
            for r in 1:n
                if labels[r] == old; labels[r] = new; end
            end
        end
    end
    uniq = sort(unique(labels));
    remap = Dict(uniq[k] => k for k in 1:length(uniq));
    return [remap[l] for l in labels];
end

function _params_n(model::MyContinuousHiddenMarkovModel, K::Int)
    μ = zeros(K); σ = zeros(K);
    for k in 1:K
        d = model.emission[k];
        μ[k] = d.μ; σ[k] = d.σ;
    end
    return μ, σ;
end

function _k_eff_chmm_n(model::MyContinuousHiddenMarkovModel, K::Int, tau::Float64)::Int
    if K == 1; return 1; end
    μ, σ = _params_n(model, K);
    μ_z = (μ .- mean(μ)) ./ std(μ);
    σ_z = (σ .- mean(σ)) ./ std(σ);
    D = zeros(K, K);
    for i in 1:K, j in 1:K
        D[i, j] = sqrt((μ_z[i] - μ_z[j])^2 + (σ_z[i] - σ_z[j])^2);
    end
    labels = _single_linkage(D, tau);
    return length(unique(labels));
end

# --------------------------------------------------------------------------------------- #
function _ic(ll::Float64, p::Int, n::Int)::NamedTuple
    aic  = -2 * ll + 2 * p;
    bic  = -2 * ll + p * log(n);
    hqc  = -2 * ll + 2 * p * log(log(n));
    caic = -2 * ll + p * (log(n) + 1);
    return (AIC=aic, BIC=bic, HQC=hqc, CAIC=caic);
end

p_count(K::Int)::Int = K * (K - 1) + 2 * K;

# --------------------------------------------------------------------------------------- #
println("\n[sweep] CHMM-N on pre-2020 estimation; computing K_eff at each K...");

results = NamedTuple[];
for K in K_GRID
    Random.seed!(SEED + K);
    m = build(MyContinuousHiddenMarkovModel,
        (observations=R_est, number_of_states=K, max_iter=MAX_ITER));
    est_ll = m.log_likelihood_history[end];

    K_eff = _k_eff_chmm_n(m, K, TAU);

    p_nom = p_count(K);
    p_eff = p_count(K_eff);
    ic_nom = _ic(est_ll, p_nom, n_est);
    ic_eff = _ic(est_ll, p_eff, n_est);

    push!(results, (
        K=K, K_eff=K_eff,
        p_nom=p_nom, p_eff=p_eff,
        est_ll=est_ll,
        AIC_nom=ic_nom.AIC, BIC_nom=ic_nom.BIC, HQC_nom=ic_nom.HQC, CAIC_nom=ic_nom.CAIC,
        AIC_eff=ic_eff.AIC, BIC_eff=ic_eff.BIC, HQC_eff=ic_eff.HQC, CAIC_eff=ic_eff.CAIC,
    ));
    @printf("  K=%2d  K_eff=%2d  p_nom=%3d  p_eff=%3d  ll=%.1f  BIC_nom=%.1f  BIC_eff=%.1f\n",
            K, K_eff, p_nom, p_eff, est_ll, ic_nom.BIC, ic_eff.BIC);
end

# --------------------------------------------------------------------------------------- #
function _argmin_K(rs, fld::Symbol)::Int
    vals = [getfield(r, fld) for r in rs];
    return rs[argmin(vals)].K;
end

ks_nom = (
    AIC=_argmin_K(results, :AIC_nom),  BIC=_argmin_K(results, :BIC_nom),
    HQC=_argmin_K(results, :HQC_nom),  CAIC=_argmin_K(results, :CAIC_nom),
);
ks_eff = (
    AIC=_argmin_K(results, :AIC_eff),  BIC=_argmin_K(results, :BIC_eff),
    HQC=_argmin_K(results, :HQC_eff),  CAIC=_argmin_K(results, :CAIC_eff),
);

println("\n  K* under K_nom: AIC=$(ks_nom.AIC)  BIC=$(ks_nom.BIC)  HQC=$(ks_nom.HQC)  CAIC=$(ks_nom.CAIC)");
println("  K* under K_eff: AIC=$(ks_eff.AIC)  BIC=$(ks_eff.BIC)  HQC=$(ks_eff.HQC)  CAIC=$(ks_eff.CAIC)");

# --------------------------------------------------------------------------------------- #
open(joinpath(OUT_DIR, "K_Eff_Corrected_IC.txt"), "w") do io
    println(io, "="^140);
    println(io, "K_eff-corrected information-criterion re-rank  (peer-review P4.7 / R2 Tier 2).");
    println(io, "="^140);
    println(io, "");
    println(io, "Estimation slice  : 2014-01-03 through 2018-06-29 ($n_est observations, pre-COVID).");
    println(io, "Method            : refit CHMM-N at each K in $K_GRID, compute K_eff via the");
    println(io, "                    standardized-distance single-linkage diagnostic of");
    println(io, "                    run_state_distinctness.jl (Euclidean in z-scored (mu, sigma),");
    println(io, "                    tau = $TAU), then re-evaluate AIC/BIC/HQC/CAIC with");
    println(io, "                    p_eff = K_eff * (K_eff - 1) + 2 * K_eff in the penalty term.");
    println(io, "                    Log-likelihood unchanged from the K_nom fit.");
    println(io, "");
    println(io, "Reading           : if K_eff < K_nom at large K, the parameter-count penalty drops");
    println(io, "                    and the IC at the corrected operating point can re-rank.  The");
    println(io, "                    expected pattern: AIC, which is least sensitive to p, will be");
    println(io, "                    near-monotone in ll; BIC/CAIC, which scale with log(n)*p, will");
    println(io, "                    favour higher K under K_eff than under K_nom whenever K_eff");
    println(io, "                    is meaningfully below K_nom.");
    println(io, "");
    println(io, rpad("K",     3), " | ", rpad("K_eff", 5), " | ",
                rpad("p_nom", 5), " | ", rpad("p_eff", 5), " | ",
                rpad("ll",     8), " | ",
                rpad("AIC_n",  9), " | ", rpad("AIC_e",  9), " | ",
                rpad("BIC_n",  9), " | ", rpad("BIC_e",  9), " | ",
                rpad("HQC_n",  9), " | ", rpad("HQC_e",  9), " | ",
                rpad("CAIC_n", 9), " | ", rpad("CAIC_e", 9));
    println(io, "-"^140);
    for r in results
        println(io, rpad(r.K,     3), " | ", rpad(r.K_eff, 5), " | ",
                    rpad(r.p_nom, 5), " | ", rpad(r.p_eff, 5), " | ",
                    rpad(round(r.est_ll, digits=1), 8), " | ",
                    rpad(round(r.AIC_nom,  digits=1), 9), " | ", rpad(round(r.AIC_eff,  digits=1), 9), " | ",
                    rpad(round(r.BIC_nom,  digits=1), 9), " | ", rpad(round(r.BIC_eff,  digits=1), 9), " | ",
                    rpad(round(r.HQC_nom,  digits=1), 9), " | ", rpad(round(r.HQC_eff,  digits=1), 9), " | ",
                    rpad(round(r.CAIC_nom, digits=1), 9), " | ", rpad(round(r.CAIC_eff, digits=1), 9));
    end
    println(io, "="^140);
    println(io, "");
    println(io, "K* under K_nom (textbook penalty):");
    println(io, "  AIC  : $(ks_nom.AIC)");
    println(io, "  BIC  : $(ks_nom.BIC)");
    println(io, "  HQC  : $(ks_nom.HQC)");
    println(io, "  CAIC : $(ks_nom.CAIC)");
    println(io, "");
    println(io, "K* under K_eff (corrected penalty):");
    println(io, "  AIC  : $(ks_eff.AIC)");
    println(io, "  BIC  : $(ks_eff.BIC)");
    println(io, "  HQC  : $(ks_eff.HQC)");
    println(io, "  CAIC : $(ks_eff.CAIC)");
end

open(joinpath(PAPER_ROBUSTNESS_DIR, "k_eff_corrected_ic.csv"), "w") do io
    println(io, "model,K,K_eff,p_nom,p_eff,est_ll,AIC_nom,AIC_eff,BIC_nom,BIC_eff,HQC_nom,HQC_eff,CAIC_nom,CAIC_eff");
    for r in results
        println(io, "CHMM-N,$(r.K),$(r.K_eff),$(r.p_nom),$(r.p_eff),",
                    "$(round(r.est_ll, digits=3)),",
                    "$(round(r.AIC_nom,  digits=3)),$(round(r.AIC_eff,  digits=3)),",
                    "$(round(r.BIC_nom,  digits=3)),$(round(r.BIC_eff,  digits=3)),",
                    "$(round(r.HQC_nom,  digits=3)),$(round(r.HQC_eff,  digits=3)),",
                    "$(round(r.CAIC_nom, digits=3)),$(round(r.CAIC_eff, digits=3))");
    end
end

println("\n" * "="^72);
println("  Done.");
println("  Outputs:");
println("    $(joinpath(OUT_DIR, "K_Eff_Corrected_IC.txt"))");
println("    $(joinpath(PAPER_ROBUSTNESS_DIR, "k_eff_corrected_ic.csv"))");
println("="^72);
