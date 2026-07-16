# ========================================================================================= #
# run_chmm_t_shared_nu.jl
#
# Reviewer Round 2 / Item B4 (peer-review.md R2#req-4, R3#req-3).
#
# Body CHMM-t fits per-state ν_k via golden-section ECM with bracket (2.1, 50). The
# resulting IS kurtosis overshoots observed by an order of magnitude (14.4 IS unpenalised)
# because ECM concentrates ν_k on a small subset of low-ν tail states. R2 / R3 ask whether
# the per-state ν_k is the binding constraint on the kurtosis overshoot, by running a
# shared-ν ablation: a single ν shared across all K states, fit by ECM (the standard
# Student-t HMM in the time-series literature, not a per-state ν_k mixture).
#
# This runner implements the shared-ν ECM in place and reports the headline panel
# (IS / OoS KS, IS / OoS sim kurt, |G_t| ACF-MAE, raw-G_t ACF-MAE) at K = 3, 6, 18 against
# the body per-state ν_k row.
#
# Output:
#   results/chmm_t_shared_nu/chmm_t_shared_nu.csv
#   results/chmm_t_shared_nu/chmm_t_shared_nu.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random
using Statistics
using HypothesisTests
using StatsBase
using Printf
using Distributions
using JLD2

const SEED      = 20260420;
const N_PATHS   = 1000;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const L_LAGS    = 252;
const NU_BOUNDS = (2.1, 50.0);
const NU_INIT   = 6.0;

Random.seed!(SEED);

const OUT_DIR = joinpath(_ROOT, "results", "chmm_t_shared_nu");
mkpath(OUT_DIR);

println("="^80)
println("  Shared-ν Student-t HMM ablation  (R2/R3 / Item B4)")
println("  Seed: $SEED  Paths: $N_PATHS  ν_bounds: $NU_BOUNDS")
println("="^80)

# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading SPY IS / OoS...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = all_R[:, idx_spy];
n_is = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE);
n_oos = length(R_oos);
println("  IS = $n_is  OoS = $n_oos")

# ----------------------------------------------------------------------------------------- #
# Shared-ν ECM lives in the library (src/Compute.jl, baum_welch_student_t_shared_nu) with the
# same evaluate-before-update / best-evaluated-checkpoint contract as the four per-family
# fitters. This runner only wraps it into the named-tuple shape the simulation code uses.

function ecm_shared_nu(obs::AbstractVector, K::Int;
        max_iter::Int=60, tol::Float64=1e-4,
        ν_init::Float64=NU_INIT, ν_bounds::Tuple{Float64,Float64}=NU_BOUNDS)
    T, μ, σ, ν, π, ll_hist, _ = baum_welch_student_t_shared_nu(collect(Float64, obs), K;
        max_iter=max_iter, tol=tol, ν_init=ν_init, ν_bounds=ν_bounds)
    return (T=T, μ=μ, σ=σ, ν=ν, π=π, ll_history=ll_hist)
end

# ----------------------------------------------------------------------------------------- #
# Stationary distribution from T (slow-power approximation; matches body convention).
function _stationary_T(T)
    π = (T^1000)[1, :]
    return Categorical(π)
end

# Sample state path from initial state via deterministic transition: piecewise multinomial.
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

function _eval_panel(R_obs::AbstractVector, sim::AbstractMatrix; L::Int=L_LAGS)
    np = size(sim, 2); n_o = length(R_obs)
    μ_o = mean(R_obs); σ_o = std(R_obs)
    kurt_o = sum(((R_obs .- μ_o) ./ σ_o).^4) / n_o - 3.0
    L_use = min(L, n_o - 1)
    acf_o = autocor(abs.(R_obs), 1:L_use)
    acf_o_raw = autocor(R_obs, 1:L_use)
    ks_pass = 0
    kurt_s = 0.0; acf_mae = 0.0; acf_mae_raw = 0.0
    for i in 1:np
        s = sim[:, i]
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= 0.05
            ks_pass += 1
        end
        μ_s = mean(s); σ_s = std(s)
        kurt_s += sum(((s .- μ_s) ./ σ_s).^4) / length(s) - 3.0
        acf_s = autocor(abs.(s), 1:L_use)
        acf_s_raw = autocor(s, 1:L_use)
        acf_mae += mean(abs.(acf_o .- acf_s))
        acf_mae_raw += mean(abs.(acf_o_raw .- acf_s_raw))
    end
    return (
        ks_pct = round(100*ks_pass/np, digits=1),
        sim_kurt = round(kurt_s/np, digits=2),
        obs_kurt = round(kurt_o, digits=2),
        acf_mae = round(acf_mae/np, digits=4),
        acf_mae_raw = round(acf_mae_raw/np, digits=4),
    )
end

# ----------------------------------------------------------------------------------------- #
results = NamedTuple[]

for K in [3, 6, 18]
    println("\n[fit] shared-ν ECM, K = $K ...")
    fit = ecm_shared_nu(R_is, K; max_iter=MAX_ITER)
    @printf("  K = %d : ν_shared = %.3f\n", K, fit.ν)
    println("[sim] $N_PATHS IS + OoS paths ...")
    sim_is  = _sim_shared_nu(fit, n_is,  N_PATHS)
    sim_oos = _sim_shared_nu(fit, n_oos, N_PATHS)
    if K == 3
        mkpath(OUT_DIR)
        JLD2.save(joinpath(OUT_DIR, "sims_K3.jld2"), "is", sim_is, "oos", sim_oos)
    end
    is_p = _eval_panel(R_is, sim_is)
    oos_p = _eval_panel(R_oos, sim_oos)
    push!(results, (K=K, ν_shared=fit.ν, is=is_p, oos=oos_p))
    @printf("  IS  KS %.1f%% kurt %.2f (obs %.2f)  |G| ACF-MAE %.4f\n",
            is_p.ks_pct, is_p.sim_kurt, is_p.obs_kurt, is_p.acf_mae)
    @printf("  OoS KS %.1f%% kurt %.2f (obs %.2f)  |G| ACF-MAE %.4f\n",
            oos_p.ks_pct, oos_p.sim_kurt, oos_p.obs_kurt, oos_p.acf_mae)
end

# ----------------------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "chmm_t_shared_nu.csv")
open(csv_path, "w") do io
    println(io, "K,nu_shared,is_KS,is_kurt,is_acf_mae,is_acf_mae_raw,oos_KS,oos_kurt,oos_acf_mae,oos_acf_mae_raw")
    for r in results
        @printf(io, "%d,%.4f,%.1f,%.2f,%.4f,%.4f,%.1f,%.2f,%.4f,%.4f\n",
                r.K, r.ν_shared, r.is.ks_pct, r.is.sim_kurt, r.is.acf_mae, r.is.acf_mae_raw,
                r.oos.ks_pct, r.oos.sim_kurt, r.oos.acf_mae, r.oos.acf_mae_raw)
    end
end

txt_path = joinpath(OUT_DIR, "chmm_t_shared_nu.txt")
open(txt_path, "w") do io
    println(io, "="^110)
    println(io, "Shared-ν Student-t HMM ablation (R2/R3 / Item B4)")
    println(io, "="^110)
    println(io)
    println(io, "Setup: SPY IS = $n_is, OoS = $n_oos, paths = $N_PATHS, seed = $SEED.")
    println(io, "ECM: shared ν across all K states, GSS on aggregate Q-function over ν_bounds = $NU_BOUNDS.")
    println(io)
    @printf(io, "  %-3s  %-8s  %-9s  %-9s  %-12s  %-12s  %-9s  %-9s  %-12s  %-12s\n",
            "K", "ν_shared", "IS_KS%", "IS_kurt", "|G|_MAE_IS", "raw_MAE_IS",
            "OoS_KS%", "OoS_kurt", "|G|_MAE_OoS", "raw_MAE_OoS")
    println(io, "  ", "-"^110)
    for r in results
        @printf(io, "  %-3d  %8.3f  %9.1f  %9.2f  %12.4f  %12.4f  %9.1f  %9.2f  %12.4f  %12.4f\n",
                r.K, r.ν_shared, r.is.ks_pct, r.is.sim_kurt, r.is.acf_mae, r.is.acf_mae_raw,
                r.oos.ks_pct, r.oos.sim_kurt, r.oos.acf_mae, r.oos.acf_mae_raw)
    end
    println(io)
    println(io, "Comparison against per-state ν_k (body Table tab:model_comparison):")
    println(io, "  K=3 per-state (penalised λ=20): IS KS 91.9%, IS kurt 18.87, OoS KS 81.4%, OoS kurt 10.61")
    println(io, "  K=18 per-state (penalised λ=20): IS KS 95.0%, IS kurt 8.56, OoS KS 85.8%, OoS kurt 7.07")
    println(io, "  K=18 per-state (unpenalised):    IS KS 95.6%, IS kurt 14.35, OoS KS 85.7%, OoS kurt 10.71")
    println(io)
    println(io, "Reading.")
    println(io, "  If shared-ν IS kurtosis matches observed (~7.7) without the penalty, the per-state ν_k")
    println(io, "  is the binding constraint on the overshoot; the body's λ-shrinkage and bracket-lift")
    println(io, "  discussion is downstream of the wrong design choice. If shared-ν also overshoots,")
    println(io, "  the per-state structure is not the issue and the per-state ν_k design is justified.")
end

println("\n[done] $csv_path")
println("[done] $txt_path")
