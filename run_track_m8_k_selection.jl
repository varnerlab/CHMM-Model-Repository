# ========================================================================================= #
# run_track_m8_k_selection.jl
#
# Track M8 (revision response to referee comment M8): clean K-selection on a pre-OoS
# validation slice, decoupled from the 2024-2026 OoS window used for VaR backtesting.
#
# The original K = 18 operating point was selected using a combined IC rank + IS/OoS
# distributional pass-rate criterion; the OoS pass rate contaminates the VaR evaluation
# that reuses the same OoS window. The fix here: carve 2022-01-03 through 2024-01-03 out
# of the original IS as a validation slice, re-fit CHMM-N at each K on the preceding
# 2014-01-03 through 2021-12-31 estimation slice (~2013 observations), and select K by
# held-out log-likelihood on the 2022-2023 validation slice (~503 observations). The
# 2024-2026 OoS window is untouched by this selection procedure.
#
# Sweep: K ∈ {3, 6, 9, 12, 15, 18, 21} (same as Table 2 in the paper).
#
# Emission family: CHMM-N for the primary K-selection (consistent with the paper's
# claim that IC rank aligns across emission families); CHMM-t and CHMM-L repeated for a
# robustness check.
#
# Outputs:
#   results/track_m8/K_Selection_Validation.txt
#   ../CHMM-paper/results/revision/M8_k_selection.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
using Dates
const SEED = 20260422;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const MAX_ITER  = 60;
const K_GRID    = [3, 6, 9, 12, 15, 18, 21];

const TRACK_M8_DIR       = joinpath(_ROOT, "results", "track_m8");
const PAPER_REVISION_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "revision"));
mkpath(TRACK_M8_DIR);
mkpath(PAPER_REVISION_DIR);

println("="^72)
println("  Track M8: pre-OoS validation K-selection (referee M8)")
println("  Seed $SEED, K grid $K_GRID")
println("="^72)

# --------------------------------------------------------------------------------------- #
# Data: reuse the existing train DataFrame, slice into estimation + validation sub-windows
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading and slicing SPY IS into estimation + validation...");
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

# Estimation: 2014-01-03 through 2021-12-31 (~8 years)
# Validation: 2022-01-03 through 2024-01-03 (~2 years)
R_est = _slice_between(dates_is, closes_is, Date(2014,1,3), Date(2021,12,31));
R_val = _slice_between(dates_is, closes_is, Date(2022,1,3), Date(2024,1,3));
n_est = length(R_est); n_val = length(R_val);
println("  estimation $n_est (2014-01-03 → 2021-12-31)  validation $n_val (2022-01-03 → 2024-01-03)");

# --------------------------------------------------------------------------------------- #
# Forward-algorithm held-out log-likelihood under a fitted CHMM
# --------------------------------------------------------------------------------------- #
"""
    forward_log_likelihood(observations, model) -> Float64

Returns log P(r_{1:T} | θ) under a fitted CHMM with uniform initial prior (matching the
Viterbi convention in src/Compute.jl). Uses the rescaled log-space forward recursion,
so it is numerically stable for long validation windows.
"""
function forward_log_likelihood(observations::Vector{Float64},
                                model::Union{MyContinuousHiddenMarkovModel,
                                             MyStudentTHiddenMarkovModel,
                                             MyLaplaceHiddenMarkovModel})::Float64
    N = length(observations);
    K = length(model.states);
    T_mat = zeros(K, K);
    for i in 1:K
        T_mat[i, :] = probs(model.transition[i]);
    end
    log_T = log.(T_mat);
    log_pi = zeros(N, K);
    log_pi[1, :] .= log(1.0 / K) .+ [logpdf(model.emission[k], observations[1]) for k in 1:K];
    ll = _logsumexp_vec(log_pi[1, :]);
    log_pi[1, :] .-= ll;
    for t in 2:N
        for k in 1:K
            log_pi[t, k] = _logsumexp_vec(log_pi[t-1, :] .+ log_T[:, k]) +
                           logpdf(model.emission[k], observations[t]);
        end
        lse = _logsumexp_vec(log_pi[t, :]);
        ll += lse;
        log_pi[t, :] .-= lse;
    end
    return ll;
end

function ks_pass_rate(R_ref::Vector{Float64}, sim::Matrix{Float64}; α::Float64=0.05)
    n_sim = size(sim, 2); n_pass = 0;
    for p in 1:n_sim
        pv = pvalue(ApproximateTwoSampleKSTest(R_ref, sim[:, p]));
        if pv >= α; n_pass += 1; end
    end
    return n_pass / n_sim;
end

# --------------------------------------------------------------------------------------- #
# Information criteria (per the paper's existing definitions)
# AIC  = -2 ll + 2 p
# BIC  = -2 ll + p log n
# HQC  = -2 ll + 2 p log log n
# CAIC = -2 ll + p (log n + 1)
# where p = K(K-1) transition + 2K emission (μ, σ) for Gaussian = K^2 + K parameters
# --------------------------------------------------------------------------------------- #
function _ic_gaussian(ll::Float64, K::Int, n::Int)::NamedTuple
    p = K * (K - 1) + 2 * K;   # K² + K
    aic  = -2 * ll + 2 * p;
    bic  = -2 * ll + p * log(n);
    hqc  = -2 * ll + 2 * p * log(log(n));
    caic = -2 * ll + p * (log(n) + 1);
    return (AIC=aic, BIC=bic, HQC=hqc, CAIC=caic, p=p);
end

# --------------------------------------------------------------------------------------- #
# Primary sweep: CHMM-N on estimation window, held-out log-lik on validation
# --------------------------------------------------------------------------------------- #
println("\n[sweep] CHMM-N on estimation window; held-out log-lik on validation window...");

results_n = NamedTuple[];

for K in K_GRID
    Random.seed!(SEED + K);
    m = build(MyContinuousHiddenMarkovModel,
        (observations=R_est, number_of_states=K, max_iter=MAX_ITER));

    # Estimation-window log-lik (last value from the EM trace)
    est_ll = m.log_likelihood_history[end];
    ic = _ic_gaussian(est_ll, K, n_est);

    # Held-out validation log-lik via the forward algorithm
    val_ll = forward_log_likelihood(R_val, m);
    val_ll_per_obs = val_ll / n_val;

    # Held-out validation KS pass rate: simulate N_paths of length n_val, compute two-sample KS
    Random.seed!(SEED + 1000 + K);
    sim_val = simulate_returns(m, n_val; n_paths=500);
    val_ks = ks_pass_rate(R_val, sim_val);

    push!(results_n, (
        K=K, p=ic.p,
        est_ll=est_ll, val_ll=val_ll, val_ll_per_obs=val_ll_per_obs,
        AIC=ic.AIC, BIC=ic.BIC, HQC=ic.HQC, CAIC=ic.CAIC,
        val_ks=val_ks,
    ));
    println("  K=$K  est_ll=$(round(est_ll, digits=1))  val_ll=$(round(val_ll, digits=1))  val_ll/obs=$(round(val_ll_per_obs, digits=4))  val_KS=$(round(100*val_ks, digits=1))%  BIC=$(round(ic.BIC, digits=1))");
end

k_star_val_ll = results_n[argmax([r.val_ll for r in results_n])].K;
k_star_val_ks = results_n[argmax([r.val_ks for r in results_n])].K;
k_star_bic    = results_n[argmin([r.BIC for r in results_n])].K;
println("\n  K* by held-out log-lik:  $k_star_val_ll");
println("  K* by held-out KS rate:  $k_star_val_ks");
println("  K* by BIC (est window):  $k_star_bic");

# --------------------------------------------------------------------------------------- #
# Robustness: CHMM-t and CHMM-L at the leading K values
# --------------------------------------------------------------------------------------- #
println("\n[robustness] CHMM-t and CHMM-L held-out log-lik at leading K values (K=12, 15, 18, 21)...");

results_tl = NamedTuple[];
for K in [12, 15, 18, 21]
    for (fam, Tname) in [(:t, "CHMM-t"), (:l, "CHMM-L")]
        Random.seed!(SEED + 100*K + (fam == :t ? 1 : 2));
        local m;
        if fam == :t
            m = build(MyStudentTHiddenMarkovModel,
                (observations=R_est, number_of_states=K, max_iter=MAX_ITER));
        else
            m = build(MyLaplaceHiddenMarkovModel,
                (observations=R_est, number_of_states=K, max_iter=MAX_ITER));
        end
        val_ll = forward_log_likelihood(R_val, m);
        push!(results_tl, (model=Tname, K=K, val_ll=val_ll, val_ll_per_obs=val_ll/n_val));
        println("  $Tname  K=$K  val_ll=$(round(val_ll, digits=1))  val_ll/obs=$(round(val_ll/n_val, digits=4))");
    end
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
open(joinpath(TRACK_M8_DIR, "K_Selection_Validation.txt"), "w") do io
    println(io, "="^140);
    println(io, "Track M8. Pre-OoS validation K-selection (referee M8 response).");
    println(io, "="^140);
    println(io, "");
    println(io, "Estimation slice  : 2014-01-03 through 2021-12-31 ($n_est observations).");
    println(io, "Validation slice  : 2022-01-03 through 2024-01-03 ($n_val observations).");
    println(io, "Selection criterion: held-out log-likelihood on the validation slice.");
    println(io, "  (Complementary metrics: held-out two-sample KS pass rate at α=0.05; BIC on the estimation slice.)");
    println(io, "");
    println(io, "CHMM-N primary sweep:");
    println(io, "");
    println(io, rpad("K", 3), " | ",
                rpad("p", 5), " | ",
                rpad("est_ll", 10), " | ",
                rpad("val_ll", 10), " | ",
                rpad("val_ll/obs", 11), " | ",
                rpad("val_KS%", 8), " | ",
                rpad("AIC", 10), " | ",
                rpad("BIC", 10), " | ",
                rpad("HQC", 10), " | ",
                rpad("CAIC", 10));
    println(io, "-"^140);
    for r in results_n
        println(io, rpad(r.K, 3), " | ",
                    rpad(r.p, 5), " | ",
                    rpad(round(r.est_ll, digits=1), 10), " | ",
                    rpad(round(r.val_ll, digits=1), 10), " | ",
                    rpad(round(r.val_ll_per_obs, digits=4), 11), " | ",
                    rpad(round(100*r.val_ks, digits=1), 8), " | ",
                    rpad(round(r.AIC, digits=1), 10), " | ",
                    rpad(round(r.BIC, digits=1), 10), " | ",
                    rpad(round(r.HQC, digits=1), 10), " | ",
                    rpad(round(r.CAIC, digits=1), 10));
    end
    println(io, "="^140);
    println(io, "");
    println(io, "K* by held-out log-lik  (primary M8 criterion) : $k_star_val_ll");
    println(io, "K* by held-out KS rate  (complementary check)  : $k_star_val_ks");
    println(io, "K* by BIC (est-window penalised log-lik)       : $k_star_bic");
    println(io, "");
    println(io, "Robustness (CHMM-t and CHMM-L at leading K values):");
    println(io, "");
    println(io, rpad("model", 7), " | ", rpad("K", 3), " | ", rpad("val_ll", 10), " | ", rpad("val_ll/obs", 11));
    println(io, "-"^50);
    for r in results_tl
        println(io, rpad(r.model, 7), " | ",
                    rpad(r.K, 3), " | ",
                    rpad(round(r.val_ll, digits=1), 10), " | ",
                    rpad(round(r.val_ll_per_obs, digits=4), 11));
    end
end

open(joinpath(PAPER_REVISION_DIR, "M8_k_selection.csv"), "w") do io
    println(io, "model,K,p,est_ll,val_ll,val_ll_per_obs,val_KS_pct,AIC,BIC,HQC,CAIC");
    for r in results_n
        println(io, "CHMM-N,$(r.K),$(r.p),$(round(r.est_ll, digits=3)),$(round(r.val_ll, digits=3)),$(round(r.val_ll_per_obs, digits=5)),$(round(100*r.val_ks, digits=2)),$(round(r.AIC, digits=3)),$(round(r.BIC, digits=3)),$(round(r.HQC, digits=3)),$(round(r.CAIC, digits=3))");
    end
    for r in results_tl
        println(io, "$(r.model),$(r.K),NA,NA,$(round(r.val_ll, digits=3)),$(round(r.val_ll_per_obs, digits=5)),NA,NA,NA,NA,NA");
    end
end

println("\n" * "="^72);
println("  Track M8 complete.");
println("  Text report : $(joinpath(TRACK_M8_DIR, "K_Selection_Validation.txt"))");
println("  Paper CSV   : $(joinpath(PAPER_REVISION_DIR, "M8_k_selection.csv"))");
println("="^72);
