# ========================================================================================= #
# run_k_selection_validation_pre2020.jl
#
# Pre-2020 held-out K-selection variant: addresses peer-review item R1-Q1 / R1-RE1.
#
# The companion run_k_selection_validation.jl uses 2022-2023 as the validation slice.
# That slice is itself partly a structural-break period (the 2022 rate-hike cycle and the
# late-2022 inflation peak), which the paper's discussion explicitly notes ("any model
# with substantial IS-specific structure generalises worse on it"). Reviewer R1 asked:
# does a pre-2020 validation slice that avoids the rate-hike confound select the same K*?
#
# This script runs the same selection procedure on a 2014-01-03 .. 2018-06-29 estimation
# window and a 2018-07-02 .. 2019-12-31 validation window, both fully pre-COVID and
# pre-2022-rate-hike. The Q4 2018 drawdown and 2019 recovery are inside the validation
# slice as a moderate-stress non-regime-shift event.
#
# Outputs:
#   results/k_selection_validation/K_Selection_Validation_Pre2020.txt
#   ../CHMM-paper/results/robustness/k_selection_validation_pre2020.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random
using Statistics
using HypothesisTests
using Dates

const SEED      = 20260422;
const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const MAX_ITER  = 60;
const K_GRID    = [3, 6, 9, 12, 15, 18, 21];

const OUT_DIR              = joinpath(_ROOT, "results", "k_selection_validation");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^72);
println("  Pre-2020 held-out K-selection (R1-Q1 / R1-RE1)");
println("  Seed $SEED, K grid $K_GRID");
println("="^72);

Random.seed!(SEED);

# --------------------------------------------------------------------------------------- #
println("\n[data] Loading and slicing SPY into pre-2020 estimation + validation...");
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

# Estimation: 2014-01-03 through 2018-06-29 (~4.5 years, pre-2018-Q4-drawdown)
# Validation: 2018-07-02 through 2019-12-31 (~1.5 years, pre-COVID, pre-rate-hike;
#                                            includes Q4 2018 drawdown + 2019 recovery)
R_est = _slice_between(dates_is, closes_is, Date(2014,1,3),  Date(2018,6,29));
R_val = _slice_between(dates_is, closes_is, Date(2018,7,2),  Date(2019,12,31));
n_est = length(R_est); n_val = length(R_val);
println("  estimation $n_est days  (2014-01-03 → 2018-06-29)");
println("  validation $n_val days  (2018-07-02 → 2019-12-31, includes Q4 2018 drawdown + 2019 recovery)");

# --------------------------------------------------------------------------------------- #
function forward_log_likelihood(observations::Vector{Float64},
                                model::Union{MyContinuousHiddenMarkovModel,
                                             MyStudentTHiddenMarkovModel,
                                             MyLaplaceHiddenMarkovModel})::Float64
    N = length(observations);
    K = length(model.states);
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
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

function _ic_gaussian(ll::Float64, K::Int, n::Int)::NamedTuple
    p = K * (K - 1) + 2 * K;
    aic  = -2 * ll + 2 * p;
    bic  = -2 * ll + p * log(n);
    hqc  = -2 * ll + 2 * p * log(log(n));
    caic = -2 * ll + p * (log(n) + 1);
    return (AIC=aic, BIC=bic, HQC=hqc, CAIC=caic, p=p);
end

# --------------------------------------------------------------------------------------- #
println("\n[sweep] CHMM-N on pre-2020 estimation; held-out log-lik on pre-2020 validation...");

results_n = NamedTuple[];
for K in K_GRID
    Random.seed!(SEED + K);
    m = build(MyContinuousHiddenMarkovModel,
        (observations=R_est, number_of_states=K, max_iter=MAX_ITER));
    est_ll = m.log_likelihood_history[end];
    ic = _ic_gaussian(est_ll, K, n_est);
    val_ll = forward_log_likelihood(R_val, m);
    val_ll_per_obs = val_ll / n_val;
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
k_star_aic    = results_n[argmin([r.AIC for r in results_n])].K;
k_star_hqc    = results_n[argmin([r.HQC for r in results_n])].K;
k_star_caic   = results_n[argmin([r.CAIC for r in results_n])].K;
println("\n  K* by held-out log-lik  : $k_star_val_ll");
println("  K* by held-out KS rate  : $k_star_val_ks");
println("  K* by AIC               : $k_star_aic");
println("  K* by BIC               : $k_star_bic");
println("  K* by HQC               : $k_star_hqc");
println("  K* by CAIC              : $k_star_caic");

# --------------------------------------------------------------------------------------- #
open(joinpath(OUT_DIR, "K_Selection_Validation_Pre2020.txt"), "w") do io
    println(io, "="^140);
    println(io, "Pre-2020 held-out K-selection  (R1-Q1 / R1-RE1).");
    println(io, "="^140);
    println(io, "");
    println(io, "Estimation slice  : 2014-01-03 through 2018-06-29 ($n_est observations).");
    println(io, "Validation slice  : 2018-07-02 through 2019-12-31 ($n_val observations).");
    println(io, "Both slices fully pre-COVID and pre-2022-rate-hike. Q4 2018 drawdown + 2019");
    println(io, "recovery sits inside the validation window as a moderate-stress non-regime-shift event.");
    println(io, "");
    println(io, "Selection criteria: held-out log-lik (primary), held-out KS, AIC/BIC/HQC/CAIC on est window.");
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
    println(io, "K* by held-out log-lik (primary) : $k_star_val_ll");
    println(io, "K* by held-out KS rate           : $k_star_val_ks");
    println(io, "K* by AIC                        : $k_star_aic");
    println(io, "K* by BIC                        : $k_star_bic");
    println(io, "K* by HQC                        : $k_star_hqc");
    println(io, "K* by CAIC                       : $k_star_caic");
end

open(joinpath(PAPER_ROBUSTNESS_DIR, "k_selection_validation_pre2020.csv"), "w") do io
    println(io, "model,K,p,est_ll,val_ll,val_ll_per_obs,val_KS_pct,AIC,BIC,HQC,CAIC");
    for r in results_n
        println(io, "CHMM-N,$(r.K),$(r.p),$(round(r.est_ll, digits=3)),$(round(r.val_ll, digits=3)),$(round(r.val_ll_per_obs, digits=5)),$(round(100*r.val_ks, digits=2)),$(round(r.AIC, digits=3)),$(round(r.BIC, digits=3)),$(round(r.HQC, digits=3)),$(round(r.CAIC, digits=3))");
    end
end

println("\n" * "="^72);
println("  Done.");
println("  Outputs:");
println("    $(joinpath(OUT_DIR, "K_Selection_Validation_Pre2020.txt"))");
println("    $(joinpath(PAPER_ROBUSTNESS_DIR, "k_selection_validation_pre2020.csv"))");
println("="^72);
