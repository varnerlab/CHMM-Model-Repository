# ========================================================================================= #
# run_k_selection_kfold_pre2020.jl
#
# k-fold rolling-origin cross-validation of K-selection on the strictly pre-2020 slice.
# Addresses peer-review item R1-RE1 (priority 1):
#
#   "Report mean +/- s.d. of held-out per-observation log-likelihood and held-out KS at
#    K in {3, 6, 9, 12, 18} over 5 or 10 folds. The current single-fold result is one
#    observation."
#
# The companion run_k_selection_validation_pre2020.jl reports a single fold (4.5y train
# 2014-01..2018-06, 1.5y val 2018-07..2019-12). Reviewer R1 W1's contingency is that if
# K = 6 does not remain preferred over K = 3 outside sampling error under k-fold CV, the
# body should be rebuilt at K* = 3.
#
# Protocol: 4 expanding-window rolling-origin folds, 1-year validation per fold.
# We use 4 instead of 5 because a 5-fold design forces one fold to have only ~1 year
# of training data, which is below the practical floor for K = 18 EM convergence on
# this dataset (252 obs vs ~342 free parameters at K = 18). Averaging is per-observation
# so fold-length differences are not an issue.
#
#   Fold 1 : train 2014-01-03 .. 2015-12-31  (~504 obs),  val 2016 (~252 obs)
#   Fold 2 : train 2014-01-03 .. 2016-12-31  (~756 obs),  val 2017 (~252 obs)
#   Fold 3 : train 2014-01-03 .. 2017-12-31  (~1008 obs), val 2018 (~252 obs)
#   Fold 4 : train 2014-01-03 .. 2018-12-31  (~1260 obs), val 2019 (~252 obs)
#
# Outputs:
#   results/k_selection_validation/K_Selection_Kfold_Pre2020.txt
#   ../CHMM-paper/results/robustness/k_selection_kfold_pre2020.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
using Dates
using Printf

const SEED      = 20260422;
const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const MAX_ITER  = 60;
const K_GRID    = [3, 6, 9, 12, 18];
const N_PATHS_KS = 500;

const OUT_DIR              = joinpath(_ROOT, "results", "k_selection_validation");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

# Fold definitions: (train_start, train_end, val_start, val_end), all inclusive.
const FOLDS = [
    (Date(2014,1,3), Date(2015,12,31), Date(2016,1,4),  Date(2016,12,30)),
    (Date(2014,1,3), Date(2016,12,30), Date(2017,1,3),  Date(2017,12,29)),
    (Date(2014,1,3), Date(2017,12,29), Date(2018,1,2),  Date(2018,12,31)),
    (Date(2014,1,3), Date(2018,12,31), Date(2019,1,2),  Date(2019,12,31)),
];

println("="^80);
println("  k-fold rolling-origin K-selection on pre-2020 slice (R1-RE1)");
println("  Seed $SEED, K grid $K_GRID, $(length(FOLDS)) expanding-window folds");
println("="^80);

Random.seed!(SEED);

# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY closes and slicing per fold...");
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

# --------------------------------------------------------------------------------------- #
# Forward log-likelihood under a fitted CHMM-N (matches single-fold script)
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

# --------------------------------------------------------------------------------------- #
# Per-fold sweep
# --------------------------------------------------------------------------------------- #
println("\n[sweep] CHMM-N across $(length(FOLDS)) folds x $(length(K_GRID)) K values...");
println();

# rows[(fold, K)] = (n_train, n_val, val_ll_per_obs, val_ks)
rows = Vector{NamedTuple}();

for (fi, (t_train_start, t_train_end, t_val_start, t_val_end)) in enumerate(FOLDS)
    R_train = _slice_between(dates_is, closes_is, t_train_start, t_train_end);
    R_val   = _slice_between(dates_is, closes_is, t_val_start,   t_val_end);
    n_train = length(R_train);
    n_val   = length(R_val);

    println("Fold $fi  train $t_train_start .. $t_train_end ($n_train obs)  val $t_val_start .. $t_val_end ($n_val obs)");

    for K in K_GRID
        Random.seed!(SEED + 1000*fi + K);
        m = build(MyContinuousHiddenMarkovModel,
                  (observations=R_train, number_of_states=K, max_iter=MAX_ITER));
        train_ll = m.log_likelihood_history[end];
        train_ll_per_obs = train_ll / n_train;
        val_ll   = forward_log_likelihood(R_val, m);
        val_ll_per_obs = val_ll / n_val;
        Random.seed!(SEED + 2000*fi + K);
        sim_val = simulate_returns(m, n_val; n_paths=N_PATHS_KS);
        val_ks  = ks_pass_rate(R_val, sim_val);
        push!(rows, (
            fold=fi, K=K, n_train=n_train, n_val=n_val,
            train_ll_per_obs=train_ll_per_obs,
            val_ll_per_obs=val_ll_per_obs,
            val_ks=val_ks,
        ));
        @printf("  K=%2d  train_ll/obs=%+.4f  val_ll/obs=%+.4f  val_KS=%5.1f%%\n",
                K, train_ll_per_obs, val_ll_per_obs, 100*val_ks);
    end
    println();
end

# --------------------------------------------------------------------------------------- #
# Aggregate per K: mean +/- s.d. across folds
# --------------------------------------------------------------------------------------- #
agg = Vector{NamedTuple}();
for K in K_GRID
    rows_K = filter(r -> r.K == K, rows);
    val_ll_per_obs = [r.val_ll_per_obs for r in rows_K];
    val_ks         = [r.val_ks         for r in rows_K];
    push!(agg, (
        K=K, n_folds=length(rows_K),
        val_ll_mean=mean(val_ll_per_obs),
        val_ll_sd  =std(val_ll_per_obs),
        val_ks_mean=mean(val_ks),
        val_ks_sd  =std(val_ks),
        per_fold_ll=val_ll_per_obs,
        per_fold_ks=val_ks,
    ));
end

# Selection rules
k_star_ll  = agg[argmax([a.val_ll_mean for a in agg])].K;
k_star_ks  = agg[argmax([a.val_ks_mean for a in agg])].K;

# Sampling-error read for K=3 vs K=6 and K=6 vs K=18
function _sampling_error_read(a3, a6, label::String)
    diff = a6.val_ll_mean - a3.val_ll_mean;
    pooled_sd = sqrt((a3.val_ll_sd^2 + a6.val_ll_sd^2) / 2);
    n = min(a3.n_folds, a6.n_folds);
    se = pooled_sd / sqrt(n);
    z = se > 0 ? diff / se : NaN;
    return (diff=diff, se=se, z=z, label=label);
end

agg_dict = Dict(a.K => a for a in agg);
read_3v6   = _sampling_error_read(agg_dict[3],  agg_dict[6],  "K=6 vs K=3");
read_6v18  = _sampling_error_read(agg_dict[6],  agg_dict[18], "K=18 vs K=6");

println("="^80);
println("  Aggregate: mean +/- s.d. across $(length(FOLDS)) folds");
println("="^80);
println();
@printf("  %-3s  %-15s  %-15s  %-7s\n", "K", "val_ll/obs (m)", "val_ll/obs (sd)", "val_KS%");
for a in agg
    @printf("  %-3d  %+.4f         %.4f          %5.1f +/- %4.1f\n",
            a.K, a.val_ll_mean, a.val_ll_sd, 100*a.val_ks_mean, 100*a.val_ks_sd);
end
println();
println("  K* by mean held-out log-lik per obs : $k_star_ll");
println("  K* by mean held-out KS pass rate    : $k_star_ks");
println();
println("  Sampling-error reads (mean diff / pooled SE / approx z):");
@printf("    %s : diff=%+.4f  se=%.4f  z=%+.2f\n", read_3v6.label,  read_3v6.diff,  read_3v6.se,  read_3v6.z);
@printf("    %s : diff=%+.4f  se=%.4f  z=%+.2f\n", read_6v18.label, read_6v18.diff, read_6v18.se, read_6v18.z);
println();

# --------------------------------------------------------------------------------------- #
# Output: human-readable
# --------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "K_Selection_Kfold_Pre2020.txt");
open(out_path, "w") do io
    println(io, "="^130);
    println(io, "k-fold rolling-origin K-selection on pre-2020 slice  (R1-RE1)");
    println(io, "="^130);
    println(io);
    println(io, "Setup");
    println(io, "  Ticker          : $TICKER");
    println(io, "  Seed            : $SEED");
    println(io, "  K grid          : $K_GRID");
    println(io, "  Folds           : $(length(FOLDS)) expanding-window rolling-origin");
    println(io, "  Validation len  : ~1 year per fold");
    println(io, "  KS sim paths    : $N_PATHS_KS");
    println(io, "  EM max_iter     : $MAX_ITER");
    println(io);
    println(io, "Folds:");
    for (fi, (ts, te, vs, ve)) in enumerate(FOLDS)
        n_train_approx = count(d -> ts <= d <= te, dates_is) - 1;
        n_val_approx   = count(d -> vs <= d <= ve, dates_is) - 1;
        @printf(io, "  Fold %d : train %s .. %s (~%d obs), val %s .. %s (~%d obs)\n",
                fi, ts, te, n_train_approx, vs, ve, n_val_approx);
    end
    println(io);

    println(io, "Per-fold results (CHMM-N):");
    println(io);
    @printf(io, "  %-4s  %-3s  %-10s  %-15s  %-15s  %-9s\n",
            "fold", "K", "n_train", "train_ll/obs", "val_ll/obs", "val_KS%");
    println(io, "  " * "-"^72);
    for r in rows
        @printf(io, "  %-4d  %-3d  %-10d  %+.4f          %+.4f          %5.1f\n",
                r.fold, r.K, r.n_train, r.train_ll_per_obs, r.val_ll_per_obs, 100*r.val_ks);
    end
    println(io);

    println(io, "Aggregate (mean +/- s.d. across folds):");
    println(io);
    @printf(io, "  %-3s  %-7s  %-19s  %-19s\n",
            "K", "n_folds", "val_ll/obs (m / sd)", "val_KS% (m / sd)");
    println(io, "  " * "-"^60);
    for a in agg
        @printf(io, "  %-3d  %-7d  %+.4f / %.4f    %5.1f / %4.1f\n",
                a.K, a.n_folds, a.val_ll_mean, a.val_ll_sd, 100*a.val_ks_mean, 100*a.val_ks_sd);
    end
    println(io);

    println(io, "Selection:");
    println(io, "  K* by mean held-out log-lik per obs : $k_star_ll");
    println(io, "  K* by mean held-out KS pass rate    : $k_star_ks");
    println(io);
    println(io, "Sampling-error reads on the held-out per-observation log-likelihood:");
    @printf(io, "  %-15s : diff=%+.4f  pooled SE=%.4f  approx z=%+.2f\n",
            read_3v6.label,  read_3v6.diff,  read_3v6.se,  read_3v6.z);
    @printf(io, "  %-15s : diff=%+.4f  pooled SE=%.4f  approx z=%+.2f\n",
            read_6v18.label, read_6v18.diff, read_6v18.se, read_6v18.z);
    println(io);
    println(io, "Substantive read:");
    if abs(read_3v6.z) < 1.96
        println(io, "  K=6 vs K=3 mean held-out log-lik gap is NOT significant at 5% (|z| < 1.96).");
        println(io, "  R1 W1's contingency is in scope: body headline rebuild at K=3 is supported");
        println(io, "  by this CV design. Companion-paper sensitivity at K=6 remains valid.");
    else
        if read_3v6.diff > 0
            println(io, "  K=6 dominates K=3 on mean held-out log-lik at |z| >= 1.96. Body");
            println(io, "  headline at K=6 is supported under k-fold CV.");
        else
            println(io, "  K=3 dominates K=6 on mean held-out log-lik at |z| >= 1.96. Body");
            println(io, "  rebuild at K=3 is the correct response per R1 W1's contingency.");
        end
    end
end

# --------------------------------------------------------------------------------------- #
# Output: machine-readable CSV (for paper-side import)
# --------------------------------------------------------------------------------------- #
csv_path = joinpath(PAPER_ROBUSTNESS_DIR, "k_selection_kfold_pre2020.csv");
open(csv_path, "w") do io
    println(io, "model,K,fold,n_train,n_val,train_ll_per_obs,val_ll_per_obs,val_KS_pct");
    for r in rows
        @printf(io, "CHMM-N,%d,%d,%d,%d,%.5f,%.5f,%.2f\n",
                r.K, r.fold, r.n_train, r.n_val,
                r.train_ll_per_obs, r.val_ll_per_obs, 100*r.val_ks);
    end
end

agg_csv_path = joinpath(PAPER_ROBUSTNESS_DIR, "k_selection_kfold_pre2020_agg.csv");
open(agg_csv_path, "w") do io
    println(io, "K,n_folds,val_ll_per_obs_mean,val_ll_per_obs_sd,val_KS_pct_mean,val_KS_pct_sd");
    for a in agg
        @printf(io, "%d,%d,%.5f,%.5f,%.2f,%.2f\n",
                a.K, a.n_folds, a.val_ll_mean, a.val_ll_sd,
                100*a.val_ks_mean, 100*a.val_ks_sd);
    end
end

println("="^80);
println("  k-fold K-selection complete.");
println("  Per-fold     : $csv_path");
println("  Aggregate    : $agg_csv_path");
println("  Human-readable: $out_path");
println("="^80);
