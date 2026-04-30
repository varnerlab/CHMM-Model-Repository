# ========================================================================================= #
# run_k_selection_kfold_h12y_pre2020.jl
#
# Robustness check on the 4-fold full-year K-selection CV
# (run_k_selection_kfold_pre2020.jl): re-runs the same K-selection diagnostic with
# six expanding-window rolling-origin folds at half-year validation cadence,
# covering 2017-2019 in non-overlapping 6-month chunks.
#
# Motivation: the 4-fold full-year design produced a sampling-error read of |z| = 0.07
# on K = 6 vs K = 3 (mean held-out per-observation log-lik). R1 W1's contingency
# triggers a body rebuild at K* = 3 if that result is robust. Before committing to a
# multi-section rebuild, we want to verify the result is not an artefact of the
# 1-year fold cadence.
#
# Folds:
#   Fold 1 : train 2014-01-03 .. 2016-12-30 (~3.0y, ~756 obs),  val 2017-01-03 .. 2017-06-30 (~125)
#   Fold 2 : train 2014-01-03 .. 2017-06-30 (~3.5y, ~881 obs),  val 2017-07-03 .. 2017-12-29 (~125)
#   Fold 3 : train 2014-01-03 .. 2017-12-29 (~4.0y, ~1006 obs), val 2018-01-02 .. 2018-06-29 (~124)
#   Fold 4 : train 2014-01-03 .. 2018-06-29 (~4.5y, ~1130 obs), val 2018-07-02 .. 2018-12-31 (~127)
#   Fold 5 : train 2014-01-03 .. 2018-12-31 (~5.0y, ~1257 obs), val 2019-01-02 .. 2019-06-28 (~123)
#   Fold 6 : train 2014-01-03 .. 2019-06-28 (~5.5y, ~1380 obs), val 2019-07-01 .. 2019-12-31 (~129)
#
# Outputs:
#   results/k_selection_validation/K_Selection_Kfold_H12y_Pre2020.txt
#   ../CHMM-paper/results/robustness/k_selection_kfold_h12y_pre2020.csv
#   ../CHMM-paper/results/robustness/k_selection_kfold_h12y_pre2020_agg.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
using Dates
using Printf

const SEED       = 20260422;
const TICKER     = "SPY";
const RISK_FREE  = 0.0;
const DT         = 1/252;
const MAX_ITER   = 60;
const K_GRID     = [3, 6, 9, 12, 18];
const N_PATHS_KS = 500;

const OUT_DIR              = joinpath(_ROOT, "results", "k_selection_validation");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

# 6 expanding-window rolling-origin folds at half-year validation cadence.
const FOLDS = [
    (Date(2014,1,3), Date(2016,12,30), Date(2017,1,3),  Date(2017,6,30)),
    (Date(2014,1,3), Date(2017,6,30),  Date(2017,7,3),  Date(2017,12,29)),
    (Date(2014,1,3), Date(2017,12,29), Date(2018,1,2),  Date(2018,6,29)),
    (Date(2014,1,3), Date(2018,6,29),  Date(2018,7,2),  Date(2018,12,31)),
    (Date(2014,1,3), Date(2018,12,31), Date(2019,1,2),  Date(2019,6,28)),
    (Date(2014,1,3), Date(2019,6,28),  Date(2019,7,1),  Date(2019,12,31)),
];

println("="^80);
println("  k-fold rolling-origin K-selection on pre-2020 slice (R1-RE1, robustness check)");
println("  Half-year validation cadence, $(length(FOLDS)) expanding-window folds");
println("="^80);

Random.seed!(SEED);

# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY closes...");
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
println("\n[sweep] CHMM-N across $(length(FOLDS)) half-year folds x $(length(K_GRID)) K values...");
println();

rows = Vector{NamedTuple}();

for (fi, (t_train_start, t_train_end, t_val_start, t_val_end)) in enumerate(FOLDS)
    R_train = _slice_between(dates_is, closes_is, t_train_start, t_train_end);
    R_val   = _slice_between(dates_is, closes_is, t_val_start,   t_val_end);
    n_train = length(R_train);
    n_val   = length(R_val);

    println("Fold $fi  train $t_train_start..$t_train_end ($n_train obs)  val $t_val_start..$t_val_end ($n_val obs)");

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
    ));
end

k_star_ll = agg[argmax([a.val_ll_mean for a in agg])].K;
k_star_ks = agg[argmax([a.val_ks_mean for a in agg])].K;

function _sampling_error_read(a3, a6, label::String)
    diff = a6.val_ll_mean - a3.val_ll_mean;
    pooled_sd = sqrt((a3.val_ll_sd^2 + a6.val_ll_sd^2) / 2);
    n = min(a3.n_folds, a6.n_folds);
    se = pooled_sd / sqrt(n);
    z = se > 0 ? diff / se : NaN;
    return (diff=diff, se=se, z=z, label=label);
end

agg_dict   = Dict(a.K => a for a in agg);
read_3v6   = _sampling_error_read(agg_dict[3], agg_dict[6],  "K=6 vs K=3");
read_6v18  = _sampling_error_read(agg_dict[6], agg_dict[18], "K=18 vs K=6");

println("="^80);
println("  Aggregate: mean +/- s.d. across $(length(FOLDS)) half-year folds");
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
out_path = joinpath(OUT_DIR, "K_Selection_Kfold_H12y_Pre2020.txt");
open(out_path, "w") do io
    println(io, "="^130);
    println(io, "k-fold rolling-origin K-selection on pre-2020 slice  (R1-RE1, half-year cadence robustness check)");
    println(io, "="^130);
    println(io);
    println(io, "Setup");
    println(io, "  Ticker          : $TICKER");
    println(io, "  Seed            : $SEED");
    println(io, "  K grid          : $K_GRID");
    println(io, "  Folds           : $(length(FOLDS)) expanding-window rolling-origin");
    println(io, "  Validation len  : ~6 months per fold (~125 obs)");
    println(io, "  KS sim paths    : $N_PATHS_KS");
    println(io, "  EM max_iter     : $MAX_ITER");
    println(io);
    println(io, "Folds:");
    for (fi, (ts, te, vs, ve)) in enumerate(FOLDS)
        n_train_approx = count(d -> ts <= d <= te, dates_is) - 1;
        n_val_approx   = count(d -> vs <= d <= ve, dates_is) - 1;
        @printf(io, "  Fold %d : train %s..%s (~%d obs), val %s..%s (~%d obs)\n",
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
end

csv_path = joinpath(PAPER_ROBUSTNESS_DIR, "k_selection_kfold_h12y_pre2020.csv");
open(csv_path, "w") do io
    println(io, "model,K,fold,n_train,n_val,train_ll_per_obs,val_ll_per_obs,val_KS_pct");
    for r in rows
        @printf(io, "CHMM-N,%d,%d,%d,%d,%.5f,%.5f,%.2f\n",
                r.K, r.fold, r.n_train, r.n_val,
                r.train_ll_per_obs, r.val_ll_per_obs, 100*r.val_ks);
    end
end

agg_csv_path = joinpath(PAPER_ROBUSTNESS_DIR, "k_selection_kfold_h12y_pre2020_agg.csv");
open(agg_csv_path, "w") do io
    println(io, "K,n_folds,val_ll_per_obs_mean,val_ll_per_obs_sd,val_KS_pct_mean,val_KS_pct_sd");
    for a in agg
        @printf(io, "%d,%d,%.5f,%.5f,%.2f,%.2f\n",
                a.K, a.n_folds, a.val_ll_mean, a.val_ll_sd,
                100*a.val_ks_mean, 100*a.val_ks_sd);
    end
end

println("="^80);
println("  6-fold half-year K-selection complete.");
println("  Per-fold     : $csv_path");
println("  Aggregate    : $agg_csv_path");
println("  Human-readable: $out_path");
println("="^80);
