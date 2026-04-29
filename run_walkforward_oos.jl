# =========================================================================== #
# run_walkforward_oos.jl
#
# A2 (REVISION_PLAN_V3_TO_ACCEPT.md): Walk-forward / rolling-origin OoS for
# CHMM-N at K ∈ {3, 18}. Six folds, each train 5 years, test 1 year (≈ 250
# trading days). At each fold refit CHMM-N from scratch on train, simulate
# N_paths paths of length |test| and report KS pass rate, simulated kurtosis,
# and |G_t| ACF-MAE on the fold's test window.
#
# Folds (calendar):
#   W1: train 2014-01..2018-12, test 2019-01..2020-01
#   W2: train 2015-01..2019-12, test 2020-01..2021-01  (COVID)
#   W3: train 2016-01..2020-12, test 2021-01..2022-01
#   W4: train 2017-01..2021-12, test 2022-01..2023-01  (rate hike)
#   W5: train 2018-01..2022-12, test 2023-01..2024-01
#   W6: train 2019-01..2023-12, test 2024-01..2025-01
#
# Output:
#   results/walkforward/walkforward_summary.csv
#   results/walkforward/walkforward_summary.txt
# =========================================================================== #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random, Statistics, StatsBase, HypothesisTests, Distributions, Printf, Dates

const SEED      = 20260420;
const K_VALUES  = [3, 18];
const N_PATHS   = 500;
const MAX_ITER  = 60;
const ALPHA_KS  = 0.05;
const RISK_FREE = 0.0;
const DT        = 1/252;
const L_LAGS    = 252;
const TICKER    = "SPY";

const OUT_DIR = joinpath(_ROOT, "results", "walkforward");
mkpath(OUT_DIR);

# Walk-forward folds: (train_start, train_end_excl, test_start, test_end_excl)
const FOLDS = [
    ("W1", Date(2014,1,1), Date(2019,1,1), Date(2019,1,1), Date(2020,1,1)),
    ("W2", Date(2015,1,1), Date(2020,1,1), Date(2020,1,1), Date(2021,1,1)),
    ("W3", Date(2016,1,1), Date(2021,1,1), Date(2021,1,1), Date(2022,1,1)),
    ("W4", Date(2017,1,1), Date(2022,1,1), Date(2022,1,1), Date(2023,1,1)),
    ("W5", Date(2018,1,1), Date(2023,1,1), Date(2023,1,1), Date(2024,1,1)),
    ("W6", Date(2019,1,1), Date(2024,1,1), Date(2024,1,1), Date(2025,1,1)),
];

Random.seed!(SEED);

println("="^70)
println("  A2: Walk-forward / rolling-origin OoS — CHMM-N at K = $K_VALUES")
println("  Folds: $(length(FOLDS)),  N_paths = $N_PATHS")
println("="^70)

# Load combined dataset (train + OoS)
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
df_train = train_dataset[TICKER];
df_oos   = oos_dataset[TICKER];
df_full  = vcat(df_train, df_oos; cols=:orderequal);
sort!(df_full, :timestamp);
println("[data] $(TICKER) full timeline: $(df_full[1, :timestamp]) → $(df_full[end, :timestamp]) ($(nrow(df_full)) rows)")

function _slice_log_returns(df::DataFrame, t_start::Date, t_end_excl::Date; Δt::Float64=DT)
    ts = Date.(df.timestamp);
    mask = (ts .>= t_start) .& (ts .< t_end_excl);
    sub = df[mask, :];
    P = sub.volume_weighted_average_price;
    R = (1 / Δt) .* (log.(P[2:end] ./ P[1:end-1])) .- 0.0;
    return Vector{Float64}(R);
end

function eval_fold(R_test::Vector{Float64}, sim::Matrix{Float64}; L::Int=L_LAGS, α::Float64=ALPHA_KS)
    np = size(sim, 2); n_o = length(R_test);
    L_use = min(L, n_o - 1);
    acf_o = autocor(abs.(R_test), 1:L_use);
    ks_pass = 0; kurt = 0.0; acf_mae = 0.0;
    for p in 1:np
        s = sim[:, p];
        ks_p = pvalue(ApproximateTwoSampleKSTest(s, R_test));
        ks_pass += (ks_p ≥ α) ? 1 : 0;
        kurt += kurtosis(s);
        acf_mae += mean(abs.(autocor(abs.(s), 1:L_use) .- acf_o));
    end
    return (
        ks_rate = 100 * ks_pass / np,
        kurt    = kurt / np,
        acf_mae = acf_mae / np,
    );
end

results = NamedTuple[];

for (fid, ts, te, vs, ve) in FOLDS
    println("\n" * "="^60)
    println("  Fold $fid:  train [$ts, $te)  test [$vs, $ve)")
    println("="^60)
    R_train = _slice_log_returns(df_full, ts, te);
    R_test  = _slice_log_returns(df_full, vs, ve);
    @printf("  T_train = %d   T_test = %d\n", length(R_train), length(R_test));
    if length(R_test) < 30
        @warn "Test slice <30 days, skipping fold $fid";
        continue;
    end
    for K in K_VALUES
        Random.seed!(SEED + 7 * K);
        m = build(MyContinuousHiddenMarkovModel,
            (observations=R_train, number_of_states=K, max_iter=MAX_ITER));
        Random.seed!(SEED + 7 * K + 100);
        sim = simulate_returns(m, length(R_test); n_paths=N_PATHS);
        metr = eval_fold(R_test, sim);
        @printf("  CHMM-N (K=%d): KS = %5.1f%%   kurt_sim = %6.3f   ACF-MAE |G| = %.4f\n",
            K, metr.ks_rate, metr.kurt, metr.acf_mae)
        push!(results, (
            fold=fid, K=K, T_train=length(R_train), T_test=length(R_test),
            ks_rate=metr.ks_rate, kurt=metr.kurt, acf_mae=metr.acf_mae,
        ));
    end
end

# Per-K summary (median, IQR across folds)
function _stats(xs)
    n = length(xs);
    if n == 0; return (median=NaN, iqr_lo=NaN, iqr_hi=NaN); end
    s = sort(xs);
    med = median(s);
    iqr_lo = s[max(1, Int(round(0.25 * (n + 1))))];
    iqr_hi = s[min(n, Int(round(0.75 * (n + 1))))];
    return (median=med, iqr_lo=iqr_lo, iqr_hi=iqr_hi);
end

open(joinpath(OUT_DIR, "walkforward_summary.csv"), "w") do io
    write(io, "fold,K,T_train,T_test,KS_rate_pct,kurt_sim,acf_mae_abs\n");
    for r in results
        write(io, @sprintf("%s,%d,%d,%d,%.2f,%.4f,%.6f\n",
            r.fold, r.K, r.T_train, r.T_test, r.ks_rate, r.kurt, r.acf_mae));
    end
end

open(joinpath(OUT_DIR, "walkforward_summary.txt"), "w") do io
    println(io, "A2: Walk-forward / rolling-origin OoS — CHMM-N at K = $K_VALUES");
    println(io, "  $(length(FOLDS)) folds, train 5y, test 1y, $N_PATHS paths/fold");
    println(io, "="^70);
    for K in K_VALUES
        rows = [r for r in results if r.K == K];
        if isempty(rows); continue; end
        @printf(io, "\nCHMM-N at K = %d:\n", K);
        @printf(io, "  %-4s %-9s %-9s %-9s %-9s %-9s\n",
            "fold", "T_train", "T_test", "KS (%)", "kurt", "ACF-MAE");
        for r in rows
            @printf(io, "  %-4s %-9d %-9d %-9.1f %-9.3f %-9.4f\n",
                r.fold, r.T_train, r.T_test, r.ks_rate, r.kurt, r.acf_mae);
        end
        ks_s = _stats([r.ks_rate for r in rows]);
        kt_s = _stats([r.kurt for r in rows]);
        ac_s = _stats([r.acf_mae for r in rows]);
        @printf(io, "  median (IQR): KS = %.1f%% [%.1f, %.1f]   kurt = %.3f [%.3f, %.3f]   ACF-MAE = %.4f [%.4f, %.4f]\n",
            ks_s.median, ks_s.iqr_lo, ks_s.iqr_hi,
            kt_s.median, kt_s.iqr_lo, kt_s.iqr_hi,
            ac_s.median, ac_s.iqr_lo, ac_s.iqr_hi);
    end
end

println("\n[done] Output: $OUT_DIR")
