# ========================================================================================= #
# run_track_m4_rolling_and_weekly.jl
#
# Track M4 (revision response to referee comment M4): broaden the empirical base beyond the
# single 572-day OoS window on SPY. This session covers two of the three M4 sub-asks:
#
#   (a) rolling-origin OoS: five rolling 1-year OoS windows across 2022-2026 with the
#       preceding 8-year window as IS, for CHMM-N / -t / -L at K = 18. Reports per-window
#       and aggregated IS/OoS KS pass rate plus unconditional Kupiec VaR calibration.
#
#   (b) weekly sampling frequency: 5-day non-overlapping aggregation of daily log returns
#       on the 2014-2024 IS window, CHMM-N / -t / -L refit at K ∈ {6, 12} (smaller K
#       because the weekly sample has ~500 observations instead of ~2500). Reports
#       simulated IS KS, simulated kurtosis, and ACF-MAE at the weekly frequency.
#
# The third M4 sub-ask (multiple indices: MSCI World, DAX, Nikkei, FTSE) requires a
# separate Alpaca fetch of URTH / EWG / EWJ / EWU daily bars; handled in
# `run_track_m4_indices.jl` once credentials are verified.
#
# Outputs:
#   results/track_m4/M4_Rolling_Origin.txt
#   results/track_m4/M4_Weekly.txt
#   ../CHMM-paper/results/revision/M4_rolling_origin.csv
#   ../CHMM-paper/results/revision/M4_weekly.csv
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
const N_PATHS   = 500;   # reduced from 1000 to keep total runtime tractable across 5 windows × 3 models
const K_MAIN    = 18;
const MAX_ITER  = 60;

const TRACK_M4_DIR       = joinpath(_ROOT, "results", "track_m4");
const PAPER_REVISION_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "revision"));
mkpath(TRACK_M4_DIR);
mkpath(PAPER_REVISION_DIR);

println("="^72)
println("  Track M4 (a, b): rolling-origin OoS + weekly-frequency robustness (M4)")
println("  Seed $SEED, K_main=$K_MAIN, N_paths=$N_PATHS")
println("="^72)

# --------------------------------------------------------------------------------------- #
# Data: concatenate IS + OoS SPY DataFrames and extract (date, close) per row
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading and concatenating SPY IS + OoS DataFrames...");
train = MyPortfolioDataSet() |> x -> x["dataset"];
oos   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
spy_is  = train["SPY"];
spy_oos = oos["SPY"];
spy_all = vcat(spy_is, spy_oos; cols=:union);
# Normalise timestamp -> Date and sort
dates_all  = Date.(spy_all.timestamp);
closes_all = Vector{Float64}(spy_all.close);
order = sortperm(dates_all);
dates_all  = dates_all[order];
closes_all = closes_all[order];
# Dedupe same-date rows (rare when IS/OoS overlap boundary-days)
keep = [true; diff(dates_all) .> Day(0)];
dates_all  = dates_all[keep];
closes_all = closes_all[keep];
println("  SPY total daily rows: $(length(dates_all))   range: $(dates_all[1]) → $(dates_all[end])");

# Log growth rate per the paper's convention: G_t = (1/Δt) * ln(P_t / P_{t-1}) - r_f, Δt = 1/252.
function _log_growth_series(closes::Vector{Float64}; Δt::Float64=DT, rf::Float64=RISK_FREE)
    N = length(closes);
    r = Vector{Float64}(undef, N - 1);
    for t in 2:N
        r[t-1] = (1/Δt) * log(closes[t] / closes[t-1]) - rf;
    end
    return r;
end

# Slice return series between two dates (inclusive).
function _slice_between(dates::Vector{Date}, closes::Vector{Float64}, t0::Date, t1::Date)
    # We want returns whose ENDPOINT date is within [t0, t1]; returns are indexed by
    # t = 2, ..., N so the start index in the return vector is (findfirst date ≥ t0)-1 +1
    # = findfirst date ≥ t0. End index: findlast date ≤ t1.
    idx0 = findfirst(d -> d >= t0, dates);
    idx1 = findlast(d -> d <= t1, dates);
    @assert idx0 !== nothing && idx1 !== nothing && idx1 >= idx0;
    closes_sub = closes[idx0:idx1];
    return _log_growth_series(closes_sub);
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
# (a) Rolling-origin OoS on SPY
# --------------------------------------------------------------------------------------- #
println("\n" * "="^72);
println("  (a) Rolling-origin OoS: 5 windows, 8-year IS + 1-year OoS");
println("="^72);

# Define 5 windows with 8-year IS and 1-year OoS, advancing by 1 year
windows = [
    (Date(2014,1,3),  Date(2021,12,31), Date(2022,1,3),  Date(2022,12,30)),
    (Date(2015,1,2),  Date(2022,12,30), Date(2023,1,3),  Date(2023,12,29)),
    (Date(2016,1,4),  Date(2023,12,29), Date(2024,1,2),  Date(2024,12,31)),
    (Date(2017,1,3),  Date(2024,12,31), Date(2025,1,2),  Date(2025,12,31)),
    (Date(2018,1,2),  Date(2025,12,31), Date(2026,1,2),  Date(2026,4,20)),
];

rolling_results = NamedTuple[];

for (widx, (is0, is1, oos0, oos1)) in enumerate(windows)
    R_is_w  = _slice_between(dates_all, closes_all, is0,  is1);
    R_oos_w = _slice_between(dates_all, closes_all, oos0, oos1);
    println("\n  [window $widx] IS $is0 → $is1 ($(length(R_is_w)) obs), OoS $oos0 → $oos1 ($(length(R_oos_w)) obs)");

    for (name, sym) in [("CHMM-N", :n), ("CHMM-t", :t), ("CHMM-L", :l)]
        Random.seed!(SEED + 10*widx + (sym == :n ? 1 : sym == :t ? 2 : 3));
        local m;
        if sym == :n
            m = build(MyContinuousHiddenMarkovModel,
                (observations=R_is_w, number_of_states=K_MAIN, max_iter=MAX_ITER));
        elseif sym == :t
            m = build(MyStudentTHiddenMarkovModel,
                (observations=R_is_w, number_of_states=K_MAIN, max_iter=MAX_ITER));
        else
            m = build(MyLaplaceHiddenMarkovModel,
                (observations=R_is_w, number_of_states=K_MAIN, max_iter=MAX_ITER));
        end

        Random.seed!(SEED + 1000 + 10*widx);
        sim_is  = simulate_returns(m, length(R_is_w);  n_paths=N_PATHS);
        Random.seed!(SEED + 2000 + 10*widx);
        sim_oos = simulate_returns(m, length(R_oos_w); n_paths=N_PATHS);

        is_ks  = ks_pass_rate(R_is_w,  sim_is);
        oos_ks = ks_pass_rate(R_oos_w, sim_oos);

        # Unconditional pooled-archive VaR on OoS
        pooled_oos = vec(sim_oos);
        v01 = quantile(pooled_oos, 0.01);
        v05 = quantile(pooled_oos, 0.05);
        br01 = R_oos_w .<= v01;
        br05 = R_oos_w .<= v05;
        k01 = kupiec_lr(br01, 0.01);
        k05 = kupiec_lr(br05, 0.05);
        c01 = christoffersen_lr(br01);
        c05 = christoffersen_lr(br05);

        push!(rolling_results, (
            window=widx, model=name,
            is_start=is0, is_end=is1, oos_start=oos0, oos_end=oos1,
            is_n=length(R_is_w), oos_n=length(R_oos_w),
            is_ks=is_ks, oos_ks=oos_ks,
            br01=k01.breach_rate, LRuc01=k01.LR, LRind01=c01.LR,
            br05=k05.breach_rate, LRuc05=k05.LR, LRind05=c05.LR,
        ));
        println("    $name  IS KS $(round(100*is_ks, digits=1))%  OoS KS $(round(100*oos_ks, digits=1))%  LR_uc01 $(round(k01.LR, digits=2))  LR_uc05 $(round(k05.LR, digits=2))");
    end
end

# --------------------------------------------------------------------------------------- #
# (b) Weekly-frequency on SPY IS (2014-2024)
# --------------------------------------------------------------------------------------- #
println("\n" * "="^72);
println("  (b) Weekly-frequency: 5-day non-overlapping aggregation, K ∈ {6, 12}");
println("="^72);

R_is_daily = _slice_between(dates_all, closes_all, Date(2014,1,3), Date(2024,1,3));
n_daily = length(R_is_daily);
# 5-day non-overlapping aggregation: sum of 5 consecutive daily log growth rates = 5-day log growth rate / Δt_weekly
# With Δt_weekly = 5/252, the weekly G_t = Σ daily G_t / 5 (mean), but following the existing annualization
# convention the sum of 5 daily G_t already integrates the weekly drift at annualized scale; take the sum.
n_weekly = div(n_daily, 5);
R_is_weekly = [sum(R_is_daily[5*(i-1)+1:5*i]) for i in 1:n_weekly];
println("\n  daily IS rows: $n_daily, weekly aggregated: $n_weekly");

weekly_results = NamedTuple[];

for K in [6, 12]
    for (name, sym) in [("CHMM-N", :n), ("CHMM-t", :t), ("CHMM-L", :l)]
        Random.seed!(SEED + 5000 + K + (sym == :n ? 1 : sym == :t ? 2 : 3));
        local m;
        if sym == :n
            m = build(MyContinuousHiddenMarkovModel,
                (observations=R_is_weekly, number_of_states=K, max_iter=MAX_ITER));
        elseif sym == :t
            m = build(MyStudentTHiddenMarkovModel,
                (observations=R_is_weekly, number_of_states=K, max_iter=MAX_ITER));
        else
            m = build(MyLaplaceHiddenMarkovModel,
                (observations=R_is_weekly, number_of_states=K, max_iter=MAX_ITER));
        end

        Random.seed!(SEED + 6000 + K);
        sim = simulate_returns(m, n_weekly; n_paths=N_PATHS);
        is_ks = ks_pass_rate(R_is_weekly, sim);
        sim_kurt = mean([kurtosis(sim[:, p]) for p in 1:N_PATHS]);
        obs_kurt = kurtosis(R_is_weekly);
        # ACF-MAE on |r| at lags 1..50 (at weekly frequency; 50 weeks ≈ 1 year)
        obs_acf = autocor(abs.(R_is_weekly), 1:50);
        sim_acf_mean = zeros(50);
        for p in 1:N_PATHS
            sim_acf_mean .+= autocor(abs.(sim[:, p]), 1:50);
        end
        sim_acf_mean ./= N_PATHS;
        acf_mae = mean(abs.(obs_acf .- sim_acf_mean));

        push!(weekly_results, (
            model=name, K=K, obs_kurt=obs_kurt,
            is_ks=is_ks, sim_kurt=sim_kurt, acf_mae=acf_mae,
        ));
        println("  K=$K  $name  IS KS $(round(100*is_ks, digits=1))%  sim kurt $(round(sim_kurt, digits=2)) (obs $(round(obs_kurt, digits=2)))  ACF-MAE $(round(acf_mae, digits=4))");
    end
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
open(joinpath(TRACK_M4_DIR, "M4_Rolling_Origin.txt"), "w") do io
    println(io, "="^160);
    println(io, "Track M4(a). Rolling-origin OoS: 5 windows × 3 emission families (referee M4 response).");
    println(io, "="^160);
    println(io, "");
    println(io, rpad("win",3), " | ", rpad("model",7), " | ",
                rpad("IS range",23), " | ", rpad("OoS range",23), " | ",
                rpad("IS_KS%",6), " | ", rpad("OoS_KS%",7), " | ",
                rpad("br%01",6), " | ", rpad("LRuc01",6), " | ",
                rpad("LRind01",7), " | ", rpad("br%05",6), " | ",
                rpad("LRuc05",6), " | ", rpad("LRind05",7));
    println(io, "-"^160);
    for r in rolling_results
        println(io, rpad(r.window,3), " | ", rpad(r.model,7), " | ",
                    rpad("$(r.is_start)→$(r.is_end)", 23), " | ",
                    rpad("$(r.oos_start)→$(r.oos_end)", 23), " | ",
                    rpad(round(100*r.is_ks, digits=1), 6), " | ",
                    rpad(round(100*r.oos_ks, digits=1), 7), " | ",
                    rpad(round(100*r.br01, digits=1), 6), " | ",
                    rpad(round(r.LRuc01, digits=2), 6), " | ",
                    rpad(round(r.LRind01, digits=2), 7), " | ",
                    rpad(round(100*r.br05, digits=1), 6), " | ",
                    rpad(round(r.LRuc05, digits=2), 6), " | ",
                    rpad(round(r.LRind05, digits=2), 7));
    end
    println(io, "="^160);
    println(io, "");
    # Aggregate mean ± std per model across 5 windows
    println(io, "Per-model aggregates across 5 rolling windows (mean ± std):");
    println(io, "");
    for model in ["CHMM-N", "CHMM-t", "CHMM-L"]
        subs = [r for r in rolling_results if r.model == model];
        is_ks_vals  = [100*r.is_ks for r in subs];
        oos_ks_vals = [100*r.oos_ks for r in subs];
        br01_vals   = [100*r.br01 for r in subs];
        br05_vals   = [100*r.br05 for r in subs];
        LRuc05_vals = [r.LRuc05 for r in subs];
        LRuc01_vals = [r.LRuc01 for r in subs];
        println(io, "  $(rpad(model, 7)) IS KS: $(round(mean(is_ks_vals), digits=1))% ± $(round(std(is_ks_vals), digits=1))%   OoS KS: $(round(mean(oos_ks_vals), digits=1))% ± $(round(std(oos_ks_vals), digits=1))%   br%01: $(round(mean(br01_vals), digits=2))% ± $(round(std(br01_vals), digits=2))%   br%05: $(round(mean(br05_vals), digits=2))% ± $(round(std(br05_vals), digits=2))%   LR_uc01: $(round(mean(LRuc01_vals), digits=2))   LR_uc05: $(round(mean(LRuc05_vals), digits=2))");
    end
end

open(joinpath(PAPER_REVISION_DIR, "M4_rolling_origin.csv"), "w") do io
    println(io, "window,model,is_start,is_end,oos_start,oos_end,is_n,oos_n,is_ks_pct,oos_ks_pct,br01_pct,LRuc01,LRind01,br05_pct,LRuc05,LRind05");
    for r in rolling_results
        println(io, "$(r.window),$(r.model),$(r.is_start),$(r.is_end),$(r.oos_start),$(r.oos_end),$(r.is_n),$(r.oos_n),$(round(100*r.is_ks, digits=2)),$(round(100*r.oos_ks, digits=2)),$(round(100*r.br01, digits=2)),$(round(r.LRuc01, digits=3)),$(round(r.LRind01, digits=3)),$(round(100*r.br05, digits=2)),$(round(r.LRuc05, digits=3)),$(round(r.LRind05, digits=3))");
    end
end

open(joinpath(TRACK_M4_DIR, "M4_Weekly.txt"), "w") do io
    println(io, "="^120);
    println(io, "Track M4(b). Weekly-frequency evaluation: 5-day non-overlapping aggregation on 2014-2024 SPY IS.");
    println(io, "="^120);
    println(io, "");
    println(io, "Weekly T = $n_weekly.  Observed weekly excess kurtosis = $(round(kurtosis(R_is_weekly), digits=3)).");
    println(io, "");
    println(io, rpad("K",3), " | ", rpad("model",7), " | ",
                rpad("IS_KS%",6), " | ", rpad("sim_kurt",8), " | ",
                rpad("obs_kurt",8), " | ", rpad("ACF_MAE",8));
    println(io, "-"^120);
    for r in weekly_results
        println(io, rpad(r.K,3), " | ", rpad(r.model,7), " | ",
                    rpad(round(100*r.is_ks, digits=1), 6), " | ",
                    rpad(round(r.sim_kurt, digits=2), 8), " | ",
                    rpad(round(r.obs_kurt, digits=2), 8), " | ",
                    rpad(round(r.acf_mae, digits=4), 8));
    end
    println(io, "="^120);
end

open(joinpath(PAPER_REVISION_DIR, "M4_weekly.csv"), "w") do io
    println(io, "model,K,obs_kurt,IS_KS_pct,sim_kurt,ACF_MAE");
    for r in weekly_results
        println(io, "$(r.model),$(r.K),$(round(r.obs_kurt, digits=3)),$(round(100*r.is_ks, digits=2)),$(round(r.sim_kurt, digits=3)),$(round(r.acf_mae, digits=5))");
    end
end

println("\n" * "="^72);
println("  Track M4 (a, b) complete.");
println("  Rolling-origin text: $(joinpath(TRACK_M4_DIR, "M4_Rolling_Origin.txt"))");
println("  Weekly text        : $(joinpath(TRACK_M4_DIR, "M4_Weekly.txt"))");
println("  Paper CSVs         : $PAPER_REVISION_DIR/M4_rolling_origin.csv, $PAPER_REVISION_DIR/M4_weekly.csv");
println("="^72);
