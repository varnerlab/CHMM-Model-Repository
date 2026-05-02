# ========================================================================================= #
# run_track_a_utility.jl
#
# Track A downstream-utility runner (DECISION-MEMO items A4, A5, A8):
#   A4  TSTR vol forecaster   train HAR(1,5,22) on synthetic paths, evaluate RMSE/QLIKE on real OoS
#   A5  Vol-target strategy   use the A4 forecaster to size a SPY position, report Sharpe/MDD/turn
#   A8  Kupiec + Christoffersen LR tests on 1 % and 5 % VaR from each generator
#
# Inputs: results/track_a/sim_archive_cache.jld2 (produced by run_track_a_metrics.jl).
# Outputs:
#   results/track_a/tstr_vol_forecaster.txt    A4 panel
#   results/track_a/vol_target_strategy.txt    A5 panel
#   results/track_a/VaR_LR_tests.txt           A8 Kupiec + Christoffersen
# ========================================================================================= #

const _PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."));
using Pkg; Pkg.activate(_PROJECT_ROOT);
include(joinpath(_PROJECT_ROOT, "Include.jl"));

using Random
const SEED = 20260422;
Random.seed!(SEED);

const TICKER       = "SPY";
const RISK_FREE    = 0.0;
const DT           = 1/252;

const TRACK_A_DIR = joinpath(_ROOT, "results", "track_a");
const CACHE_PATH  = joinpath(TRACK_A_DIR, "sim_archive_cache.jld2");

if !isfile(CACHE_PATH)
    error("Archive cache $CACHE_PATH not found. Run run_track_a_metrics.jl first.");
end

println("="^72)
println("  Track A utility runner")
println("="^72)

# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY IS + OoS...");
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
println("  IS: $n_is obs; OoS: $n_oos obs");

# Load archive
println("[cache] Loading simulation archives...");
archive = load(CACHE_PATH)["archive"];

MODEL_ORDER = [
    "Bootstrap", "Gaussian", "Laplace",
    "DiscreteNJ", "DiscreteWJ", "GARCH",
    "CHMM-N", "CHMM-t", "CHMM-L",
];

# --------------------------------------------------------------------------------------- #
# A4: TSTR vol forecaster (HAR)
#
# Use daily squared excess returns as a proxy for realized variance.
# HAR(1,5,22): RV_t = β0 + β1 RV_{t-1} + β5 avg(RV_{t-5..t-1}) + β22 avg(RV_{t-22..t-1}) + ε
# Train on synthetic paths concatenated, evaluate on real OoS.
# --------------------------------------------------------------------------------------- #
println("\n[A4] TSTR vol forecaster...");

function _build_har_features(r::AbstractVector)
    rv = r .^ 2;
    T = length(rv);
    # Require 22 lags
    idx = 23:T;
    X = Array{Float64,2}(undef, length(idx), 4);
    y = Vector{Float64}(undef, length(idx));
    for (i, t) in enumerate(idx)
        X[i, 1] = 1.0;
        X[i, 2] = rv[t-1];
        X[i, 3] = mean(rv[t-5:t-1]);
        X[i, 4] = mean(rv[t-22:t-1]);
        y[i]    = rv[t];
    end
    return X, y;
end

function _har_fit(X, y)
    β = X \ y;
    return β;
end

function _har_predict(β, r::AbstractVector)
    X, y = _build_har_features(r);
    yhat = X * β;
    # Guard against negative predictions (RV should be non-negative)
    yhat = max.(yhat, 1e-10);
    return yhat, y;
end

# Real-trained benchmark: HAR fit on R_is, evaluated on R_oos
X_is,  y_is  = _build_har_features(R_is);
β_real = _har_fit(X_is, y_is);

function _rmse(yhat, y); return sqrt(mean((yhat .- y) .^ 2)); end
function _qlike(yhat, y)
    yhat = max.(yhat, 1e-10);
    y    = max.(y, 1e-10);
    return mean(y ./ yhat .- log.(y ./ yhat) .- 1.0);
end

yhat_real, y_oos_targets = _har_predict(β_real, R_oos);
real_rmse  = _rmse(yhat_real, y_oos_targets);
real_qlike = _qlike(yhat_real, y_oos_targets);
println("  Real-trained HAR   OoS RMSE=$(round(real_rmse, digits=5))  QLIKE=$(round(real_qlike, digits=5))");

# Synth-trained rows: for each model, concatenate its IS synthetic paths and refit HAR
n_paths_use = 50;    # use 50 paths per model; length n_is each; gives 2516 * 50 = ~125 k training rows
tstr_results = Dict{String,NamedTuple}();
for m in MODEL_ORDER
    s_is = archive[m].is;
    rs = vec(s_is[:, 1:min(n_paths_use, size(s_is, 2))]);
    Xs, ys = _build_har_features(rs);
    if size(Xs, 1) < 100
        tstr_results[m] = (rmse=NaN, qlike=NaN);
        continue;
    end
    β_s = _har_fit(Xs, ys);
    yhat_s, _ = _har_predict(β_s, R_oos);
    tstr_results[m] = (
        rmse=_rmse(yhat_s, y_oos_targets),
        qlike=_qlike(yhat_s, y_oos_targets),
        β0=β_s[1], β_d=β_s[2], β_w=β_s[3], β_m=β_s[4],
    );
    println("  $m HAR  β=(d=$(round(β_s[2], digits=3)), w=$(round(β_s[3], digits=3)), m=$(round(β_s[4], digits=3)))  OoS RMSE=$(round(tstr_results[m].rmse, digits=5))  QLIKE=$(round(tstr_results[m].qlike, digits=5))");
end

open(joinpath(TRACK_A_DIR, "tstr_vol_forecaster.txt"), "w") do io
    println(io, "="^110);
    println(io, "A4. TSTR (Train on Synthetic, Test on Real): HAR(1,5,22) vol forecaster");
    println(io, "="^110);
    println(io, "");
    println(io, "Setup    : Target = daily squared return on SPY OoS window ($n_oos obs).");
    println(io, "Features : (intercept, lag-1 RV, mean RV over lags 1-5, mean RV over lags 1-22) (Corsi 2009).");
    println(io, "Training : $n_paths_use synthetic paths per model, concatenated into one long series for OLS.");
    println(io, "Metrics  : RMSE on y vs ŷ; QLIKE = mean(y/ŷ − log(y/ŷ) − 1). Lower is better on both.");
    println(io, "Benchmark: HAR trained on real R_is (2014-2024) and evaluated on real R_oos (2024-2026).");
    println(io, "");
    println(io, rpad("Training data", 16), " | ", rpad("RMSE", 10), " | ", rpad("RMSE vs real", 13), " | ", rpad("QLIKE", 10), " | ", rpad("QLIKE vs real", 14));
    println(io, "-"^110);
    println(io, rpad("Real IS",   16), " | ",
                rpad(round(real_rmse,  digits=5), 10), " | ", rpad("—", 13), " | ",
                rpad(round(real_qlike, digits=5), 10), " | ", rpad("—", 14));
    for m in MODEL_ORDER
        r = tstr_results[m];
        rmse_ratio  = r.rmse  / real_rmse;
        qlike_ratio = r.qlike / real_qlike;
        println(io, rpad(m, 16), " | ",
                    rpad(round(r.rmse,  digits=5), 10), " | ",
                    rpad(round(rmse_ratio,  digits=3), 13), " | ",
                    rpad(round(r.qlike, digits=5), 10), " | ",
                    rpad(round(qlike_ratio, digits=3), 14));
    end
    println(io, "="^110);
    println(io, "");
    println(io, "Ratio column: synth-trained RMSE / real-trained RMSE. A ratio near 1.0 means the synthetic");
    println(io, "data is adequate training substrate for a downstream vol forecaster.");
end

# --------------------------------------------------------------------------------------- #
# A5: Vol-target strategy back-test
#
# Use HAR forecast from each generator to estimate next-day volatility σ̂_t = sqrt(ŷ_t).
# Position w_t = TARGET_VOL / σ̂_t, clipped to [0.1, 3.0].
# Daily P&L = w_t * R_oos[t] (in annualized-excess-growth convention).
# Report annualized Sharpe, max drawdown, annualized turnover.
# --------------------------------------------------------------------------------------- #
println("\n[A5] Vol-target strategy back-test...");

const TARGET_ANNUAL_VOL = 0.15;
const W_MIN             = 0.1;
const W_MAX             = 3.0;

function _run_strategy(yhat::AbstractVector, R::AbstractVector; dt::Float64=1/252)
    # yhat is forecast of daily squared excess growth rate. Annualized vol = sqrt(yhat).
    # But since our returns already divide by dt, a simpler approach: compute realized IS σ
    # on yhat's training support (already annualized) and scale.
    σ̂ = sqrt.(max.(yhat, 1e-10));
    w = clamp.(TARGET_ANNUAL_VOL ./ σ̂, W_MIN, W_MAX);
    pnl = w .* R;
    # Annualized Sharpe (daily * sqrt(252))
    sharpe = mean(pnl) / (std(pnl) + 1e-12) * sqrt(252) * dt;
    # Need to be careful with units: R already in annualized units (dt normalized)
    # Daily effective return = R * dt
    daily_pnl = w .* R .* dt;
    ann_ret = mean(daily_pnl) * 252;
    ann_vol = std(daily_pnl) * sqrt(252);
    sharpe_simple = ann_vol > 0 ? ann_ret / ann_vol : 0.0;
    # Max drawdown on cumulative equity curve
    eq = cumsum(daily_pnl);
    peak = accumulate(max, eq);
    drawdown = peak .- eq;
    mdd = maximum(drawdown);
    # Turnover: sum of |Δw|
    turnover = sum(abs.(diff(w))) * 252 / length(w);
    return (sharpe=sharpe_simple, ann_ret=ann_ret, ann_vol=ann_vol, mdd=mdd, turnover=turnover, mean_w=mean(w));
end

# Align prediction window: yhat has length n_oos - 22
R_oos_target = R_oos[23:end];
yhat_real_full, _ = _har_predict(β_real, R_oos);
strategy_real = _run_strategy(yhat_real_full, R_oos_target);
println("  Real-trained  Sharpe=$(round(strategy_real.sharpe, digits=2))  MDD=$(round(strategy_real.mdd, digits=4))  Turn=$(round(strategy_real.turnover, digits=2))");

strategy_results = Dict{String,NamedTuple}();
for m in MODEL_ORDER
    s_is = archive[m].is;
    rs = vec(s_is[:, 1:min(n_paths_use, size(s_is, 2))]);
    Xs, ys = _build_har_features(rs);
    if size(Xs, 1) < 100; strategy_results[m] = (sharpe=NaN, ann_ret=NaN, ann_vol=NaN, mdd=NaN, turnover=NaN, mean_w=NaN); continue; end
    β_s = _har_fit(Xs, ys);
    yhat_s, _ = _har_predict(β_s, R_oos);
    strategy_results[m] = _run_strategy(yhat_s, R_oos_target);
    r = strategy_results[m];
    println("  $m  Sharpe=$(round(r.sharpe, digits=2))  AnnRet=$(round(r.ann_ret, digits=3))  AnnVol=$(round(r.ann_vol, digits=3))  MDD=$(round(r.mdd, digits=4))  Turn=$(round(r.turnover, digits=2))");
end

# Passive buy-and-hold
passive_daily = R_oos_target .* DT;
passive = (sharpe = mean(passive_daily)*252 / (std(passive_daily)*sqrt(252)+1e-12),
           ann_ret = mean(passive_daily)*252,
           ann_vol = std(passive_daily)*sqrt(252),
           mdd = let eq = cumsum(passive_daily); maximum(accumulate(max, eq) .- eq); end,
           turnover = 0.0, mean_w = 1.0);

open(joinpath(TRACK_A_DIR, "vol_target_strategy.txt"), "w") do io
    println(io, "="^120);
    println(io, "A5. Vol-target SPY strategy using HAR(1,5,22) vol forecast (target σ = $(TARGET_ANNUAL_VOL))");
    println(io, "="^120);
    println(io, "");
    println(io, "Policy  : w_t = $TARGET_ANNUAL_VOL / σ̂_t, clipped to [$(W_MIN), $(W_MAX)].");
    println(io, "σ̂_t    : sqrt of HAR next-day forecast of squared return (same HAR model as A4).");
    println(io, "Return  : daily PnL = w_t * R_oos[t] * (1/252). Annualized Sharpe = ann_ret / ann_vol.");
    println(io, "Window  : OoS ($(length(R_oos_target)) days) after dropping 22 days of HAR warm-up.");
    println(io, "");
    println(io, rpad("Training data",  16), " | ", rpad("Sharpe", 8), " | ", rpad("AnnRet", 8), " | ", rpad("AnnVol", 8), " | ", rpad("MDD", 8), " | ", rpad("Turnover", 10), " | ", rpad("Mean w", 8));
    println(io, "-"^120);
    println(io, rpad("Passive buy-hold", 16), " | ",
                rpad(round(passive.sharpe,   digits=2), 8), " | ",
                rpad(round(passive.ann_ret,  digits=3), 8), " | ",
                rpad(round(passive.ann_vol,  digits=3), 8), " | ",
                rpad(round(passive.mdd,      digits=4), 8), " | ",
                rpad(round(passive.turnover, digits=2), 10), " | ",
                rpad(round(passive.mean_w,   digits=2), 8));
    println(io, rpad("Real IS",          16), " | ",
                rpad(round(strategy_real.sharpe,   digits=2), 8), " | ",
                rpad(round(strategy_real.ann_ret,  digits=3), 8), " | ",
                rpad(round(strategy_real.ann_vol,  digits=3), 8), " | ",
                rpad(round(strategy_real.mdd,      digits=4), 8), " | ",
                rpad(round(strategy_real.turnover, digits=2), 10), " | ",
                rpad(round(strategy_real.mean_w,   digits=2), 8));
    for m in MODEL_ORDER
        r = strategy_results[m];
        println(io, rpad(m, 16), " | ",
                    rpad(round(r.sharpe,   digits=2), 8), " | ",
                    rpad(round(r.ann_ret,  digits=3), 8), " | ",
                    rpad(round(r.ann_vol,  digits=3), 8), " | ",
                    rpad(round(r.mdd,      digits=4), 8), " | ",
                    rpad(round(r.turnover, digits=2), 10), " | ",
                    rpad(round(r.mean_w,   digits=2), 8));
    end
    println(io, "="^120);
    println(io, "");
    println(io, "A good synthetic generator has Sharpe close to the real-trained benchmark with similar MDD and turnover.");
end

# --------------------------------------------------------------------------------------- #
# A8: Kupiec + Christoffersen LR tests on 1 % and 5 % VaR from each generator
# --------------------------------------------------------------------------------------- #
println("\n[A8] Kupiec + Christoffersen LR tests on 1 % and 5 % VaR...");

# For each model, compute the unconditional VaR at α from the pooled simulation archive
# (IS archive). Apply to R_oos breaches. Then repeat with OoS archive for a "walk-forward" cousin.
function _var_levels(sim_archive::AbstractMatrix, α::Float64)
    return quantile(vec(sim_archive), α);
end

# Also the historical VaR as a sanity check
hist_var_01 = quantile(R_is, 0.01);
hist_var_05 = quantile(R_is, 0.05);

var_lr_results = Dict{String,NamedTuple}();
for m in MODEL_ORDER
    # Use IS archive to estimate generator VaR, then count breaches on R_oos
    v01 = _var_levels(archive[m].is, 0.01);
    v05 = _var_levels(archive[m].is, 0.05);
    br_01 = R_oos .<= v01;
    br_05 = R_oos .<= v05;
    k01 = kupiec_lr(br_01, 0.01);
    k05 = kupiec_lr(br_05, 0.05);
    c01 = christoffersen_lr(br_01);
    c05 = christoffersen_lr(br_05);
    cc01 = christoffersen_cc(br_01, 0.01);
    cc05 = christoffersen_cc(br_05, 0.05);
    var_lr_results[m] = (
        v01=v01, v05=v05,
        k01=k01, k05=k05, c01=c01, c05=c05, cc01=cc01, cc05=cc05,
    );
    println("  $m VaR01=$(round(v01, digits=3)) breach=$(round(100*k01.breach_rate, digits=2))% (LR_uc=$(round(k01.LR, digits=2))) | VaR05=$(round(v05, digits=3)) breach=$(round(100*k05.breach_rate, digits=2))% (LR_uc=$(round(k05.LR, digits=2)))");
end

# Historical baseline (using R_is quantile directly)
hist_br_01 = R_oos .<= hist_var_01;
hist_br_05 = R_oos .<= hist_var_05;
hist_k01 = kupiec_lr(hist_br_01, 0.01);
hist_k05 = kupiec_lr(hist_br_05, 0.05);
hist_c01 = christoffersen_lr(hist_br_01);
hist_c05 = christoffersen_lr(hist_br_05);

open(joinpath(TRACK_A_DIR, "VaR_LR_tests.txt"), "w") do io
    println(io, "="^150);
    println(io, "A8. Kupiec (unconditional coverage) and Christoffersen (independence) LR tests on VaR");
    println(io, "="^150);
    println(io, "");
    println(io, "Setup   : VaR_α estimated as α-quantile of each generator's $(_PATH_TO_DATA == "" ? "" : "")in-sample simulated archive.");
    println(io, "Target  : α ∈ {0.01, 0.05}. Expected breach rate = α. LR_uc ~ χ²(1), crit 3.84.");
    println(io, "Christoffersen: LR_ind ~ χ²(1) for serial independence of breaches; LR_cc = LR_uc + LR_ind ~ χ²(2), crit 5.99.");
    println(io, "Data    : SPY OoS window ($n_oos daily observations).");
    println(io, "");
    println(io, rpad("Model",      12), " | ",
                rpad("VaR01",       7), " | ", rpad("br%01",   6), " | ", rpad("LR_uc01", 7), " | ", rpad("p_uc01", 6), " | ",
                rpad("LR_ind01",   8), " | ", rpad("LR_cc01", 7), " | ", rpad("VaR05",   7), " | ",
                rpad("br%05",       6), " | ", rpad("LR_uc05", 7), " | ", rpad("p_uc05", 6), " | ",
                rpad("LR_ind05",   8), " | ", rpad("LR_cc05", 7));
    println(io, "-"^150);
    # Historical row
    println(io, rpad("Historical",  12), " | ",
                rpad(round(hist_var_01,      digits=3), 7), " | ",
                rpad(round(100*hist_k01.breach_rate, digits=1), 6), " | ",
                rpad(round(hist_k01.LR,       digits=2), 7), " | ",
                rpad(round(hist_k01.pvalue,   digits=3), 6), " | ",
                rpad(round(hist_c01.LR,       digits=2), 8), " | ",
                rpad("-",                    7), " | ",
                rpad(round(hist_var_05,       digits=3), 7), " | ",
                rpad(round(100*hist_k05.breach_rate, digits=1), 6), " | ",
                rpad(round(hist_k05.LR,       digits=2), 7), " | ",
                rpad(round(hist_k05.pvalue,   digits=3), 6), " | ",
                rpad(round(hist_c05.LR,       digits=2), 8), " | ",
                rpad("-",                    7));
    for m in MODEL_ORDER
        r = var_lr_results[m];
        println(io, rpad(m, 12), " | ",
                    rpad(round(r.v01,        digits=3), 7), " | ",
                    rpad(round(100*r.k01.breach_rate, digits=1), 6), " | ",
                    rpad(round(r.k01.LR,      digits=2), 7), " | ",
                    rpad(round(r.k01.pvalue,  digits=3), 6), " | ",
                    rpad(round(r.c01.LR,      digits=2), 8), " | ",
                    rpad(round(r.cc01.LR,     digits=2), 7), " | ",
                    rpad(round(r.v05,         digits=3), 7), " | ",
                    rpad(round(100*r.k05.breach_rate, digits=1), 6), " | ",
                    rpad(round(r.k05.LR,      digits=2), 7), " | ",
                    rpad(round(r.k05.pvalue,  digits=3), 6), " | ",
                    rpad(round(r.c05.LR,      digits=2), 8), " | ",
                    rpad(round(r.cc05.LR,     digits=2), 7));
    end
    println(io, "="^150);
    println(io, "");
    println(io, "Reading: LR_uc tests whether breach rate equals α. LR_ind tests independence of consecutive breaches.");
    println(io, "A well-calibrated VaR passes both tests: LR_uc < 3.84 (5% crit) and LR_ind < 3.84.");
end

println("\n" * "="^72);
println("  Track A utility run complete.");
println("  Results: $TRACK_A_DIR");
println("="^72);
