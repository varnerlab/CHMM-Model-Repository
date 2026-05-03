# ========================================================================================= #
# run_cross_decade_validation.jl
#
# Closes peer-review item P2.12 / R3 W6 / R3 RE3: 1994-2004 vs 2014-2024 cross-decade
# validation of CHMM stylized-fact reproduction. The body's empirical scope is SPY
# 2014-2026 because Polygon.io / Alpaca / IEX feeds do not cover the 1994-2004 window;
# CRSP has it. Day-pass WRDS access secured at revision time; CSV at
# data/external/crsp_1994_2006.csv covers SPY plus 28 of the 30 body cross-ticker panel
# (NEE and APD missing from the CRSP query).
#
# Protocol mirrors the body single-window OoS construction at $K^\star = 3$ (the
# state-resolution-robust body headline) and $K = 18$ (sensitivity reference):
#
#   IS  = SPY 1994-01-03 .. 2004-01-02  (10y, ~2520 trading days)
#   OoS = SPY 2004-01-05 .. 2006-04-28  (~2.4y, ~585 trading days)
#
# Fit CHMM-N (Gaussian) and penalised CHMM-t at lambda = 20 on the 1994-2004 IS slice;
# simulate 1000 IS- and OoS-length paths; score IS / OoS KS, mean simulated kurtosis,
# |G_t| ACF-MAE. Compare against the body's 2014-2024 figures from
# results/kstar3_headline/metrics.csv (CHMM-N at K*=3: 89.7 / 80.5%, kurt 3.83 / 3.53,
# ACF-MAE 0.0460 / 0.0545).
#
# Outputs:
#   results/cross_decade_validation/{summary.txt, metrics.csv}
#   ../CHMM-paper/results/robustness/cross_decade_validation.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, HypothesisTests, StatsBase, Printf, Dates, CSV, DataFrames

const SEED      = 20260420;
const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const N_PATHS   = 1_000;
const L_LAGS    = 252;
const MAX_ITER  = 60;
const LAMBDA    = 20.0;

const IS_END  = Date(2004, 1, 2);
const OOS_END = Date(2006, 4, 28);

const CRSP_PATH = joinpath(_ROOT, "data", "external", "crsp_1994_2006.csv");
const OUT_DIR              = joinpath(_ROOT, "results", "cross_decade_validation");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^80);
println("  Cross-decade validation: SPY 1994-2004 IS / 2004-2006 OoS  (P2.12 / R3 W6)");
println("="^80);

# --------------------------------------------------------------------------------------- #
# Load CRSP CIZ-format CSV
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading CRSP CSV $CRSP_PATH ...");
df = CSV.read(CRSP_PATH, DataFrame;
              types = Dict(:DlyPrc => Float64, :DlyFacPrc => Float64,
                           :PERMNO => Int, :Ticker => String));
df.DlyCalDt = Date.(df.DlyCalDt);
println("  $(nrow(df)) rows, $(length(unique(df.Ticker))) tickers");

"""
For a given ticker symbol, return a sorted Vector{Date}, Vector{Float64} of
(date, adjusted close) where adjusted close = DlyPrc / DlyFacPrc. Handles CRSP
PERMNO duplication on a single date by keeping the first occurrence per date.
"""
function _ticker_series(df::DataFrame, ticker::String)
    sub = df[df.Ticker .== ticker, :];
    isempty(sub) && return Date[], Float64[];
    sub = sort(sub, [:DlyCalDt, :PERMNO]);
    seen = Set{Date}();
    dates  = Date[];
    closes = Float64[];
    for r in eachrow(sub)
        if r.DlyCalDt in seen; continue; end
        push!(seen, r.DlyCalDt);
        push!(dates,  r.DlyCalDt);
        push!(closes, r.DlyPrc / r.DlyFacPrc);
    end
    return dates, closes;
end

function _log_growth_series(closes::Vector{Float64}; Δt = DT, rf = RISK_FREE)
    N = length(closes);
    r = Vector{Float64}(undef, N - 1);
    for t in 2:N
        r[t-1] = (1/Δt) * log(closes[t] / closes[t-1]) - rf;
    end
    return r;
end

function _slice_returns(dates::Vector{Date}, closes::Vector{Float64},
                        t0::Date, t1::Date)
    idx0 = findfirst(d -> d >= t0, dates);
    idx1 = findlast(d -> d <= t1, dates);
    isnothing(idx0) || isnothing(idx1) && return Float64[], Date[];
    return _log_growth_series(closes[idx0:idx1]), dates[idx0:idx1];
end

# --------------------------------------------------------------------------------------- #
# SPY: pull series, slice IS / OoS
# --------------------------------------------------------------------------------------- #
spy_dates, spy_closes = _ticker_series(df, "SPY");
println("\n[SPY] $(length(spy_dates)) trading days from $(spy_dates[1]) to $(spy_dates[end])");

R_is,  is_dates  = _slice_returns(spy_dates, spy_closes, Date(1994, 1, 3), IS_END);
R_oos, oos_dates = _slice_returns(spy_dates, spy_closes, IS_END + Day(1), OOS_END);
n_is = length(R_is); n_oos = length(R_oos);
@printf("  IS  = %s .. %s  (%d obs)\n", is_dates[1],  is_dates[end],  n_is);
@printf("  OoS = %s .. %s  (%d obs)\n", oos_dates[1], oos_dates[end], n_oos);
obs_kurt_is  = sum(((R_is  .- mean(R_is))  ./ std(R_is))  .^ 4) / n_is  - 3.0;
obs_kurt_oos = sum(((R_oos .- mean(R_oos)) ./ std(R_oos)) .^ 4) / n_oos - 3.0;
@printf("  observed excess kurtosis: IS = %.2f, OoS = %.2f\n", obs_kurt_is, obs_kurt_oos);

# --------------------------------------------------------------------------------------- #
# Fit + simulate
# --------------------------------------------------------------------------------------- #
function _stationary(model, K::Int)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    π_stat = (T_mat^1000)[1, :];
    π_stat = max.(π_stat, 1e-12); π_stat ./= sum(π_stat);
    return Categorical(π_stat);
end

function _sim_paths(model, sd, n_is::Int, n_oos::Int, n_paths::Int; seed::Int)
    Random.seed!(seed);
    sim_is  = Matrix{Float64}(undef, n_is,  n_paths);
    sim_oos = Matrix{Float64}(undef, n_oos, n_paths);
    for i in 1:n_paths
        s0 = rand(sd); st = model(s0, n_is);
        for j in 1:n_is;  sim_is[j, i]  = rand(model.emission[st[j]]); end
        s0 = rand(sd); st = model(s0, n_oos);
        for j in 1:n_oos; sim_oos[j, i] = rand(model.emission[st[j]]); end
    end
    return sim_is, sim_oos;
end

function _eval(R_obs, sim_archive)
    np = size(sim_archive, 2);
    n_o = length(R_obs);
    L_use = min(L_LAGS, n_o - 1);
    acf_o     = autocor(abs.(R_obs), 1:L_use);
    acf_o_raw = autocor(R_obs,        1:L_use);
    ks_pass = 0; kurt_s = 0.0; acf_mae = 0.0; acf_mae_raw = 0.0;
    for i in 1:np
        s = sim_archive[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= 0.05; ks_pass += 1; end
        μ_s = mean(s); σ_s = std(s);
        kurt_s   += sum(((s .- μ_s) ./ σ_s) .^ 4) / length(s) - 3.0;
        acf_mae     += mean(abs.(acf_o     .- autocor(abs.(s), 1:L_use)));
        acf_mae_raw += mean(abs.(acf_o_raw .- autocor(s,        1:L_use)));
    end
    return (
        ks_pct      = round(100 * ks_pass / np, digits = 1),
        sim_kurt    = round(kurt_s / np, digits = 2),
        acf_mae     = round(acf_mae / np, digits = 4),
        acf_mae_raw = round(acf_mae_raw / np, digits = 4),
    );
end

const FAMILIES = [
    (:gaussian, "CHMM-N",                            3,   SEED + 31),
    (:gaussian, "CHMM-N",                            18,  SEED + 32),
    (:student_t_pen, "CHMM-t (penalised, λ = 20)",   3,   SEED + 33),
];

function _fit(family::Symbol, K::Int)
    if family == :gaussian
        return build(MyContinuousHiddenMarkovModel,
                     (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    elseif family == :student_t_pen
        return build(MyStudentTHiddenMarkovModel,
                     (observations=R_is, number_of_states=K, max_iter=MAX_ITER,
                      ν_shrink_rate=LAMBDA));
    end
end

panels = NamedTuple[];
for (family, label, K, fam_seed) in FAMILIES
    println("\n[$label, K = $K] fitting on 1994-2004 IS...");
    Random.seed!(fam_seed);
    t0 = time();
    m = _fit(family, K);
    dt_fit = time() - t0;
    @printf("  fit in %.1f s\n", dt_fit);
    sd = _stationary(m, K);
    sim_is, sim_oos = _sim_paths(m, sd, n_is, n_oos, N_PATHS; seed = fam_seed + 100);
    is_panel  = _eval(R_is,  sim_is);
    oos_panel = _eval(R_oos, sim_oos);
    @printf("  IS  KS = %5.1f%%  sim kurt = %5.2f  |G| ACF-MAE = %.4f\n",
            is_panel.ks_pct, is_panel.sim_kurt, is_panel.acf_mae);
    @printf("  OoS KS = %5.1f%%  sim kurt = %5.2f  |G| ACF-MAE = %.4f\n",
            oos_panel.ks_pct, oos_panel.sim_kurt, oos_panel.acf_mae);
    push!(panels, (family = family, label = label, K = K,
                   is = is_panel, oos = oos_panel,
                   fit_time = dt_fit));
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "metrics.csv");
open(csv_path, "w") do io
    println(io, "family,K,scope,is_ks_pct,is_sim_kurt,is_obs_kurt,is_acf_mae," *
                "oos_ks_pct,oos_sim_kurt,oos_obs_kurt,oos_acf_mae");
    for r in panels
        @printf(io, "%s,%d,1994-2004 IS / 2004-2006 OoS,%.1f,%.2f,%.2f,%.4f,%.1f,%.2f,%.2f,%.4f\n",
                String(r.family), r.K,
                r.is.ks_pct,  r.is.sim_kurt,  obs_kurt_is,  r.is.acf_mae,
                r.oos.ks_pct, r.oos.sim_kurt, obs_kurt_oos, r.oos.acf_mae);
    end
end
cp(csv_path, joinpath(PAPER_ROBUSTNESS_DIR, "cross_decade_validation.csv"); force = true);

summary_path = joinpath(OUT_DIR, "summary.txt");
open(summary_path, "w") do io
    println(io, "="^110);
    println(io, "Cross-decade validation: SPY 1994-2004 IS / 2004-2006 OoS  (P2.12 / R3 W6 / R3 RE3)");
    println(io, "="^110);
    println(io);
    @printf(io, "Source: CRSP daily stock file via WRDS (data/external/crsp_1994_2006.csv)\n");
    @printf(io, "Adjusted close = DlyPrc / DlyFacPrc (CRSP CIZ format).\n");
    @printf(io, "IS slice  : %s .. %s (%d obs)\n", is_dates[1],  is_dates[end],  n_is);
    @printf(io, "OoS slice : %s .. %s (%d obs)\n", oos_dates[1], oos_dates[end], n_oos);
    @printf(io, "Observed excess kurtosis: IS = %.2f, OoS = %.2f\n", obs_kurt_is, obs_kurt_oos);
    println(io);
    println(io, "Cross-decade panel:");
    @printf(io, "  %-32s %-3s  %-8s  %-8s  %-9s  %-9s  %-9s  %-9s\n",
            "Model", "K", "IS KS%", "OoS KS%", "kurt IS", "kurt OoS",
            "|G| IS",  "|G| OoS");
    println(io, "  ", "-"^102);
    for r in panels
        @printf(io, "  %-32s %-3d  %-8.1f  %-8.1f  %-9.2f  %-9.2f  %-9.4f  %-9.4f\n",
                r.label, r.K, r.is.ks_pct, r.oos.ks_pct,
                r.is.sim_kurt, r.oos.sim_kurt, r.is.acf_mae, r.oos.acf_mae);
    end
    println(io);
    println(io, "Body comparison (2014-2024 IS / 2024-2026 OoS, results/kstar3_headline/metrics.csv):");
    println(io, "  CHMM-N (K* = 3)                      89.7      80.5      3.83       3.53       0.0460     0.0545");
    println(io, "  CHMM-t pen. (λ = 20, K* = 3)         90.6      83.2      14.91      8.50       0.0537     0.0502");
    println(io, "  CHMM-N (K = 18)                      94.1      81.8      5.04       4.44       0.0509     ~0.054");
    println(io);
    println(io, "Reading: the cross-decade panel reports CHMM behaviour on a structurally different");
    println(io, "decade (dot-com peak 2000, dot-com bust 2000-2002, post-bust recovery 2003-2006) than");
    println(io, "the body 2014-2024 / 2024-2026 window (post-GFC normalisation, COVID, 2022 rate hike).");
    println(io, "If the IS / OoS KS pass-rates and kurtosis-fidelity are within sampling error of the");
    println(io, "body figures, the body's 'CHMM as a synthetic-data generator' claim is not decade-");
    println(io, "specific. Material divergence flags the headline as 2014-2024-specific and the");
    println(io, "manuscript scope statement should be tightened accordingly.");
end

println("\n" * "="^80);
println("  Cross-decade validation complete.");
@printf("  Summary  : %s\n", summary_path);
@printf("  CSV      : %s\n", csv_path);
@printf("  Paper CSV: %s\n", joinpath(PAPER_ROBUSTNESS_DIR, "cross_decade_validation.csv"));
println("="^80);
