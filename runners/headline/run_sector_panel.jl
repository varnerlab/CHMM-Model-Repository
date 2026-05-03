# ========================================================================================= #
# run_sector_panel.jl
#
# T1.2 (REVISION_PLAN_V5_TO_ACCEPT.md): 30-ticker sector-balanced cross-ticker panel.
# Replaces the six-ticker spot-check (`run_cross_ticker_penalised.jl`) with a 30-ticker
# panel constructed as three large-cap representatives from each of ten GICS sectors,
# fit independently under the same penalised CHMM-t scaffold (K = 18, λ = 20). Reports
# aggregate IS / OoS KS pass-rate distribution, |G_t| ACF-MAE, kurtosis residual, and a
# per-sector rollup.
#
# The runner reuses the helper functions (`_stationary`, `_sim_paths`, `_eval`) and
# constants (SEED, DT, RISK_FREE, K_MAIN, SHRINK_RATE, N_PATHS, MAX_ITER) from
# `run_cross_ticker_penalised.jl` so the two cross-ticker panels are directly comparable.
#
# GICS sector partition (top three large-cap representatives per sector at IS-window
# median market cap; explicit list rather than a programmatic look-up to keep the panel
# reproducible without an external sector-classification dependency):
#
#   Information Technology   : AAPL, MSFT, NVDA
#   Health Care              : JNJ,  UNH,  LLY
#   Financials               : JPM,  BAC,  WFC
#   Consumer Discretionary   : AMZN, HD,   MCD
#   Communication Services   : NFLX, VZ,   DIS
#   Industrials              : CAT,  BA,   HON
#   Consumer Staples         : PG,   KO,   WMT
#   Energy                   : XOM,  CVX,  COP
#   Utilities                : NEE,  DUK,  SO
#   Materials                : FCX,  NEM,  APD
#
# Output:
#   results/sector_panel/sector_panel_summary.csv
#   results/sector_panel/sector_panel_summary.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random
using Statistics
using HypothesisTests
using StatsBase
using Printf

const SEED        = 20260420;
const K_MAIN      = 18;
const N_PATHS     = 1000;
const MAX_ITER    = 60;
const DT          = 1/252;
const RISK_FREE   = 0.0;
const L_LAGS      = 252;
const SHRINK_RATE = 20.0;
const ALPHA_KS    = 0.05;
const KS_FAIL_PP  = 60.0;            # OoS KS threshold below which a ticker counts as a "failure"

const OUT_DIR = joinpath(_ROOT, "results", "sector_panel");
mkpath(OUT_DIR);

# Sector partition: 10 GICS sectors × 3 large-cap representatives = 30 tickers
const SECTOR_PANEL = [
    ("Information Technology",   ["AAPL", "MSFT", "NVDA"]),
    ("Health Care",              ["JNJ",  "UNH",  "LLY"]),
    ("Financials",               ["JPM",  "BAC",  "WFC"]),
    ("Consumer Discretionary",   ["AMZN", "HD",   "MCD"]),
    ("Communication Services",   ["NFLX", "VZ",   "DIS"]),
    ("Industrials",              ["CAT",  "BA",   "HON"]),
    ("Consumer Staples",         ["PG",   "KO",   "WMT"]),
    ("Energy",                   ["XOM",  "CVX",  "COP"]),
    ("Utilities",                ["NEE",  "DUK",  "SO" ]),
    ("Materials",                ["FCX",  "NEM",  "APD"]),
];

println("="^80)
println("  Sector-balanced 30-ticker panel  [peer-review V5 T1.2]")
println("  Penalised CHMM-t  K = $K_MAIN  λ = $SHRINK_RATE")
println("  Seed: $SEED  Paths: $N_PATHS")
println("="^80)

# ----------------------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading IS / OoS portfolios...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
println("  $(length(all_tickers)) tickers with full IS history; T_IS = $(size(all_R, 1))")

# ----------------------------------------------------------------------------------------- #
# Helpers (mirror those in run_cross_ticker_penalised.jl for cross-comparability)
# ----------------------------------------------------------------------------------------- #
function _stationary(model, K::Int)
    T = zeros(K, K);
    for i in 1:K; T[i, :] = probs(model.transition[i]); end
    π = (T^1000)[1, :];
    return Categorical(π);
end

function _sim_paths(model, sd, n::Int, np::Int)
    sim = Matrix{Float64}(undef, n, np);
    for p in 1:np
        s0 = rand(sd);
        st = model(s0, n);
        for j in 1:n; sim[j, p] = rand(model.emission[st[j]]); end
    end
    return sim;
end

function _eval(R_obs::AbstractVector, sim::AbstractMatrix; L::Int=L_LAGS, α::Float64=ALPHA_KS)
    np = size(sim, 2);
    n_o = length(R_obs);
    μ_o = mean(R_obs); σ_o = std(R_obs);
    kurt_o = sum(((R_obs .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L, n_o - 1);
    acf_o = autocor(abs.(R_obs), 1:L_use);

    ks_pass = 0;
    kurt_s = 0.0; acf_mae_s = 0.0;
    for i in 1:np
        s = sim[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= α
            ks_pass += 1;
        end
        μ_s = mean(s); σ_s = std(s);
        kurt_s += sum(((s .- μ_s) ./ σ_s).^4) / length(s) - 3.0;
        acf_s = autocor(abs.(s), 1:L_use);
        acf_mae_s += mean(abs.(acf_o .- acf_s));
    end
    return (
        ks_pct   = round(100*ks_pass/np, digits=1),
        sim_kurt = round(kurt_s/np, digits=2),
        obs_kurt = round(kurt_o, digits=2),
        acf_mae  = round(acf_mae_s/np, digits=4),
    );
end

# ----------------------------------------------------------------------------------------- #
# Per-ticker fit + evaluate
# ----------------------------------------------------------------------------------------- #
results_rows = NamedTuple[];
let
    n_fitted = 0
    skipped = String[]

for (sector, tickers) in SECTOR_PANEL
    println("\n[sector] $sector")
    for tk in tickers
        t_idx = findfirst(==(tk), all_tickers);
        if isnothing(t_idx); push!(skipped, tk); println("  $tk: not in dataset, skipping."); continue; end
        if !haskey(oos_dataset, tk); push!(skipped, tk); println("  $tk: OoS missing, skipping."); continue; end

        R_t_is  = Vector{Float64}(all_R[:, t_idx]);
        R_t_oos = log_growth_matrix(oos_dataset, tk; Δt=DT, risk_free_rate=RISK_FREE);
        n_is = length(R_t_is); n_oos = length(R_t_oos);

        Random.seed!(SEED + 13 * n_fitted + hash(tk) % 1000);
        chmm_t_pen = build(MyStudentTHiddenMarkovModel,
            (observations=R_t_is, number_of_states=K_MAIN, max_iter=MAX_ITER,
             ν_shrink_rate=SHRINK_RATE));
        sd_t = _stationary(chmm_t_pen, K_MAIN);

        nus = [chmm_t_pen.emission[k].ρ.ν for k in 1:K_MAIN];

        Random.seed!(SEED + 11 + 13 * n_fitted);
        sim_is  = _sim_paths(chmm_t_pen, sd_t, n_is,  N_PATHS);
        sim_oos = _sim_paths(chmm_t_pen, sd_t, n_oos, N_PATHS);

        is_panel  = _eval(R_t_is,  sim_is);
        oos_panel = _eval(R_t_oos, sim_oos);

        @printf("  %-5s  IS KS = %5.1f%%  OoS KS = %5.1f%%  kurt obs/sim = %6.2f / %6.2f  |G| ACF-MAE = %.4f\n",
            tk, is_panel.ks_pct, oos_panel.ks_pct, is_panel.obs_kurt, is_panel.sim_kurt, is_panel.acf_mae)

        push!(results_rows, (
            sector       = sector,
            ticker       = tk,
            ks_is        = is_panel.ks_pct,
            ks_oos       = oos_panel.ks_pct,
            kurt_obs     = is_panel.obs_kurt,
            kurt_sim     = is_panel.sim_kurt,
            kurt_resid   = round(is_panel.sim_kurt - is_panel.obs_kurt, digits=2),
            acf_mae      = is_panel.acf_mae,
            nu_median    = round(median(nus), digits=2),
        ));
        n_fitted += 1;
    end
end

global g_skipped = skipped
end  # let

# ----------------------------------------------------------------------------------------- #
# Aggregate statistics
# ----------------------------------------------------------------------------------------- #
function _stats(xs::Vector{<:Real})
    n = length(xs);
    if n == 0; return (median=NaN, q1=NaN, q3=NaN, mean=NaN, std=NaN); end
    s = sort(xs);
    return (
        median = median(s),
        q1     = s[max(1, Int(round(0.25 * (n + 1))))],
        q3     = s[min(n, Int(round(0.75 * (n + 1))))],
        mean   = mean(s),
        std    = std(s),
    );
end

ks_is_v   = [r.ks_is for r in results_rows];
ks_oos_v  = [r.ks_oos for r in results_rows];
acf_v     = [r.acf_mae for r in results_rows];
kurt_r_v  = [r.kurt_resid for r in results_rows];

agg_ks_is  = _stats(ks_is_v);
agg_ks_oos = _stats(ks_oos_v);
agg_acf    = _stats(acf_v);
agg_kurt_r = _stats(kurt_r_v);
n_fail     = sum(ks_oos_v .< KS_FAIL_PP);
fail_rate  = round(100 * n_fail / length(ks_oos_v), digits=1);

# Per-sector rollup
sectors = unique([r.sector for r in results_rows]);
sector_rollup = Dict{String,Any}();
for s in sectors
    rows = [r for r in results_rows if r.sector == s];
    sector_rollup[s] = (
        n         = length(rows),
        ks_is_med = round(median([r.ks_is for r in rows]), digits=1),
        ks_oos_med= round(median([r.ks_oos for r in rows]), digits=1),
        acf_med   = round(median([r.acf_mae for r in rows]), digits=4),
        worst_oos = sort(rows, by = r -> r.ks_oos)[1],
    );
end

# ----------------------------------------------------------------------------------------- #
# Output
# ----------------------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "sector_panel_summary.csv");
open(csv_path, "w") do io
    write(io, "sector,ticker,ks_is_pct,ks_oos_pct,kurt_obs,kurt_sim,kurt_resid,acf_mae_abs,nu_median\n");
    for r in results_rows
        write(io, @sprintf("%s,%s,%.1f,%.1f,%.2f,%.2f,%.2f,%.4f,%.2f\n",
            r.sector, r.ticker, r.ks_is, r.ks_oos, r.kurt_obs, r.kurt_sim,
            r.kurt_resid, r.acf_mae, r.nu_median));
    end
end

txt_path = joinpath(OUT_DIR, "sector_panel_summary.txt");
open(txt_path, "w") do io
    println(io, "="^96);
    println(io, "Sector-balanced 30-ticker panel (penalised CHMM-t, K = $K_MAIN, λ = $SHRINK_RATE)");
    println(io, "[peer-review V5 T1.2]");
    println(io, "="^96);
    println(io, "Setup: per-ticker independent fit, paths = $N_PATHS, seed = $SEED, T_IS = $(size(all_R,1)).");
    println(io, "Universe: 10 GICS sectors × 3 large-cap representatives.");
    if !isempty(g_skipped)
        println(io, "Skipped (data not available): ", join(g_skipped, ", "));
    end
    println(io);
    println(io, "Per-ticker rows ranked by sector then ticker:");
    println(io);
    println(io, "Sector                   | Ticker | IS KS% | OoS KS% | Kurt obs | Kurt sim | Kurt resid | |G| ACF-MAE | ν median");
    println(io, "-"^120);
    for r in results_rows
        println(io,
            rpad(r.sector, 24), " | ",
            rpad(r.ticker, 6), " | ",
            lpad(r.ks_is, 6), " | ",
            lpad(r.ks_oos, 7), " | ",
            lpad(r.kurt_obs, 8), " | ",
            lpad(r.kurt_sim, 8), " | ",
            lpad(r.kurt_resid, 10), " | ",
            lpad(r.acf_mae, 11), " | ",
            lpad(r.nu_median, 8));
    end
    println(io);
    println(io, "Aggregate distribution across $(length(results_rows)) tickers:");
    println(io, "-"^96);
    @printf(io, "  IS KS%%        median = %5.1f   [Q1, Q3] = [%5.1f, %5.1f]   mean ± sd = %5.1f ± %4.1f\n",
        agg_ks_is.median, agg_ks_is.q1, agg_ks_is.q3, agg_ks_is.mean, agg_ks_is.std);
    @printf(io, "  OoS KS%%       median = %5.1f   [Q1, Q3] = [%5.1f, %5.1f]   mean ± sd = %5.1f ± %4.1f\n",
        agg_ks_oos.median, agg_ks_oos.q1, agg_ks_oos.q3, agg_ks_oos.mean, agg_ks_oos.std);
    @printf(io, "  |G| ACF-MAE   median = %.4f  [Q1, Q3] = [%.4f, %.4f]\n",
        agg_acf.median, agg_acf.q1, agg_acf.q3);
    @printf(io, "  Kurt residual median = %5.2f   [Q1, Q3] = [%5.2f, %5.2f]\n",
        agg_kurt_r.median, agg_kurt_r.q1, agg_kurt_r.q3);
    @printf(io, "  Tickers with OoS KS < %.0f%%: %d / %d  (%.1f%%)\n",
        KS_FAIL_PP, n_fail, length(ks_oos_v), fail_rate);
    println(io);
    println(io, "Per-sector rollup (median across the 3 representatives + worst OoS ticker):");
    println(io, "-"^96);
    for (sector, _) in SECTOR_PANEL
        if !haskey(sector_rollup, sector); continue; end
        s = sector_rollup[sector];
        @printf(io, "  %-24s  n=%d  IS KS med = %5.1f%%   OoS KS med = %5.1f%%   |G| ACF-MAE med = %.4f   worst OoS = %s (%.1f%%)\n",
            sector, s.n, s.ks_is_med, s.ks_oos_med, s.acf_med, s.worst_oos.ticker, s.worst_oos.ks_oos);
    end
    println(io);
    println(io, "Top-three by OoS KS failure (lowest OoS KS pass rate):");
    println(io, "-"^96);
    sorted_by_oos = sort(results_rows, by = r -> r.ks_oos);
    for i in 1:min(3, length(sorted_by_oos))
        r = sorted_by_oos[i];
        @printf(io, "  %-5s (%-24s)  OoS KS = %5.1f%%   IS KS = %5.1f%%   kurt resid = %5.2f\n",
            r.ticker, r.sector, r.ks_oos, r.ks_is, r.kurt_resid);
    end
end

println("\n[done] Output:")
println("  $csv_path")
println("  $txt_path")
