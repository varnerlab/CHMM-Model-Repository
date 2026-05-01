# ========================================================================================= #
# run_sector_panel_n6.jl
#
# Reviewer Round 2 / Item P10 (peer-review.md R3#req-4, R2#3 follow-up).
#
# The body 30-ticker sector-balanced cross-ticker panel uses n = 3 per sector. The body
# ANOVA F(9, 20) = 0.44, p = 0.90, η² = 0.16 is severely underpowered to detect sector
# effects of moderate-to-large magnitude. Round-2 reviewers asked for an n = 6 per sector
# expansion (60 tickers total) to give the sector ANOVA adequate power.
#
# Strategy: fit the 30 *additional* tickers (3 new representatives per sector) under
# identical settings to the body panel and combine with the cached 30-ticker results to
# produce the full n = 6 panel. Re-run the one-way ANOVA on OoS KS by sector at the
# expanded sample size.
#
# Additional 30 tickers (3 large-cap representatives per sector, complementing the body
# panel; chosen for liquidity and continuous IS coverage):
#
#   Information Technology   : ADBE, CRM,  ORCL
#   Health Care              : ABBV, MRK,  PFE
#   Financials               : GS,   MS,   BLK
#   Consumer Discretionary   : NKE,  SBUX, TSLA
#   Communication Services   : GOOG, T,    CMCSA
#   Industrials              : UNP,  MMM,  LMT
#   Consumer Staples         : PEP,  MO,   COST
#   Energy                   : SLB,  EOG,  MPC
#   Utilities                : D,    AEP,  EXC
#   Materials                : SHW,  ECL,  NUE
#
# Output:
#   results/sector_panel_n6/sector_panel_n6.csv       (full 60-ticker rollup)
#   results/sector_panel_n6/sector_panel_n6.txt       (human-readable + ANOVA)
#   results/sector_panel_n6/sector_panel_n6_extra.csv (just the 30 new tickers)
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
using StatsBase
using Printf
using CSV
using DataFrames
using Distributions
using Distances

const SEED        = 20260420;
const K_MAIN      = 18;
const N_PATHS     = 1000;
const MAX_ITER    = 60;
const DT          = 1/252;
const RISK_FREE   = 0.0;
const L_LAGS      = 252;
const SHRINK_RATE = 20.0;
const ALPHA_KS    = 0.05;
const KS_FAIL_PP  = 60.0;

const OUT_DIR = joinpath(_ROOT, "results", "sector_panel_n6");
mkpath(OUT_DIR);

# Additional 30 tickers (3 per sector, complementing the body 30-ticker panel)
const SECTOR_PANEL_EXTRA = [
    ("Information Technology",   ["ADBE", "CRM",  "ORCL"]),
    ("Health Care",              ["ABBV", "MRK",  "PFE"]),
    ("Financials",               ["GS",   "MS",   "BLK"]),
    ("Consumer Discretionary",   ["NKE",  "SBUX", "TSLA"]),
    ("Communication Services",   ["GOOG", "T",    "CMCSA"]),
    ("Industrials",              ["UNP",  "MMM",  "LMT"]),
    ("Consumer Staples",         ["PEP",  "MO",   "COST"]),
    ("Energy",                   ["SLB",  "EOG",  "MPC"]),
    ("Utilities",                ["D",    "AEP",  "EXC"]),
    ("Materials",                ["SHW",  "ECL",  "NUE"]),
];

println("="^80)
println("  60-ticker sector-balanced panel (n=6 per sector)  (R3#req-4 / Item P10)")
println("  Penalised CHMM-t  K = $K_MAIN  λ = $SHRINK_RATE")
println("  Fitting 30 additional tickers and combining with cached body 30.")
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
all_tickers_full = keys(dataset) |> collect |> sort;
all_R_full = log_growth_matrix(dataset, all_tickers_full; Δt=DT, risk_free_rate=RISK_FREE);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
println("  $(length(all_tickers_full)) tickers with full IS history; T_IS = $(size(all_R_full, 1))")

# Verify all extra tickers are present
extra_flat = vcat([tks for (_, tks) in SECTOR_PANEL_EXTRA]...);
missing_t = [t for t in extra_flat if !(t in all_tickers_full)];
if !isempty(missing_t)
    error("Tickers missing from IS dataset: $missing_t")
end
println("  All 30 additional tickers present in dataset.")

# ----------------------------------------------------------------------------------------- #
# Helpers (mirror run_sector_panel.jl)
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
    ks_pass = 0; kurt_s = 0.0; acf_mae = 0.0;
    for i in 1:np
        s = sim[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= α
            ks_pass += 1;
        end
        μ_s = mean(s); σ_s = std(s);
        kurt_s += sum(((s .- μ_s) ./ σ_s).^4) / length(s) - 3.0;
        acf_s = autocor(abs.(s), 1:L_use);
        acf_mae += mean(abs.(acf_o .- acf_s));
    end
    return (
        ks_pct  = round(100*ks_pass/np, digits=1),
        kurt_o  = round(kurt_o, digits=2),
        kurt_s  = round(kurt_s/np, digits=2),
        acf_mae = round(acf_mae/np, digits=4),
    );
end

# ----------------------------------------------------------------------------------------- #
# Per-ticker fit + IS / OoS evaluation
# ----------------------------------------------------------------------------------------- #
function fit_and_eval_ticker(ticker::String)
    R_is = collect(all_R_full[:, findfirst(==(ticker), all_tickers_full)])
    R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, ticker; Δt=DT, risk_free_rate=RISK_FREE))
    Random.seed!(SEED + Int(hash(ticker) % 10000))
    chmm = build(MyStudentTHiddenMarkovModel,
        (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER,
         ν_shrink_rate=SHRINK_RATE))
    sd = _stationary(chmm, K_MAIN)
    sim_is  = _sim_paths(chmm, sd, length(R_is),  N_PATHS)
    sim_oos = _sim_paths(chmm, sd, length(R_oos), N_PATHS)
    is_e  = _eval(R_is,  sim_is)
    oos_e = _eval(R_oos, sim_oos)
    nus = [params(chmm.emission[k].ρ)[1] for k in 1:K_MAIN]
    return (
        ticker = ticker,
        ks_is_pct = is_e.ks_pct,
        ks_oos_pct = oos_e.ks_pct,
        kurt_obs = is_e.kurt_o,
        kurt_sim = is_e.kurt_s,
        kurt_resid = round(is_e.kurt_s - is_e.kurt_o, digits=2),
        acf_mae_abs = is_e.acf_mae,
        nu_median = round(median(nus), digits=2),
    )
end

# ----------------------------------------------------------------------------------------- #
# Run the 30 additional tickers
# ----------------------------------------------------------------------------------------- #
extra_results = Tuple{String,NamedTuple}[]   # (sector, row)

for (sector, tickers) in SECTOR_PANEL_EXTRA
    println("\n[sector] $sector")
    for tk in tickers
        @printf("  fit %-5s ...", tk)
        flush(stdout)
        r = fit_and_eval_ticker(tk)
        push!(extra_results, (sector, r))
        @printf(" IS KS=%5.1f%%  OoS KS=%5.1f%%  kurt_obs=%5.2f  kurt_sim=%6.2f  acf_mae=%.4f\n",
                r.ks_is_pct, r.ks_oos_pct, r.kurt_obs, r.kurt_sim, r.acf_mae_abs)
    end
end

# ----------------------------------------------------------------------------------------- #
# Combine with cached body 30-ticker rows
# ----------------------------------------------------------------------------------------- #
println("\n[combine] Reading cached body 30-ticker results from results/sector_panel/sector_panel_summary.csv ...")
body_csv = joinpath(_ROOT, "results", "sector_panel", "sector_panel_summary.csv")
body_df = DataFrame(CSV.File(body_csv))
println("  cached body rows: $(nrow(body_df))")

# Build full 60-row panel (body 30 + extra 30)
all_rows = NamedTuple[]
for r in eachrow(body_df)
    push!(all_rows, (
        sector=r.sector, ticker=r.ticker,
        ks_is_pct=r.ks_is_pct, ks_oos_pct=r.ks_oos_pct,
        kurt_obs=r.kurt_obs, kurt_sim=r.kurt_sim, kurt_resid=r.kurt_resid,
        acf_mae_abs=r.acf_mae_abs, nu_median=r.nu_median, src="body",
    ))
end
for (sector, r) in extra_results
    push!(all_rows, (
        sector=sector, ticker=r.ticker,
        ks_is_pct=r.ks_is_pct, ks_oos_pct=r.ks_oos_pct,
        kurt_obs=r.kurt_obs, kurt_sim=r.kurt_sim, kurt_resid=r.kurt_resid,
        acf_mae_abs=r.acf_mae_abs, nu_median=r.nu_median, src="P10",
    ))
end
@assert length(all_rows) == 60
println("  combined panel: $(length(all_rows)) tickers")

# ----------------------------------------------------------------------------------------- #
# One-way ANOVA on OoS KS by sector at n = 6
# ----------------------------------------------------------------------------------------- #
sectors = sort(unique([r.sector for r in all_rows]))
ks_by_sector = Dict(s => [r.ks_oos_pct for r in all_rows if r.sector == s] for s in sectors)
all_ks = [r.ks_oos_pct for r in all_rows]
grand_mean = mean(all_ks)
N = length(all_ks); k = length(sectors)
# Use sum-comprehensions to avoid Julia 1.12 soft-scope assignment ambiguity on
# accumulator variables inside a top-level for-loop.
ss_between = sum(length(ks_by_sector[s]) * (mean(ks_by_sector[s]) - grand_mean)^2 for s in sectors)
ss_within  = sum(sum((x - mean(ks_by_sector[s]))^2 for x in ks_by_sector[s]) for s in sectors)
F = (ss_between / (k - 1)) / (ss_within / (N - k))
p_F = 1 - cdf(FDist(k - 1, N - k), F)
η² = ss_between / (ss_between + ss_within)
println(@sprintf("\n[ANOVA] F(%d, %d) = %.3f   p = %.4f   η² = %.3f   (n = 6 per sector, n_total = %d)",
        k - 1, N - k, F, p_F, η², N))

# Per-sector summary
sector_summary = Tuple{String, Int, Float64, Float64, Float64, Float64, Int}[]
for s in sectors
    ks = ks_by_sector[s]
    push!(sector_summary, (s, length(ks), median(ks), mean(ks), std(ks), minimum(ks),
                           count(<(KS_FAIL_PP), ks)))
end

# ----------------------------------------------------------------------------------------- #
# Output
# ----------------------------------------------------------------------------------------- #
csv_full = joinpath(OUT_DIR, "sector_panel_n6.csv")
open(csv_full, "w") do io
    println(io, "sector,ticker,src,ks_is_pct,ks_oos_pct,kurt_obs,kurt_sim,kurt_resid,acf_mae_abs,nu_median")
    for r in all_rows
        @printf(io, "%s,%s,%s,%.1f,%.1f,%.2f,%.2f,%.2f,%.4f,%.2f\n",
                r.sector, r.ticker, r.src, r.ks_is_pct, r.ks_oos_pct,
                r.kurt_obs, r.kurt_sim, r.kurt_resid, r.acf_mae_abs, r.nu_median)
    end
end

csv_extra = joinpath(OUT_DIR, "sector_panel_n6_extra.csv")
open(csv_extra, "w") do io
    println(io, "sector,ticker,ks_is_pct,ks_oos_pct,kurt_obs,kurt_sim,kurt_resid,acf_mae_abs,nu_median")
    for (sector, r) in extra_results
        @printf(io, "%s,%s,%.1f,%.1f,%.2f,%.2f,%.2f,%.4f,%.2f\n",
                sector, r.ticker, r.ks_is_pct, r.ks_oos_pct, r.kurt_obs, r.kurt_sim,
                r.kurt_resid, r.acf_mae_abs, r.nu_median)
    end
end

txt_path = joinpath(OUT_DIR, "sector_panel_n6.txt")
open(txt_path, "w") do io
    println(io, "="^110)
    println(io, "60-ticker sector-balanced panel (n=6 per sector)  (Reviewer Round 2 / Item P10)")
    println(io, "="^110)
    println(io, "Setup: penalised CHMM-t at K=$K_MAIN, λ=$SHRINK_RATE, $N_PATHS paths, seed = $SEED.")
    println(io, "Body 30-ticker rows are read from results/sector_panel/sector_panel_summary.csv;")
    println(io, "the additional 30 tickers are fit by this runner.")
    println(io)
    println(io, "Per-sector summary at n = 6:")
    @printf(io, "  %-26s  %3s  %7s  %7s  %7s  %7s  %5s\n",
            "sector", "n", "median", "mean", "sd", "min", "fail")
    println(io, "  ", "-"^85)
    for (s, n, med, m, sd, mn, fail) in sector_summary
        @printf(io, "  %-26s  %3d  %7.2f  %7.2f  %7.2f  %7.2f  %5d\n",
                s, n, med, m, sd, mn, fail)
    end
    println(io)
    @printf(io, "Aggregate (n = %d):  median = %.2f%%   mean = %.2f%%   sd = %.2f%%   failures = %d / %d\n",
            length(all_ks), median(all_ks), mean(all_ks), std(all_ks),
            count(<(KS_FAIL_PP), all_ks), length(all_ks))
    println(io)
    println(io, "One-way ANOVA on OoS KS by sector:")
    @printf(io, "  F(%d, %d) = %.3f   p = %.4f   η² = %.3f\n", k-1, N-k, F, p_F, η²)
    println(io, "  Body n=3 result was F(9, 20) = 0.44, p = 0.90, η² = 0.16 (severely underpowered).")
    if p_F < 0.05
        println(io, "  At n=6 per sector, the test rejects the no-sector-effect null at the 5% level.")
    else
        println(io, "  At n=6 per sector, the test does not reject the no-sector-effect null at the 5% level.")
        println(io, "  The η² magnitude relative to the body's η²=0.16 quantifies whether the cross-")
        println(io, "  sector dispersion is consistent with sampling at the larger n; a substantially")
        println(io, "  smaller η² at n=6 would indicate the body's apparent sector concentration was")
        println(io, "  small-sample noise.")
    end
end

println("\n[done] $csv_full")
println("[done] $csv_extra")
println("[done] $txt_path")
