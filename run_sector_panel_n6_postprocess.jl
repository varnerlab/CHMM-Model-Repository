# ========================================================================================= #
# run_sector_panel_n6_postprocess.jl
#
# Post-processor for run_sector_panel_n6.jl. The main runner completed all 30 fits but hit
# a Julia 1.12 soft-scope bug (`ss_between`/`ss_within` ambiguity in the ANOVA loop) before
# writing CSVs. Rather than re-fit 30 K=18 models (30-45 min), this post-processor uses the
# stdout-captured per-ticker numbers and re-runs the aggregation + ANOVA cleanly.
#
# Output:
#   results/sector_panel_n6/sector_panel_n6.csv
#   results/sector_panel_n6/sector_panel_n6_extra.csv
#   results/sector_panel_n6/sector_panel_n6.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
using Statistics
using Distributions
using Printf
using CSV
using DataFrames

const _ROOT = @__DIR__
const OUT_DIR = joinpath(_ROOT, "results", "sector_panel_n6")
mkpath(OUT_DIR)
const KS_FAIL_PP = 60.0

# Captured from run_sector_panel_n6.jl stdout (30 additional tickers fit at K=18, λ=20):
# (sector, ticker, ks_is_pct, ks_oos_pct, kurt_obs, kurt_sim, acf_mae)
# nu_median = 50.00 (upper bracket) for all 30, consistent with body 30 under λ=20 shrinkage.
const EXTRA = [
    ("Information Technology", "ADBE",  99.5, 56.6,   6.74,  5.12, 0.0420),
    ("Information Technology", "CRM",   99.1, 92.6,   8.34,  6.62, 0.0383),
    ("Information Technology", "ORCL",  99.5,  0.0,  12.32, 14.05, 0.0320),
    ("Health Care",            "ABBV",  99.3, 99.2,  11.02, 16.80, 0.0286),
    ("Health Care",            "MRK",   99.5, 67.4,   5.78,  4.66, 0.0245),
    ("Health Care",            "PFE",   98.3, 54.6,   5.46,  5.58, 0.0436),
    ("Financials",             "GS",    99.6, 68.7,   5.05,  6.85, 0.0379),
    ("Financials",             "MS",    99.5, 68.0,   4.84,  3.60, 0.0358),
    ("Financials",             "BLK",   98.2, 88.8,   7.24,  6.20, 0.0448),
    ("Consumer Discretionary", "NKE",   99.6, 39.7,   8.86, 20.45, 0.0358),
    ("Consumer Discretionary", "SBUX",  99.7, 48.5,   9.67, 16.20, 0.0426),
    ("Consumer Discretionary", "TSLA",  99.6, 85.7,   3.46,  2.74, 0.0518),
    ("Communication Services", "GOOG",  99.3, 32.3, 514.73,  8.75, 0.0213),
    ("Communication Services", "T",     98.9, 21.4,  33.10, 19.30, 0.0325),
    ("Communication Services", "CMCSA", 99.4, 81.5,   4.50, 11.01, 0.0348),
    ("Industrials",            "UNP",   98.8, 86.6, 550.53, 30.03, 0.0192),
    ("Industrials",            "MMM",   98.9, 44.8,   8.41, 13.38, 0.0595),
    ("Industrials",            "LMT",   99.5, 90.1,  11.60, 34.73, 0.0356),
    ("Consumer Staples",       "PEP",   99.6, 13.7,   8.23,  9.99, 0.0350),
    ("Consumer Staples",       "MO",    99.4, 92.4,   8.30,  6.54, 0.0443),
    ("Consumer Staples",       "COST",  99.5, 91.6,   7.91,  6.92, 0.0302),
    ("Energy",                 "SLB",   98.7, 95.3,  30.00,  8.76, 0.0847),
    ("Energy",                 "EOG",   99.2, 11.3, 180.14, 13.89, 0.0338),
    ("Energy",                 "MPC",   98.9, 86.9,  12.93,  8.77, 0.0570),
    ("Utilities",              "D",     99.4, 54.5,  11.32,  9.42, 0.0393),
    ("Utilities",              "AEP",   99.5, 97.4,  10.97,  8.41, 0.0414),
    ("Utilities",              "EXC",   99.9, 95.8,  86.70, 15.63, 0.0326),
    ("Materials",              "SHW",   98.6, 65.6,  13.98, 28.68, 0.0389),
    ("Materials",              "ECL",   97.8, 87.9,  12.33, 33.26, 0.0593),
    ("Materials",              "NUE",   99.5, 95.6,   3.55,  3.74, 0.0670),
]
@assert length(EXTRA) == 30

# Read cached body 30-ticker rows
body_csv = joinpath(_ROOT, "results", "sector_panel", "sector_panel_summary.csv")
body_df = DataFrame(CSV.File(body_csv))
@assert nrow(body_df) == 30

# Build full 60-row panel
all_rows = NamedTuple[]
for r in eachrow(body_df)
    push!(all_rows, (
        sector=r.sector, ticker=r.ticker, src="body",
        ks_is_pct=r.ks_is_pct, ks_oos_pct=r.ks_oos_pct,
        kurt_obs=r.kurt_obs, kurt_sim=r.kurt_sim,
        kurt_resid=r.kurt_resid,
        acf_mae_abs=r.acf_mae_abs, nu_median=r.nu_median,
    ))
end
for (sector, ticker, ks_is, ks_oos, kurt_o, kurt_s, acf) in EXTRA
    push!(all_rows, (
        sector=sector, ticker=ticker, src="P10",
        ks_is_pct=ks_is, ks_oos_pct=ks_oos,
        kurt_obs=kurt_o, kurt_sim=kurt_s,
        kurt_resid=round(kurt_s - kurt_o, digits=2),
        acf_mae_abs=acf, nu_median=50.00,
    ))
end
@assert length(all_rows) == 60

# One-way ANOVA on OoS KS by sector (using comprehensions; avoids Julia 1.12 soft-scope)
sectors = sort(unique([r.sector for r in all_rows]))
ks_by_sector = Dict(s => [r.ks_oos_pct for r in all_rows if r.sector == s] for s in sectors)
all_ks = [r.ks_oos_pct for r in all_rows]
N, k = length(all_ks), length(sectors)
grand = mean(all_ks)
ss_between = sum(length(ks_by_sector[s]) * (mean(ks_by_sector[s]) - grand)^2 for s in sectors)
ss_within  = sum(sum((x - mean(ks_by_sector[s]))^2 for x in ks_by_sector[s]) for s in sectors)
F = (ss_between / (k - 1)) / (ss_within / (N - k))
p_F = 1 - cdf(FDist(k - 1, N - k), F)
η² = ss_between / (ss_between + ss_within)

println(@sprintf("\n[ANOVA] F(%d, %d) = %.3f   p = %.4f   η² = %.3f   (n = 6 per sector, n_total = %d)",
    k - 1, N - k, F, p_F, η², N))

# Per-sector summary
println("\nPer-sector OoS KS distribution (n=6 each):")
@printf("  %-26s  %3s  %7s  %7s  %7s  %5s\n", "sector", "n", "median", "mean", "min", "fail")
println("  ", "-"^75)
sector_summary = NamedTuple[]
for s in sectors
    ks = ks_by_sector[s]
    row = (sector=s, n=length(ks), med=median(ks), mn=mean(ks), sd=std(ks),
           min=minimum(ks), fail=count(<(KS_FAIL_PP), ks))
    push!(sector_summary, row)
    @printf("  %-26s  %3d  %7.2f  %7.2f  %7.2f  %5d\n",
        s, row.n, row.med, row.mn, row.min, row.fail)
end

println(@sprintf("\nAggregate (n = %d):  median = %.2f%%  mean = %.2f%%  sd = %.2f%%  failures = %d / %d",
    length(all_ks), median(all_ks), mean(all_ks), std(all_ks),
    count(<(KS_FAIL_PP), all_ks), length(all_ks)))

# CSVs
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
    for (sector, ticker, ks_is, ks_oos, kurt_o, kurt_s, acf) in EXTRA
        @printf(io, "%s,%s,%.1f,%.1f,%.2f,%.2f,%.2f,%.4f,%.2f\n",
            sector, ticker, ks_is, ks_oos, kurt_o, kurt_s,
            kurt_s - kurt_o, acf, 50.00)
    end
end

txt_path = joinpath(OUT_DIR, "sector_panel_n6.txt")
open(txt_path, "w") do io
    println(io, "="^110)
    println(io, "60-ticker sector-balanced panel (n=6 per sector)  (Reviewer Round 2 / Item P10)")
    println(io, "="^110)
    println(io, "Setup: penalised CHMM-t at K=18, λ=20, 1000 paths, seed = 20260420.")
    println(io, "Body 30-ticker rows are read from results/sector_panel/sector_panel_summary.csv;")
    println(io, "the additional 30 tickers were fit by run_sector_panel_n6.jl. This file is the")
    println(io, "post-processed combined panel; per-ticker numbers in sector_panel_n6.csv.")
    println(io)
    println(io, "Per-sector summary at n = 6:")
    @printf(io, "  %-26s  %3s  %7s  %7s  %7s  %7s  %5s\n",
        "sector", "n", "median", "mean", "sd", "min", "fail")
    println(io, "  ", "-"^85)
    for r in sector_summary
        @printf(io, "  %-26s  %3d  %7.2f  %7.2f  %7.2f  %7.2f  %5d\n",
            r.sector, r.n, r.med, r.mn, r.sd, r.min, r.fail)
    end
    println(io)
    @printf(io, "Aggregate (n = %d):  median = %.2f%%   mean = %.2f%%   sd = %.2f%%   failures = %d / %d\n",
        length(all_ks), median(all_ks), mean(all_ks), std(all_ks),
        count(<(KS_FAIL_PP), all_ks), length(all_ks))
    println(io)
    println(io, "One-way ANOVA on OoS KS by sector at n = 6:")
    @printf(io, "  F(%d, %d) = %.3f   p = %.4f   η² = %.3f\n", k-1, N-k, F, p_F, η²)
    println(io, "  Body n=3 result was F(9, 20) = 0.44, p = 0.90, η² = 0.16 (severely underpowered).")
    if p_F < 0.05
        println(io, "  At n=6 per sector, the test rejects the no-sector-effect null at the 5% level.")
    else
        @printf(io, "  At n=6 per sector, the test does not reject the no-sector-effect null at α=0.05.\n")
        @printf(io, "  η² = %.3f: cross-sector dispersion accounts for %.1f%% of OoS KS variance, mostly\n", η², 100*η²)
        println(io, "  attributable to per-ticker variation rather than sector-level structure.")
    end
end

println("\n[done] $csv_full")
println("[done] $csv_extra")
println("[done] $txt_path")
