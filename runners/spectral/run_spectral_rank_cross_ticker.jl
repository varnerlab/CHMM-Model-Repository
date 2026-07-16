# ========================================================================================= #
# run_spectral_rank_cross_ticker.jl
#
# Cross-ticker spectral effective-rank diagnostic at K = 18 across the 30-ticker sector-
# balanced panel. Addresses peer-review item P2.1 (R2.W1, R3.Q1): the abstract claim that
# "the algebraic rank bound is non-binding at $K \ge 3$ on equity-return data" rests on
# n = 1 ticker (SPY) in run_spectral_rank.jl. This script repeats the diagnostic on the
# 30-ticker panel using the grouped-component definition of spectral_common.jl (complex-
# conjugate eigenvalue pairs combined into single real damped-oscillatory components;
# third-review item 7) and reports the cross-ticker distribution of:
#   1. The dominant component's share of the absolute component-magnitude budget
#      B = Σ_c |contrib_c(1)| at lag 1.
#   2. Number of components carrying ≥ 95% / 99% of cumulative B.
#   3. The max reconstruction error of the spectral ACF sum vs the direct matrix formula.
#
# Output: results/diagnostics/spectral_rank_cross_ticker.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
include(joinpath(@__DIR__, "spectral_common.jl"));

using Printf

const SEED      = 20260420;
const K_MAIN    = 18;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const N_M_DRAW  = 200_000;

const OUT_DIR = joinpath(_ROOT, "results", "diagnostics");
mkpath(OUT_DIR);

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
const ALL_TICKERS = vcat([t for (_, ts) in SECTOR_PANEL for t in ts], ["SPY"]);

println("="^88);
println("  Cross-ticker spectral effective-rank diagnostic at K = $K_MAIN  (peer-review P2.1)");
println("="^88);

# ----------------------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading IS panel...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
panel_tickers = sort(keys(filtered) |> collect);
all_R = log_growth_matrix(filtered, panel_tickers; Δt=DT, risk_free_rate=RISK_FREE);

# ----------------------------------------------------------------------------------------- #
# Per-ticker spectral helpers come from spectral_common.jl: _T_pibar_m,
# _spectral_components (grouped conjugate pairs), _component_summary.
# ----------------------------------------------------------------------------------------- #
# Loop over tickers
# ----------------------------------------------------------------------------------------- #
panel_results = Dict{String, NamedTuple}();
for sector_name in vcat([s for (s, _) in SECTOR_PANEL], ["SPY (control)"])
    if sector_name == "SPY (control)"
        ticks = ["SPY"];
        sec = "Index";
    else
        sec = sector_name;
        ticks = first([ts for (s, ts) in SECTOR_PANEL if s == sec]);
    end
    for ticker in ticks
        idx = findfirst(==(ticker), panel_tickers);
        if idx === nothing
            @warn "ticker $ticker not in panel_tickers; skipping";
            continue;
        end
        R_is = all_R[:, idx];
        Random.seed!(SEED);
        try
            mdl = build(MyContinuousHiddenMarkovModel,
                (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
            T, π̄, m, M = _T_pibar_m(mdl, K_MAIN; n_draw=N_M_DRAW, seed=SEED);
            σ²_G, comps, κ_V, recon_err = _spectral_components(T, π̄, m, M);
            s = _component_summary(comps);
            panel_results[ticker] = (sector=sec, n_comps=length(comps),
                                     recon_err=recon_err, s...);
            @printf("  %-6s [%s]  dom_share = %.3f   n95 = %d   n99 = %d   n>1%%= %d   recon = %.1e\n",
                    ticker, sec, s.dom_share, s.n_for_95, s.n_for_99, s.n_above_1pct, recon_err);
        catch e
            @warn "fit failed for $ticker: $e";
        end
    end
end

# ----------------------------------------------------------------------------------------- #
# Summary stats
# ----------------------------------------------------------------------------------------- #
dom_shares = [r.dom_share for r in values(panel_results)];
n95s = [r.n_for_95 for r in values(panel_results)];
n99s = [r.n_for_99 for r in values(panel_results)];
println();
println("Cross-ticker distribution (n = $(length(dom_shares)) tickers):");
@printf("  dominant non-unit mode lag-1 share : median %.3f, [Q1 %.3f, Q3 %.3f], min %.3f\n",
        median(dom_shares), quantile(dom_shares, 0.25), quantile(dom_shares, 0.75),
        minimum(dom_shares));
@printf("  modes for 95%% cumulative           : median %.1f, [Q1 %.1f, Q3 %.1f], max %d\n",
        median(n95s), quantile(n95s, 0.25), quantile(n95s, 0.75), maximum(n95s));
@printf("  modes for 99%% cumulative           : median %.1f, [Q1 %.1f, Q3 %.1f], max %d\n",
        median(n99s), quantile(n99s, 0.25), quantile(n99s, 0.75), maximum(n99s));

# ----------------------------------------------------------------------------------------- #
# Output
# ----------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "spectral_rank_cross_ticker.txt");
open(out_path, "w") do io
    println(io, "="^96);
    println(io, "Cross-ticker spectral effective-rank diagnostic  (grouped components; third-review item 7)");
    println(io, "="^96);
    println(io, "Setup: CHMM-N at K = $K_MAIN, sector-balanced 30-ticker panel + SPY control,");
    println(io, "       seed = $SEED, n_draw = $N_M_DRAW per state for m_k.");
    println(io, "Complex-conjugate eigenvalue pairs are grouped into single real damped-oscillatory");
    println(io, "components; contrib_c(1) is each component's SIGNED real lag-1 ACF contribution.");
    println(io, "Per-ticker columns: dom_share = max_c |contrib_c(1)| / B with");
    println(io, "       B = Σ_c |contrib_c(1)| the absolute component-magnitude budget (B is not a");
    println(io, "       percentage of ρ(1) itself: signed contributions can cancel).");
    println(io, "       n95/n99 = number of components carrying ≥ 95% / 99% of cumulative B.");
    println(io, "       n>1%% = number of components with > 1% of B.");
    println(io, "       ncomp = grouped component count (≤ K - 1); recon = max |spectral - direct|");
    println(io, "       ACF reconstruction error over τ = 1..252.");
    println(io);
    @printf(io, "%-6s %-26s %-10s %-6s %-6s %-7s %-6s %-9s\n",
            "ticker", "sector", "dom_share", "n95", "n99", "n>1%", "ncomp", "recon");
    println(io, "-"^96);
    for sector_name in vcat([s for (s, _) in SECTOR_PANEL], ["Index"])
        for ticker in sort([t for (t, r) in panel_results if r.sector == sector_name])
            r = panel_results[ticker];
            @printf(io, "%-6s %-26s %-10.3f %-6d %-6d %-7d %-6d %-9.1e\n",
                    ticker, r.sector, r.dom_share, r.n_for_95, r.n_for_99, r.n_above_1pct,
                    r.n_comps, r.recon_err);
        end
    end
    println(io);
    println(io, "-"^96);
    println(io, "Cross-ticker distribution (n = $(length(dom_shares)) tickers)");
    println(io, "-"^96);
    @printf(io, "  dominant lag-1 share : median %.3f  Q1 %.3f  Q3 %.3f  min %.3f\n",
            median(dom_shares), quantile(dom_shares, 0.25), quantile(dom_shares, 0.75),
            minimum(dom_shares));
    @printf(io, "  n_for_95             : median %.1f  Q1 %.1f  Q3 %.1f  max %d\n",
            median(n95s), quantile(n95s, 0.25), quantile(n95s, 0.75), maximum(n95s));
    @printf(io, "  n_for_99             : median %.1f  Q1 %.1f  Q3 %.1f  max %d\n",
            median(n99s), quantile(n99s, 0.25), quantile(n99s, 0.75), maximum(n99s));
    println(io);
    println(io, "Reading: if median dom_share is ≥ 0.90, the rank-non-binding claim of");
    println(io, "Section 3 (theory.tex) is supported across the cross-ticker panel rather than");
    println(io, "on SPY alone, addressing peer-review item P2.1 / R2.W1.");
end
println();
println("[done] Wrote $out_path");
