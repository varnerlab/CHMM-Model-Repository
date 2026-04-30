# ========================================================================================= #
# run_spectral_rank_cross_ticker.jl
#
# Cross-ticker spectral effective-rank diagnostic at K = 18 across the 30-ticker sector-
# balanced panel. Addresses peer-review item P2.1 (R2.W1, R3.Q1): the abstract claim that
# "the algebraic rank bound is non-binding at $K \ge 3$ on equity-return data" rests on
# n = 1 ticker (SPY) in run_spectral_rank.jl. This script repeats the diagnostic on the
# 30-ticker panel and reports the cross-ticker distribution of:
#   1. The dominant non-unit eigenvalue's lag-1 ACF contribution.
#   2. Number of modes carrying ≥ 95% / 99% of cumulative |w_k λ_k|.
#
# Output: results/diagnostics/spectral_rank_cross_ticker.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random, LinearAlgebra, Statistics, Printf

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
# Per-ticker spectral mode helpers
# ----------------------------------------------------------------------------------------- #
function _T_pibar_m(model, K::Int; seed::Int=0, n_draw::Int=N_M_DRAW)
    Random.seed!(seed);
    T = zeros(K, K);
    for i in 1:K; T[i, :] = probs(model.transition[i]); end
    π̄ = (T^2000)[1, :];
    m = zeros(K); M = zeros(K);
    for k in 1:K
        s = [rand(model.emission[k]) for _ in 1:n_draw];
        m[k] = mean(abs.(s));
        M[k] = mean(s.^2);
    end
    return T, π̄, m, M;
end

function _spectral_modes(T::AbstractMatrix, π̄::AbstractVector, m::AbstractVector,
                          M::AbstractVector)
    K = length(m);
    F = eigen(T);
    λ = F.values;
    V = F.vectors;
    W = inv(V);
    σ²_G = π̄' * M - (π̄' * m)^2;
    idx_one = argmin(abs.(λ .- 1.0));
    rest = setdiff(1:K, idx_one);
    rows = NamedTuple[];
    for k in rest
        v_k = V[:, k];
        w_k = W[k, :];
        c_k = (m' * Diagonal(π̄) * v_k) * dot(w_k, m);
        w_k_norm = c_k / σ²_G;
        push!(rows, (
            lambda  = λ[k],
            abs_lam = abs(λ[k]),
            w_k     = w_k_norm,
        ));
    end
    sort!(rows, by = r -> -r.abs_lam);
    return σ²_G, rows;
end

function _summarise(rows)
    contribs_t1 = [real(r.w_k * r.lambda) for r in rows];   # τ = 1 contribution
    abs_t1_sum = sum(abs.(contribs_t1));
    sorted_idx = sortperm(abs.(contribs_t1); rev=true);
    sorted_contribs = abs.(contribs_t1[sorted_idx]);
    cum = cumsum(sorted_contribs) ./ abs_t1_sum;
    dom_share = sorted_contribs[1] / abs_t1_sum;
    n_for_95 = findfirst(x -> x >= 0.95, cum);
    n_for_99 = findfirst(x -> x >= 0.99, cum);
    n_above_1pct = count(x -> x / abs_t1_sum > 0.01, sorted_contribs);
    return (dom_share=dom_share,
            n_for_95=isnothing(n_for_95) ? length(rows) : n_for_95,
            n_for_99=isnothing(n_for_99) ? length(rows) : n_for_99,
            n_above_1pct=n_above_1pct);
end

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
            T, π̄, m, M = _T_pibar_m(mdl, K_MAIN; seed=SEED);
            σ²_G, rows = _spectral_modes(T, π̄, m, M);
            s = _summarise(rows);
            panel_results[ticker] = (sector=sec, s...);
            @printf("  %-6s [%s]  dom_share = %.3f   n95 = %d   n99 = %d   n>1%%= %d\n",
                    ticker, sec, s.dom_share, s.n_for_95, s.n_for_99, s.n_above_1pct);
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
    println(io, "Cross-ticker spectral effective-rank diagnostic  (peer-review P2.1 / R2.W1)");
    println(io, "="^96);
    println(io, "Setup: CHMM-N at K = $K_MAIN, sector-balanced 30-ticker panel + SPY control,");
    println(io, "       seed = $SEED, n_draw = $N_M_DRAW per state for m_k.");
    println(io, "Per-ticker columns: dom_share = (dominant non-unit eigenvalue's |w_k λ_k|) /");
    println(io, "       (sum of |w_k λ_k| over all non-unit eigenvalues, lag = 1).");
    println(io, "       n95 = number of modes carrying ≥ 95% of cumulative |w_k λ_k|.");
    println(io, "       n>1%% = number of modes with > 1% of total |w_k λ_k|.");
    println(io);
    @printf(io, "%-6s %-26s %-10s %-6s %-6s %-7s\n",
            "ticker", "sector", "dom_share", "n95", "n99", "n>1%");
    println(io, "-"^96);
    for sector_name in vcat([s for (s, _) in SECTOR_PANEL], ["Index"])
        for ticker in sort([t for (t, r) in panel_results if r.sector == sector_name])
            r = panel_results[ticker];
            @printf(io, "%-6s %-26s %-10.3f %-6d %-6d %-7d\n",
                    ticker, r.sector, r.dom_share, r.n_for_95, r.n_for_99, r.n_above_1pct);
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
