# ========================================================================================= #
# run_mode_capacity_ceiling.jl
#
# Best attainable m-mode geometric approximation of the observed |G| sample ACF
# (2026-07-16 sixth review, finding 1: comparing likelihood fits at K = 3 vs K = 18 shows
# what ML fitting delivers, not what the decay-mode BUDGET could deliver; this runner
# measures the budget's attainable-accuracy ceiling directly).
#
# For each ticker of the 30-ticker panel (+ SPY), and each mode count m, fit
#
#     rho_hat(tau) = sum_{k=1..m} a_k * lambda_k^tau ,   tau = 1..252,
#
# to the observed IS sample |G| ACF by least squares, with a_k free-sign and
# lambda_k in (0, 0.9995). Selection: orthogonal matching pursuit over a fine
# lambda dictionary (log-spaced half-lives), exact least-squares coefficient refit after
# every selection, then local per-atom lambda refinement sweeps. The result upper-bounds
# what ANY real-mode combination with that budget can achieve on this sample curve
# (a fitted K-state HMM's modes satisfy additional moment constraints and may include
# damped-oscillatory pairs, so its error can only be >= this ceiling's error).
#
#   m = 2  is the mode budget of a K = 3 chain; m = 17 that of a K = 18 chain.
#
# Reading rule (pre-specified): the K = 3 budget is non-binding on a band exactly when the
# m = 2 ceiling's band MAE is close to the m = 17 ceiling's; a material gap means the
# budget does bind there in principle, whether or not likelihood fits exploit it.
#
# Output: results/diagnostics/mode_capacity_ceiling.txt
#         results/diagnostics/mode_capacity_ceiling.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Printf, LinearAlgebra

const DT        = 1/252;
const RISK_FREE = 0.0;
const MAXLAG    = 252;
const M_LIST    = [1, 2, 4, 8, 17];
const LAM_MAX   = 0.9995;

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

# Dictionary of candidate decay rates: log-spaced half-lives from 0.4 to 600 trading days.
const HALF_LIVES = exp.(range(log(0.4), log(600.0), length=400));
const LAMBDAS    = clamp.(exp.(log(0.5) ./ HALF_LIVES), 1e-6, LAM_MAX);

_atom(λ) = [λ^τ for τ in 1:MAXLAG];

function _lls(ρ::Vector{Float64}, lams::Vector{Float64})
    G = hcat([_atom(λ) for λ in lams]...);
    a = G \ ρ;
    return a, ρ .- G * a;
end

"""
Orthogonal matching pursuit for m atoms + local λ refinement, returning the
selected λ's, refit coefficients, and the fitted curve.
"""
function _fit_m_modes(ρ::Vector{Float64}, m::Int)
    selected = Float64[];
    resid = copy(ρ);
    norms = Dict(λ => norm(_atom(λ)) for λ in LAMBDAS);
    for _ in 1:m
        best_λ = LAMBDAS[1]; best_score = -Inf;
        for λ in LAMBDAS
            λ in selected && continue;
            sc = abs(dot(resid, _atom(λ))) / norms[λ];
            if sc > best_score; best_score = sc; best_λ = λ; end
        end
        push!(selected, best_λ);
        _, resid = _lls(ρ, selected);
    end
    # local refinement: per-atom multiplicative half-life perturbations, 3 sweeps
    factors = [0.90, 0.95, 0.98, 1.02, 1.05, 1.10];
    _, r0 = _lls(ρ, selected);
    sse = sum(abs2, r0);
    for _ in 1:3
        for i in 1:m
            h_i = log(0.5) / log(selected[i]);
            for f in factors
                cand = copy(selected);
                cand[i] = clamp(exp(log(0.5) / (h_i * f)), 1e-6, LAM_MAX);
                _, r = _lls(ρ, cand);
                s = sum(abs2, r);
                if s < sse - 1e-14; sse = s; selected = cand; end
            end
        end
    end
    a, resid = _lls(ρ, selected);
    return selected, a, ρ .- resid;
end

function _sample_acf_abs(x::AbstractVector, maxlag::Int)
    a = abs.(x); n = length(a); μ = mean(a);
    v = sum((a .- μ).^2);
    return [sum((a[1:n-τ] .- μ) .* (a[1+τ:n] .- μ)) / v for τ in 1:maxlag];
end

# ----------------------------------------------------------------------------------------- #
println("="^88);
println("  Mode-capacity ceiling: best m-mode geometric approximation of the sample |G| ACF");
println("="^88);

train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
panel_tickers = sort(keys(filtered) |> collect);
all_R = log_growth_matrix(filtered, panel_tickers; Δt=DT, risk_free_rate=RISK_FREE);

results = Dict{String, Dict{Int, NamedTuple}}();
sectors = Dict{String, String}();
for sector_name in vcat([s for (s, _) in SECTOR_PANEL], ["SPY (control)"])
    ticks = sector_name == "SPY (control)" ? ["SPY"] :
            first([ts for (s, ts) in SECTOR_PANEL if s == sector_name]);
    sec = sector_name == "SPY (control)" ? "Index" : sector_name;
    for ticker in ticks
        idx = findfirst(==(ticker), panel_tickers);
        idx === nothing && continue;
        ρ = _sample_acf_abs(all_R[:, idx], MAXLAG);
        results[ticker] = Dict{Int, NamedTuple}();
        sectors[ticker] = sec;
        for m in M_LIST
            lams, a, fit = _fit_m_modes(ρ, m);
            results[ticker][m] = (near = mean(abs.(fit[1:63] .- ρ[1:63])),
                                  far  = mean(abs.(fit[64:MAXLAG] .- ρ[64:MAXLAG])),
                                  lams = sort(lams; rev=true), a = a);
        end
        r2 = results[ticker][2]; r17 = results[ticker][17];
        @printf("  %-6s m=2: near %.4f far %.4f   m=17: near %.4f far %.4f\n",
                ticker, r2.near, r2.far, r17.near, r17.far);
    end
end

med(m, f) = median([getfield(results[t][m], f) for t in keys(results)]);

open(joinpath(OUT_DIR, "mode_capacity_ceiling.csv"), "w") do io
    println(io, "ticker,m,near_mae,far_mae,lambdas");
    for t in sort(collect(keys(results))), m in M_LIST
        r = results[t][m];
        @printf(io, "%s,%d,%.6f,%.6f,%s\n", t, m, r.near, r.far,
                join([@sprintf("%.4f", l) for l in r.lams], ";"));
    end
end

out_path = joinpath(OUT_DIR, "mode_capacity_ceiling.txt");
open(out_path, "w") do io
    println(io, "="^96);
    println(io, "Mode-capacity ceiling: best m-mode geometric approximation of the observed IS sample |G| ACF");
    println(io, "="^96);
    println(io, "Fit: least squares over free-sign coefficients and lambda in (0, $LAM_MAX), lags 1-$MAXLAG,");
    println(io, "selection by orthogonal matching pursuit over a $(length(LAMBDAS))-atom log-spaced half-life");
    println(io, "dictionary with exact LLS refit after each selection and 3 local per-atom refinement sweeps.");
    println(io, "The m-mode ceiling UPPER-BOUNDS what any real-mode K-state HMM budget (m = K - 1) can achieve");
    println(io, "on this sample curve: HMM modes carry additional moment constraints, so fitted-model error >=");
    println(io, "ceiling error. m = 2 is the K = 3 budget; m = 17 the K = 18 budget.");
    println(io);
    println(io, "Cross-ticker medians of the ceiling MAE (n = $(length(results)) tickers):");
    println(io, "-"^72);
    @printf(io, "%-4s | %-22s | %-22s\n", "m", "near band (lags 1-63)", "far band (lags 64-252)");
    println(io, "-"^72);
    for m in M_LIST
        @printf(io, "%-4d | %-22.4f | %-22.4f\n", m, med(m, :near), med(m, :far));
    end
    println(io);
    println(io, "SPY detail:");
    for m in M_LIST
        r = results["SPY"][m];
        @printf(io, "  m = %-2d  near %.4f  far %.4f   lambdas: %s\n", m, r.near, r.far,
                join([@sprintf("%.4f", l) for l in r.lams], ", "));
    end
    println(io);
    g_near = med(2, :near) - med(17, :near);
    g_far  = med(2, :far)  - med(17, :far);
    println(io, "Reading (pre-specified rule: the K = 3 budget is non-binding on a band exactly when the");
    println(io, "m = 2 ceiling is close to the m = 17 ceiling there):");
    @printf(io, "  near band: median ceiling gap m=2 - m=17 = %.4f\n", g_near);
    @printf(io, "  far band : median ceiling gap m=2 - m=17 = %.4f\n", g_far);
    println(io, "  This ceiling is an in-sample approximation bound: with many free modes the fit can");
    println(io, "  track sample noise, so ceiling gaps are generous to the richer budget; a SMALL gap is");
    println(io, "  therefore strong evidence of a non-binding budget, while a large gap shows the budget");
    println(io, "  binds in principle even if likelihood fits do not exploit it.");
end
println();
println("[done] Wrote $out_path");
