# ========================================================================================= #
# run_spectral_rank_cross_ticker.jl
#
# Cross-ticker spectral effective-rank diagnostic at K = 3 AND K = 18 across the 30-ticker
# sector-balanced panel (+ SPY control), with CONVERGED MULTISTART fits and a HELD-OUT
# out-of-sample ACF criterion (2026-07-16 sixth review, findings 1-2: the previous version
# compared a converged K = 3 fit against K = 18 fits capped at 60 iterations — final
# likelihood increments thousands of times the tolerance — from one deterministic start,
# scored in-sample only).
#
# Design:
#   - Fits: baum_welch_multistart, tol = 1e-4, max_iter = 4000; K = 3 uses 3 starts,
#     K = 18 uses 5 starts (canonical quantile start + jittered starts). Per ticker and
#     per K the artifact records the best start, its evaluation count, final likelihood
#     increment, a converged flag (tolerance rule, not the cap), and the LL spread
#     across starts.
#   - Shares: grouped-component budgets under the lag-1 norm B = Σ_c |contrib_c(1)| and
#     the horizon-aware norm B∫ = Σ_c Σ_{τ=1..252} |contrib_c(τ)|.
#   - ACF criterion: per ticker, MAE between the fitted population |G| ACF and the sample
#     |G| ACF, near band (lags 1-63) and far band (64-252), on BOTH the IS window (fit
#     window; descriptive) and the HELD-OUT OoS window (pre-specified criterion:
#     near-band OoS MAE). The zero curve (i.i.d.) is the reference on each window.
#
# What this does and does not identify: both state counts are likelihood fits, so the
# comparison shows what ML fitting delivers at each capacity; the attainable-accuracy
# question ("could ANY m-mode combination do better?") is answered separately by
# runners/spectral/run_mode_capacity_ceiling.jl.
#
# Output: results/diagnostics/spectral_rank_cross_ticker.txt
#         results/diagnostics/spectral_rank_cross_ticker_fits.csv (per-ticker optimizer table)
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
include(joinpath(@__DIR__, "spectral_common.jl"));

using Printf

const SEED      = 20260420;
const K_LIST    = [3, 18];
const N_STARTS  = Dict(3 => 3, 18 => 5);
const MAX_ITER  = 4000;
const TOL       = 1e-4;
const DT        = 1/252;
const RISK_FREE = 0.0;
const N_M_DRAW  = 200_000;   # non-Normal fallback only; Normal moments are analytic
const MAXLAG    = 252;

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

println("="^88);
println("  Cross-ticker spectral diagnostic, K in $(K_LIST), converged multistart + held-out ACF");
println("="^88);

# ----------------------------------------------------------------------------------------- #
# Data (IS fit window + held-out OoS window)
# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading IS + OoS panels...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
panel_tickers = sort(keys(filtered) |> collect);
all_R = log_growth_matrix(filtered, panel_tickers; Δt=DT, risk_free_rate=RISK_FREE);

function _sample_acf_abs(x::AbstractVector, maxlag::Int)
    a = abs.(x); n = length(a); μ = mean(a);
    v = sum((a .- μ).^2);
    return [sum((a[1:n-τ] .- μ) .* (a[1+τ:n] .- μ)) / v for τ in 1:maxlag];
end

_band(v, lo, hi) = mean(abs.(v[lo:hi]));

# ----------------------------------------------------------------------------------------- #
# Loop over tickers x K
# ----------------------------------------------------------------------------------------- #
panel_results = Dict{Int, Dict{String, NamedTuple}}(K => Dict{String, NamedTuple}() for K in K_LIST);
acf_fit  = Dict{String, NamedTuple}();   # per-ticker ACF MAEs at both K, IS + OoS, + zero refs
fit_rows = NamedTuple[];                 # per-ticker x K optimizer evidence

for (si, sector_name) in enumerate(vcat([s for (s, _) in SECTOR_PANEL], ["SPY (control)"]))
    if sector_name == "SPY (control)"
        ticks = ["SPY"]; sec = "Index";
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
        R_oos = try
            collect(Float64, log_growth_matrix(oos_dataset, ticker; Δt=DT, risk_free_rate=RISK_FREE));
        catch e
            @warn "no OoS series for $ticker: $e"; Float64[];
        end
        ρ_is = _sample_acf_abs(R_is, MAXLAG);
        ρ_oos = isempty(R_oos) ? Float64[] : _sample_acf_abs(R_oos, MAXLAG);
        maes = Dict{Int, NamedTuple}();
        for K in K_LIST
            try
                T, μ, σ, πv, llh, γ, diags = baum_welch_multistart(R_is, K;
                    n_starts=N_STARTS[K], max_iter=MAX_ITER, tol=TOL,
                    seed=SEED + 100 * idx + K);
                best = argmax([d.ll for d in diags]);
                d = diags[best];
                lls = [x.ll for x in diags];
                push!(fit_rows, (ticker=ticker, K=K, best_start=d.start,
                                 n_evals=d.n_evals, final_inc=d.final_increment,
                                 converged=d.converged, ll_best=d.ll,
                                 ll_spread=maximum(lls) - minimum(lls),
                                 n_converged=count(x -> x.converged, diags)));
                # spectral quantities from the best fit
                mdl_T = T;
                π̄ = _stationary_pi(mdl_T);
                m = zeros(K); M = zeros(K);
                for k in 1:K
                    z = μ[k] / σ[k];
                    m[k] = σ[k] * sqrt(2 / π) * exp(-z^2 / 2) + μ[k] * (1 - 2 * cdf(Normal(), -z));
                    M[k] = μ[k]^2 + σ[k]^2;
                end
                σ²_G, comps, κ_V, recon_err = _spectral_components(mdl_T, π̄, m, M);
                s = _component_summary(comps);
                panel_results[K][ticker] = (sector=sec, n_comps=length(comps),
                                            recon_err=recon_err, s...);
                ρ_mod = _theoretical_acf(mdl_T, π̄, m, M, MAXLAG);
                maes[K] = (is_near  = mean(abs.(ρ_mod[1:63] .- ρ_is[1:63])),
                           is_far   = mean(abs.(ρ_mod[64:MAXLAG] .- ρ_is[64:MAXLAG])),
                           oos_near = isempty(ρ_oos) ? NaN : mean(abs.(ρ_mod[1:63] .- ρ_oos[1:63])),
                           oos_far  = isempty(ρ_oos) ? NaN : mean(abs.(ρ_mod[64:MAXLAG] .- ρ_oos[64:MAXLAG])));
                @printf("  %-6s [%s] K=%-2d  conv=%s n=%d inc=%.1e  dom=%.3f (int %.3f)  oos-near=%.4f\n",
                        ticker, sec, K, d.converged ? "y" : "N", d.n_evals, d.final_increment,
                        s.dom_share, s.dom_share_int, maes[K].oos_near);
            catch e
                @warn "fit failed for $ticker at K=$K: $e";
            end
        end
        if all(K -> haskey(maes, K), K_LIST)
            acf_fit[ticker] = (sector=sec,
                               is_near_k3 = maes[3].is_near,   is_near_k18 = maes[18].is_near,
                               is_far_k3  = maes[3].is_far,    is_far_k18  = maes[18].is_far,
                               oos_near_k3 = maes[3].oos_near, oos_near_k18 = maes[18].oos_near,
                               oos_far_k3  = maes[3].oos_far,  oos_far_k18  = maes[18].oos_far,
                               is_near_zero = _band(ρ_is, 1, 63),  is_far_zero = _band(ρ_is, 64, MAXLAG),
                               oos_near_zero = isempty(ρ_oos) ? NaN : _band(ρ_oos, 1, 63),
                               oos_far_zero  = isempty(ρ_oos) ? NaN : _band(ρ_oos, 64, MAXLAG));
        end
    end
end

# ----------------------------------------------------------------------------------------- #
# Summaries
# ----------------------------------------------------------------------------------------- #
function _dist_summary(K)
    rs = collect(values(panel_results[K]));
    return (n=length(rs),
            dom  = [r.dom_share for r in rs],  domI = [r.dom_share_int for r in rs],
            n95  = [r.n_for_95 for r in rs],   n95I = [r.n_for_95_int for r in rs]);
end

vals(f) = [getfield(x, f) for x in values(acf_fit)];
Δ_is_near  = vals(:is_near_k3)  .- vals(:is_near_k18);
Δ_oos_near = vals(:oos_near_k3) .- vals(:oos_near_k18);

conv_rows(K) = [r for r in fit_rows if r.K == K];

# ----------------------------------------------------------------------------------------- #
# Output
# ----------------------------------------------------------------------------------------- #
open(joinpath(OUT_DIR, "spectral_rank_cross_ticker_fits.csv"), "w") do io
    println(io, "ticker,K,best_start,n_evals,final_increment,converged,ll_best,ll_spread,n_converged_starts");
    for r in sort(fit_rows, by = x -> (x.ticker, x.K))
        @printf(io, "%s,%d,%d,%d,%.3e,%d,%.4f,%.4f,%d\n",
                r.ticker, r.K, r.best_start, r.n_evals, r.final_inc,
                r.converged ? 1 : 0, r.ll_best, r.ll_spread, r.n_converged);
    end
end

out_path = joinpath(OUT_DIR, "spectral_rank_cross_ticker.txt");
open(out_path, "w") do io
    println(io, "="^96);
    println(io, "Cross-ticker spectral effective-rank diagnostic (converged multistart; held-out ACF criterion)");
    println(io, "="^96);
    println(io, "Setup: CHMM-N at K in $(K_LIST), 30-ticker sector panel + SPY control.");
    println(io, "Fits: baum_welch_multistart, tol = $TOL, max_iter = $MAX_ITER; starts per K:");
    println(io, "      K = 3 -> $(N_STARTS[3]) starts, K = 18 -> $(N_STARTS[18]) starts (canonical quantile + jittered).");
    println(io, "Emission moments are analytic (folded normal); stationary vector by checked left-");
    println(io, "eigenvector solve. Held-out criterion (pre-specified): near-band (lags 1-63) MAE of");
    println(io, "the fitted population |G| ACF against the OUT-OF-SAMPLE window's sample ACF.");
    println(io);
    println(io, "-"^96);
    println(io, "Optimizer evidence (per K, across tickers; per-ticker rows in spectral_rank_cross_ticker_fits.csv)");
    println(io, "-"^96);
    for K in K_LIST
        rows = conv_rows(K);
        n_conv = count(r -> r.converged, rows);
        incs = [abs(r.final_inc) for r in rows];
        nev  = [r.n_evals for r in rows];
        sprd = [r.ll_spread for r in rows];
        @printf(io, "  K = %-2d : converged (tol rule) %d / %d fits;  n_evals median %d  max %d;\n",
                K, n_conv, length(rows), round(Int, median(nev)), maximum(nev));
        @printf(io, "           |final increment| median %.1e  max %.1e;  cross-start LL spread median %.3f  max %.3f\n",
                median(incs), maximum(incs), median(sprd), maximum(sprd));
    end
    for K in K_LIST
        d = _dist_summary(K);
        println(io);
        println(io, "-"^96);
        println(io, "K = $K grouped-component budget shares (n = $(d.n) tickers)");
        println(io, "-"^96);
        @printf(io, "  dominant lag-1 share    : median %.3f  Q1 %.3f  Q3 %.3f  min %.3f\n",
                median(d.dom), quantile(d.dom, 0.25), quantile(d.dom, 0.75), minimum(d.dom));
        @printf(io, "  dominant horizon share  : median %.3f  Q1 %.3f  Q3 %.3f  min %.3f\n",
                median(d.domI), quantile(d.domI, 0.25), quantile(d.domI, 0.75), minimum(d.domI));
        @printf(io, "  n_for_95 (lag-1)        : median %.1f  max %d\n", median(d.n95), maximum(d.n95));
        @printf(io, "  n_for_95 (horizon)      : median %.1f  max %d\n", median(d.n95I), maximum(d.n95I));
    end
    println(io);
    println(io, "-"^96);
    println(io, "Model-vs-sample |G| ACF MAE (fitted population ACF against the window's sample ACF)");
    println(io, "-"^96);
    @printf(io, "%-6s %-24s %-8s %-8s %-8s %-9s %-9s %-9s\n",
            "ticker", "sector", "IS_K3", "IS_K18", "IS_zero", "OoS_K3", "OoS_K18", "OoS_zero");
    println(io, "-"^96);
    for sector_name in vcat([s for (s, _) in SECTOR_PANEL], ["Index"])
        for ticker in sort([t for (t, f) in acf_fit if f.sector == sector_name])
            f = acf_fit[ticker];
            @printf(io, "%-6s %-24s %-8.4f %-8.4f %-8.4f %-9.4f %-9.4f %-9.4f\n",
                    ticker, f.sector, f.is_near_k3, f.is_near_k18, f.is_near_zero,
                    f.oos_near_k3, f.oos_near_k18, f.oos_near_zero);
        end
    end
    println(io);
    n = length(acf_fit);
    println(io, "Cross-ticker medians over n = $n tickers (near band, lags 1-63):");
    @printf(io, "  IS  (fit window)  : K3 %.4f   K18 %.4f   zero %.4f   Δ(K3 - K18) median %.4f   K18 better on %d / %d\n",
            median(vals(:is_near_k3)), median(vals(:is_near_k18)), median(vals(:is_near_zero)),
            median(Δ_is_near), count(>(0), Δ_is_near), n);
    @printf(io, "  OoS (held out)    : K3 %.4f   K18 %.4f   zero %.4f   Δ(K3 - K18) median %.4f   K18 better on %d / %d\n",
            median(vals(:oos_near_k3)), median(vals(:oos_near_k18)), median(vals(:oos_near_zero)),
            median(Δ_oos_near), count(>(0), Δ_oos_near), n);
    println(io, "Far band (lags 64-252) medians:");
    @printf(io, "  IS  : K3 %.4f   K18 %.4f   zero %.4f      OoS : K3 %.4f   K18 %.4f   zero %.4f\n",
            median(vals(:is_far_k3)), median(vals(:is_far_k18)), median(vals(:is_far_zero)),
            median(vals(:oos_far_k3)), median(vals(:oos_far_k18)), median(vals(:oos_far_zero)));
    println(io);
    println(io, "Reading (scope):");
    println(io, "  - These are converged multistart LIKELIHOOD fits, so the comparison shows what ML");
    println(io, "    fitting delivers at each state count, on the fit window and on a held-out window.");
    println(io, "  - It does NOT by itself bound what any m-mode combination could achieve; the");
    println(io, "    attainable-accuracy ceiling is measured by run_mode_capacity_ceiling.jl, and the");
    println(io, "    two artifacts should be read together.");
    println(io, "  - The ticker count is descriptive (no cross-sectional dependence correction; the");
    println(io, "    panel shares market-wide factors, and SPY is included).");
end
println();
println("[done] Wrote $out_path");
println("[done] Wrote $(joinpath(OUT_DIR, "spectral_rank_cross_ticker_fits.csv"))");
