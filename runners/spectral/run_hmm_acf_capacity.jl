# ========================================================================================= #
# run_hmm_acf_capacity.jl
#
# Realizable ACF-targeted HMM capacity experiment (2026-07-16 seventh review, finding 1).
#
# Motivation: the free-exponential diagnostic in run_exp_mode_diagnostic.jl fits
# sum_k a_k lambda_k^tau with free-sign coefficients over positive-real atoms — a curve
# family that no HMM need realize (HMM ACF components satisfy joint stochastic-matrix,
# stationary-weight, and emission-moment constraints, and can be signed or damped-
# oscillatory). That fit is therefore an unrestricted approximation DIAGNOSTIC, not a
# capacity bound, and cannot by itself identify what a K-state HMM can attain.
#
# This runner measures attainability directly: for each ticker it optimizes an ACTUAL
# stationary Gaussian-emission K-state HMM (row-softmax stochastic T, exact stationary
# law, analytic folded-normal moments) so that its population |G| ACF matches the
# observed IS sample ACF, by multistart finite-difference Adam on the SSE over lags
# 1..252 (see acf_capacity_common.jl). Every reported error is achieved by a valid
# HMM, so it upper-bounds the K-state class's best error on that curve: it certifies
# ATTAINABILITY without claiming global optimality.
#
# Bracketing: for each K the likelihood fit shows what maximum-likelihood estimation
# delivers with no ACF targeting; the ACF-targeted fit shows what the same class
# attains when the criterion is the ACF itself; the unrestricted exponential
# diagnostic is a no-constraints reference curve fit. One start of the ACF-targeted
# optimizer is seeded at the likelihood fit, so by construction the attained SSE is
# no worse than the likelihood fit's own population-ACF SSE on the same target.
#
# Reading rule (declared before the run): the estimation criterion, not the state
# budget, is implicated on a band exactly when (a) the ACF-targeted K = 3 error is
# materially below the likelihood-fit error at both K, and (b) the ACF-targeted
# K = 3 and K = 18 errors are close to each other. If ACF-targeted K = 3 cannot
# approach ACF-targeted K = 18, the state budget itself binds.
#
# Output: results/diagnostics/hmm_acf_capacity.csv
#         results/diagnostics/hmm_acf_capacity.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
include(joinpath(@__DIR__, "spectral_common.jl"));
include(joinpath(@__DIR__, "acf_capacity_common.jl"));

using Printf, LinearAlgebra, Statistics, Random

const SEED      = 20260420;
const DT        = 1/252;
const RISK_FREE = 0.0;
const MAXLAG    = 252;
const K_LIST    = [3, 18];
const N_STARTS  = Dict(3 => 20, 18 => 8);
const N_ITER    = 4000;
const ML_SEED_MAX_ITER = 1000;   # cap on the likelihood fit used only as a seed
const ML_SEED_TOL      = 1e-4;

const SUMMARY_ONLY = "--summary-only" in ARGS;   # rebuild the txt from the CSV, no refits

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

function _sample_acf_abs(x::AbstractVector, maxlag::Int)
    a = abs.(x); n = length(a); μ = mean(a);
    v = sum((a .- μ).^2);
    return [sum((a[1:n-τ] .- μ) .* (a[1+τ:n] .- μ)) / v for τ in 1:maxlag];
end

# Scale-free marginal disclosure: model excess kurtosis of G from state raw moments.
function _model_excess_kurtosis(π̄, μ, σ)
    m1 = sum(π̄[k] * μ[k] for k in eachindex(μ));
    m2 = sum(π̄[k] * (μ[k]^2 + σ[k]^2) for k in eachindex(μ));
    m3 = sum(π̄[k] * (μ[k]^3 + 3μ[k]*σ[k]^2) for k in eachindex(μ));
    m4 = sum(π̄[k] * (μ[k]^4 + 6μ[k]^2*σ[k]^2 + 3σ[k]^4) for k in eachindex(μ));
    c2 = m2 - m1^2;
    c4 = m4 - 4m1*m3 + 6m1^2*m2 - 3m1^4;
    return c4 / c2^2 - 3.0;
end

# ----------------------------------------------------------------------------------------- #
println("="^88);
println("  Realizable ACF-targeted HMM capacity: valid K-state HMMs optimized on the sample ACF");
println("="^88);

if !SUMMARY_ONLY

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
        R = all_R[:, idx];
        ρ̂ = _sample_acf_abs(R, MAXLAG);
        kurt_sample = let z = (R .- mean(R))
            sum(z.^4) / length(R) / (sum(z.^2) / length(R))^2 - 3.0
        end;
        cv_sample = std(abs.(R)) / mean(abs.(R));
        results[ticker] = Dict{Int, NamedTuple}();
        sectors[ticker] = sec;
        for K in K_LIST
            # Likelihood seed: single quantile-start Baum-Welch, used only as a start.
            Tm, μm, σm, πm, llh, _ = baum_welch(R, K; max_iter=ML_SEED_MAX_ITER, tol=ML_SEED_TOL);
            fit = fit_acf_hmm(ρ̂, K; n_starts=N_STARTS[K], seed=SEED + 100*idx + K,
                              R=R, ml_seed=(T=Tm, μ=μm, σ=σm), n_iter=N_ITER);
            # Sanity-check the reported optimum with the checked stationary solve.
            _ = _stationary_pi(fit.T);
            kurt_model = _model_excess_kurtosis(fit.π̄, fit.μ, fit.σ);
            m_opt, M_opt = _folded_moments(fit.μ, fit.σ);
            μG = dot(fit.π̄, m_opt); σ²G = dot(fit.π̄, M_opt) - μG^2;
            cv_model = sqrt(max(σ²G, 0.0)) / μG;
            # The ML-seed start's final SSE certifies the seeded-start guarantee.
            sse_mlseed = fit.diagnostics[2].sse_final;
            results[ticker][K] = (fit=fit, kurt_model=kurt_model, kurt_sample=kurt_sample,
                                  cv_model=cv_model, cv_sample=cv_sample,
                                  sse_mlseed=sse_mlseed);
            @printf("  %-6s K=%-2d  near %.4f  far %.4f  sse %.5f  (best start %d/%d, %s)\n",
                    ticker, K, fit.near_mae, fit.far_mae, fit.sse,
                    fit.best_start, fit.n_starts, fit.converged ? "early-stop" : "iter-cap");
        end
    end
end

open(joinpath(OUT_DIR, "hmm_acf_capacity.csv"), "w") do io
    println(io, "ticker,sector,K,n_starts,best_start,converged,n_iter,sse,near_mae,far_mae," *
                "kurt_model,kurt_sample,cv_abs_model,cv_abs_sample,pi_min,abs_lams");
    for t in sort(collect(keys(results))), K in K_LIST
        r = results[t][K]; fit = r.fit;
        lam_str = join([@sprintf("%.4f", l) for l in fit.abs_lams[1:min(5, end)]], ";");
        @printf(io, "%s,%s,%d,%d,%d,%s,%d,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.6f,%s\n",
                t, sectors[t], K, fit.n_starts, fit.best_start, fit.converged, fit.n_iter,
                fit.sse, fit.near_mae, fit.far_mae, r.kurt_model, r.kurt_sample,
                r.cv_model, r.cv_sample, minimum(fit.π̄), lam_str);
    end
end

end  # !SUMMARY_ONLY

# ----------------------------------------------------------------------------------------- #
# Summary txt is rebuilt from the per-ticker CSV (single source of truth), so it can be
# regenerated without refitting: `julia run_hmm_acf_capacity.jl --summary-only`.
# Cross-artifact comparisons (likelihood-fit medians, unrestricted exponential diagnostic)
# are NOT duplicated here; the paper combines the sibling artifacts directly
# (spectral_rank_cross_ticker.txt, exp_mode_diagnostic.txt), each consistency-checked
# against its own source.
# ----------------------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "hmm_acf_capacity.csv");
rows = NamedTuple[];
for ln in readlines(csv_path)[2:end]
    f = split(ln, ",");
    push!(rows, (ticker=String(f[1]), K=parse(Int, f[3]), n_starts=parse(Int, f[4]),
                 best_start=parse(Int, f[5]), converged=f[6] == "true",
                 sse=parse(Float64, f[8]), near=parse(Float64, f[9]),
                 far=parse(Float64, f[10]), kurt_model=parse(Float64, f[11]),
                 kurt_sample=parse(Float64, f[12]), pi_min=parse(Float64, f[15]),
                 abs_lams=String(f[16])));
end
medr(K, f) = median([getfield(r, f) for r in rows if r.K == K]);

out_path = joinpath(OUT_DIR, "hmm_acf_capacity.txt");
open(out_path, "w") do io
    println(io, "="^96);
    println(io, "Realizable ACF-targeted HMM capacity: valid K-state Gaussian-emission HMMs optimized so");
    println(io, "their population |G| ACF matches the observed IS sample ACF (SSE over lags 1-252),");
    println(io, "multistart finite-difference Adam; every reported error is ATTAINED by a valid HMM.");
    println(io, "="^96);
    println(io);
    println(io, "Starts per K: $(N_STARTS) (start 1 = sticky quantile, start 2 = likelihood-fit seed,");
    println(io, "rest = random persistence-diverse); Adam cap $(N_ITER) iterations, early stop on");
    println(io, "300-iteration stall. The likelihood-seeded start guarantees attained SSE <= the");
    println(io, "likelihood fit's own population-ACF SSE on the same target curve.");
    println(io);
    println(io, "Cross-ticker medians of model-vs-sample |G| ACF MAE (n = $(length(unique(r.ticker for r in rows))) tickers),");
    println(io, "every error ATTAINED by a valid K-state HMM:");
    println(io, "-"^78);
    @printf(io, "%-28s | %-10s | %-10s\n", "row", "near 1-63", "far 64-252");
    println(io, "-"^78);
    for K in K_LIST
        @printf(io, "ACF-targeted HMM  K = %-2d      | %-10.4f | %-10.4f\n",
                K, medr(K, :near), medr(K, :far));
    end
    println(io);
    println(io, "Comparison rows (likelihood-fit medians; unrestricted exponential diagnostic) live in");
    println(io, "their own artifacts: spectral_rank_cross_ticker.txt and exp_mode_diagnostic.txt.");
    println(io);
    println(io, "SPY detail:");
    for K in K_LIST
        r = only(filter(r -> r.ticker == "SPY" && r.K == K, rows));
        @printf(io, "  K = %-2d  near %.4f  far %.4f  sse %.5f  |lambda|: %s\n",
                K, r.near, r.far, r.sse, replace(r.abs_lams, ";" => ", "));
    end
    println(io);
    tickers = sort(unique(r.ticker for r in rows));
    d3  = Dict(r.ticker => r for r in rows if r.K == 3);
    d18 = Dict(r.ticker => r for r in rows if r.K == 18);
    g_near = median([d3[t].near - d18[t].near for t in tickers]);
    g_far  = median([d3[t].far  - d18[t].far  for t in tickers]);
    println(io, "Paired per-ticker ACF-targeted gap K=3 minus K=18 (median of differences):");
    @printf(io, "  near band: %+.4f\n", g_near);
    @printf(io, "  far band : %+.4f\n", g_far);
    println(io, "Marginal disclosure (the ACF-targeted optimum need not fit the marginal; scale-free):");
    for K in K_LIST
        spy = only(filter(r -> r.ticker == "SPY" && r.K == K, rows));
        @printf(io, "  K = %-2d: median model excess kurtosis %.2f vs sample median %.2f (SPY %.2f vs %.2f); median min stationary mass %.1e\n",
                K, medr(K, :kurt_model), medr(K, :kurt_sample), spy.kurt_model, spy.kurt_sample, medr(K, :pi_min));
    end
    println(io);
    n_conv = count(r.converged for r in rows);
    @printf(io, "\nOptimizer evidence: %d / %d best-start fits early-stopped (rest hit the iteration cap);\n",
            n_conv, length(rows));
    println(io, "per-ticker per-start diagnostics in the CSV.");
    println(io);
    println(io, "Reading rule (declared before the run): the estimation criterion, not the state");
    println(io, "budget, is implicated on a band exactly when (a) ACF-targeted K = 3 error is");
    println(io, "materially below the likelihood-fit error at both K, and (b) ACF-targeted K = 3");
    println(io, "and K = 18 are close. If (b) fails, the state budget itself binds. These fits are");
    println(io, "multistart heuristic optima: attained errors are upper bounds on the class's best");
    println(io, "error (attainability certificates), not certified global minima.");
end
println();
println("[done] Wrote $out_path");
