# ========================================================================================= #
# run_exp_mode_diagnostic.jl
#
# Unrestricted exponential-mode approximation DIAGNOSTIC of the observed |G| sample ACF
# (2026-07-16 seventh review, finding 1). Supersedes run_mode_capacity_ceiling.jl, whose
# positive-real free-coefficient fit was presented as an HMM capacity bound: it is not one.
# A K-state HMM's ACF components satisfy joint stochastic-matrix / stationary-weight /
# emission-moment constraints that the free fit ignores, and the fit itself is a greedy
# heuristic with no global certificate — so its error neither lower- nor upper-bounds what
# an HMM can attain. Realizable attainability is measured separately by
# run_hmm_acf_capacity.jl, which optimizes actual valid HMMs.
#
# What this diagnostic DOES measure: how well m unconstrained exponential decay modes can
# track the sample ACF, where the mode shapes now span every shape a K-state HMM component
# can take —
#     positive real     a · λ^τ                       (cost 1 mode)
#     negative real     a · (−λ)^τ                    (cost 1 mode)
#     damped oscillation λ^τ (a·cos θτ + b·sin θτ)    (cost 2 modes, conjugate pair)
# with free-sign coefficients, λ in (0, 0.9995), θ over a fixed angle grid. Any K-state
# HMM ACF is a combination of at most m = K − 1 such shapes, so the fitted CLASS contains
# every K-state HMM ACF curve; the achieved fit remains a heuristic optimum (greedy
# budget-aware matching pursuit + exact LLS refit + local λ refinement), reported as an
# exploratory descriptive result, not a bound.
#
# Aggregation: per-m cross-ticker medians AND the paired per-ticker m=2 − m=17 gap
# (median of differences; a difference of separate medians is also printed, labeled as
# such). Dictionary-resolution sensitivity re-runs m ∈ {2, 17} at half and double λ
# resolution and reports the median absolute change in band MAEs.
#
# Output: results/diagnostics/exp_mode_diagnostic.txt
#         results/diagnostics/exp_mode_diagnostic.csv         (components + coefficients)
#         results/diagnostics/exp_mode_diagnostic_curves.csv  (fitted curves, m ∈ {2, 17})
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
include(joinpath(@__DIR__, "exp_mode_common.jl"));

using Printf, LinearAlgebra, Statistics

const DT        = 1/252;
const RISK_FREE = 0.0;
const MAXLAG    = 252;
const M_LIST    = [1, 2, 4, 8, 17];
const N_LAM     = 400;

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

# ----------------------------------------------------------------------------------------- #
println("="^88);
println("  Unrestricted exponential-mode approximation diagnostic of the sample |G| ACF");
println("="^88);

train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
panel_tickers = sort(keys(filtered) |> collect);
all_R = log_growth_matrix(filtered, panel_tickers; Δt=DT, risk_free_rate=RISK_FREE);

const COMPS, BASES = _build_dictionary(N_LAM, MAXLAG);
println("[setup] dictionary: $(length(COMPS)) components ($(N_LAM) lambda x {pos, neg, $(length(THETAS)) osc angles})");

results = Dict{String, Dict{Int, NamedTuple}}();
acfs    = Dict{String, Vector{Float64}}();
for sector_name in vcat([s for (s, _) in SECTOR_PANEL], ["SPY (control)"])
    ticks = sector_name == "SPY (control)" ? ["SPY"] :
            first([ts for (s, ts) in SECTOR_PANEL if s == sector_name]);
    for ticker in ticks
        idx = findfirst(==(ticker), panel_tickers);
        idx === nothing && continue;
        ρ = _sample_acf_abs(all_R[:, idx], MAXLAG);
        acfs[ticker] = ρ;
        results[ticker] = Dict{Int, NamedTuple}();
        for m in M_LIST
            sel, a, fit = _fit_m_modes(ρ, m, COMPS, BASES, MAXLAG);
            near, far = _band_maes(fit, ρ);
            results[ticker][m] = (near=near, far=far, sel=sel, a=a, fit=fit);
        end
        r2 = results[ticker][2]; r17 = results[ticker][17];
        @printf("  %-6s m=2: near %.4f far %.4f   m=17: near %.4f far %.4f\n",
                ticker, r2.near, r2.far, r17.near, r17.far);
    end
end

# Dictionary-resolution sensitivity: rerun m in {2, 17} at half/double lambda resolution.
sens = Dict{Int, Vector{Float64}}();
for n_lam in [N_LAM ÷ 2, N_LAM * 2]
    comps_s, bases_s = _build_dictionary(n_lam, MAXLAG);
    deltas = Float64[];
    for t in keys(results), m in [2, 17]
        _, _, fit = _fit_m_modes(acfs[t], m, comps_s, bases_s, MAXLAG);
        near, far = _band_maes(fit, acfs[t]);
        push!(deltas, abs(near - results[t][m].near));
        push!(deltas, abs(far  - results[t][m].far));
    end
    sens[n_lam] = deltas;
    @printf("[sensitivity] n_lam = %d: median |delta MAE| = %.5f, max = %.5f\n",
            n_lam, median(deltas), maximum(deltas));
end

med(m, f) = median([getfield(results[t][m], f) for t in keys(results)]);
tickers_sorted = sort(collect(keys(results)));

open(joinpath(OUT_DIR, "exp_mode_diagnostic.csv"), "w") do io
    println(io, "ticker,m,near_mae,far_mae,components");
    for t in tickers_sorted, m in M_LIST
        r = results[t][m];
        @printf(io, "%s,%d,%.6f,%.6f,%s\n", t, m, r.near, r.far, _comp_string(r.sel, r.a));
    end
end

open(joinpath(OUT_DIR, "exp_mode_diagnostic_curves.csv"), "w") do io
    println(io, "ticker,m,tau,rho_sample,rho_fit");
    for t in tickers_sorted, m in [2, 17]
        r = results[t][m];
        for τ in 1:MAXLAG
            @printf(io, "%s,%d,%d,%.6f,%.6f\n", t, m, τ, acfs[t][τ], r.fit[τ]);
        end
    end
end

out_path = joinpath(OUT_DIR, "exp_mode_diagnostic.txt");
open(out_path, "w") do io
    println(io, "="^96);
    println(io, "Unrestricted exponential-mode approximation diagnostic of the observed IS sample |G| ACF");
    println(io, "="^96);
    println(io, "EXPLORATORY DESCRIPTIVE DIAGNOSTIC — NOT an HMM capacity bound. The fitted class spans");
    println(io, "every shape a K-state HMM ACF component can take (positive/negative real decay and");
    println(io, "damped-oscillatory pairs at 2-mode cost, free-sign coefficients), so it CONTAINS all");
    println(io, "K-state HMM ACF curves at budget m = K - 1; but (a) most curves in the class are not");
    println(io, "HMM-realizable, and (b) the fit is a greedy heuristic (budget-aware matching pursuit");
    println(io, "over a $(length(COMPS))-component dictionary, exact LLS refit, $(N_SWEEPS) local lambda-refinement");
    println(io, "sweeps) with no global certificate. Realizable attainability is measured separately by");
    println(io, "run_hmm_acf_capacity.jl. Lags 1-$(MAXLAG); near band 1-63, far band 64-252.");
    println(io);
    println(io, "Cross-ticker medians of the diagnostic fit MAE (n = $(length(results)) tickers):");
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
        @printf(io, "  m = %-2d  near %.4f  far %.4f   components: %s\n", m, r.near, r.far,
                _comp_string(r.sel, r.a));
    end
    println(io);
    g_near_paired = median([results[t][2].near - results[t][17].near for t in tickers_sorted]);
    g_far_paired  = median([results[t][2].far  - results[t][17].far  for t in tickers_sorted]);
    println(io, "m = 2 minus m = 17 diagnostic gap:");
    @printf(io, "  near band: paired per-ticker median of differences = %.4f  (difference of separate medians = %.4f)\n",
            g_near_paired, med(2, :near) - med(17, :near));
    @printf(io, "  far band : paired per-ticker median of differences = %.4f  (difference of separate medians = %.4f)\n",
            g_far_paired, med(2, :far) - med(17, :far));
    println(io);
    println(io, "Dictionary-resolution sensitivity (m in {2, 17}, all tickers, band MAEs):");
    for n_lam in sort(collect(keys(sens)))
        @printf(io, "  n_lam = %-4d : median |delta MAE| = %.5f, max = %.5f\n",
                n_lam, median(sens[n_lam]), maximum(sens[n_lam]));
    end
    println(io);
    println(io, "Reading: small m = 2 vs m = 17 gaps say only that a few unconstrained exponential");
    println(io, "modes track this sample curve nearly as well as many; they do NOT establish that the");
    println(io, "HMM feasible set at K = 3 attains these errors, nor identify which constraint binds a");
    println(io, "likelihood fit. See run_hmm_acf_capacity.jl for the realizable (valid-HMM) comparison.");
end
println();
println("[done] Wrote $out_path");
