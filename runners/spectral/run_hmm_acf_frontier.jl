# ========================================================================================= #
# run_hmm_acf_frontier.jl
#
# Marginal-versus-ACF Pareto-frontier experiment at K = 3 (see CHANGELOG.md).
#
# Question: the realizable ACF-targeted experiment (run_hmm_acf_capacity.jl) shows a
# valid three-state HMM can track the finite-band sample |G| ACF far better than the
# likelihood fit does, and that the particular ACF-only optimum found has a poor
# marginal. That single point cannot establish whether the two axes genuinely COMPETE
# inside the three-state class (no fit is good at both) or whether a jointly good fit
# exists that neither criterion happens to select. This runner estimates the trade-off
# curve directly.
#
# Design: for each ticker, minimize the weighted objective
#
#     J_lambda(theta) = ACF_SSE(theta) + lambda * s * CvM(theta)
#
# over valid stationary Gaussian-emission 3-state HMMs (the same softmax/exp
# parametrization, exact stationary law, and multistart finite-difference Adam as the
# capacity runner), where CvM is the Cramer-von Mises-type distance between the
# stationary mixture CDF and the empirical CDF on a 500-point empirical-quantile grid,
# and s is a fixed balance scale = ACF_SSE / CvM of the single-start SPY likelihood fit
# (computed once at runner start, printed, and stored in the artifact). Arms:
# lambda in {0, 0.1, 0.3, 1, 3, 10, 30, 100} plus a pure-marginal arm (CvM only).
# Six starts per arm (sticky quantile, single-start Baum-Welch likelihood seed, four
# random persistence-diverse). Per-arm metrics: near/far ACF MAE, CvM, model excess
# kurtosis, stationary-mixture marginal log-likelihood per observation (marginal-density
# fit; ignores dependence), and tail-quantile errors at q in {0.01, 0.05, 0.95, 0.99};
# the single-start likelihood fit's own metrics ship as the "ml_ref" row.
#
# Reading rule (declared before the run): per ticker, the regret of a fit on an axis is
# metric / (best value of that metric attained across all arms of that ticker). Taking
# cross-ticker medians per arm:
#   - the axes COMPETE at K = 3 when every arm has max(ACF-regret, CvM-regret) >= 2
#     at the median (no arm is within a factor 2 of both single-axis optima);
#   - JOINT ATTAINABILITY is indicated when some arm has both median regrets <= 1.5.
# Anything between is reported descriptively. All fits are multistart heuristic optima:
# achieved points bound the true frontier from above (toward worse values), so a
# jointly good achieved fit is conclusive for attainability, while an empty middle is
# evidence subject to the optimizer caveat.
#
# Output: results/diagnostics/hmm_acf_frontier.csv     (ticker x arm rows + ml_ref)
#         results/diagnostics/hmm_acf_frontier.txt     (medians per arm, reading)
#         results/hmm_acf_frontier/frontier_<ticker>.jld2  (all arm winners: full models)
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
include(joinpath(@__DIR__, "spectral_common.jl"));
include(joinpath(@__DIR__, "acf_capacity_common.jl"));

using Printf, LinearAlgebra, Statistics, Random, JLD2

const SEED      = 20260420;
const DT        = 1/252;
const RISK_FREE = 0.0;
const MAXLAG    = 252;
const K         = 3;
const N_STARTS  = 6;
const N_ITER    = 4000;
const NQ        = 500;                       # empirical-quantile grid size for CvM
const LAMBDAS_ARM = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0];   # + pure-marginal arm
const QTAILS    = [0.01, 0.05, 0.95, 0.99];
const ML_SEED_MAX_ITER = 1000;
const ML_SEED_TOL      = 1e-4;

const OUT_DIR  = joinpath(_ROOT, "results", "diagnostics");
const FRT_DIR  = joinpath(_ROOT, "results", "hmm_acf_frontier");
mkpath(OUT_DIR);
mkpath(FRT_DIR);

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

_arm_label(λ) = λ === :marginal ? "marginal" : @sprintf("%.4g", λ);

function _fit_metrics(T, π̄, μ, σ, ρ̂, xq, R)
    m, M = _folded_moments(μ, σ);
    μG = dot(π̄, m); σ²G = dot(π̄, M) - μG^2;
    ρ = σ²G > 0 ? _population_acf(T, π̄, m, M, length(ρ̂)) : zeros(length(ρ̂));
    near = mean(abs.(ρ[1:63] .- ρ̂[1:63]));
    far  = mean(abs.(ρ[64:end] .- ρ̂[64:end]));
    sse  = sum(abs2, ρ .- ρ̂);
    cvm  = _marginal_cvm(π̄, μ, σ, xq);
    kurt = _model_excess_kurtosis(π̄, μ, σ);
    mll  = _mixture_loglik_perobs(π̄, μ, σ, R);
    qerr = [abs(_mixture_quantile(π̄, μ, σ, q) - quantile(R, q)) for q in QTAILS];
    return (rho=ρ, near=near, far=far, sse=sse, cvm=cvm, kurt=kurt, mll=mll, qerr=qerr);
end

# ----------------------------------------------------------------------------------------- #
println("="^88);
println("  Marginal-vs-ACF Pareto frontier at K = 3: weighted-objective sweep over valid HMMs");
println("="^88);

train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
panel_tickers = sort(keys(filtered) |> collect);
all_R = log_growth_matrix(filtered, panel_tickers; Δt=DT, risk_free_rate=RISK_FREE);

# Balance scale s from the SPY single-start likelihood fit: s = ACF_SSE_ml / CvM_ml,
# so lambda = 1 weighs the two terms comparably at that reference point.
spy_idx = findfirst(==("SPY"), panel_tickers);
R_spy   = all_R[:, spy_idx];
ρ̂_spy   = _sample_acf_abs(R_spy, MAXLAG);
xq_spy  = [quantile(R_spy, (i - 0.5) / NQ) for i in 1:NQ];
Tm0, μm0, σm0, _, _, _ = baum_welch(R_spy, K; max_iter=ML_SEED_MAX_ITER, tol=ML_SEED_TOL);
π̄m0 = _stationary_pi_unchecked(Tm0);
ref0 = _fit_metrics(Tm0, π̄m0, μm0, σm0, ρ̂_spy, xq_spy, R_spy);
const SCALE_S = ref0.sse / ref0.cvm;
@printf("[scale] SPY likelihood fit: ACF_SSE = %.5f, CvM = %.3e  ->  s = %.4g\n",
        ref0.sse, ref0.cvm, SCALE_S);

arms = vcat(Any[λ for λ in LAMBDAS_ARM], Any[:marginal]);
results = Dict{String, Dict{Any, Any}}();
mlrefs  = Dict{String, Any}();
for sector_name in vcat([s for (s, _) in SECTOR_PANEL], ["SPY (control)"])
    ticks = sector_name == "SPY (control)" ? ["SPY"] :
            first([ts for (s, ts) in SECTOR_PANEL if s == sector_name]);
    for ticker in ticks
        idx = findfirst(==(ticker), panel_tickers);
        idx === nothing && continue;
        R  = all_R[:, idx];
        ρ̂  = _sample_acf_abs(R, MAXLAG);
        xq = [quantile(R, (i - 0.5) / NQ) for i in 1:NQ];
        Tm, μm, σm, _, _, _ = baum_welch(R, K; max_iter=ML_SEED_MAX_ITER, tol=ML_SEED_TOL);
        π̄m = _stationary_pi_unchecked(Tm);
        mlrefs[ticker] = _fit_metrics(Tm, π̄m, μm, σm, ρ̂, xq, R);
        results[ticker] = Dict{Any, Any}();
        jld = Dict{String, Any}("ticker" => ticker, "K" => K, "rho_target" => ρ̂,
                                "xq" => xq, "scale_s" => SCALE_S,
                                "ml_ref" => (T=Tm, pi=π̄m, mu=μm, sigma=σm));
        for (ai, arm) in enumerate(arms)
            λs = arm === :marginal ? Inf : Float64(arm) * SCALE_S;
            obj = θ -> _joint_objective(θ, ρ̂, xq, λs, K);
            fit = fit_acf_hmm(ρ̂, K; n_starts=N_STARTS, seed=SEED + 1000*ai + idx,
                              R=R, ml_seed=(T=Tm, μ=μm, σ=σm), n_iter=N_ITER,
                              objective=obj);
            met = _fit_metrics(fit.T, fit.π̄, fit.μ, fit.σ, ρ̂, xq, R);
            results[ticker][arm] = (fit=fit, met=met);
            jld["arm_$(_arm_label(arm))"] = Dict(
                "lambda" => _arm_label(arm), "T" => fit.T, "pi" => fit.π̄,
                "mu" => fit.μ, "sigma" => fit.σ, "rho_fit" => met.rho,
                "near" => met.near, "far" => met.far, "acf_sse" => met.sse,
                "cvm" => met.cvm, "kurt" => met.kurt, "mll" => met.mll,
                "qerr" => met.qerr, "diagnostics" => fit.diagnostics,
                "best_start" => fit.best_start, "stop_reason" => String(fit.stop_reason));
        end
        save(joinpath(FRT_DIR, "frontier_$(ticker).jld2"), jld);
        r0 = results[ticker][0.0].met; rM = results[ticker][:marginal].met;
        @printf("  %-6s  acf-only: near %.4f cvm %.2e | marginal-only: near %.4f cvm %.2e | ml: near %.4f cvm %.2e\n",
                ticker, r0.near, r0.cvm, rM.near, rM.cvm,
                mlrefs[ticker].near, mlrefs[ticker].cvm);
    end
end

tickers = sort(collect(keys(results)));

open(joinpath(OUT_DIR, "hmm_acf_frontier.csv"), "w") do io
    println(io, "ticker,arm,near_mae,far_mae,acf_sse,cvm,kurt_model,mll_perobs," *
                "q01_err,q05_err,q95_err,q99_err,best_start,n_iter,stop_reason");
    for t in tickers
        for arm in arms
            r = results[t][arm]; met = r.met; fit = r.fit;
            @printf(io, "%s,%s,%.6f,%.6f,%.6f,%.6e,%.4f,%.5f,%.4f,%.4f,%.4f,%.4f,%d,%d,%s\n",
                    t, _arm_label(arm), met.near, met.far, met.sse, met.cvm, met.kurt,
                    met.mll, met.qerr[1], met.qerr[2], met.qerr[3], met.qerr[4],
                    fit.best_start, fit.n_iter, String(fit.stop_reason));
        end
        met = mlrefs[t];
        @printf(io, "%s,ml_ref,%.6f,%.6f,%.6f,%.6e,%.4f,%.5f,%.4f,%.4f,%.4f,%.4f,0,0,ml\n",
                t, met.near, met.far, met.sse, met.cvm, met.kurt, met.mll,
                met.qerr[1], met.qerr[2], met.qerr[3], met.qerr[4]);
    end
end

# Regrets: per ticker, best-attained value of each metric across arms; per-arm regret
# = metric / best; cross-ticker medians per arm.
reg = Dict{Any, NamedTuple}();
for arm in arms
    ra = Float64[]; rm = Float64[];
    for t in tickers
        best_acf = minimum(results[t][a].met.near for a in arms);
        best_cvm = minimum(results[t][a].met.cvm for a in arms);
        push!(ra, results[t][arm].met.near / best_acf);
        push!(rm, results[t][arm].met.cvm / best_cvm);
    end
    reg[arm] = (acf=median(ra), cvm=median(rm));
end

out_path = joinpath(OUT_DIR, "hmm_acf_frontier.txt");
open(out_path, "w") do io
    println(io, "="^96);
    println(io, "Marginal-vs-ACF Pareto frontier at K = 3: weighted-objective sweep over valid HMMs");
    println(io, "J_lambda = ACF_SSE + lambda * s * CvM; s = SPY-likelihood-fit balance scale (see header).");
    println(io, "="^96);
    @printf(io, "\nBalance scale s = %.4g (SPY single-start likelihood fit: ACF_SSE %.5f / CvM %.3e).\n",
            SCALE_S, ref0.sse, ref0.cvm);
    println(io, "All fits are multistart heuristic optima (achieved points bound the true frontier");
    println(io, "from above); n = $(length(tickers)) tickers, $(N_STARTS) starts per arm, K = $K.");
    println(io);
    println(io, "Cross-ticker medians per arm:");
    println(io, "-"^96);
    @printf(io, "%-10s | %-10s | %-10s | %-10s | %-10s | %-9s | %-9s\n",
            "lambda", "near 1-63", "far", "CvM", "exc.kurt", "regret-A", "regret-M");
    println(io, "-"^96);
    for arm in arms
        meds = [median([getfield(results[t][arm].met, f) for t in tickers])
                for f in (:near, :far, :cvm, :kurt)];
        @printf(io, "%-10s | %-10.4f | %-10.4f | %-10.3e | %-10.2f | %-9.2f | %-9.2f\n",
                _arm_label(arm), meds[1], meds[2], meds[3], meds[4],
                reg[arm].acf, reg[arm].cvm);
    end
    let meds = [median([getfield(mlrefs[t], f) for t in tickers]) for f in (:near, :far, :cvm, :kurt)]
        @printf(io, "%-10s | %-10.4f | %-10.4f | %-10.3e | %-10.2f | %-9s | %-9s\n",
                "ml_ref", meds[1], meds[2], meds[3], meds[4], "-", "-");
    end
    println(io);
    # Declared reading rule.
    max_regrets = [max(reg[arm].acf, reg[arm].cvm) for arm in arms];
    compete = all(r >= 2.0 for r in max_regrets);
    joint_arms = [arm for arm in arms if reg[arm].acf <= 1.5 && reg[arm].cvm <= 1.5];
    println(io, "Reading (rule declared before the run: regret of a fit on an axis = metric / best");
    println(io, "attained across arms per ticker; medians across tickers):");
    @printf(io, "  min over arms of max(median ACF-regret, median CvM-regret) = %.2f\n",
            minimum(max_regrets));
    if !isempty(joint_arms)
        println(io, "  JOINT ATTAINABILITY INDICATED: arm(s) $(join(_arm_label.(joint_arms), ", "))");
        println(io, "  reach both median regrets <= 1.5 - the axes do not strictly compete at K = 3;");
        println(io, "  a valid three-state chain can be close to both single-axis optima at once.");
    elseif compete
        println(io, "  COMPETITION SUPPORTED: every arm has max(median regret) >= 2 - no achieved fit");
        println(io, "  is within a factor 2 of both single-axis optima; subject to the optimizer");
        println(io, "  caveat (achieved points only).");
    else
        println(io, "  INTERMEDIATE: some arm is within a factor 2 of both optima on one reading but");
        println(io, "  not within 1.5 of both; the trade-off is partial. Descriptive only.");
    end
    println(io);
    println(io, "Per-ticker nondominated arms (near-band ACF MAE vs CvM), by ticker:");
    for t in tickers
        pts = [(results[t][a].met.near, results[t][a].met.cvm) for a in arms];
        nd = _nondominated(pts);
        println(io, "  $(rpad(t, 6)): $(join([_arm_label(arms[i]) for i in nd], ", "))");
    end
    println(io);
    println(io, "Full per-arm models (T, pi, mu, sigma, fitted curves, per-start diagnostics) are");
    println(io, "the JLD2 files in results/hmm_acf_frontier/. The marginal log-likelihood column is");
    println(io, "a stationary-mixture marginal-density measure and ignores serial dependence.");
end
println();
println("[done] Wrote $out_path");
