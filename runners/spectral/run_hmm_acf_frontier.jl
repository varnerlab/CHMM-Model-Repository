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
# (computed once at runner start, printed, and stored in the artifact). The scale s is
# a declared constant that DEFINES the arm objectives; it is deliberately kept at its
# original single-start provenance so the arm fits stay deterministic across reruns
# (changing s would change every weighted objective). Arms:
# lambda in {0, 0.1, 0.3, 1, 3, 10, 30, 100} plus a pure-marginal arm (CvM only).
# Six starts per arm (sticky quantile, single-start Baum-Welch likelihood seed, four
# random persistence-diverse). Per-arm metrics: near/far ACF MAE, CvM, model excess
# kurtosis, stationary-mixture marginal log-likelihood per observation (marginal-density
# fit; ignores dependence), and tail-quantile errors at q in {0.01, 0.05, 0.95, 0.99}.
#
# Comparators (two, labeled):
#   ml_ref   - the internal single-start Baum-Welch reference (cap 1,000 iterations):
#              the seed used inside the optimizer and the source of the scale s.
#   ml_multi - the PUBLISHED converged multistart likelihood fit: the exact fit of
#              run_spectral_rank_cross_ticker.jl, recomputed deterministically with the
#              same constants (3 starts at K = 3, max_iter 4,000, tol 1e-4, seed
#              SEED + 100*idx + K on the identical data path) and evaluated on the same
#              metrics. Paper comparisons quote ml_multi; ml_ref is internal.
#
# Reading rules: the ORIGINAL rule (declared before the first run) takes cross-ticker
# medians per arm of the per-ticker regrets (regret of a fit on an axis = metric / best
# value of that metric attained across all arms of that ticker):
#   - the axes COMPETE at K = 3 when every arm has max(ACF-regret, CvM-regret) >= 2
#     at the median (no arm is within a factor 2 of both single-axis optima);
#   - JOINT ATTAINABILITY is indicated when some arm has both median regrets <= 1.5.
# Two separate medians need not be attained by the same tickers, so the PRIMARY panel
# summary (added at the ninth review; stricter, per-ticker joint statistic) is:
#   - per arm, the median over tickers of max(ACF-regret, CvM-regret), and the count
#     of tickers with BOTH regrets <= 1.5; joint attainability requires some arm's
#     median per-ticker max regret <= 1.5.
# Anything between is reported descriptively. All fits are achieved feasible points
# from capped multistart heuristic searches (most weighted-arm winners hit the
# iteration cap; per-arm stop-reason counts are in the artifact): achieved points
# bound the true frontier from above (toward worse values), so a jointly good achieved
# fit is conclusive for attainability, while an empty middle is evidence subject to
# the optimizer caveat.
#
# Output: results/diagnostics/hmm_acf_frontier.csv        (ticker x arm + ml_ref, ml_multi)
#         results/diagnostics/hmm_acf_frontier_regret.csv (per-ticker per-arm regrets)
#         results/diagnostics/hmm_acf_frontier.txt        (medians, stop counts, reading)
#         results/hmm_acf_frontier/frontier_<ticker>.jld2 (all arm winners + comparators)
#
# `julia run_hmm_acf_frontier.jl --summary-only` rebuilds the txt and the regret CSV
# from the main CSV (single source of truth), with no refits.
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
include(joinpath(@__DIR__, "spectral_common.jl"));
include(joinpath(@__DIR__, "acf_capacity_common.jl"));

using Printf, LinearAlgebra, Statistics, Random, JLD2

const SUMMARY_ONLY = "--summary-only" in ARGS;   # rebuild txt + regret CSV, no refits

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

# Published multistart likelihood fit, mirrored from run_spectral_rank_cross_ticker.jl
# (N_STARTS[3] = 3, MAX_ITER = 4000, TOL = 1e-4, seed = SEED + 100*idx + K).
const ML_MULTI_N_STARTS = 3;
const ML_MULTI_MAX_ITER = 4000;
const ML_MULTI_TOL      = 1e-4;

const OUT_DIR  = joinpath(_ROOT, "results", "diagnostics");
const FRT_DIR  = joinpath(_ROOT, "results", "hmm_acf_frontier");
mkpath(OUT_DIR);
mkpath(FRT_DIR);

const ARM_LABELS = ["0", "0.1", "0.3", "1", "3", "10", "30", "100", "marginal"];
const JOINT_REGRET_THRESHOLD = 1.5;

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
# Summary layer: everything below the main CSV is derived from it (single source of
# truth), so `--summary-only` can rebuild the txt and regret CSV without refits.
# ----------------------------------------------------------------------------------------- #

function _read_frontier_rows(csv_path::String)
    rows = NamedTuple[];
    for ln in readlines(csv_path)[2:end]
        p = split(ln, ',');
        push!(rows, (ticker=String(p[1]), arm=String(p[2]),
                     near=parse(Float64, p[3]), far=parse(Float64, p[4]),
                     sse=parse(Float64, p[5]), cvm=parse(Float64, p[6]),
                     kurt=parse(Float64, p[7]), mll=parse(Float64, p[8]),
                     q01=parse(Float64, p[9]), q05=parse(Float64, p[10]),
                     q95=parse(Float64, p[11]), q99=parse(Float64, p[12]),
                     stop=String(p[15])));
    end
    return rows;
end

function _write_summary()
    csv_path = joinpath(OUT_DIR, "hmm_acf_frontier.csv");
    rows = _read_frontier_rows(csv_path);
    byta = Dict((r.ticker, r.arm) => r for r in rows);
    tickers = sort(unique([r.ticker for r in rows]));
    n_t = length(tickers);

    # Balance scale as printed: recovered from the SPY ml_ref row (the runner's exact
    # SCALE_S is stored in every JLD2; the CSV-derived value is identical to the
    # printed precision).
    sref = byta[("SPY", "ml_ref")];
    s_print = sref.sse / sref.cvm;

    # Per-ticker per-arm regrets over the nine sweep arms (comparator rows excluded
    # from the best-attained denominators).
    regrows = NamedTuple[];
    reg = Dict{String, NamedTuple}();
    for arm in ARM_LABELS
        ra = Float64[]; rm = Float64[]; mx = Float64[]; nboth = 0;
        for t in tickers
            best_acf = minimum(byta[(t, a)].near for a in ARM_LABELS);
            best_cvm = minimum(byta[(t, a)].cvm for a in ARM_LABELS);
            r = byta[(t, arm)];
            a_r = r.near / best_acf; c_r = r.cvm / best_cvm;
            both = (a_r <= JOINT_REGRET_THRESHOLD && c_r <= JOINT_REGRET_THRESHOLD);
            push!(ra, a_r); push!(rm, c_r); push!(mx, max(a_r, c_r));
            nboth += both ? 1 : 0;
            push!(regrows, (ticker=t, arm=arm, acf_regret=a_r, cvm_regret=c_r,
                            max_regret=max(a_r, c_r), both=both));
        end
        reg[arm] = (acf=median(ra), cvm=median(rm), med_max=median(mx), n_both=nboth);
    end

    open(joinpath(OUT_DIR, "hmm_acf_frontier_regret.csv"), "w") do io
        println(io, "ticker,arm,acf_regret,cvm_regret,max_regret,both_le_1p5");
        for r in regrows
            @printf(io, "%s,%s,%.4f,%.4f,%.4f,%d\n",
                    r.ticker, r.arm, r.acf_regret, r.cvm_regret, r.max_regret,
                    r.both ? 1 : 0);
        end
    end

    out_path = joinpath(OUT_DIR, "hmm_acf_frontier.txt");
    open(out_path, "w") do io
        println(io, "="^96);
        println(io, "Marginal-vs-ACF Pareto frontier at K = 3: weighted-objective sweep over valid HMMs");
        println(io, "J_lambda = ACF_SSE + lambda * s * CvM; s = SPY-likelihood-fit balance scale (see header).");
        println(io, "="^96);
        @printf(io, "\nBalance scale s = %.4g (SPY single-start likelihood fit: ACF_SSE %.5f / CvM %.3e).\n",
                s_print, sref.sse, sref.cvm);
        println(io, "All fits are achieved feasible points from capped multistart heuristic searches");
        println(io, "(achieved points bound the true frontier from above); n = $(n_t) tickers,");
        println(io, "$(N_STARTS) starts per arm, K = $K. Comparators: ml_ref = internal single-start");
        println(io, "Baum-Welch reference (optimizer seed and scale source); ml_multi = published");
        println(io, "converged multistart likelihood fit (run_spectral_rank_cross_ticker.jl constants,");
        println(io, "recomputed deterministically). Paper comparisons quote ml_multi.");
        println(io);
        println(io, "Cross-ticker medians per arm:");
        println(io, "-"^96);
        @printf(io, "%-10s | %-10s | %-10s | %-10s | %-10s | %-9s | %-9s\n",
                "lambda", "near 1-63", "far", "CvM", "exc.kurt", "regret-A", "regret-M");
        println(io, "-"^96);
        for arm in vcat(ARM_LABELS, ["ml_ref", "ml_multi"])
            meds = [median([getfield(byta[(t, arm)], f) for t in tickers])
                    for f in (:near, :far, :cvm, :kurt)];
            if haskey(reg, arm)
                @printf(io, "%-10s | %-10.4f | %-10.4f | %-10.3e | %-10.2f | %-9.2f | %-9.2f\n",
                        arm, meds[1], meds[2], meds[3], meds[4],
                        reg[arm].acf, reg[arm].cvm);
            else
                @printf(io, "%-10s | %-10.4f | %-10.4f | %-10.3e | %-10.2f | %-9s | %-9s\n",
                        arm, meds[1], meds[2], meds[3], meds[4], "-", "-");
            end
        end
        println(io);
        println(io, "Comparator medians (marginal-density and tail axes):");
        for arm in ["ml_ref", "ml_multi"]
            meds = [median([getfield(byta[(t, arm)], f) for t in tickers])
                    for f in (:mll, :q01, :q99)];
            @printf(io, "  %-8s : mll/obs %-9.4f | q01 err %-7.4f | q99 err %-7.4f\n",
                    arm, meds[1], meds[2], meds[3]);
        end
        let meds = [median([getfield(byta[(t, "0.1")], f) for t in tickers])
                    for f in (:mll, :q01, :q99)]
            @printf(io, "  %-8s : mll/obs %-9.4f | q01 err %-7.4f | q99 err %-7.4f\n",
                    "0.1", meds[1], meds[2], meds[3]);
        end
        println(io);
        println(io, "Optimizer stop reasons per arm (stall / iter_cap of $(N_ITER)); stall is an");
        println(io, "objective-stall early stop, NOT a first-order stationarity certificate:");
        for arm in ARM_LABELS
            ns = count(t -> byta[(t, arm)].stop == "stall", tickers);
            nc = count(t -> byta[(t, arm)].stop == "iter_cap", tickers);
            @printf(io, "  lambda %-8s : %2d stalled, %2d hit the iteration cap\n", arm, ns, nc);
        end
        println(io);
        println(io, "Per-ticker joint regret (PRIMARY panel summary; added at the ninth review -");
        println(io, "the same ticker must be good on both axes):");
        println(io, "-"^96);
        @printf(io, "%-10s | %-24s | %-28s\n",
                "lambda", "median max(regret)", "tickers with both <= $(JOINT_REGRET_THRESHOLD)");
        println(io, "-"^96);
        for arm in ARM_LABELS
            @printf(io, "%-10s | %-24.4f | %d / %d\n", arm, reg[arm].med_max,
                    reg[arm].n_both, n_t);
        end
        println(io);
        # Reading: primary per-ticker joint rule, then the original two-median rule.
        joint_primary = [arm for arm in ARM_LABELS
                         if reg[arm].med_max <= JOINT_REGRET_THRESHOLD];
        max_regrets = [max(reg[arm].acf, reg[arm].cvm) for arm in ARM_LABELS];
        joint_arms = [arm for arm in ARM_LABELS
                      if reg[arm].acf <= 1.5 && reg[arm].cvm <= 1.5];
        compete = all(r >= 2.0 for r in max_regrets);
        println(io, "Reading (regret of a fit on an axis = metric / best attained across arms per");
        println(io, "ticker):");
        println(io, "PRIMARY (per-ticker joint statistic):");
        if !isempty(joint_primary)
            best_arm = joint_primary[argmin([reg[a].med_max for a in joint_primary])];
            @printf(io, "  JOINT ATTAINABILITY INDICATED: arm(s) %s have median per-ticker\n",
                    join(joint_primary, ", "));
            @printf(io, "  max(ACF-regret, CvM-regret) <= %.1f; best arm %s at %.2f with %d/%d\n",
                    JOINT_REGRET_THRESHOLD, best_arm, reg[best_arm].med_max,
                    reg[best_arm].n_both, n_t);
            println(io, "  tickers within the threshold on BOTH axes - the typical ticker is close");
            println(io, "  to both single-axis optima with one and the same chain.");
        else
            println(io, "  no arm reaches median per-ticker max regret <= $(JOINT_REGRET_THRESHOLD); see the");
            println(io, "  original rule below and the per-arm table above (descriptive).");
        end
        println(io, "ORIGINAL rule (two cross-ticker medians; declared before the first run):");
        @printf(io, "  min over arms of max(median ACF-regret, median CvM-regret) = %.2f\n",
                minimum(max_regrets));
        if !isempty(joint_arms)
            println(io, "  JOINT ATTAINABILITY INDICATED: arm(s) $(join(joint_arms, ", "))");
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
            pts = [(byta[(t, a)].near, byta[(t, a)].cvm) for a in ARM_LABELS];
            nd = _nondominated(pts);
            println(io, "  $(rpad(t, 6)): $(join([ARM_LABELS[i] for i in nd], ", "))");
        end
        println(io);
        println(io, "Full per-arm models (T, pi, mu, sigma, fitted curves, per-start diagnostics)");
        println(io, "and both comparators' parameters are the JLD2 files in");
        println(io, "results/hmm_acf_frontier/. The marginal log-likelihood column is a");
        println(io, "stationary-mixture marginal-density measure and ignores serial dependence.");
    end
    println();
    println("[done] Wrote $out_path");
end

# ----------------------------------------------------------------------------------------- #
if !SUMMARY_ONLY

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
# so lambda = 1 weighs the two terms comparably at that reference point. Kept at the
# original single-start provenance deliberately: s defines the arm objectives, and the
# arms must stay byte-deterministic across reruns (see header).
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
mlmulti = Dict{String, Any}();
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
        # Published converged multistart likelihood fit (the paper's comparator),
        # recomputed with the exact run_spectral_rank_cross_ticker.jl constants.
        Tmm, μmm, σmm, _, _, _, diags_mm = baum_welch_multistart(R, K;
            n_starts=ML_MULTI_N_STARTS, max_iter=ML_MULTI_MAX_ITER, tol=ML_MULTI_TOL,
            seed=SEED + 100 * idx + K);
        π̄mm = _stationary_pi_unchecked(Tmm);
        d_mm = diags_mm[argmax([d.ll for d in diags_mm])];
        mlmulti[ticker] = (met=_fit_metrics(Tmm, π̄mm, μmm, σmm, ρ̂, xq, R),
                          T=Tmm, π̄=π̄mm, μ=μmm, σ=σmm, diags=diags_mm, best=d_mm);
        results[ticker] = Dict{Any, Any}();
        jld = Dict{String, Any}("ticker" => ticker, "K" => K, "rho_target" => ρ̂,
                                "xq" => xq, "scale_s" => SCALE_S,
                                "ml_ref" => (T=Tm, pi=π̄m, mu=μm, sigma=σm));
        let mm = mlmulti[ticker]
            jld["ml_multi"] = Dict(
                "T" => mm.T, "pi" => mm.π̄, "mu" => mm.μ, "sigma" => mm.σ,
                "rho_fit" => mm.met.rho, "near" => mm.met.near, "far" => mm.met.far,
                "acf_sse" => mm.met.sse, "cvm" => mm.met.cvm, "kurt" => mm.met.kurt,
                "mll" => mm.met.mll, "qerr" => mm.met.qerr,
                "n_starts" => ML_MULTI_N_STARTS, "seed" => SEED + 100 * idx + K,
                "ll_best" => mm.best.ll, "diagnostics" => mm.diags);
        end
        for (ai, arm) in enumerate(arms)
            λs = arm === :marginal ? Inf : Float64(arm) * SCALE_S;
            obj = θ -> _joint_objective(θ, ρ̂, xq, λs, K);
            fit = fit_acf_hmm(ρ̂, K; n_starts=N_STARTS, seed=SEED + 1000*ai + idx,
                              R=R, ml_seed=(T=Tm, μ=μm, σ=σm), n_iter=N_ITER,
                              objective=obj);
            met = _fit_metrics(fit.T, fit.π̄, fit.μ, fit.σ, ρ̂, xq, R);
            results[ticker][arm] = (fit=fit, met=met);
            jld["arm_$(_arm_label(arm))"] = Dict(
                "lambda" => _arm_label(arm), "lambda_scaled" => λs,
                "T" => fit.T, "pi" => fit.π̄,
                "mu" => fit.μ, "sigma" => fit.σ, "rho_fit" => met.rho,
                "near" => met.near, "far" => met.far, "acf_sse" => met.sse,
                "cvm" => met.cvm, "kurt" => met.kurt, "mll" => met.mll,
                "qerr" => met.qerr, "diagnostics" => fit.diagnostics,
                "objective_value" => fit.objective_value,
                "best_start" => fit.best_start, "stop_reason" => String(fit.stop_reason));
        end
        save(joinpath(FRT_DIR, "frontier_$(ticker).jld2"), jld);
        r0 = results[ticker][0.0].met; rM = results[ticker][:marginal].met;
        @printf("  %-6s  acf-only: near %.4f cvm %.2e | marginal-only: near %.4f cvm %.2e | ml_multi: near %.4f cvm %.2e\n",
                ticker, r0.near, r0.cvm, rM.near, rM.cvm,
                mlmulti[ticker].met.near, mlmulti[ticker].met.cvm);
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
        mm = mlmulti[t]; met = mm.met;
        @printf(io, "%s,ml_multi,%.6f,%.6f,%.6f,%.6e,%.4f,%.5f,%.4f,%.4f,%.4f,%.4f,%d,%d,ml_multi\n",
                t, met.near, met.far, met.sse, met.cvm, met.kurt, met.mll,
                met.qerr[1], met.qerr[2], met.qerr[3], met.qerr[4],
                mm.best.start, mm.best.n_evals);
    end
end

end  # !SUMMARY_ONLY

_write_summary();
