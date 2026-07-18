# =========================================================================== #
# run_msgarch_conditional_var.jl
#
# Traced (see CHANGELOG.md):, forecast-origin-aligned MS-GARCH
# conditional-VaR block for the paper's main VaR table. The previous paper
# artefact results/robustness/msgarch_conditional_var.csv had no runner in
# this repository and was computed on 572 OoS forecasts (it omitted the
# boundary return from the last IS session into the first held-out session),
# whereas the CHMM / filtered-bootstrap / CAViaR rows use 573.
#
# This runner:
#   1. Fits MS-GARCH(1,1) (Haas-Mittnik-Paolella 2004, path-independent
#      per-regime recursion) at K in {2, 3, 4} on the SPY IS window. K = 2, 3
#      use the canonical src/MSGARCH.jl fitters; K = 4 replicates the
#      general-K softmax-logit architecture of
#      runners/headline/run_msgarch_higher_k.jl (src/ is not modified).
#   2. Produces one-step-ahead conditional VaR on the SAME 573-forecast
#      origin set as run_conditional_var_all_families.jl (boundary return
#      prepended): a single static IS fit, Hamilton filter propagated through
#      each OoS day, predictive state weights T' * gamma_{t-1}, per-regime
#      conditional sigma^2_{k,t} from the recursion, mixture-Normal quantile
#      at alpha in {0.01, 0.05} by bisection.
#   3. Back-tests each (K, alpha) row: Kupiec LR_uc, Christoffersen LR_ind /
#      LR_cc, Engle-Manganelli DQ (q = 4 lags + intercept + VaR_t, chi^2_6,
#      matching run_engle_manganelli_dq.jl).
#
# Output:
#   results/msgarch_conditional_var/msgarch_conditional_var.{txt,csv}
#   results/msgarch_conditional_var/var_series_K{2,3,4}.csv   (per-day series)
#   results/msgarch_conditional_var/msgarch_params_K{2,3,4}.txt (fitted params,
#       consumed by run_var_inference_upgrade.jl)
#   ../CHMM-Paper-Repository/results/robustness/msgarch_conditional_var.csv
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, LinearAlgebra, Distributions, Printf

const SEED      = 20260420;      # same seed as the CHMM conditional-VaR harness
const DT        = 1/252;
const RISK_FREE = 0.0;
const Q_LAGS    = 4;             # Engle-Manganelli DQ lag count
const K_GRID    = [2, 3, 4];
const ALPHAS    = [0.01, 0.05];

const OUT_DIR              = joinpath(_ROOT, "results", "msgarch_conditional_var");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-Paper-Repository", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^80)
println("  MS-GARCH conditional VaR, K = $K_GRID, aligned 573-forecast origin set")
println("="^80)

# -------- data (identical to run_conditional_var_all_families.jl) --------
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
all_tickers = keys(filtered) |> collect |> sort;
all_R = log_growth_matrix(filtered, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));

# Boundary fix (2026-07 audit): the IS and OoS price sets were differenced separately,
# omitting the last-IS-session -> first-held-out-session return (2024-01-03 -> 2024-01-04).
# Prepend it so the first OoS forecast target is the return into 2024-01-04 and the
# forecast-origin set matches the CHMM harness (n_oos = 573).
R_oos = vcat((1/DT) * log(oos_dataset["SPY"][1, :volume_weighted_average_price] /
                          train_dataset["SPY"][end, :volume_weighted_average_price]) - RISK_FREE,
             R_oos);

n_is = length(R_is); n_oos = length(R_oos);
println("[setup] IS = $n_is, OoS = $n_oos")

# --------------------------------------------------------------------------- #
# General-K MS-GARCH fit for K = 4 (replicates run_msgarch_higher_k.jl:
# softmax-logit transition parameterisation + Nelder-Mead over the joint
# parameter vector, Hamilton filter from src/MSGARCH.jl for the likelihood).
# --------------------------------------------------------------------------- #
function unpack_T(logits::Vector{Float64}, K::Int)::Matrix{Float64}
    @assert length(logits) == K * (K - 1);
    T = zeros(K, K);
    for k in 1:K
        es = zeros(K - 1);
        for l in 1:(K - 1)
            es[l] = exp(logits[(k - 1) * (K - 1) + l]);
        end
        denom = 1.0 + sum(es);
        T[k, k] = 1.0 / denom;
        for l in 1:(K - 1)
            j = ((k + l - 1) % K) + 1;
            T[k, j] = es[l] / denom;
        end
    end
    return T;
end

function nll_msgarch_kg(params::Vector{Float64}, obs::Vector{Float64}, μ::Float64, K::Int)
    @assert length(params) == 3 * K + K * (K - 1);
    ω = params[1:3:(3 * K)];
    α = params[2:3:(3 * K)];
    β = params[3:3:(3 * K)];
    for k in 1:K
        if ω[k] <= 0 || α[k] < 0 || β[k] < 0 || (α[k] + β[k]) >= 0.999
            return 1e10;
        end
    end
    T = unpack_T(params[(3 * K + 1):end], K);
    ll, _, _ = _hamilton_filter_msgarch(obs, ω, α, β, μ, T);
    return isfinite(ll) ? -ll : 1e10;
end

function fit_msgarch_kg(obs::Vector{Float64}, K::Int; max_iter::Int=5000)
    μ_obs = mean(obs); var_obs = var(obs);
    nparams = 3 * K + K * (K - 1);
    fn(p) = nll_msgarch_kg(p, obs, μ_obs, K);

    best_nll = Inf; best_params = zeros(nparams);
    for α_try in [0.05, 0.10]
        for β_try in [0.85, 0.90]
            denom = (1 - α_try - β_try);
            if denom < 0.05; continue; end
            for spread in [2.5, 4.0, 6.0]
                ratios = [spread^((k - (K + 1) / 2) / max(K / 2, 1)) for k in 1:K];
                g = exp(mean(log.(ratios))); ratios ./= g;
                params = zeros(nparams);
                for k in 1:K
                    params[3 * (k - 1) + 1] = var_obs * denom * ratios[k];
                    params[3 * (k - 1) + 2] = α_try;
                    params[3 * (k - 1) + 3] = β_try;
                end
                logit0 = log(0.05 / 0.90);
                for j in 1:(K * (K - 1))
                    params[3 * K + j] = logit0;
                end
                v = fn(params);
                if v < best_nll
                    best_nll = v;
                    best_params = copy(params);
                end
            end
        end
    end

    best, _ = _nelder_mead(fn, best_params; max_iter=max_iter, tol=1e-6);

    ω = best[1:3:(3 * K)];
    α = best[2:3:(3 * K)];
    β = best[3:3:(3 * K)];
    T = unpack_T(best[(3 * K + 1):end], K);
    uv = [ω[k] / max(1 - α[k] - β[k], 1e-6) for k in 1:K];
    order = sortperm(uv);
    ω = ω[order]; α = α[order]; β = β[order];
    P = zeros(K, K);
    for i in 1:K, j in 1:K; P[i, j] = T[order[i], order[j]]; end
    T = P;
    ll, _, _ = _hamilton_filter_msgarch(obs, ω, α, β, μ_obs, T);
    return (K=K, ω=ω, α=α, β=β, μ=μ_obs, T=T, ll=ll);
end

function fit_msgarch_at(obs::Vector{Float64}, K::Int)
    Random.seed!(SEED);   # fits are deterministic (grid + Nelder-Mead); seed kept by convention
    if K == 2
        m = fit_msgarch_k2(obs; max_iter=3000);
        return (K=2, ω=m.ω, α=m.α, β=m.β, μ=m.μ, T=m.T, ll=m.log_likelihood);
    elseif K == 3
        m = fit_msgarch_k3(obs; max_iter=3000);
        return (K=3, ω=m.ω, α=m.α, β=m.β, μ=m.μ, T=m.T, ll=m.log_likelihood);
    else
        return fit_msgarch_kg(obs, K; max_iter=5000);
    end
end

# --------------------------------------------------------------------------- #
# One-step-ahead conditional VaR series on the OoS window.
# Hamilton filter over vcat(R_is, R_oos) under IS-fitted parameters; at each
# OoS day t the predictive state distribution is T' * gamma_{t-1} and the
# per-regime conditional variance sigma^2_{k,t} comes from the path-
# independent recursion (both functions of information through t-1).
# --------------------------------------------------------------------------- #
function mixture_quantile(α::Float64, weights, components;
                          lo::Float64=-50.0, hi::Float64=50.0,
                          tol::Float64=1e-6, max_iter::Int=80)
    cdf_at(x) = sum(w * cdf(c, x) for (w, c) in zip(weights, components));
    a, b = lo, hi;
    if cdf_at(a) - α > 0; a = -200.0; end
    if cdf_at(b) - α < 0; b = 200.0; end
    for _ in 1:max_iter
        m = 0.5 * (a + b);
        fm = cdf_at(m) - α;
        if abs(fm) < tol; return m; end
        if fm < 0; a = m; else; b = m; end
    end
    return 0.5 * (a + b);
end

function msgarch_var_series(model, R_is, R_oos, alphas)
    R_full = vcat(R_is, R_oos);
    _, σ2, γ = _hamilton_filter_msgarch(R_full, model.ω, model.α, model.β, model.μ, model.T);
    n_i = length(R_is); n_o = length(R_oos); K = model.K;
    out = Dict(α => zeros(n_o) for α in alphas);
    for j in 1:n_o
        t = n_i + j;
        π_pred = model.T' * γ[t - 1, :];
        π_pred = max.(π_pred, 0.0); π_pred ./= sum(π_pred);
        comps = [Normal(model.μ, sqrt(σ2[t, k])) for k in 1:K];
        for α in alphas
            out[α][j] = mixture_quantile(α, π_pred, comps);
        end
    end
    return out;
end

# -------- Engle-Manganelli DQ test (mirrors run_engle_manganelli_dq.jl) --------
function dq_test(breaches::AbstractVector{Bool}, var_thr::AbstractVector{Float64},
                 α::Float64; q::Int = Q_LAGS)
    n = length(breaches);
    @assert n == length(var_thr);
    hit = Float64.(breaches) .- α;
    n_eff = n - q;
    p = q + 2;
    X = zeros(n_eff, p);
    Y = zeros(n_eff);
    for t in (q+1):n
        row = t - q;
        X[row, 1] = 1.0;
        for i in 1:q
            X[row, 1 + i] = hit[t - i];
        end
        X[row, p] = var_thr[t];
        Y[row] = hit[t];
    end
    XtX = X' * X;
    Xty = X' * Y;
    if det(XtX) == 0
        return (DQ = NaN, p_value = NaN, dof = p, n_eff = n_eff, singular = true);
    end
    beta_hat = XtX \ Xty;
    DQ = (beta_hat' * XtX * beta_hat) / (α * (1.0 - α));
    p_value = 1.0 - cdf(Chisq(p), DQ);
    return (DQ = DQ, p_value = p_value, dof = p, n_eff = n_eff, singular = false);
end

# --------------------------------------------------------------------------- #
# Run the K grid
# --------------------------------------------------------------------------- #
panels = NamedTuple[];
for K in K_GRID
    println("\n[fit] MS-GARCH K = $K on SPY IS...");
    t0 = time();
    model = fit_msgarch_at(R_is, K);
    println(@sprintf("  fit in %.1f s  ll = %.2f", time() - t0, model.ll));
    println("  uncond sigma per regime: ",
        [round(sqrt(model.ω[k] / max(1 - model.α[k] - model.β[k], 1e-6)), digits=3) for k in 1:K]);
    println("  T diagonal: ", [round(model.T[k, k], digits=3) for k in 1:K]);

    # Persist fitted parameters (consumed by run_var_inference_upgrade.jl)
    open(joinpath(OUT_DIR, "msgarch_params_K$(K).txt"), "w") do io
        println(io, "K $(K)");
        println(io, "ll $(model.ll)");
        println(io, "mu $(model.μ)");
        println(io, "omega " * join(string.(model.ω), " "));
        println(io, "alpha " * join(string.(model.α), " "));
        println(io, "beta " * join(string.(model.β), " "));
        for i in 1:K
            println(io, "T$(i) " * join(string.(model.T[i, :]), " "));
        end
    end

    vars = msgarch_var_series(model, R_is, R_oos, ALPHAS);

    # Per-day series
    br = Dict(α => (R_oos .< vars[α]) for α in ALPHAS);
    open(joinpath(OUT_DIR, "var_series_K$(K).csv"), "w") do io
        println(io, "idx,R_oos,var_05,var_01,breach_05,breach_01");
        for j in 1:n_oos
            @printf(io, "%d,%.5f,%.5f,%.5f,%d,%d\n",
                j, R_oos[j], vars[0.05][j], vars[0.01][j],
                br[0.05][j] ? 1 : 0, br[0.01][j] ? 1 : 0);
        end
    end

    for α in ALPHAS
        var_thr = vars[α];
        breaches = br[α];
        n_br = sum(breaches);
        br_rate = n_br / n_oos;
        med_var = median(var_thr);
        k_uc = kupiec_lr(breaches, α);
        c_in = christoffersen_lr(breaches);
        c_cc = christoffersen_cc(breaches, α);
        dq   = dq_test(breaches, var_thr, α);
        @printf("  K=%d  α=%.2f  br=%2d (%.2f%%)  med VaR=%.3f  LR_uc=%.3f (p=%.3f)  LR_cc=%.3f (p=%.3f)  DQ=%.3f (p=%.3f)\n",
            K, α, n_br, 100*br_rate, med_var, k_uc.LR, k_uc.pvalue,
            c_cc.LR, c_cc.pvalue, dq.DQ, dq.p_value);
        push!(panels, (
            K=K, α=α, breaches=n_br, breach_rate=br_rate, med_var=med_var,
            LR_uc=k_uc.LR, p_uc=k_uc.pvalue,
            LR_ind=c_in.LR, p_ind=c_in.pvalue,
            LR_cc=c_cc.LR, p_cc=c_cc.pvalue,
            DQ=dq.DQ, DQ_p=dq.p_value, DQ_dof=dq.dof,
        ));
    end
end

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
# CSV (column layout matches the paper artefact results/robustness/msgarch_conditional_var.csv)
csv_header = "K,alpha,breaches,breach_rate,median_var,LR_uc,p_uc,LR_ind,p_ind,LR_cc,p_cc,DQ,DQ_p,DQ_dof";
function write_panel_csv(path)
    open(path, "w") do io
        println(io, csv_header);
        for r in panels
            @printf(io, "%d,%.2f,%d,%.5f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                r.K, r.α, r.breaches, r.breach_rate, r.med_var,
                r.LR_uc, r.p_uc, r.LR_ind, r.p_ind, r.LR_cc, r.p_cc,
                r.DQ, r.DQ_p, r.DQ_dof);
        end
    end
end
write_panel_csv(joinpath(OUT_DIR, "msgarch_conditional_var.csv"));
write_panel_csv(joinpath(PAPER_ROBUSTNESS_DIR, "msgarch_conditional_var.csv"));

open(joinpath(OUT_DIR, "msgarch_conditional_var.txt"), "w") do io
    println(io, "MS-GARCH(1,1) conditional VaR back-test on SPY OoS, K in {2, 3, 4}");
    println(io, "  Haas-Mittnik-Paolella 2004 path-independent recursion; static IS fit,");
    println(io, "  Hamilton filter propagated through the OoS window; mixture-Normal quantile.");
    println(io, "  Forecast-origin set aligned with the CHMM harness: OoS T = $n_oos days");
    println(io, "  (boundary return from the last IS session into the first held-out session included).");
    println(io, "  Seed = $SEED. Critical values: chi^2_1(0.05) = 3.841, chi^2_2(0.05) = 5.991,");
    println(io, "  chi^2_6(0.05) = 12.592. DQ: q = $Q_LAGS lagged hits + intercept + VaR_t.");
    println(io, "="^112);
    @printf(io, "%-3s %-5s %-9s %-9s %-9s %-9s %-7s %-9s %-7s %-9s %-7s\n",
        "K", "α", "breaches", "br rate", "med VaR", "LR_uc", "p_uc", "LR_cc", "p_cc", "DQ", "p_DQ");
    println(io, "-"^112);
    for r in panels
        @printf(io, "%-3d %-5.2f %-9d %-9.4f %-9.3f %-9.3f %-7.3f %-9.3f %-7.3f %-9.3f %-7.3f\n",
            r.K, r.α, r.breaches, r.breach_rate, r.med_var,
            r.LR_uc, r.p_uc, r.LR_cc, r.p_cc, r.DQ, r.DQ_p);
    end
    println(io);
    println(io, "Provenance note: replaces the untraced 572-forecast panel previously shipped in the");
    println(io, "paper repository (old 1% rows: breaches 11/11/7, p_cc 0.06/0.06/0.80, p_DQ 0.03/0.03/0.59).");
end

println("\n[done] Output: $OUT_DIR");
println("[done] Paper CSV overwritten: $(joinpath(PAPER_ROBUSTNESS_DIR, "msgarch_conditional_var.csv"))");
