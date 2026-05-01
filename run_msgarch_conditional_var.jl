# ========================================================================================= #
# run_msgarch_conditional_var.jl
#
# MS-GARCH regime-conditional VaR via the same state-filter pipeline as the body
# CHMM regime-conditional VaR (Eq. (9) in var_backtest.tex, applied to MS-GARCH
# state probabilities and per-state conditional GARCH(1,1) variances instead of
# CHMM state probabilities and per-state Gaussian emissions).
#
# Closes peer-review item E7 (MS-GARCH-via-state-filter conditional VaR). Tests
# whether the conditional-coverage value proposition is CHMM-specific or generic
# to multi-state regime-switching models on this dataset.
#
# Pipeline:
#   1. Load IS-fit MSGARCH(1,1)-norm parameters from results/msgarch_reference/
#      models_K{K}.jld2 (Bayesian posterior-mean point estimates from Ardia 2019
#      MSGARCH R package, fitted by run_msgarch_reference.jl).
#   2. Apply a Haas et al. (2004) path-independent MSGARCH state filter to
#      R_full = (R_is, R_oos), holding parameters fixed at IS values and
#      propagating only the per-state conditional variances σ²_{t,k} and the
#      state belief P(s_t = k | F_{t-1}) through the OoS observations.
#   3. At each OoS day j, build the predictive density
#         f(R_t | F_{t-1}) = Σ_k P(s_t = k | F_{t-1}) · N(0, σ²_{t,k}),
#      compute its α-quantile (mixture VaR), and check whether R_oos[j] breaches.
#   4. Run Kupiec, Christoffersen-ind, Christoffersen-cc, Engle-Manganelli DQ on
#      the breach series.
#
# Parameter convention (matches MSGARCH 2.51 sGARCH/norm output):
#   per state k: σ²_{t,k} = α0_k + α1_k · R²_{t-1} + β_k · σ²_{t-1,k}
#   stationary unconditional: σ²_k = α0_k / (1 - α1_k - β_k)  (used for t = 1)
#
# Output:
#   results/diagnostics/msgarch_conditional_var.txt       human-readable
#   ../CHMM-paper/results/robustness/msgarch_conditional_var.csv   paper CSV
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using LinearAlgebra
using Distributions
using Printf
using JLD2
using FileIO

const SEED        = 20260420;
const DT          = 1/252;
const RISK_FREE   = 0.0;
const Q_LAGS_DQ   = 4;
const K_GRID      = (2, 3, 4);                 # the three K values for which we have
                                               # an IS-fit MSGARCH reference model
const ALPHA_GRID  = (0.01, 0.05);

const MSGARCH_DIR = joinpath(_ROOT, "results", "msgarch_reference");
const OUT_DIR     = joinpath(_ROOT, "results", "diagnostics");
const PAPER_DIR   = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_DIR);

println("="^80);
println("  MS-GARCH regime-conditional VaR (peer-review E7)");
println("  Same Eq. (9) state-filter pipeline as run_conditional_var.jl, ");
println("  with MSGARCH(1,1)-norm state forecasts in place of CHMM-N forecasts.");
println("="^80);

# -------------------------------------------------------------------------------------- #
# Data — same convention as run_conditional_var.jl and run_engle_manganelli_dq.jl
# -------------------------------------------------------------------------------------- #
println("\n[setup] Loading SPY IS / OoS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String, DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
all_tickers = keys(filtered) |> collect |> sort;
all_R = log_growth_matrix(filtered, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is  = Vector{Float64}(all_R[:, idx_spy]);
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
n_is  = length(R_is); n_oos = length(R_oos);
println("  IS = $n_is   OoS = $n_oos");

# -------------------------------------------------------------------------------------- #
# MSGARCH parameter extraction from the saved JLD2 model object
# (run_msgarch_reference.jl stores par_post_mean + par_names + transition matrix).
# -------------------------------------------------------------------------------------- #
"""
    load_msgarch_params(K)

Read the IS-fit MSGARCH-K parameters from results/msgarch_reference/models_K{K}.jld2,
parse the (α0, α1, β) triplets per state, and return them alongside the K x K
transition matrix.

Returns NamedTuple with fields :K, :α0, :α1, :β, :T_mat, :π̄.
"""
function load_msgarch_params(K::Int)
    f = joinpath(MSGARCH_DIR, "models_K$(K).jld2");
    isfile(f) || error("missing MSGARCH fit at $f; run run_msgarch_reference.jl first");
    d = load(f);
    par_names    = String.(d["par_names"]);
    par_means    = Float64.(d["par_post_mean"]);
    T_mat        = Matrix{Float64}(d["transition"]);

    α0 = zeros(K); α1 = zeros(K); β = zeros(K);
    for k in 1:K
        for (nm, v) in zip(par_names, par_means)
            if nm == "alpha0_$k";  α0[k] = v;
            elseif nm == "alpha1_$k"; α1[k] = v;
            elseif nm == "beta_$k";   β[k]  = v;
            end
        end
    end
    # Stationary distribution of T (left-eigenvector for eigenvalue 1)
    π̄ = (T_mat^2000)[1, :];
    π̄ ./= sum(π̄);
    return (K=K, α0=α0, α1=α1, β=β, T_mat=T_mat, π̄=π̄);
end

# -------------------------------------------------------------------------------------- #
# Haas et al. (2004) path-independent MS-GARCH state filter.
#
# State k variance recursion (each state has its own σ²_{t,k}, all states see the same
# realised return R_{t-1}):
#     σ²_{t,k} = α0_k + α1_k · R²_{t-1} + β_k · σ²_{t-1,k}
#
# Hidden-state filter (Hamilton 1994 §22):
#     ψ_{t}(k)   = P(s_t = k | F_{t-1})  = Σ_i ψ_{t-1|t-1}(i) · T_{ik}     (predict)
#     L_{t}(k)   = N(R_t; 0, σ²_{t,k})                                      (likelihood)
#     ψ_{t|t}(k) ∝ ψ_{t}(k) · L_{t}(k)                                      (filter)
#
# Predictive density at t given F_{t-1}:
#     f(R_t | F_{t-1}) = Σ_k ψ_{t}(k) · N(R_t; 0, σ²_{t,k})
# This is the Eq. (9) construction the body's CHMM regime-conditional VaR uses, with
# CHMM state probabilities replaced by MS-GARCH state probabilities and CHMM Gaussian
# emissions replaced by MS-GARCH state-conditional GARCH(1,1) Gaussian innovations.
# -------------------------------------------------------------------------------------- #
"""
    msgarch_filter(R, params)

Run the MSGARCH(1,1) state filter through `R` under the IS-fixed parameters in `params`
(returned by `load_msgarch_params`). Returns a NamedTuple with:

  σ²_pred[t, k] : per-state conditional variance σ²_{t,k} given F_{t-1}
                  (i.e. the variance that goes into the predictive density at t).
  ψ_pred[t, k]  : P(s_t = k | F_{t-1}) — the predictive state probability.
  ψ_filt[t, k]  : P(s_t = k | F_{t}).
"""
function msgarch_filter(R::AbstractVector, params)
    K = params.K; α0 = params.α0; α1 = params.α1; β = params.β;
    T_mat = params.T_mat; π̄ = params.π̄;
    n = length(R);

    σ²_pred = zeros(n, K);     # σ²_{t,k} | F_{t-1}
    ψ_pred  = zeros(n, K);     # P(s_t | F_{t-1})
    ψ_filt  = zeros(n, K);     # P(s_t | F_t)

    # Initialisation: at t = 1, no past returns observed, so σ²_{1,k} is the
    # unconditional per-state variance and the predictive state belief is the
    # stationary distribution.
    σ²_curr = α0 ./ max.(1.0 .- α1 .- β, 1e-12);   # unconditional σ²_k
    ψ_pred[1, :] = π̄;

    for t in 1:n
        # Save predictive σ²_{t,k} (used in the predictive density at t)
        σ²_pred[t, :] = σ²_curr;
        # Likelihood per state at R_t given σ²_{t,k}
        L = [pdf(Normal(0.0, sqrt(σ²_curr[k])), R[t]) for k in 1:K];
        post = ψ_pred[t, :] .* L;
        Z = sum(post);
        if Z <= 0 || !isfinite(Z)
            post .= ψ_pred[t, :];
        else
            post ./= Z;
        end
        ψ_filt[t, :] = post;

        # Update per-state variance for t+1 using R_t (Haas 2004: shared past return)
        σ²_curr = α0 .+ α1 .* R[t]^2 .+ β .* σ²_curr;

        # Predict next-step state belief
        if t < n
            ψ_pred[t+1, :] = vec(post' * T_mat);
        end
    end
    return (σ²_pred=σ²_pred, ψ_pred=ψ_pred, ψ_filt=ψ_filt);
end

"""
    mixture_normal_quantile(α, weights, σ; lo, hi, tol, max_iter)

α-quantile of a zero-mean Gaussian mixture Σ_k weights[k] · N(0, σ[k]²).
"""
function mixture_normal_quantile(α::Float64, weights::AbstractVector, σ::AbstractVector;
                                  lo::Float64=-100.0, hi::Float64=100.0,
                                  tol::Float64=1e-7, max_iter::Int=80)
    cdf_at(x) = sum(w * cdf(Normal(0.0, sk), x) for (w, sk) in zip(weights, σ));
    a = lo; b = hi;
    for _ in 1:max_iter
        m = (a + b) / 2;
        fm = cdf_at(m) - α;
        if abs(fm) < tol; return m; end
        if fm < 0; a = m; else; b = m; end
    end
    return (a + b) / 2;
end

# -------------------------------------------------------------------------------------- #
# Engle-Manganelli (2004) DQ test (mirrors run_engle_manganelli_dq.jl)
# -------------------------------------------------------------------------------------- #
function dq_test(breaches::AbstractVector{Bool}, var_thr::AbstractVector{Float64},
                 α::Float64; q::Int = Q_LAGS_DQ)
    n = length(breaches); @assert n == length(var_thr);
    hit = Float64.(breaches) .- α;
    n_eff = n - q;
    p = q + 2;
    X = zeros(n_eff, p);
    Y = zeros(n_eff);
    for t in (q+1):n
        row = t - q;
        X[row, 1] = 1.0;
        for i in 1:q; X[row, 1 + i] = hit[t - i]; end
        X[row, p] = var_thr[t];
        Y[row] = hit[t];
    end
    XtX = X' * X; Xty = X' * Y;
    if det(XtX) == 0
        return (DQ=NaN, p_value=NaN, dof=p, n_eff=n_eff, singular=true);
    end
    β̂ = XtX \ Xty;
    DQ = (β̂' * XtX * β̂) / (α * (1.0 - α));
    pval = 1.0 - cdf(Chisq(p), DQ);
    return (DQ=DQ, p_value=pval, dof=p, n_eff=n_eff, singular=false);
end

# -------------------------------------------------------------------------------------- #
# Run K x α grid
# -------------------------------------------------------------------------------------- #
panels = NamedTuple[];

R_full = vcat(R_is, R_oos);

for K in K_GRID
    println("\n[fit] loading IS-fit MSGARCH-K = $K parameters...");
    params = load_msgarch_params(K);
    @printf("  α0 = %s\n", join([@sprintf("%.4f", x) for x in params.α0], "  "));
    @printf("  α1 = %s\n", join([@sprintf("%.4f", x) for x in params.α1], "  "));
    @printf("  β  = %s\n", join([@sprintf("%.4f", x) for x in params.β],  "  "));
    @printf("  π̄ = %s\n",  join([@sprintf("%.4f", x) for x in params.π̄],  "  "));

    # Filter through R_is then R_oos as a single timeline
    println("  [filter] running MSGARCH state filter through (IS, OoS)...");
    filt = msgarch_filter(R_full, params);

    # OoS slice
    σ²_pred_oos = filt.σ²_pred[(n_is+1):(n_is+n_oos), :];
    ψ_pred_oos  = filt.ψ_pred[(n_is+1):(n_is+n_oos),  :];

    for α in ALPHA_GRID
        println("\n[eval] K = $K, α = $α: regime-conditional VaR back-test...");
        var_thr = zeros(n_oos);
        breaches = falses(n_oos);
        for t in 1:n_oos
            σ_t = sqrt.(σ²_pred_oos[t, :]);
            v = mixture_normal_quantile(α, ψ_pred_oos[t, :], σ_t);
            var_thr[t] = v;
            breaches[t] = R_oos[t] < v;
        end
        n_br = sum(breaches);
        br_rate = n_br / n_oos;
        med_var = median(var_thr);

        k_uc = kupiec_lr(breaches, α);
        c_in = christoffersen_lr(breaches);
        c_cc = christoffersen_cc(breaches, α);
        dq   = dq_test(breaches, var_thr, α; q=Q_LAGS_DQ);

        @printf("  breaches = %d / %d   (%.2f%%)\n", n_br, n_oos, 100*br_rate);
        @printf("  median VaR threshold = %.3f\n", med_var);
        @printf("  Kupiec   LR_uc = %.3f  p = %.3f\n", k_uc.LR, k_uc.pvalue);
        @printf("  Christ.  LR_ind = %.3f p = %.3f\n", c_in.LR, c_in.pvalue);
        @printf("  Christ.  LR_cc  = %.3f p = %.3f\n", c_cc.LR, c_cc.pvalue);
        @printf("  Engle-M. DQ    = %.3f p = %.3f  (chi^2_%d crit @ 5%% = %.3f)\n",
                dq.DQ, dq.p_value, dq.dof, quantile(Chisq(dq.dof), 0.95));

        push!(panels, (
            K=K, α=α, breaches=n_br, breach_rate=br_rate, median_var=med_var,
            LR_uc=k_uc.LR, p_uc=k_uc.pvalue,
            LR_ind=c_in.LR, p_ind=c_in.pvalue,
            LR_cc=c_cc.LR, p_cc=c_cc.pvalue,
            DQ=dq.DQ, DQ_p=dq.p_value, DQ_dof=dq.dof,
        ));
    end
end

# -------------------------------------------------------------------------------------- #
# Output: human-readable
# -------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "msgarch_conditional_var.txt");
open(out_path, "w") do io
    println(io, "="^110);
    println(io, "MS-GARCH regime-conditional VaR back-test (peer-review E7)");
    println(io, "="^110);
    println(io, "Setup");
    println(io, "  Reference fit : MSGARCH 2.51 (Ardia 2019), Bayesian MCMC posterior-mean,");
    println(io, "                  variance = sGARCH, distribution = norm.");
    println(io, "                  Parameters loaded from results/msgarch_reference/models_K{K}.jld2.");
    println(io, "  Filter        : Haas et al. (2004) path-independent MSGARCH state filter,");
    println(io, "                  per-state conditional variance recursion");
    println(io, "                    σ²_{t,k} = α0_k + α1_k R²_{t-1} + β_k σ²_{t-1,k}");
    println(io, "                  with stationary initialisation σ²_{1,k} = α0_k / (1 - α1_k - β_k).");
    println(io, "  Predictive    : f(R_t | F_{t-1}) = Σ_k P(s_t = k | F_{t-1}) · N(0, σ²_{t,k}).");
    println(io, "                  VaR_α(t) = α-quantile of this Gaussian mixture, by binary");
    println(io, "                  search on the mixture CDF.  Same Eq. (9) construction as the");
    println(io, "                  body CHMM regime-conditional VaR (run_conditional_var.jl), with");
    println(io, "                  CHMM state belief / Gaussian emissions replaced by MSGARCH ones.");
    println(io, "  Tests         : Kupiec LR_uc (~chi²_1, 3.841)");
    println(io, "                  Christoffersen LR_ind (~chi²_1, 3.841)");
    println(io, "                  Christoffersen LR_cc  (~chi²_2, 5.991)");
    println(io, "                  Engle-Manganelli DQ at q = $(Q_LAGS_DQ) lags (~chi²_$(Q_LAGS_DQ + 2), $(round(quantile(Chisq(Q_LAGS_DQ + 2), 0.95), digits=2)))");
    println(io);
    println(io, "OoS = $n_oos days, IS = $n_is days, seed = $SEED.");
    println(io);

    @printf(io, "%-3s %-5s %-9s %-9s %-9s %-7s %-6s %-7s %-6s %-7s %-6s %-9s %-7s\n",
            "K", "α", "breaches", "br rate", "med VaR",
            "LR_uc", "p", "LR_ind", "p", "LR_cc", "p", "DQ", "DQ p");
    println(io, "-"^120);
    for r in panels
        @printf(io, "%-3d %-5.2f %-9d %-9.3f %-9.3f %-7.3f %-6.3f %-7.3f %-6.3f %-7.3f %-6.3f %-9.3f %-7.3f\n",
                r.K, r.α, r.breaches, r.breach_rate, r.median_var,
                r.LR_uc, r.p_uc, r.LR_ind, r.p_ind,
                r.LR_cc, r.p_cc, r.DQ, r.DQ_p);
    end
    println(io);
    println(io, "Reading. The body CHMM regime-conditional VaR (Table 6 in var_backtest.tex)");
    println(io, "passes Christoffersen-cc cleanly at α = 0.05 on every (K, family) row, and");
    println(io, "the higher-power Engle-Manganelli DQ test rejects at α = 0.01 only for the");
    println(io, "K = 18 row. The MSGARCH panel above tests whether that conditional-coverage");
    println(io, "behaviour is CHMM-specific or generic to multi-state regime-switching models");
    println(io, "on this dataset, by reusing the same Eq. (9) state-filter pipeline on MSGARCH");
    println(io, "state forecasts. Side-by-side reading: any row where MSGARCH passes Kupiec but");
    println(io, "rejects Christoffersen-ind / -cc indicates that breach clustering is present in");
    println(io, "the MSGARCH regime-conditional construction too, isolating the conditional-");
    println(io, "coverage benefit to the multi-state structure rather than to CHMM specifically.");
end

# -------------------------------------------------------------------------------------- #
# Output: paper CSV
# -------------------------------------------------------------------------------------- #
csv_path = joinpath(PAPER_DIR, "msgarch_conditional_var.csv");
open(csv_path, "w") do io
    println(io, "K,alpha,breaches,breach_rate,median_var,LR_uc,p_uc,LR_ind,p_ind,LR_cc,p_cc,DQ,DQ_p,DQ_dof");
    for r in panels
        @printf(io, "%d,%.2f,%d,%.5f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                r.K, r.α, r.breaches, r.breach_rate, r.median_var,
                r.LR_uc, r.p_uc, r.LR_ind, r.p_ind, r.LR_cc, r.p_cc,
                r.DQ, r.DQ_p, r.DQ_dof);
    end
end

println("\n" * "="^80);
println("  MS-GARCH regime-conditional VaR back-test complete.");
@printf("  Human-readable: %s\n", out_path);
@printf("  Paper CSV     : %s\n", csv_path);
println("="^80);
