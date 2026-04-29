# ========================================================================================= #
# run_conditional_var.jl
#
# Regime-conditional VaR back-test under one-step-ahead state forecast propagation.
#
# At each OoS time t, run the forward filter through the full history F_t = (IS) ∪
# (OoS up to t-1) under IS-fixed CHMM-N parameters to get the filtered posterior
# Pr(s_t | F_t), then propagate one step to obtain
#
#     Pr(s_{t+1} = k | F_t) = Σ_i Pr(s_t = i | F_t) · T_{ik}.
#
# The conditional predictive density at t+1 is the K-component Gaussian mixture
#
#     f(x | F_t) = Σ_k Pr(s_{t+1} = k | F_t) · N(x; μ_k, σ_k),
#
# and the conditional VaR is its α-quantile, found by binary search on the mixture CDF.
# Run the breach series through Kupiec / Christoffersen-ind / Christoffersen-cc.
#
# Addresses peer-review Tier-1 / B6: the body Kupiec-only panel does not exercise the
# CHMM's regime-switching capability; this conditional variant does.
#
# Output: results/diagnostics/conditional_var.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using LinearAlgebra
using Distributions
using Printf

const SEED      = 20260420;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;

const OUT_DIR = joinpath(_ROOT, "results", "diagnostics");
mkpath(OUT_DIR);

println("="^80);
println("  Regime-conditional VaR back-test  (peer-review B6)");
println("="^80);

# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading SPY IS / OoS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
all_tickers = keys(filtered) |> collect |> sort;
all_R = log_growth_matrix(filtered, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = all_R[:, idx_spy];

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE);

n_is  = length(R_is);
n_oos = length(R_oos);
println("  IS = $n_is  OoS = $n_oos");

# ----------------------------------------------------------------------------------------- #
"""
Filter posterior Pr(s_t | y_{1:t}) for an IS-fixed CHMM-N over the observation series y.
Returns a (length(y)+1) × K matrix `pred` where `pred[t, :]` = Pr(s_t | y_{1:t-1}) is the
one-step-ahead predictive state distribution that knows y_{1:t-1} but NOT y_t. The first
row is π̄ (initial); the (T+1)-th row predicts step T+1.
"""
function filter_predictive(y::AbstractVector, T::AbstractMatrix, μ::AbstractVector,
                            σ::AbstractVector, π_init::AbstractVector)
    K = length(μ);
    n = length(y);
    pred = zeros(n + 1, K);
    pred[1, :] = π_init;
    for t in 1:n
        # filter step: posterior Pr(s_t | y_{1:t}) ∝ pred[t, :] .* b(y_t)
        b = [pdf(Normal(μ[k], σ[k]), y[t]) for k in 1:K];
        post = pred[t, :] .* b;
        Z = sum(post);
        if Z <= 0
            post .= pred[t, :];
        else
            post ./= Z;
        end
        # predict step: Pr(s_{t+1} | y_{1:t}) = post' * T
        pred[t + 1, :] = vec(post' * T);
    end
    return pred;
end

"""
Quantile of a Gaussian mixture by binary search on the CDF over a wide bracket.
"""
function mixture_quantile(α::Float64, weights::AbstractVector, μ::AbstractVector,
                          σ::AbstractVector; lo::Float64 = -50.0, hi::Float64 = 50.0,
                          tol::Float64 = 1e-7, max_iter::Int = 80)
    cdf_at(x) = sum(w * cdf(Normal(μk, σk), x) for (w, μk, σk) in zip(weights, μ, σ));
    a = lo; b = hi;
    fa = cdf_at(a) - α;
    fb = cdf_at(b) - α;
    fa <= 0 || error("lower bracket fails: cdf($a) = $(fa + α) > $α");
    fb >= 0 || error("upper bracket fails: cdf($b) = $(fb + α) < $α");
    for _ in 1:max_iter
        m = (a + b) / 2;
        fm = cdf_at(m) - α;
        if abs(fm) < tol; return m; end
        if fm < 0; a = m; else; b = m; end
    end
    return (a + b) / 2;
end

# ----------------------------------------------------------------------------------------- #
"""
Refit CHMM-N at K and return the IS-fixed parameters plus the predictive state
distributions over the full timeline (IS then OoS), all under those frozen parameters.
"""
function fit_and_filter(R_is::AbstractVector, R_oos::AbstractVector, K::Int)
    println("\n[fit] CHMM-N at K = $K on SPY IS...");
    Random.seed!(SEED);
    chmm = build(MyContinuousHiddenMarkovModel,
                 (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(chmm.transition[i]); end
    π̄ = (T_mat^2000)[1, :];
    μ = zeros(K); σ = zeros(K);
    for k in 1:K
        d = chmm.emission[k];
        # MyContinuousHiddenMarkovModel emissions are Normal(μ, σ)
        μ[k] = mean(d);
        σ[k] = std(d);
    end

    # Filter through IS + OoS as a single timeline; the predictive state distribution at
    # OoS day j is pred[n_is + j, :], using F_{t-1} = IS ∪ {R_oos[1:j-1]}.
    R_full = vcat(R_is, R_oos);
    pred = filter_predictive(R_full, T_mat, μ, σ, π̄);

    return (K=K, T_mat=T_mat, π̄=π̄, μ=μ, σ=σ, pred=pred);
end

function eval_conditional_var(fit, R_oos::AbstractVector, α::Float64;
                               n_is::Int)
    K = fit.K; μ = fit.μ; σ = fit.σ;
    n_o = length(R_oos);
    var_thresholds = zeros(n_o);
    breaches = falses(n_o);
    for j in 1:n_o
        # predictive Pr(s_{t+1} | F_t) where t+1 is the j-th OoS observation; this lives
        # at row n_is + j of `fit.pred` (which holds Pr(s_{t+1} | y_{1:t})).
        w = fit.pred[n_is + j, :];
        var_α = mixture_quantile(α, w, μ, σ);
        var_thresholds[j] = var_α;
        breaches[j] = R_oos[j] < var_α;
    end
    return var_thresholds, breaches;
end

# ----------------------------------------------------------------------------------------- #
panels = NamedTuple[];

for K in (3, 18)
    fit = fit_and_filter(R_is, R_oos, K);

    for α in (0.01, 0.05)
        println("\n[eval] K = $K, α = $α: conditional-VaR back-test...");
        var_thr, br = eval_conditional_var(fit, R_oos, α; n_is=n_is);
        n_br = sum(br);
        br_rate = n_br / length(br);
        med_var = median(var_thr);

        k_uc = kupiec_lr(br, α);
        c_in = christoffersen_lr(br);
        c_cc = christoffersen_cc(br, α);

        println("  breaches = $n_br / $(length(br))   ($(round(100*br_rate, digits=2))%)");
        println("  median VaR threshold = $(round(med_var, digits=3))");
        @printf("  Kupiec   LR_uc = %.3f  (p = %.3f)\n", k_uc.LR, k_uc.pvalue);
        @printf("  Christ.  LR_ind = %.3f (p = %.3f)\n", c_in.LR, c_in.pvalue);
        @printf("  Christ.  LR_cc = %.3f  (p = %.3f)\n", c_cc.LR, c_cc.pvalue);

        push!(panels, (
            K = K, α = α,
            breaches = n_br, breach_rate = br_rate, median_var = med_var,
            LR_uc = k_uc.LR, p_uc = k_uc.pvalue,
            LR_ind = c_in.LR, p_ind = c_in.pvalue,
            LR_cc = c_cc.LR, p_cc = c_cc.pvalue,
        ));
    end
end

# ----------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "conditional_var.txt");
open(out_path, "w") do io
    println(io, "="^110);
    println(io, "Regime-conditional VaR back-test under one-step-ahead state forecast propagation");
    println(io, "(peer-review B6)");
    println(io, "="^110);
    println(io, "Setup: CHMM-N at K = 3 and K = 18 fit on SPY IS (T = $n_is days, seed = $SEED).");
    println(io, "OoS  : T = $n_oos days. Parameters frozen at IS estimate; only the latent state");
    println(io, "       belief updates as each new R_t arrives. At each OoS day t, conditional VaR");
    println(io, "       VaR_α(t) = F^{-1}(α) where F is the K-component Gaussian mixture predictive");
    println(io, "       density at t given F_{t-1} = IS ∪ R_oos[1..t-1] under the forward filter.");
    println(io, "Tests: Kupiec LR_uc (unconditional coverage, ~χ²_1, critical 3.841)");
    println(io, "       Christoffersen LR_ind (breach independence, ~χ²_1, critical 3.841)");
    println(io, "       Christoffersen LR_cc = LR_uc + LR_ind (joint, ~χ²_2, critical 5.991)");
    println(io);
    @printf(io, "%4s | %-5s | %-9s | %-10s | %-11s | %-7s %-6s | %-7s %-6s | %-7s %-6s\n",
            "K","α","breaches","br rate","med VaR","LR_uc","p","LR_ind","p","LR_cc","p");
    println(io, "-"^110);
    for r in panels
        @printf(io, "%4d | %-5.2f | %-9d | %-10.3f | %-11.3f | %-7.3f %-6.3f | %-7.3f %-6.3f | %-7.3f %-6.3f\n",
                r.K, r.α, r.breaches, r.breach_rate, r.median_var,
                r.LR_uc, r.p_uc, r.LR_ind, r.p_ind, r.LR_cc, r.p_cc);
    end
    println(io);
    println(io, "Comparison panel: unconditional pooled-archive VaR for the same windows is in");
    println(io, "results/diagnostics/var_christoffersen.txt and paper Appendix tab:christoffersen_var.");
    println(io, "Both unconditional CHMM-N and GARCH(1,1) reject Christoffersen-ind on OoS at α=0.05;");
    println(io, "the question this panel answers is whether regime forecast propagation closes that gap.");
end

println("\n[done] $out_path");
