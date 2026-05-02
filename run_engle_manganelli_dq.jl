# ========================================================================================= #
# run_engle_manganelli_dq.jl
#
# Engle-Manganelli (2004, JBES) Dynamic Quantile (DQ) test as a higher-power conditional-
# coverage alternative to Christoffersen-cc on the same OoS window. Closes peer-review
# item P2.10 / R3 RE2.
#
# Test specification (Engle-Manganelli 2004, equation 16): regress the centred breach
# indicator on its lags plus the contemporaneous VaR estimate,
#
#     Hit_t - α = β_0 + sum_{i=1}^q β_i (Hit_{t-i} - α) + β_{q+1} VaR_t + u_t,
#
# and reject correct conditional coverage if any β coefficient is non-zero.  Test
# statistic DQ = beta-hat' (X'X) beta-hat / [α(1-α)] ~ chi^2(p) under the null where
# p = q + 2 (lags + intercept + VaR).  We use q = 4 lags as in the original Engle-
# Manganelli paper, giving p = 6.
#
# Run on the same regime-conditional VaR construction as run_conditional_var.jl: refit
# CHMM-N at K in {3, 18} on SPY IS, propagate the forward filter through the OoS
# window under IS-fixed parameters, and compute conditional VaR_t at α in {0.01, 0.05}.
#
# Output:
#   results/diagnostics/engle_manganelli_dq.txt
#   ../CHMM-paper/results/robustness/engle_manganelli_dq.csv
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
const Q_LAGS    = 4;             # number of lagged Hit indicators

const OUT_DIR              = joinpath(_ROOT, "results", "diagnostics");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^80);
println("  Engle-Manganelli (2004) Dynamic Quantile test (P2.10, R3 RE2)");
println("  q = $Q_LAGS lagged Hit indicators + intercept + VaR_t");
println("="^80);

# --------------------------------------------------------------------------------------- #
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
println("  IS = $n_is  OoS = $n_oos");

# --------------------------------------------------------------------------------------- #
# Forward filter + mixture quantile (mirrors run_conditional_var.jl)
# --------------------------------------------------------------------------------------- #
function filter_predictive(y::AbstractVector, T::AbstractMatrix, μ::AbstractVector,
                            σ::AbstractVector, π_init::AbstractVector)
    K = length(μ); n = length(y);
    pred = zeros(n + 1, K);
    pred[1, :] = π_init;
    for t in 1:n
        b = [pdf(Normal(μ[k], σ[k]), y[t]) for k in 1:K];
        post = pred[t, :] .* b;
        Z = sum(post);
        if Z <= 0; post .= pred[t, :]; else; post ./= Z; end
        pred[t + 1, :] = vec(post' * T);
    end
    return pred;
end

function mixture_quantile(α::Float64, weights::AbstractVector, μ::AbstractVector,
                          σ::AbstractVector; lo=-50.0, hi=50.0, tol=1e-7, max_iter=80)
    cdf_at(x) = sum(w * cdf(Normal(μk, σk), x) for (w, μk, σk) in zip(weights, μ, σ));
    a = lo; b = hi;
    for _ in 1:max_iter
        m = (a + b) / 2;
        fm = cdf_at(m) - α;
        if abs(fm) < tol; return m; end
        if fm < 0; a = m; else; b = m; end
    end
    return (a + b) / 2;
end

function fit_and_filter(R_is, R_oos, K)
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
        μ[k] = mean(d); σ[k] = std(d);
    end
    R_full = vcat(R_is, R_oos);
    pred = filter_predictive(R_full, T_mat, μ, σ, π̄);
    return (K=K, T_mat=T_mat, π̄=π̄, μ=μ, σ=σ, pred=pred);
end

function eval_conditional_var(fit, R_oos, α; n_is)
    μ = fit.μ; σ = fit.σ;
    n_o = length(R_oos);
    var_thr = zeros(n_o);
    breaches = falses(n_o);
    for j in 1:n_o
        w = fit.pred[n_is + j, :];
        v = mixture_quantile(α, w, μ, σ);
        var_thr[j] = v;
        breaches[j] = R_oos[j] < v;
    end
    return var_thr, breaches;
end

# --------------------------------------------------------------------------------------- #
# Engle-Manganelli (2004) DQ test
# Hit_t = β_0 + sum_{i=1}^q β_i Hit_{t-i} + β_{q+1} VaR_t + u_t
# DQ statistic: DQ = beta-hat' (X'X) beta-hat / [α(1-α)] ~ chi^2(p) under H0
# --------------------------------------------------------------------------------------- #
function dq_test(breaches::AbstractVector{Bool}, var_thr::AbstractVector{Float64},
                 α::Float64; q::Int = Q_LAGS)
    n = length(breaches);
    @assert n == length(var_thr);
    # Centred Hit: Hit_t - α
    hit = Float64.(breaches) .- α;
    # Build regression on t = (q+1)..n
    n_eff = n - q;
    p = q + 2;                      # intercept + q lags + VaR_t
    X = zeros(n_eff, p);
    Y = zeros(n_eff);
    for t in (q+1):n
        row = t - q;
        X[row, 1] = 1.0;            # intercept
        for i in 1:q
            X[row, 1 + i] = hit[t - i];
        end
        X[row, p] = var_thr[t];     # contemporaneous VaR
        Y[row] = hit[t];
    end
    # OLS: beta = (X'X)^{-1} X'Y
    XtX = X' * X;
    Xty = X' * Y;
    if det(XtX) == 0
        return (DQ = NaN, p_value = NaN, dof = p, n_eff = n_eff,
                singular = true);
    end
    beta_hat = XtX \ Xty;
    # Engle-Manganelli statistic: beta-hat' (X'X) beta-hat / [α(1-α)]
    DQ = (beta_hat' * XtX * beta_hat) / (α * (1.0 - α));
    p_value = 1.0 - cdf(Chisq(p), DQ);
    return (DQ = DQ, p_value = p_value, dof = p, n_eff = n_eff,
            singular = false);
end

# --------------------------------------------------------------------------------------- #
# Run K x α grid
# --------------------------------------------------------------------------------------- #
panels = NamedTuple[];

for K in (3, 18)
    fit = fit_and_filter(R_is, R_oos, K);
    for α in (0.01, 0.05)
        println("\n[eval] K = $K, α = $α");
        var_thr, br = eval_conditional_var(fit, R_oos, α; n_is = n_is);
        n_br = sum(br);
        br_rate = n_br / length(br);

        # Christoffersen-cc for cross-reference
        c_cc = christoffersen_cc(br, α);

        # DQ test
        dq = dq_test(br, var_thr, α; q = Q_LAGS);

        @printf("  breaches = %d / %d  (%.2f%%)\n", n_br, length(br), 100*br_rate);
        @printf("  Christoffersen-cc  LR = %.3f  p = %.3f\n", c_cc.LR, c_cc.pvalue);
        @printf("  Engle-Manganelli DQ = %.3f  p = %.3f  (chi^2_%d critical at 5%% = %.3f)\n",
                dq.DQ, dq.p_value, dq.dof, quantile(Chisq(dq.dof), 0.95));

        push!(panels, (
            K = K, α = α, q_lags = Q_LAGS,
            breaches = n_br, breach_rate = br_rate,
            cc_LR = c_cc.LR, cc_p = c_cc.pvalue,
            DQ = dq.DQ, DQ_p = dq.p_value, DQ_dof = dq.dof,
        ));
    end
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "engle_manganelli_dq.txt");
open(out_path, "w") do io
    println(io, "="^110);
    println(io, "Engle-Manganelli (2004) DQ test on the regime-conditional VaR back-test");
    println(io, "Higher-power conditional-coverage alternative to Christoffersen-cc (P2.10 / R3 RE2)");
    println(io, "="^110);
    println(io);
    println(io, "Specification: Hit_t - α = β_0 + sum_{i=1}^q β_i (Hit_{t-i} - α) + β_{q+1} VaR_t + u_t");
    println(io, "Test statistic: DQ = β̂' (X'X) β̂ / [α(1-α)] ~ chi^2($(Q_LAGS + 2)) under H0");
    println(io, "(q = $Q_LAGS lagged hits, intercept, contemporaneous VaR_t; matches Engle-Manganelli 2004 eq.\\ 16)");
    println(io);
    @printf(io, "%-3s  %-5s  %-8s  %-7s  %-9s  %-9s  %-9s  %-9s\n",
            "K", "α", "breach%", "n_br", "cc LR", "cc p", "DQ", "DQ p");
    println(io, "-"^90);
    for r in panels
        @printf(io, "%-3d  %-5.2f  %-8.2f  %-7d  %-9.3f  %-9.3f  %-9.3f  %-9.3f\n",
                r.K, r.α, 100*r.breach_rate, r.breaches,
                r.cc_LR, r.cc_p, r.DQ, r.DQ_p);
    end
    println(io);
    println(io, "Reading: each row reports both the Christoffersen-cc statistic and the");
    println(io, "Engle-Manganelli DQ statistic on the same breach series. Reject correct");
    println(io, "conditional coverage if p < 0.05 under either test. The DQ test has higher");
    println(io, "power against longer-memory or quantile-dependent miscalibration than the");
    println(io, "Christoffersen-cc Markov-chain alternative.");
end

csv_path = joinpath(PAPER_ROBUSTNESS_DIR, "engle_manganelli_dq.csv");
open(csv_path, "w") do io
    println(io, "K,alpha,q_lags,breaches,breach_rate,cc_LR,cc_p,DQ,DQ_p,DQ_dof");
    for r in panels
        @printf(io, "%d,%.2f,%d,%d,%.5f,%.4f,%.4f,%.4f,%.4f,%d\n",
                r.K, r.α, r.q_lags, r.breaches, r.breach_rate,
                r.cc_LR, r.cc_p, r.DQ, r.DQ_p, r.DQ_dof);
    end
end

println("\n" * "="^80);
println("  Engle-Manganelli DQ test complete.");
@printf("  Human-readable: %s\n", out_path);
@printf("  Paper CSV     : %s\n", csv_path);
println("="^80);
