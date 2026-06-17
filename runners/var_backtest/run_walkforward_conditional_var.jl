# ========================================================================================= #
# run_walkforward_conditional_var.jl
#
# T2.1 (REVISION_PLAN_V5_TO_ACCEPT.md): Walk-forward extension of the regime-conditional
# VaR back-test. Six folds × K ∈ {3, 18} × α ∈ {0.01, 0.05} for CHMM-N (24 rows). At
# each fold we refit CHMM-N from scratch on the train slice, run the forward filter
# through the test slice under fold-IS-fixed parameters, and back-test the conditional
# VaR via Kupiec / Christoffersen-ind / Christoffersen-cc on the fold's test breaches.
#
# Each row reports breaches, breach rate, median VaR, LR_uc, LR_ind, LR_cc, p_cc.
#
# The runner reuses
#   * the fold structure of `run_walkforward_oos.jl`, and
#   * the conditional-VaR construction (`filter_predictive`, `mixture_quantile`) from
#     `run_conditional_var.jl`, lifted verbatim so the two diagnostics agree on the
#     headline OoS window when the test slice is W6/2024.
#
# Output:
#   results/walkforward/walkforward_conditional_var.csv
#   results/walkforward/walkforward_conditional_var.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random
using Statistics
using LinearAlgebra
using Distributions
using Printf
using Dates

const SEED      = 20260420;
const K_VALUES  = [3, 18];
const ALPHAS    = [0.01, 0.05];
const MAX_ITER  = 60;
const RISK_FREE = 0.0;
const DT        = 1/252;
const TICKER    = "SPY";

const OUT_DIR = joinpath(_ROOT, "results", "walkforward");
mkpath(OUT_DIR);

# Same fold calendar as run_walkforward_oos.jl
const FOLDS = [
    ("W1", Date(2014,1,1), Date(2019,1,1), Date(2019,1,1), Date(2020,1,1)),
    ("W2", Date(2015,1,1), Date(2020,1,1), Date(2020,1,1), Date(2021,1,1)),
    ("W3", Date(2016,1,1), Date(2021,1,1), Date(2021,1,1), Date(2022,1,1)),
    ("W4", Date(2017,1,1), Date(2022,1,1), Date(2022,1,1), Date(2023,1,1)),
    ("W5", Date(2018,1,1), Date(2023,1,1), Date(2023,1,1), Date(2024,1,1)),
    ("W6", Date(2019,1,1), Date(2024,1,1), Date(2024,1,1), Date(2025,1,1)),
];

println("="^80)
println("  Walk-forward regime-conditional VaR back-test  [peer-review V5 T2.1]")
println("  CHMM-N at K = $K_VALUES  α = $ALPHAS  $(length(FOLDS)) folds, seed = $SEED")
println("="^80)

# ----------------------------------------------------------------------------------------- #
# Data loading (mirrors run_walkforward_oos.jl)
# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading SPY full timeline...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
df_train = train_dataset[TICKER];
df_oos   = oos_dataset[TICKER];
df_full  = vcat(df_train, df_oos; cols=:orderequal);
sort!(df_full, :timestamp);
println("  $(TICKER) full timeline: $(df_full[1, :timestamp]) → $(df_full[end, :timestamp]) ($(nrow(df_full)) rows)")

function _slice_log_returns(df::DataFrame, t_start::Date, t_end_excl::Date; Δt::Float64=DT)
    ts = Date.(df.timestamp);
    mask = (ts .>= t_start) .& (ts .< t_end_excl);
    sub = df[mask, :];
    P = sub.volume_weighted_average_price;
    return Vector{Float64}((1 / Δt) .* (log.(P[2:end] ./ P[1:end-1])));
end

# ----------------------------------------------------------------------------------------- #
# Conditional VaR helpers (verbatim from run_conditional_var.jl for consistency)
# ----------------------------------------------------------------------------------------- #
"""
Filter posterior Pr(s_t | y_{1:t}) for an IS-fixed CHMM-N over the observation series y.
Returns a (length(y)+1) × K matrix `pred` where `pred[t, :]` = Pr(s_t | y_{1:t-1}) is the
one-step-ahead predictive state distribution.
"""
function filter_predictive(y::AbstractVector, T::AbstractMatrix, μ::AbstractVector,
                            σ::AbstractVector, π_init::AbstractVector)
    K = length(μ);
    n = length(y);
    pred = zeros(n + 1, K);
    pred[1, :] = π_init;
    for t in 1:n
        b = [pdf(Normal(μ[k], σ[k]), y[t]) for k in 1:K];
        post = pred[t, :] .* b;
        Z = sum(post);
        if Z <= 0
            post .= pred[t, :];
        else
            post ./= Z;
        end
        pred[t + 1, :] = vec(post' * T);
    end
    return pred;
end

function mixture_quantile(α::Float64, weights::AbstractVector, μ::AbstractVector,
                          σ::AbstractVector; lo::Float64 = -50.0, hi::Float64 = 50.0,
                          tol::Float64 = 1e-7, max_iter::Int = 80)
    cdf_at(x) = sum(w * cdf(Normal(μk, σk), x) for (w, μk, σk) in zip(weights, μ, σ));
    a = lo; b = hi;
    fa = cdf_at(a) - α;
    fb = cdf_at(b) - α;
    if !(fa <= 0); error("lower bracket fails: cdf($a)=$(fa+α) > $α"); end
    if !(fb >= 0); error("upper bracket fails: cdf($b)=$(fb+α) < $α"); end
    for _ in 1:max_iter
        m = (a + b) / 2;
        fm = cdf_at(m) - α;
        if abs(fm) < tol; return m; end
        if fm < 0; a = m; else; b = m; end
    end
    return (a + b) / 2;
end

# ----------------------------------------------------------------------------------------- #
# Per-fold per-K fit + filter under fold-IS-fixed parameters
# ----------------------------------------------------------------------------------------- #
function fit_and_filter_fold(R_train::AbstractVector, R_test::AbstractVector, K::Int)
    Random.seed!(SEED + 7 * K);
    chmm = build(MyContinuousHiddenMarkovModel,
                 (observations=R_train, number_of_states=K, max_iter=MAX_ITER));
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(chmm.transition[i]); end
    π̄ = (T_mat^2000)[1, :];
    μ = zeros(K); σ = zeros(K);
    for k in 1:K
        d = chmm.emission[k];
        μ[k] = mean(d); σ[k] = std(d);
    end
    R_full = vcat(R_train, R_test);
    pred = filter_predictive(R_full, T_mat, μ, σ, π̄);
    return (K=K, T_mat=T_mat, π̄=π̄, μ=μ, σ=σ, pred=pred,
            n_train=length(R_train), n_test=length(R_test));
end

function eval_conditional_var(fit, R_test::AbstractVector, α::Float64)
    n_o = length(R_test);
    var_thr = zeros(n_o);
    breaches = falses(n_o);
    for j in 1:n_o
        # Predictive Pr(s_{t+1} | F_t) for the j-th test observation; row n_train + j of pred.
        w = fit.pred[fit.n_train + j, :];
        v = mixture_quantile(α, w, fit.μ, fit.σ);
        var_thr[j] = v;
        breaches[j] = R_test[j] < v;
    end
    return var_thr, breaches;
end

# ----------------------------------------------------------------------------------------- #
# Walk-forward loop
# ----------------------------------------------------------------------------------------- #
panels = NamedTuple[];

for (fid, ts, te, vs, ve) in FOLDS
    println("\n" * "="^60)
    println("  Fold $fid:  train [$ts, $te)  test [$vs, $ve)")
    println("="^60)
    R_train = _slice_log_returns(df_full, ts, te);
    R_test  = _slice_log_returns(df_full, vs, ve);
    @printf("  T_train = %d   T_test = %d\n", length(R_train), length(R_test))

    if length(R_test) < 30
        @warn "Test slice <30 days, skipping fold $fid";
        continue;
    end

    for K in K_VALUES
        println("\n  [fit] CHMM-N at K = $K on fold $fid train slice...")
        fit = fit_and_filter_fold(R_train, R_test, K);
        for α in ALPHAS
            var_thr, br = eval_conditional_var(fit, R_test, α);
            n_br = sum(br);
            br_rate = n_br / length(br);
            med_var = median(var_thr);
            k_uc = kupiec_lr(br, α);
            c_in = christoffersen_lr(br);
            c_cc = christoffersen_cc(br, α);
            @printf("  α=%.2f  breaches = %3d / %3d (%5.2f%%)  med VaR = %7.3f  LR_uc = %5.3f  LR_ind = %5.3f  LR_cc = %5.3f  p_cc = %.3f\n",
                α, n_br, length(br), 100*br_rate, med_var,
                k_uc.LR, c_in.LR, c_cc.LR, c_cc.pvalue)
            push!(panels, (
                fold = fid, K = K, α = α,
                T_train = length(R_train), T_test = length(R_test),
                breaches = n_br, breach_rate = br_rate, median_var = med_var,
                LR_uc  = k_uc.LR,  p_uc  = k_uc.pvalue,
                LR_ind = c_in.LR,  p_ind = c_in.pvalue,
                LR_cc  = c_cc.LR,  p_cc  = c_cc.pvalue,
            ));
        end
    end
end

# ----------------------------------------------------------------------------------------- #
# Output
# ----------------------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "walkforward_conditional_var.csv");
open(csv_path, "w") do io
    write(io, "fold,K,alpha,T_train,T_test,breaches,breach_rate,median_var,LR_uc,p_uc,LR_ind,p_ind,LR_cc,p_cc\n");
    for r in panels
        write(io, @sprintf("%s,%d,%.2f,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            r.fold, r.K, r.α, r.T_train, r.T_test,
            r.breaches, r.breach_rate, r.median_var,
            r.LR_uc, r.p_uc, r.LR_ind, r.p_ind, r.LR_cc, r.p_cc));
    end
end

txt_path = joinpath(OUT_DIR, "walkforward_conditional_var.txt");
open(txt_path, "w") do io
    println(io, "="^120);
    println(io, "Walk-forward regime-conditional VaR back-test, CHMM-N");
    println(io, "[peer-review V5 T2.1]");
    println(io, "="^120);
    println(io, "Setup: $(length(FOLDS)) folds, train 5y / test 1y, CHMM-N refit per fold from scratch.");
    println(io, "       At each test day t, conditional VaR_α(t) = α-quantile of the K-component");
    println(io, "       Gaussian mixture predictive density under fold-IS-fixed parameters,");
    println(io, "       evaluated on the full timeline filter Pr(s_{t+1} | F_t = train ∪ test[1..t-1]).");
    println(io, "Tests: Kupiec LR_uc (~χ²_1, critical 3.841)");
    println(io, "       Christoffersen LR_ind (breach independence, ~χ²_1)");
    println(io, "       Christoffersen LR_cc = LR_uc + LR_ind (joint, ~χ²_2, critical 5.991)");
    println(io);
    @printf(io, "%-4s | %2s | %-5s | %-7s | %-7s | %-9s | %-10s | %-9s | %-7s %-6s | %-7s %-6s | %-7s %-6s\n",
        "fold","K","α","T_tr","T_te","breaches","br rate","med VaR","LR_uc","p","LR_ind","p","LR_cc","p");
    println(io, "-"^120);
    for r in panels
        @printf(io, "%-4s | %2d | %-5.2f | %-7d | %-7d | %-9d | %-10.4f | %-9.3f | %-7.3f %-6.3f | %-7.3f %-6.3f | %-7.3f %-6.3f\n",
            r.fold, r.K, r.α, r.T_train, r.T_test, r.breaches, r.breach_rate, r.median_var,
            r.LR_uc, r.p_uc, r.LR_ind, r.p_ind, r.LR_cc, r.p_cc);
    end
    println(io);
    println(io, "Reading: a row passes Christoffersen-cc at the 5% level when LR_cc < 5.991 (p_cc > 0.05).");
    println(io, "Stress folds (W2 COVID, W4 2022 rate-hike onset) are out-of-distribution by KS in the");
    println(io, "univariate walkforward (results/walkforward/walkforward_summary.txt).");
end

println("\n[done] Output:")
println("  $csv_path")
println("  $txt_path")
