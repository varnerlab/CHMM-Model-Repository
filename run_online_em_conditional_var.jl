# ========================================================================================= #
# run_online_em_conditional_var.jl
#
# Online-EM (≡ daily-refit) regime-conditional VaR back-test on the W2 (COVID 2020) and
# W4 (2022 rate-hike onset) stress folds. Closes peer-review item E3 (online-EM cond
# VaR for W2 / W4): the body's fold-IS-fixed and existing run_walkforward_cond_var_refit_
# cadence.jl monthly (21d) and weekly (5d) cadences all reject Christoffersen-cc on W2;
# this runner tests whether daily refit, the practical limit of the cadence sweep and the
# closest tractable approximation to the Cappé (2011, JCGS) stochastic-recursion online-EM
# for HMMs, closes the W2 / W4 gap.
#
# Pipeline at each test day j ∈ [1, n_test]:
#   1. Rolling 5y train window ending the day before R_test[j] (TRAIN_LEN = 1260 returns).
#   2. Fresh Baum-Welch / EM fit (CHMM-N at K = 3, max 60 iterations, quantile init seeded
#      from the body global seed plus a per-day deterministic offset for reproducibility).
#   3. Forward filter through the new train window to prime the predictive state belief.
#   4. Predictive VaR_α(j) = α-quantile of the K-component Gaussian mixture under
#      pred_state and fresh emission parameters.
#   5. Update breach indicator R_test[j] < VaR_α(j); update pred_state via one filter step.
#
# This runner re-runs CHMM-N at K = 3 per day per fold rather than using a stochastic
# online recursion. The intent (does continuous parameter adaptation close W2 / W4?) is
# preserved at the cost of compute. The full Cappé (2011) stochastic-recursion online-EM
# is a deferred follow-up because the conclusion below holds at daily refit and the
# stochastic-recursion variant cannot be more responsive than daily batch EM by
# construction.
#
# Output:
#   results/walkforward_online_em/walkforward_online_em.csv
#   results/walkforward_online_em/walkforward_online_em.txt
#   ../CHMM-paper/results/robustness/walkforward_online_em.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using LinearAlgebra
using Distributions
using Printf
using Dates
using CSV
using DataFrames

const SEED       = 20260420;
const K          = 3;
const ALPHAS     = (0.01, 0.05);
const Q_LAGS_DQ  = 4;
const MAX_ITER   = 60;
const RISK_FREE  = 0.0;
const DT         = 1/252;
const TICKER     = "SPY";
const TRAIN_LEN  = 1260;          # 5y rolling-window batch EM at each step
const CADENCE    = 1;             # daily refit ≡ online-EM at the cadence limit
const STRESS_FOLDS = [
    ("W2", Date(2015,1,1), Date(2020,1,1), Date(2020,1,1), Date(2021,1,1)),
    ("W4", Date(2017,1,1), Date(2022,1,1), Date(2022,1,1), Date(2023,1,1)),
];

const OUT_DIR    = joinpath(_ROOT, "results", "walkforward_online_em");
const PAPER_DIR  = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_DIR);

println("="^80);
println("  Online-EM (daily-refit) regime-conditional VaR on W2 / W4  (peer-review E3)");
println("  K = $K, α ∈ $ALPHAS, cadence = $CADENCE day, train window = $TRAIN_LEN");
println("="^80);

# -------------------------------------------------------------------------------------- #
# Data
# -------------------------------------------------------------------------------------- #
println("\n[setup] Loading SPY full timeline...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
df_train = train_dataset[TICKER];
df_oos   = oos_dataset[TICKER];
df_full  = vcat(df_train, df_oos; cols = :orderequal);
sort!(df_full, :timestamp);
ts_full = Date.(df_full.timestamp);
P_full  = Vector{Float64}(df_full.volume_weighted_average_price);
R_full  = Vector{Float64}((1/DT) .* (log.(P_full[2:end] ./ P_full[1:end-1])));
ts_full_R = ts_full[2:end];
n_full = length(R_full);
@printf("  R_full series: %d log returns from %s to %s\n", n_full, ts_full_R[1], ts_full_R[end]);

function _idx_range(t_start::Date, t_end_excl::Date)
    mask = (ts_full_R .>= t_start) .& (ts_full_R .< t_end_excl);
    idx_first = findfirst(mask); idx_last = findlast(mask);
    return (idx_first, idx_last);
end

# -------------------------------------------------------------------------------------- #
# CHMM-N fit + predictive helpers (mirrors run_walkforward_cond_var_refit_cadence.jl)
# -------------------------------------------------------------------------------------- #
function _fit_chmm_n(R_train::AbstractVector, K::Int, fit_seed::Int)
    Random.seed!(fit_seed);
    chmm = build(MyContinuousHiddenMarkovModel,
                 (observations=R_train, number_of_states=K, max_iter=MAX_ITER));
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(chmm.transition[i]); end
    π̄ = (T_mat^2000)[1, :]; π̄ ./= sum(π̄);
    μ = zeros(K); σ = zeros(K);
    for k in 1:K
        d = chmm.emission[k];
        μ[k] = mean(d); σ[k] = std(d);
    end
    return (T = T_mat, π̄ = π̄, μ = μ, σ = σ);
end

function _filter_one_step(prev_pred::AbstractVector, y::Real, fit)
    K = length(fit.μ);
    b = [pdf(Normal(fit.μ[k], fit.σ[k]), y) for k in 1:K];
    post = prev_pred .* b;
    Z = sum(post);
    if Z <= 0; post .= prev_pred; else; post ./= Z; end
    return vec(post' * fit.T);
end

function _filter_through(R::AbstractVector, fit)
    K = length(fit.μ);
    pred = zeros(length(R) + 1, K);
    pred[1, :] = fit.π̄;
    for t in 1:length(R)
        pred[t + 1, :] = _filter_one_step(pred[t, :], R[t], fit);
    end
    return pred;
end

function _mixture_quantile(α::Float64, weights::AbstractVector, μ::AbstractVector,
                            σ::AbstractVector; lo::Float64 = -50.0, hi::Float64 = 50.0,
                            tol::Float64 = 1e-7, max_iter::Int = 80)
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

# -------------------------------------------------------------------------------------- #
# Engle-Manganelli DQ (mirrors run_engle_manganelli_dq.jl)
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
        return (DQ = NaN, p_value = NaN, dof = p, n_eff = n_eff, singular = true);
    end
    β̂ = XtX \ Xty;
    DQ = (β̂' * XtX * β̂) / (α * (1.0 - α));
    pval = 1.0 - cdf(Chisq(p), DQ);
    return (DQ = DQ, p_value = pval, dof = p, n_eff = n_eff, singular = false);
end

# -------------------------------------------------------------------------------------- #
# Daily-refit pass on a single fold
# -------------------------------------------------------------------------------------- #
function daily_refit_pass(fold_id::String,
                          train_start_date::Date, train_end_date::Date,
                          test_start_date::Date,  test_end_date::Date,
                          α::Float64)
    test_first, test_last = _idx_range(test_start_date, test_end_date);
    n_test = test_last - test_first + 1;
    @printf("  [%s α=%.2f] test idx [%d, %d] (T = %d)  daily refit on rolling %d-day window...\n",
            fold_id, α, test_first, test_last, n_test, TRAIN_LEN);

    var_thr = zeros(n_test);
    breaches = falses(n_test);

    t0 = time();
    for j in 1:n_test
        # Rolling 5y train window ending the day BEFORE R_test[j]
        calendar_train_end   = test_first + j - 2;
        calendar_train_start = max(1, calendar_train_end - TRAIN_LEN + 1);
        R_train_now = R_full[calendar_train_start:calendar_train_end];

        # Fresh CHMM-N fit (Baum-Welch, max 60 iters). Seed is the body global seed
        # plus the calendar-train-end index so each daily fit is reproducible and
        # distinct (the calendar index is monotonic so seeds are unique per refit).
        fit = _fit_chmm_n(R_train_now, K, SEED + 7 * K + calendar_train_end);

        # Prime predictive state belief by filtering through the new train window
        pred_train = _filter_through(R_train_now, fit);
        pred_state = vec(pred_train[end, :]);

        # Predictive VaR at j
        var_α = _mixture_quantile(α, pred_state, fit.μ, fit.σ);
        y_j = R_full[test_first + j - 1];
        var_thr[j] = var_α;
        breaches[j] = y_j < var_α;

        # Progress every 50 days to flag long-running run
        if j % 50 == 0
            elapsed = time() - t0;
            @printf("    [%s α=%.2f] day %d / %d  elapsed %.1fs\n",
                    fold_id, α, j, n_test, elapsed);
        end
    end
    elapsed = time() - t0;
    @printf("  [%s α=%.2f] total elapsed %.1fs (%.2f s/day)\n",
            fold_id, α, elapsed, elapsed / n_test);

    return var_thr, breaches, n_test;
end

# -------------------------------------------------------------------------------------- #
# Main loop
# -------------------------------------------------------------------------------------- #
panels = NamedTuple[];

for (fid, ts, te, vs, ve) in STRESS_FOLDS
    println("\n" * "="^60);
    println("  Fold $fid  train [$ts, $te)  test [$vs, $ve)");
    println("="^60);
    for α in ALPHAS
        var_thr, br, n_test = daily_refit_pass(fid, ts, te, vs, ve, α);
        n_br = sum(br); br_rate = n_br / n_test;
        med_var = median(var_thr);
        k_uc = kupiec_lr(br, α);
        c_in = christoffersen_lr(br);
        c_cc = christoffersen_cc(br, α);
        dq   = dq_test(br, var_thr, α; q = Q_LAGS_DQ);
        @printf("    breaches = %d / %d (%.2f%%)  med VaR = %.3f\n",
                n_br, n_test, 100*br_rate, med_var);
        @printf("    Kupiec LR_uc = %.3f  p = %.3f\n", k_uc.LR, k_uc.pvalue);
        @printf("    Christ. LR_ind = %.3f p = %.3f\n", c_in.LR, c_in.pvalue);
        @printf("    Christ. LR_cc  = %.3f p_cc = %.3f\n", c_cc.LR, c_cc.pvalue);
        @printf("    Engle-M. DQ    = %.3f p = %.3f  (chi^2_%d crit @ 5%% = %.3f)\n",
                dq.DQ, dq.p_value, dq.dof, quantile(Chisq(dq.dof), 0.95));

        push!(panels, (
            fold = fid, K = K, α = α, cadence = CADENCE,
            n_test = n_test, breaches = n_br, breach_rate = br_rate,
            median_var = med_var,
            LR_uc = k_uc.LR, p_uc = k_uc.pvalue,
            LR_ind = c_in.LR, p_ind = c_in.pvalue,
            LR_cc = c_cc.LR,  p_cc = c_cc.pvalue,
            DQ = dq.DQ, DQ_p = dq.p_value, DQ_dof = dq.dof,
        ));
    end
end

# -------------------------------------------------------------------------------------- #
# Output
# -------------------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "walkforward_online_em.csv");
open(csv_path, "w") do io
    println(io, "fold,K,alpha,cadence,n_test,breaches,breach_rate,median_var," *
                "LR_uc,p_uc,LR_ind,p_ind,LR_cc,p_cc,DQ,DQ_p,DQ_dof");
    for r in panels
        @printf(io, "%s,%d,%.2f,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                r.fold, r.K, r.α, r.cadence, r.n_test, r.breaches, r.breach_rate, r.median_var,
                r.LR_uc, r.p_uc, r.LR_ind, r.p_ind, r.LR_cc, r.p_cc,
                r.DQ, r.DQ_p, r.DQ_dof);
    end
end

txt_path = joinpath(OUT_DIR, "walkforward_online_em.txt");
open(txt_path, "w") do io
    println(io, "="^120);
    println(io, "Online-EM (daily-refit) regime-conditional VaR on W2 / W4 stress folds  (peer-review E3)");
    println(io, "="^120);
    println(io);
    println(io, "Setup");
    println(io, "  Folds       : W2 (test 2020 incl. COVID); W4 (test 2022 rate-hike onset)");
    println(io, "  K           : $K (CHMM-N)");
    println(io, "  α           : $ALPHAS");
    println(io, "  Cadence     : $CADENCE day (daily refit ≡ online-EM cadence limit)");
    println(io, "  Train window: rolling $TRAIN_LEN-day batch EM ending the day before each test day");
    println(io, "  Tests       : Kupiec LR_uc (~chi²_1, 3.841)");
    println(io, "                Christoffersen LR_ind (~chi²_1, 3.841)");
    println(io, "                Christoffersen LR_cc  (~chi²_2, 5.991)");
    println(io, "                Engle-Manganelli DQ at q = $(Q_LAGS_DQ) lags (~chi²_$(Q_LAGS_DQ + 2), $(round(quantile(Chisq(Q_LAGS_DQ + 2), 0.95), digits=2)))");
    println(io);
    @printf(io, "%-4s %-5s %-9s %-7s %-9s %-7s %-6s %-7s %-6s %-7s %-6s %-9s %-7s\n",
            "fold", "α", "breaches", "br rate", "med VaR",
            "LR_uc", "p", "LR_ind", "p", "LR_cc", "p_cc", "DQ", "DQ p");
    println(io, "-"^120);
    for r in panels
        @printf(io, "%-4s %-5.2f %-9d %-7.4f %-9.3f %-7.3f %-6.3f %-7.3f %-6.3f %-7.3f %-6.3f %-9.3f %-7.3f\n",
                r.fold, r.α, r.breaches, r.breach_rate, r.median_var,
                r.LR_uc, r.p_uc, r.LR_ind, r.p_ind, r.LR_cc, r.p_cc,
                r.DQ, r.DQ_p);
    end
    println(io);
    println(io, "Reading. The body's existing within-fold refit-cadence sweep");
    println(io, "(run_walkforward_cond_var_refit_cadence.jl, monthly = 21d / weekly = 5d) leaves W2");
    println(io, "rejected at every cadence (p_cc ∈ {0.011, 0.017, 0.023}) and W4 borderline-passing");
    println(io, "or rejecting depending on cadence. The body interpretation is that W2 is an");
    println(io, "intrinsic regime-break failure rather than a tracking-lag artefact. The daily-refit");
    println(io, "panel above is the practical limit of the cadence sweep and the closest tractable");
    println(io, "approximation to a Cappé (2011) stochastic-recursion online-EM. If daily refit also");
    println(io, "rejects W2, the body's intrinsic-regime-break reading is supported with no remaining");
    println(io, "cadence-side margin to argue otherwise; if daily refit closes W2, the body's reading");
    println(io, "should be qualified to ``cadences slower than daily''.");
end

paper_csv = joinpath(PAPER_DIR, "walkforward_online_em.csv");
open(paper_csv, "w") do io
    println(io, "fold,K,alpha,cadence,n_test,breaches,breach_rate,median_var," *
                "LR_uc,p_uc,LR_ind,p_ind,LR_cc,p_cc,DQ,DQ_p,DQ_dof");
    for r in panels
        @printf(io, "%s,%d,%.2f,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                r.fold, r.K, r.α, r.cadence, r.n_test, r.breaches, r.breach_rate, r.median_var,
                r.LR_uc, r.p_uc, r.LR_ind, r.p_ind, r.LR_cc, r.p_cc,
                r.DQ, r.DQ_p, r.DQ_dof);
    end
end

println("\n" * "="^80);
println("  Online-EM (daily-refit) cond VaR back-test complete.");
@printf("  CSV (model repo) : %s\n", csv_path);
@printf("  TXT (model repo) : %s\n", txt_path);
@printf("  CSV (paper repo) : %s\n", paper_csv);
println("="^80);
