# =========================================================================== #
# run_quarterly_refit_conditional_var.jl
#
# Peer-review item 8 (R2 W3 / RE3): regime-conditional VaR back-test under
# QUARTERLY REFIT, comparing against the IS-fixed conditional VaR baseline
# (results/diagnostics/conditional_var.txt). Refit cadence: every 63 trading
# days CHMM-N is re-fit on the most recent 5y window and the OoS forward filter
# resumes from the new parameters.
#
# Output:
#   results/diagnostics/quarterly_refit_conditional_var.csv
#   results/diagnostics/quarterly_refit_conditional_var.txt
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, LinearAlgebra, Distributions, Printf, Dates;

const SEED      = 20260420;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const REFIT_CADENCE = 63;        # ~quarterly (252/4)
const TRAIN_LEN     = 1260;       # 5y rolling train window
const OUT_DIR = joinpath(_ROOT, "results", "diagnostics");
mkpath(OUT_DIR);

println("="^70);
println("  Item 8: Quarterly-refit regime-conditional VaR back-test");
println("  Refit cadence: every $REFIT_CADENCE OoS days, train window = $TRAIN_LEN");
println("="^70);

# --- Load IS + OoS SPY ---
train_dataset = MyPortfolioDataSet()["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet()["dataset"];
R_is  = log_growth_matrix(train_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE);
R_oos = log_growth_matrix(oos_dataset,   "SPY"; Δt=DT, risk_free_rate=RISK_FREE);
n_is  = length(R_is);
n_oos = length(R_oos);
@printf("[data] T_IS = %d   T_OoS = %d\n", n_is, n_oos);

# --- Helpers ---
function _fit_chmm_n(R_train::AbstractVector, K::Int)
    Random.seed!(SEED + 7 * K);
    chmm = build(MyContinuousHiddenMarkovModel,
                 (observations=R_train, number_of_states=K, max_iter=MAX_ITER));
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(chmm.transition[i]); end
    π_bar = (T_mat ^ 2000)[1, :];
    π_bar ./= sum(π_bar);
    μ = zeros(K); σ = zeros(K);
    for k in 1:K
        d = chmm.emission[k];
        μ[k] = mean(d);
        σ[k] = std(d);
    end
    return (T_mat=T_mat, π_bar=π_bar, μ=μ, σ=σ);
end

function _filter_predictive(y::AbstractVector, T::AbstractMatrix, μ::AbstractVector,
                            σ::AbstractVector, prior::AbstractVector)
    K = length(μ);
    n = length(y);
    pred = zeros(n + 1, K);
    pred[1, :] = prior;
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

# --- Quarterly-refit forward filter on OoS ---
# At OoS day j, use the model fit on the rolling 5y window ending at calendar day
# (n_is + j - 1), refit every REFIT_CADENCE days starting at j = 1.
function quarterly_refit_pass(K::Int, α::Float64)
    R_full = vcat(R_is, R_oos);
    var_thresholds = zeros(n_oos);
    breaches = falses(n_oos);

    last_refit_at = -1;        # OoS index of most recent refit (1-based)
    fit = nothing;
    pred_state = nothing;       # the last predictive Pr(s_{t+1} | F_t) row

    for j in 1:n_oos
        # decide whether to refit
        need_refit = (j == 1) || ((j - last_refit_at) >= REFIT_CADENCE);
        if need_refit
            train_end = n_is + j - 1;             # latest data we are allowed to use
            train_start = max(1, train_end - TRAIN_LEN + 1);
            R_train = R_full[train_start:train_end];
            fit = _fit_chmm_n(R_train, K);
            # Re-prime the filter: run filter through R_train to get the posterior at
            # the last train-day, then propagate one step to start filtering on j.
            # Since the train ends at OoS day (j-1), we resume the OoS filter on day j.
            pred_train = _filter_predictive(R_train, fit.T_mat, fit.μ, fit.σ, fit.π_bar);
            pred_state = vec(pred_train[end, :]);
            last_refit_at = j;
            @printf("  [refit] j=%4d   train [%d, %d] (T=%d)   π̄ ratio = %.3f\n",
                    j, train_start, train_end, length(R_train),
                    pred_state[argmax(fit.π_bar)] / fit.π_bar[argmax(fit.π_bar)]);
        end

        # predictive at day j: pred_state already holds Pr(s_j | F_{j-1}) under current fit
        var_α = _mixture_quantile(α, pred_state, fit.μ, fit.σ);
        var_thresholds[j] = var_α;
        breaches[j] = R_oos[j] < var_α;

        # observe R_oos[j], update posterior, propagate one step for j+1.
        b = [pdf(Normal(fit.μ[k], fit.σ[k]), R_oos[j]) for k in 1:length(fit.μ)];
        post = pred_state .* b;
        Z = sum(post);
        if Z <= 0
            post .= pred_state;
        else
            post ./= Z;
        end
        pred_state = vec(post' * fit.T_mat);
    end

    return var_thresholds, breaches;
end

# --- Run ---
results = NamedTuple[];
for K in (3, 18)
    for α in (0.01, 0.05)
        @printf("\n[run] K = %d, α = %.2f\n", K, α);
        var_thr, br = quarterly_refit_pass(K, α);
        n_br = sum(br);
        br_rate = n_br / length(br);
        med_var = median(var_thr);
        k_uc = kupiec_lr(br, α);
        c_in = christoffersen_lr(br);
        c_cc = christoffersen_cc(br, α);
        @printf("  breaches = %d / %d (%.2f%%)\n", n_br, length(br), 100*br_rate);
        @printf("  med VaR = %.3f   LR_uc = %.3f   LR_ind = %.3f   LR_cc = %.3f\n",
                med_var, k_uc.LR, c_in.LR, c_cc.LR);
        push!(results, (K=K, α=α, breaches=n_br, br_rate=br_rate, med_var=med_var,
                        LR_uc=k_uc.LR, LR_ind=c_in.LR, LR_cc=c_cc.LR,
                        p_uc=k_uc.pvalue, p_ind=c_in.pvalue, p_cc=c_cc.pvalue));
    end
end

# --- Output ---
open(joinpath(OUT_DIR, "quarterly_refit_conditional_var.csv"), "w") do io
    write(io, "K,alpha,breaches,br_rate,med_VaR,LR_uc,LR_ind,LR_cc,p_uc,p_ind,p_cc\n");
    for r in results
        write(io, @sprintf("%d,%.2f,%d,%.4f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f\n",
                r.K, r.α, r.breaches, r.br_rate, r.med_var, r.LR_uc, r.LR_ind, r.LR_cc,
                r.p_uc, r.p_ind, r.p_cc));
    end
end

open(joinpath(OUT_DIR, "quarterly_refit_conditional_var.txt"), "w") do io
    println(io, "Quarterly-refit regime-conditional VaR back-test (peer-review item 8)");
    println(io, "Refit cadence: every $REFIT_CADENCE OoS days, train window = $TRAIN_LEN obs");
    println(io, "Critical values: chi^2_1(0.05) = 3.841, chi^2_2(0.05) = 5.991");
    println(io, "="^80);
    @printf(io, "%-5s %-7s %-9s %-9s %-9s %-9s %-9s %-9s %-7s %-7s %-7s\n",
            "K", "α", "breaches", "br_rate", "med_VaR", "LR_uc", "LR_ind", "LR_cc", "p_uc", "p_ind", "p_cc");
    println(io, "-"^80);
    for r in results
        @printf(io, "%-5d %-7.2f %-9d %-9.4f %-9.3f %-9.3f %-9.3f %-9.3f %-7.4f %-7.4f %-7.4f\n",
                r.K, r.α, r.breaches, r.br_rate, r.med_var, r.LR_uc, r.LR_ind, r.LR_cc,
                r.p_uc, r.p_ind, r.p_cc);
    end
    println(io);
    println(io, "Reading: compare against the IS-fixed conditional-VaR results in");
    println(io, "results/diagnostics/conditional_var.txt. The refit construction does not");
    println(io, "discard any IS information; it just gives the model the opportunity to track");
    println(io, "regime drift across the OoS window.");
end

println("\n[done] Output: $OUT_DIR/quarterly_refit_conditional_var.{csv,txt}");
