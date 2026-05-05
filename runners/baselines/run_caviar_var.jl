# =========================================================================== #
# run_caviar_var.jl
#
# Review-response Item 6b: Engle-Manganelli (2004, JBES) symmetric-absolute-
# value (SAV) CAViaR baseline for OoS VaR. Provides a non-state-space
# conditional-quantile contender against the regime-conditional CHMM
# Christoffersen-cc result of body Section 5.
#
# Specification (Engle-Manganelli 2004, Eq. 9, SAV):
#   VaR_t(alpha) = beta_1 + beta_2 * VaR_{t-1}(alpha) + beta_3 * |R_{t-1}|
#
# beta = (b1, b2, b3) is fit on IS by minimising the asymmetric "tick"
# loss:
#   Q_T(beta) = (1/T) * sum_t [ alpha - 1{R_t < VaR_t(beta)} ] * (R_t - VaR_t(beta))
#
# A 2-step grid + Nelder-Mead local search. Fit IS on each alpha; for OoS
# roll the recursion forward through observed R_oos using the IS-fitted beta.
#
# Output: results/caviar_var/caviar_var.{txt,csv}
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, LinearAlgebra, Distributions, Printf

const SEED      = 20260420;
const DT        = 1/252;
const RISK_FREE = 0.0;
const Q_LAGS    = 4;            # DQ lag count

const OUT_DIR = joinpath(_ROOT, "results", "caviar_var");
mkpath(OUT_DIR);

println("="^80)
println("  Item 6b: CAViaR (Engle-Manganelli 2004) SAV VaR baseline")
println("="^80)

# -------- data --------
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
n_is = length(R_is); n_oos = length(R_oos);
println("[setup] IS = $n_is, OoS = $n_oos")

# -------- CAViaR SAV recursion --------
# var_t = b1 + b2 * var_{t-1} + b3 * |R_{t-1}|
# Initial var_0 set to the empirical alpha-quantile of the first 250 IS days.
function caviar_sav_path(beta::Vector{Float64}, R::AbstractVector,
                         var_init::Float64)
    b1, b2, b3 = beta[1], beta[2], beta[3];
    T = length(R);
    var_t = zeros(T);
    var_t[1] = var_init;
    for t in 2:T
        var_t[t] = b1 + b2 * var_t[t-1] + b3 * abs(R[t-1]);
    end
    return var_t;
end

# tick (asymmetric quantile) loss
function tick_loss(beta::Vector{Float64}, R::AbstractVector, alpha::Float64,
                   var_init::Float64)
    var_t = caviar_sav_path(beta, R, var_init);
    T = length(R);
    s = 0.0;
    for t in 1:T
        diff = R[t] - var_t[t];
        s += (alpha - (R[t] < var_t[t] ? 1.0 : 0.0)) * diff;
    end
    return s / T;
end

# Nelder-Mead (reuse the project's helper)
function _fit_caviar_sav(R::AbstractVector, alpha::Float64)
    var_init = quantile(R[1:min(250, length(R))], alpha);
    # Coarse grid for warm start
    best_loss = Inf; best_beta = [var_init * (1 - 0.9), 0.9, 0.05];
    for b1_try in [-0.20, -0.10, -0.05, -0.01, 0.0]
        for b2_try in [0.85, 0.90, 0.95]
            for b3_try in [-0.30, -0.15, -0.05, 0.05, 0.15]
                p = [b1_try, b2_try, b3_try];
                l = tick_loss(p, R, alpha, var_init);
                if isfinite(l) && l < best_loss
                    best_loss = l; best_beta = copy(p);
                end
            end
        end
    end
    best, _ = _nelder_mead(p -> tick_loss(p, R, alpha, var_init), best_beta;
                           max_iter=2000);
    return (beta = best, var_init = var_init, loss = tick_loss(best, R, alpha, var_init));
end

# -------- DQ test --------
function dq_test(breaches::AbstractVector{Bool}, var_thr::AbstractVector{Float64},
                 alpha::Float64; q::Int = Q_LAGS)
    n = length(breaches);
    @assert n == length(var_thr);
    hit = Float64.(breaches) .- alpha;
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
    XtX = X' * X; Xty = X' * Y;
    if det(XtX) == 0
        return (DQ = NaN, p_value = NaN, dof = p, n_eff = n_eff, singular = true);
    end
    beta_hat = XtX \ Xty;
    DQ = (beta_hat' * XtX * beta_hat) / (alpha * (1.0 - alpha));
    p_value = 1.0 - cdf(Chisq(p), DQ);
    return (DQ = DQ, p_value = p_value, dof = p, n_eff = n_eff, singular = false);
end

# -------- score VaR at alpha in {0.01, 0.05} --------
panels = NamedTuple[];
for alpha in (0.01, 0.05)
    println("\n[fit] alpha = $alpha")
    Random.seed!(SEED);
    fit = _fit_caviar_sav(R_is, alpha);
    println(@sprintf("  beta = (%.5f, %.4f, %.5f)  IS tick loss = %.6f",
        fit.beta[1], fit.beta[2], fit.beta[3], fit.loss));

    # Roll the recursion through R_is then continue into R_oos (continuity)
    R_full = vcat(R_is, R_oos);
    var_full = caviar_sav_path(fit.beta, R_full, fit.var_init);
    var_thr = var_full[(n_is+1):end];

    breaches = R_oos .< var_thr;
    n_br = sum(breaches);
    br_rate = n_br / n_oos;
    med_var = median(var_thr);

    k_uc = kupiec_lr(breaches, alpha);
    c_in = christoffersen_lr(breaches);
    c_cc = christoffersen_cc(breaches, alpha);
    dq = dq_test(breaches, var_thr, alpha);

    @printf("  alpha=%.2f  br=%2d (%.2f%%)  med VaR=%.3f  LR_uc=%.3f  LR_ind=%.3f  LR_cc=%.3f (p=%.3f)  DQ=%.2f (p=%.3f)\n",
        alpha, n_br, 100*br_rate, med_var,
        k_uc.LR, c_in.LR, c_cc.LR, c_cc.pvalue, dq.DQ, dq.p_value);

    push!(panels, (
        alpha=alpha, b1=fit.beta[1], b2=fit.beta[2], b3=fit.beta[3],
        breaches=n_br, breach_rate=br_rate, med_var=med_var,
        LR_uc=k_uc.LR, p_uc=k_uc.pvalue,
        LR_ind=c_in.LR, p_ind=c_in.pvalue,
        LR_cc=c_cc.LR, p_cc=c_cc.pvalue,
        DQ=dq.DQ, p_DQ=dq.p_value,
    ));
end

# -------- emit CSV + TXT --------
open(joinpath(OUT_DIR, "caviar_var.csv"), "w") do io
    write(io, "alpha,b1,b2,b3,breaches,breach_rate,med_var,LR_uc,p_uc,LR_ind,p_ind,LR_cc,p_cc,DQ,p_DQ\n");
    for r in panels
        write(io, @sprintf("%.2f,%.5f,%.4f,%.5f,%d,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            r.alpha, r.b1, r.b2, r.b3, r.breaches, r.breach_rate, r.med_var,
            r.LR_uc, r.p_uc, r.LR_ind, r.p_ind, r.LR_cc, r.p_cc, r.DQ, r.p_DQ));
    end
end

open(joinpath(OUT_DIR, "caviar_var.txt"), "w") do io
    println(io, "Item 6b: CAViaR (Engle-Manganelli 2004) SAV VaR back-test on SPY OoS");
    println(io, "  Specification: VaR_t = b1 + b2 * VaR_{t-1} + b3 * |R_{t-1}|");
    println(io, "  OoS T = $n_oos days. Critical chi^2_1(0.05) = 3.841, chi^2_2(0.05) = 5.991.");
    println(io, "="^110);
    @printf(io, "%-6s %-9s %-9s %-9s %-9s %-9s %-9s %-9s\n",
        "alpha", "breaches", "br rate", "med VaR", "LR_uc", "LR_ind", "LR_cc", "p_DQ");
    println(io, "-"^110);
    for r in panels
        @printf(io, "%-6.2f %-9d %-9.4f %-9.3f %-9.3f %-9.3f %-9.3f %-9.3f\n",
            r.alpha, r.breaches, r.breach_rate, r.med_var,
            r.LR_uc, r.LR_ind, r.LR_cc, r.p_DQ);
    end
end

println("\n[done] Output: $OUT_DIR")
