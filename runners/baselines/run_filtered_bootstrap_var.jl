# =========================================================================== #
# run_filtered_bootstrap_var.jl
#
# Review-response Item 6a: filtered (Hull-White-style) historical-simulation
# VaR baseline as a fair non-state-space contender against the regime-
# conditional CHMM Christoffersen-cc result of body Section 5.
#
# Recipe (Barone-Adesi, Giannopoulos, Vosper 1999):
#   1. Fit GARCH(1,1)-t on IS returns; recover IS standardised residuals
#      z_t = (R_t - mu) / sigma_t.
#   2. For each OoS day t roll the GARCH recursion forward through OoS data
#      to get sigma_t.
#   3. VaR_alpha(t) = mu + sigma_t * F^{-1}_z(alpha) where F_z is the
#      empirical CDF of IS standardised residuals.
#   4. Score breach series with Kupiec, Christoffersen-cc, Engle-Manganelli DQ.
#
# This is the cleanest "non-CHMM, time-varying conditional" VaR baseline:
# it inherits GARCH conditional volatility but bootstraps the standardised
# innovation distribution, so it is not assumed Gaussian or t.
#
# Output: results/filtered_bootstrap_var/filtered_bootstrap_var.{txt,csv}
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, LinearAlgebra, Distributions, Printf

const SEED      = 20260420;
const DT        = 1/252;
const RISK_FREE = 0.0;
const Q_LAGS    = 4;            # Engle-Manganelli DQ test lag count

const OUT_DIR = joinpath(_ROOT, "results", "filtered_bootstrap_var");
mkpath(OUT_DIR);

println("="^80)
println("  Item 6a: Filtered (Hull-White) historical-simulation VaR baseline")
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

# -------- fit GARCH(1,1)-t on IS --------
println("\n[fit] GARCH(1,1)-t on IS")
Random.seed!(SEED)
garcht = fit_garcht11(R_is);
println(@sprintf("  omega=%.6f  alpha=%.4f  beta=%.4f  mu=%+.6f  nu=%.2f  ll=%.2f",
    garcht.ω, garcht.α, garcht.β, garcht.μ, garcht.ν, garcht.ll))

# IS standardised residuals (the bootstrap pool)
sigma_is = sqrt.(garcht.σ2_hist);
z_is = (R_is .- garcht.μ) ./ sigma_is;
println(@sprintf("  IS z-resid: mean=%+.4f sd=%.4f kurt=%.2f",
    mean(z_is), std(z_is), kurtosis(z_is)))

# -------- roll the GARCH recursion through OoS to get sigma_t --------
# sigma_oos[t] is the conditional variance forecast for OoS day t given
# information through day t-1. We seed with the IS terminal variance and the
# IS terminal return as the lag, then roll with R_oos lags one-step-ahead.
function _roll_sigma(garcht_fit, R_is_v, R_oos_v)
    n = length(R_oos_v);
    out = zeros(n);
    s2 = garcht_fit.σ2_hist[end];
    r_lag = R_is_v[end];
    for t in 1:n
        s2 = max(garcht_fit.ω + garcht_fit.α * (r_lag - garcht_fit.μ)^2 + garcht_fit.β * s2, 1e-12);
        out[t] = sqrt(s2);
        r_lag = R_oos_v[t];
    end
    return out;
end
sigma_oos = _roll_sigma(garcht, R_is, R_oos);

# -------- empirical alpha-quantile of IS standardised residuals --------
function emp_quantile(x::AbstractVector, alpha::Float64)
    sorted = sort(x);
    n = length(sorted);
    pos = max(1, min(n, ceil(Int, alpha * n)));
    return sorted[pos];
end

# -------- DQ test (mirrors run_engle_manganelli_dq.jl) --------
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
    XtX = X' * X;
    Xty = X' * Y;
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
    z_alpha = emp_quantile(z_is, alpha);
    var_thr = garcht.μ .+ sigma_oos .* z_alpha;
    breaches = R_oos .< var_thr;
    n_br = sum(breaches);
    br_rate = n_br / n_oos;
    med_var = median(var_thr);

    k_uc = kupiec_lr(breaches, alpha);
    c_in = christoffersen_lr(breaches);
    c_cc = christoffersen_cc(breaches, alpha);
    dq = dq_test(breaches, var_thr, alpha);

    @printf("  alpha=%.2f  z_alpha=%+.3f  br=%2d (%.2f%%)  med VaR=%.3f  LR_uc=%.3f  LR_ind=%.3f  LR_cc=%.3f (p=%.3f)  DQ=%.2f (p=%.3f)\n",
        alpha, z_alpha, n_br, 100*br_rate, med_var,
        k_uc.LR, c_in.LR, c_cc.LR, c_cc.pvalue, dq.DQ, dq.p_value);

    push!(panels, (
        alpha=alpha, z_alpha=z_alpha,
        breaches=n_br, breach_rate=br_rate, med_var=med_var,
        LR_uc=k_uc.LR, p_uc=k_uc.pvalue,
        LR_ind=c_in.LR, p_ind=c_in.pvalue,
        LR_cc=c_cc.LR, p_cc=c_cc.pvalue,
        DQ=dq.DQ, p_DQ=dq.p_value,
    ));
end

# -------- emit CSV + TXT --------
open(joinpath(OUT_DIR, "filtered_bootstrap_var.csv"), "w") do io
    write(io, "alpha,z_alpha,breaches,breach_rate,med_var,LR_uc,p_uc,LR_ind,p_ind,LR_cc,p_cc,DQ,p_DQ\n");
    for r in panels
        write(io, @sprintf("%.2f,%.4f,%d,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            r.alpha, r.z_alpha, r.breaches, r.breach_rate, r.med_var,
            r.LR_uc, r.p_uc, r.LR_ind, r.p_ind, r.LR_cc, r.p_cc, r.DQ, r.p_DQ));
    end
end

open(joinpath(OUT_DIR, "filtered_bootstrap_var.txt"), "w") do io
    println(io, "Item 6a: Filtered (Hull-White) historical-simulation VaR back-test on SPY OoS");
    println(io, "  GARCH(1,1)-t conditional volatility + bootstrap of IS standardised residuals.");
    println(io, "  OoS T = $n_oos days. Critical chi^2_1(0.05) = 3.841, chi^2_2(0.05) = 5.991.");
    println(io, "="^110);
    @printf(io, "%-6s %-8s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s\n",
        "alpha", "z_alpha", "breaches", "br rate", "med VaR", "LR_uc", "LR_ind", "LR_cc", "p_cc", "p_DQ");
    println(io, "-"^110);
    for r in panels
        @printf(io, "%-6.2f %-+8.4f %-9d %-9.4f %-9.3f %-9.3f %-9.3f %-9.3f %-9.3f %-9.3f\n",
            r.alpha, r.z_alpha, r.breaches, r.breach_rate, r.med_var,
            r.LR_uc, r.LR_ind, r.LR_cc, r.p_cc, r.p_DQ);
    end
end

println("\n[done] Output: $OUT_DIR")
