# =========================================================================== #
# run_engle_manganelli_dq_all_families.jl
#
# Companion to run_engle_manganelli_dq.jl (CHMM-N only) and
# run_conditional_var_all_families.jl (Christoffersen only). Computes the
# Engle-Manganelli (2004) Dynamic Quantile (DQ) test for ALL FOUR CHMM emission
# families (CHMM-N, CHMM-t penalised at λ=20, CHMM-L, CHMM-GED) at K ∈ {3, 18},
# α ∈ {0.01, 0.05} on the SPY OoS window, so the strict-tail DQ verdict can be
# compared across families (not just CHMM-N).
#
# Reuses the family-appropriate conditional-VaR construction from
# run_conditional_var_all_families.jl (generic mixture components + forward
# filter) and the DQ statistic from run_engle_manganelli_dq.jl. Neither of those
# scripts nor their committed outputs are modified; this writes a new file:
#   results/diagnostics/engle_manganelli_dq_all_families.txt / .csv
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, LinearAlgebra, Distributions, Printf

const SEED      = 20260420;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const LAMBDA_T  = 20.0;      # penalised CHMM-t shrinkage rate (matches panel runner)
const Q_LAGS    = 4;         # Engle-Manganelli 2004 lags -> chi^2(q+2)

const OUT_DIR = joinpath(_ROOT, "results", "diagnostics");
mkpath(OUT_DIR);

println("="^80)
println("  Engle-Manganelli DQ across CHMM-N / -t / -L / -GED on SPY OoS")
println("="^80)

# -------- data (identical to the two source runners) --------
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
# Prepend it so the first OoS forecast target is the return into 2024-01-04.
R_oos = vcat((1/DT) * log(oos_dataset["SPY"][1, :volume_weighted_average_price] /
                          train_dataset["SPY"][end, :volume_weighted_average_price]) - RISK_FREE,
             R_oos);

n_is = length(R_is); n_oos = length(R_oos);
println("[setup] IS = $n_is, OoS = $n_oos")

# -------- generic forward filter + mixture quantile (from all-families runner) --------
function filter_predictive(y::AbstractVector, T::AbstractMatrix, components, π_init)
    K = length(components); n = length(y);
    pred = zeros(n + 1, K);
    pred[1, :] = π_init;
    for t in 1:n
        b = [pdf(components[k], y[t]) for k in 1:K];
        post = pred[t, :] .* b; Z = sum(post);
        if Z <= 0; post .= pred[t, :]; else; post ./= Z; end
        pred[t + 1, :] = vec(post' * T);
    end
    return pred;
end

function mixture_quantile(α::Float64, weights, components;
                          lo::Float64=-50.0, hi::Float64=50.0,
                          tol::Float64=1e-6, max_iter::Int=80)
    cdf_at(x) = sum(w * cdf(c, x) for (w, c) in zip(weights, components));
    a, b = lo, hi;
    fa, fb = cdf_at(a) - α, cdf_at(b) - α;
    if fa > 0; a = -200.0; end
    if fb < 0; b = 200.0; end
    for _ in 1:max_iter
        m = 0.5 * (a + b);
        fm = cdf_at(m) - α;
        if abs(fm) < tol; return m; end
        if fm < 0; a = m; else; b = m; end
    end
    return 0.5 * (a + b);
end

# -------- family fits (from run_conditional_var_all_families.jl) --------
function _pack(m)
    K_actual = length(m.states);
    T_mat = zeros(K_actual, K_actual);
    for i in 1:K_actual; T_mat[i, :] = probs(m.transition[i]); end
    π̄ = (T_mat^2000)[1, :]; π̄ ./= sum(π̄);
    components = [m.emission[k] for k in 1:K_actual];
    return T_mat, π̄, components;
end
fit_chmm_n(R_is, K)  = (Random.seed!(SEED); _pack(build(MyContinuousHiddenMarkovModel, (observations=R_is, number_of_states=K, max_iter=MAX_ITER))));
fit_chmm_t(R_is, K)  = (Random.seed!(SEED); _pack(build(MyStudentTHiddenMarkovModel, (observations=R_is, number_of_states=K, max_iter=MAX_ITER, ν_shrink_rate=LAMBDA_T))));
fit_chmm_l(R_is, K)  = (Random.seed!(SEED); _pack(build(MyLaplaceHiddenMarkovModel, (observations=R_is, number_of_states=K, max_iter=MAX_ITER))));
fit_chmm_ged(R_is, K)= (Random.seed!(SEED); _pack(build(MyGEDHiddenMarkovModel, (observations=R_is, number_of_states=K, max_iter=MAX_ITER))));

const FAMILIES = [
    ("CHMM-N", fit_chmm_n),
    ("CHMM-t (λ=$LAMBDA_T)", fit_chmm_t),
    ("CHMM-L", fit_chmm_l),
    ("CHMM-GED", fit_chmm_ged),
];

# -------- Engle-Manganelli DQ (from run_engle_manganelli_dq.jl) --------
function dq_test(breaches::AbstractVector{Bool}, var_thr::AbstractVector{Float64},
                 α::Float64; q::Int = Q_LAGS)
    n = length(breaches);
    hit = Float64.(breaches) .- α;
    n_eff = n - q; p = q + 2;
    X = zeros(n_eff, p); Y = zeros(n_eff);
    for t in (q+1):n
        row = t - q;
        X[row, 1] = 1.0;
        for i in 1:q; X[row, 1 + i] = hit[t - i]; end
        X[row, p] = var_thr[t];
        Y[row] = hit[t];
    end
    XtX = X' * X; Xty = X' * Y;
    if det(XtX) == 0
        return (DQ = NaN, p_value = NaN, dof = p, singular = true);
    end
    beta_hat = XtX \ Xty;
    DQ = (beta_hat' * XtX * beta_hat) / (α * (1.0 - α));
    p_value = 1.0 - cdf(Chisq(p), DQ);
    return (DQ = DQ, p_value = p_value, dof = p, singular = false);
end

# -------- run family x K x α --------
panels = NamedTuple[];
for (label, fitfn) in FAMILIES
    for K in (3, 18)
        println("\n[fit] $label at K = $K")
        T_mat, π̄, components = fitfn(R_is, K);
        R_full = vcat(R_is, R_oos);
        pred = filter_predictive(R_full, T_mat, components, π̄);
        for α in (0.01, 0.05)
            var_thr = zeros(n_oos); breaches = falses(n_oos);
            for j in 1:n_oos
                w = pred[n_is + j, :];
                v = mixture_quantile(α, w, components);
                var_thr[j] = v; breaches[j] = R_oos[j] < v;
            end
            n_br = sum(breaches);
            c_cc = christoffersen_cc(breaches, α);
            dq = dq_test(breaches, var_thr, α; q = Q_LAGS);
            @printf("  %-16s K=%2d α=%.2f  br=%2d  cc p=%.3f  DQ=%.3f DQ p=%.3f\n",
                label, K, α, n_br, c_cc.pvalue, dq.DQ, dq.p_value);
            push!(panels, (
                family=label, K=K, α=α, breaches=n_br, breach_rate=n_br/n_oos,
                cc_LR=c_cc.LR, cc_p=c_cc.pvalue, DQ=dq.DQ, DQ_p=dq.p_value, DQ_dof=dq.dof,
            ));
        end
    end
end

# -------- output --------
open(joinpath(OUT_DIR, "engle_manganelli_dq_all_families.txt"), "w") do io
    println(io, "="^104);
    println(io, "Engle-Manganelli (2004) DQ test across four CHMM emission families");
    println(io, "Regime-conditional VaR, SPY OoS T = $n_oos, seed $SEED, q = $Q_LAGS lags -> chi^2($(Q_LAGS+2))");
    println(io, "CHMM-t is the penalised lambda=$LAMBDA_T variant (matches conditional_var_panel).");
    println(io, "="^104);
    @printf(io, "%-18s %-3s %-5s %-9s %-9s %-9s %-9s\n",
        "family", "K", "α", "breaches", "cc LR", "cc p", "DQ p");
    println(io, "-"^80);
    for r in panels
        @printf(io, "%-18s %-3d %-5.2f %-9d %-9.3f %-9.3f %-9.3f\n",
            r.family, r.K, r.α, r.breaches, r.cc_LR, r.cc_p, r.DQ_p);
    end
    println(io);
    println(io, "Reject conditional coverage at 5% if p < 0.05 (either test).");
end

open(joinpath(OUT_DIR, "engle_manganelli_dq_all_families.csv"), "w") do io
    println(io, "family,K,alpha,breaches,breach_rate,cc_LR,cc_p,DQ,DQ_p,DQ_dof");
    for r in panels
        @printf(io, "%s,%d,%.2f,%d,%.5f,%.4f,%.4f,%.4f,%.4f,%d\n",
            r.family, r.K, r.α, r.breaches, r.breach_rate, r.cc_LR, r.cc_p, r.DQ, r.DQ_p, r.DQ_dof);
    end
end

println("\n[done] Output: $OUT_DIR/engle_manganelli_dq_all_families.{txt,csv}")
