# =========================================================================== #
# run_conditional_var_all_families.jl
#
# B4 (REVISION_PLAN_V3_TO_ACCEPT.md): regime-conditional VaR back-test extended
# to all four CHMM emission families (CHMM-N, CHMM-t penalised at λ = 20,
# CHMM-L, CHMM-GED) at K ∈ {3, 18}, α ∈ {0.01, 0.05} on the SPY OoS window.
#
# Mirrors run_conditional_var.jl (which is Gaussian-only) but uses the
# family-appropriate predictive density: for state s the conditional mixture
# component is N(μ_s, σ_s) (CHMM-N), t_{ν_s}(μ_s, σ_s) (CHMM-t), Laplace(μ_s, b_s)
# (CHMM-L), or GED(μ_s, α_s, p_s) (CHMM-GED). The mixture quantile uses binary
# search on the mixture CDF.
#
# Output: results/conditional_var_all_families/conditional_var_panel.txt
#         results/conditional_var_all_families/conditional_var_panel.csv
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, LinearAlgebra, Distributions, Printf

const SEED      = 20260420;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const LAMBDA_T  = 20.0;       # penalised CHMM-t shrinkage rate (body recommendation)

const OUT_DIR = joinpath(_ROOT, "results", "conditional_var_all_families");
mkpath(OUT_DIR);

println("="^80)
println("  B4: Regime-conditional VaR for CHMM-N / -t / -L / -GED on SPY OoS")
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

# -------- generic forward filter on family-specific emissions --------
function filter_predictive(y::AbstractVector, T::AbstractMatrix, components, π_init)
    K = length(components);
    n = length(y);
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

# -------- families: each returns (T_mat, π̄, components::Vector) --------
function fit_chmm_n(R_is, K)
    Random.seed!(SEED);
    m = build(MyContinuousHiddenMarkovModel,
        (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    K_actual = length(m.states);
    T_mat = zeros(K_actual, K_actual);
    for i in 1:K_actual; T_mat[i, :] = probs(m.transition[i]); end
    π̄ = (T_mat^2000)[1, :]; π̄ ./= sum(π̄);
    components = [m.emission[k] for k in 1:K_actual];
    return T_mat, π̄, components;
end

function fit_chmm_t(R_is, K; λ=LAMBDA_T)
    Random.seed!(SEED);
    m = build(MyStudentTHiddenMarkovModel,
        (observations=R_is, number_of_states=K, max_iter=MAX_ITER, ν_shrink_rate=λ));
    K_actual = length(m.states);
    T_mat = zeros(K_actual, K_actual);
    for i in 1:K_actual; T_mat[i, :] = probs(m.transition[i]); end
    π̄ = (T_mat^2000)[1, :]; π̄ ./= sum(π̄);
    components = [m.emission[k] for k in 1:K_actual];
    return T_mat, π̄, components;
end

function fit_chmm_l(R_is, K)
    Random.seed!(SEED);
    m = build(MyLaplaceHiddenMarkovModel,
        (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    K_actual = length(m.states);
    T_mat = zeros(K_actual, K_actual);
    for i in 1:K_actual; T_mat[i, :] = probs(m.transition[i]); end
    π̄ = (T_mat^2000)[1, :]; π̄ ./= sum(π̄);
    components = [m.emission[k] for k in 1:K_actual];
    return T_mat, π̄, components;
end

function fit_chmm_ged(R_is, K)
    Random.seed!(SEED);
    m = build(MyGEDHiddenMarkovModel,
        (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    K_actual = length(m.states);
    T_mat = zeros(K_actual, K_actual);
    for i in 1:K_actual; T_mat[i, :] = probs(m.transition[i]); end
    π̄ = (T_mat^2000)[1, :]; π̄ ./= sum(π̄);
    components = [m.emission[k] for k in 1:K_actual];
    return T_mat, π̄, components;
end

const FAMILIES = [
    ("CHMM-N", fit_chmm_n),
    ("CHMM-t (λ=$LAMBDA_T)", fit_chmm_t),
    ("CHMM-L", fit_chmm_l),
    ("CHMM-GED", fit_chmm_ged),
];

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
                var_thr[j] = v;
                breaches[j] = R_oos[j] < v;
            end
            n_br = sum(breaches);
            br_rate = n_br / n_oos;
            med_var = median(var_thr);
            k_uc = kupiec_lr(breaches, α);
            c_in = christoffersen_lr(breaches);
            c_cc = christoffersen_cc(breaches, α);
            @printf("  K=%2d  α=%.2f  br=%2d (%.2f%%)  med VaR=%.3f  LR_uc=%.3f  LR_ind=%.3f  LR_cc=%.3f (p=%.3f)\n",
                K, α, n_br, 100*br_rate, med_var, k_uc.LR, c_in.LR, c_cc.LR, c_cc.pvalue);
            push!(panels, (
                family=label, K=K, α=α,
                breaches=n_br, breach_rate=br_rate, med_var=med_var,
                LR_uc=k_uc.LR, p_uc=k_uc.pvalue,
                LR_ind=c_in.LR, p_ind=c_in.pvalue,
                LR_cc=c_cc.LR, p_cc=c_cc.pvalue,
            ));
        end
    end
end

# CSV
open(joinpath(OUT_DIR, "conditional_var_panel.csv"), "w") do io
    write(io, "family,K,alpha,breaches,breach_rate,med_var,LR_uc,p_uc,LR_ind,p_ind,LR_cc,p_cc\n");
    for r in panels
        write(io, @sprintf("%s,%d,%.2f,%d,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            r.family, r.K, r.α, r.breaches, r.breach_rate, r.med_var,
            r.LR_uc, r.p_uc, r.LR_ind, r.p_ind, r.LR_cc, r.p_cc));
    end
end

open(joinpath(OUT_DIR, "conditional_var_panel.txt"), "w") do io
    println(io, "B4: Regime-conditional VaR back-test across four CHMM emission families");
    println(io, "  K ∈ {3, 18}, α ∈ {0.01, 0.05}, OoS T = $n_oos days");
    println(io, "  Critical values: χ²_1(0.05) = 3.841, χ²_2(0.05) = 5.991");
    println(io, "="^110);
    @printf(io, "%-22s %-3s %-5s %-9s %-9s %-9s %-9s %-9s %-9s %-9s\n",
        "family", "K", "α", "breaches", "br rate", "med VaR", "LR_uc", "LR_ind", "LR_cc", "p_cc");
    println(io, "-"^110);
    for r in panels
        @printf(io, "%-22s %-3d %-5.2f %-9d %-9.4f %-9.3f %-9.3f %-9.3f %-9.3f %-9.3f\n",
            r.family, r.K, r.α, r.breaches, r.breach_rate, r.med_var,
            r.LR_uc, r.LR_ind, r.LR_cc, r.p_cc);
    end
end

println("\n[done] Output: $OUT_DIR")
