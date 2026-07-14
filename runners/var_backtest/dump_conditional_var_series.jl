# =========================================================================== #
# dump_conditional_var_series.jl
#
# Dump the per-day one-step-ahead regime-conditional VaR series for CHMM-N at
# K = 3 (the risk head) on the SPY OoS window, for the moving-VaR time-series
# figure. Reuses the exact conditional-VaR construction from
# run_conditional_var_all_families.jl (fit CHMM-N K=3, forward-filter the
# predictive state weights, take the mixture alpha-quantile each OoS day), but
# writes the daily series rather than summary statistics.
#
# Output: results/conditional_var_all_families/conditional_var_series_chmmN_k3.csv
#   columns: idx, date, R_oos, var_05, var_01, breach_05, breach_01, p_highvol
#   (p_highvol = filtered predictive weight on the highest-sigma state)
# Neither existing runner nor its committed output is modified.
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, LinearAlgebra, Distributions, Printf

const SEED      = 20260420;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const K         = 3;

const OUT_DIR = joinpath(_ROOT, "results", "conditional_var_all_families");
mkpath(OUT_DIR);

println("="^80)
println("  Daily regime-conditional VaR series, CHMM-N K=$K, SPY OoS")
println("="^80)

# -------- data (identical to run_conditional_var_all_families.jl) --------
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

# OoS dates: returns lose the first price, so align to the trailing n_oos dates.
spy_oos = oos_dataset["SPY"];
dates_full = hasproperty(spy_oos, :date) ? spy_oos[!, :date] : collect(1:nrow(spy_oos));
dates = length(dates_full) >= n_oos ? dates_full[end-n_oos+1:end] : collect(1:n_oos);

# -------- forward filter + mixture quantile (from the panel runner) --------
function filter_predictive(y, T, components, π_init)
    Kc = length(components); n = length(y);
    pred = zeros(n + 1, Kc); pred[1, :] = π_init;
    for t in 1:n
        b = [pdf(components[k], y[t]) for k in 1:Kc];
        post = pred[t, :] .* b; Z = sum(post);
        if Z <= 0; post .= pred[t, :]; else; post ./= Z; end
        pred[t + 1, :] = vec(post' * T);
    end
    return pred;
end
function mixture_quantile(α, weights, components; lo=-50.0, hi=50.0, tol=1e-6, max_iter=80)
    cdf_at(x) = sum(w * cdf(c, x) for (w, c) in zip(weights, components));
    a, b = lo, hi;
    if cdf_at(a) - α > 0; a = -200.0; end
    if cdf_at(b) - α < 0; b = 200.0; end
    for _ in 1:max_iter
        m = 0.5 * (a + b); fm = cdf_at(m) - α;
        if abs(fm) < tol; return m; end
        if fm < 0; a = m; else; b = m; end
    end
    return 0.5 * (a + b);
end

# -------- fit CHMM-N K=3, filter, build daily series --------
Random.seed!(SEED);
m = build(MyContinuousHiddenMarkovModel, (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
Kact = length(m.states);
T_mat = zeros(Kact, Kact);
for i in 1:Kact; T_mat[i, :] = probs(m.transition[i]); end
π̄ = (T_mat^2000)[1, :]; π̄ ./= sum(π̄);
components = [m.emission[k] for k in 1:Kact];
sigmas = [std(components[k]) for k in 1:Kact];
hi_state = argmax(sigmas);
println("[fit] K_actual=$Kact  sigmas=$(round.(sigmas, digits=3))  high-vol state=$hi_state")

pred = filter_predictive(vcat(R_is, R_oos), T_mat, components, π̄);

var_05 = zeros(n_oos); var_01 = zeros(n_oos); p_hi = zeros(n_oos);
for j in 1:n_oos
    w = pred[n_is + j, :];
    var_05[j] = mixture_quantile(0.05, w, components);
    var_01[j] = mixture_quantile(0.01, w, components);
    p_hi[j]   = w[hi_state];
end
br05 = R_oos .< var_05; br01 = R_oos .< var_01;
@printf("[series] breaches: 5%% = %d (%.2f%%)   1%% = %d (%.2f%%)\n",
        sum(br05), 100*mean(br05), sum(br01), 100*mean(br01));

open(joinpath(OUT_DIR, "conditional_var_series_chmmN_k3.csv"), "w") do io
    println(io, "idx,date,R_oos,var_05,var_01,breach_05,breach_01,p_highvol");
    for j in 1:n_oos
        @printf(io, "%d,%s,%.5f,%.5f,%.5f,%d,%d,%.5f\n",
            j, string(dates[j]), R_oos[j], var_05[j], var_01[j],
            br05[j] ? 1 : 0, br01[j] ? 1 : 0, p_hi[j]);
    end
end
println("[done] $OUT_DIR/conditional_var_series_chmmN_k3.csv")
