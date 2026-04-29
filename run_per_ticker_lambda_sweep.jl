# =========================================================================== #
# run_per_ticker_lambda_sweep.jl
#
# B1 (REVISION_PLAN_V3_TO_ACCEPT.md): per-ticker λ* shrinkage sweep for CHMM-t.
# For each of the six body-panel tickers (SPY, NVDA, JNJ, JPM, AAPL, QQQ),
# sweep λ ∈ {0, 5, 10, 20, 50, 100} and define
#
#     λ̂* = argmin_λ |kurt_sim(λ) − kurt_obs|   subject to IS-KS_λ ≥ IS-KS_0 − 1.5pp.
#
# Output:
#   results/per_ticker_lambda/per_ticker_lambda_summary.csv
#   results/per_ticker_lambda/per_ticker_lambda_summary.txt
# =========================================================================== #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using StatsBase
using HypothesisTests
using Printf
using Distributions

const SEED        = 20260420;
const TICKERS     = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];
const LAMBDA_GRID = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0];
const K_MAIN      = 18;
const N_PATHS     = 500;
const MAX_ITER    = 60;
const ALPHA_KS    = 0.05;
const RISK_FREE   = 0.0;
const DT          = 1/252;
const KS_TOL_PP   = 1.5;             # max IS KS degradation vs λ = 0 baseline (pp)

const OUT_DIR = joinpath(_ROOT, "results", "per_ticker_lambda");
mkpath(OUT_DIR);

Random.seed!(SEED);

println("="^70)
println("  B1: Per-ticker λ* sweep for CHMM-t (K = $K_MAIN, λ grid $(LAMBDA_GRID))")
println("="^70)

# Data
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
println("[data] Loaded $(length(all_tickers)) tickers, T_IS = $(size(all_R, 1))")

function ks_pass_rate(R_ref::Vector{Float64}, sim::Matrix{Float64}; α::Float64=0.05)
    n = size(sim, 2); pass = 0;
    for p in 1:n
        pv = pvalue(ApproximateTwoSampleKSTest(R_ref, sim[:, p]));
        pass += (pv ≥ α) ? 1 : 0;
    end
    return 100 * pass / n;
end

per_ticker_results = Dict{String,Any}();

for (ix, tk) in enumerate(TICKERS)
    println("\n" * "="^60)
    println("  Ticker $tk ($ix/$(length(TICKERS)))")
    println("="^60)
    idx_tk = findfirst(==(tk), all_tickers);
    R_tk = Vector{Float64}(all_R[:, idx_tk]);
    n_tk = length(R_tk);
    kurt_obs = kurtosis(R_tk);  # excess kurtosis
    @printf("  obs excess kurtosis = %.3f, T = %d\n", kurt_obs, n_tk)

    rows = Vector{NamedTuple}();
    base_ks = NaN;
    for λ in LAMBDA_GRID
        Random.seed!(SEED + 1000 * ix + Int(λ));
        m = build(MyStudentTHiddenMarkovModel,
            (observations=R_tk, number_of_states=K_MAIN, max_iter=MAX_ITER,
             ν_shrink_rate=λ));
        Random.seed!(SEED + 1000 * ix + Int(λ) + 7);
        sim = simulate_returns(m, n_tk; n_paths=N_PATHS);
        ks_rate = ks_pass_rate(R_tk, sim);
        kurt_sim = mean([kurtosis(sim[:, p]) for p in 1:N_PATHS]);
        if λ == LAMBDA_GRID[1]; base_ks = ks_rate; end
        @printf("  λ=%6.1f  IS KS = %5.1f%%  kurt_sim = %7.3f  |Δkurt| = %6.3f\n",
            λ, ks_rate, kurt_sim, abs(kurt_sim - kurt_obs))
        push!(rows, (λ=λ, ks=ks_rate, kurt=kurt_sim));
    end
    feasible = [r for r in rows if r.ks ≥ base_ks - KS_TOL_PP];
    if isempty(feasible); feasible = rows; end
    best = argmin([abs(r.kurt - kurt_obs) for r in feasible]);
    λ_star = feasible[best].λ;
    @printf("  >> λ̂* = %.1f (kurt_sim = %.3f vs obs %.3f, KS = %.1f%%)\n",
        λ_star, feasible[best].kurt, kurt_obs, feasible[best].ks)
    per_ticker_results[tk] = (rows=rows, λ_star=λ_star, kurt_obs=kurt_obs, base_ks=base_ks);
end

# Summary CSV + text
open(joinpath(OUT_DIR, "per_ticker_lambda_summary.csv"), "w") do io
    write(io, "ticker,kurt_obs,lambda_star,kurt_at_star,KS_at_star\n");
    for tk in TICKERS
        r = per_ticker_results[tk];
        b = first([row for row in r.rows if row.λ == r.λ_star]);
        write(io, @sprintf("%s,%.3f,%.1f,%.3f,%.2f\n", tk, r.kurt_obs, r.λ_star, b.kurt, b.ks));
    end
end

open(joinpath(OUT_DIR, "per_ticker_lambda_summary.txt"), "w") do io
    println(io, "B1: Per-ticker λ* shrinkage sweep for CHMM-t (K = $K_MAIN, $N_PATHS paths)");
    println(io, "  KS-degradation tolerance: $KS_TOL_PP pp from λ = 0 baseline");
    println(io, "="^70);
    for tk in TICKERS
        r = per_ticker_results[tk];
        @printf(io, "\n%s (kurt_obs = %.3f, base IS KS = %.1f%%):\n", tk, r.kurt_obs, r.base_ks);
        @printf(io, "  %-8s %-12s %-12s %-12s\n", "λ", "IS-KS (%)", "kurt_sim", "|Δkurt|");
        for row in r.rows
            @printf(io, "  %-8.1f %-12.1f %-12.3f %-12.3f%s\n",
                row.λ, row.ks, row.kurt, abs(row.kurt - r.kurt_obs),
                row.λ == r.λ_star ? "  <-- λ*" : "")
        end
    end
    println(io);
    println(io, "Summary:");
    @printf(io, "  %-8s %-10s %-12s %-12s %-12s\n", "Ticker", "kurt_obs", "λ*", "kurt(λ*)", "KS(λ*)");
    for tk in TICKERS
        r = per_ticker_results[tk];
        b = first([row for row in r.rows if row.λ == r.λ_star]);
        @printf(io, "  %-8s %-10.3f %-12.1f %-12.3f %-12.1f\n", tk, r.kurt_obs, r.λ_star, b.kurt, b.ks);
    end
end

println("\n[done] Output: $OUT_DIR")
