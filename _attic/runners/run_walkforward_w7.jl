# =========================================================================== #
# run_walkforward_w7.jl
#
# Peer-review item 15 (R2 W6): seventh walk-forward fold covering 2017-2018
# (Q4 2018 drawdown + 2019 trade-war volatility). Same protocol as the existing
# six-fold panel: 5y train / 1y test, CHMM-N at K ∈ {3, 18}, N_paths = 500,
# seed = 20260420 + 7K (matches run_walkforward_oos.jl).
#
# Output: results/walkforward/walkforward_w7.csv,
#         results/walkforward/walkforward_w7.txt
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, StatsBase, HypothesisTests, Distributions, Printf, Dates;

const SEED      = 20260420;
const K_VALUES  = [3, 18];
const N_PATHS   = 500;
const MAX_ITER  = 60;
const ALPHA_KS  = 0.05;
const RISK_FREE = 0.0;
const DT        = 1/252;
const L_LAGS    = 252;
const TICKER    = "SPY";

const OUT_DIR = joinpath(_ROOT, "results", "walkforward");
mkpath(OUT_DIR);

# Two new folds covering periods not in the existing W1-W6 layout.
# W7 (item 15): the 2017-2018 / 2019 trade-war / Q4 2018 drawdown band.
const FOLDS = [
    ("W7a", Date(2013,1,1), Date(2018,1,1), Date(2018,1,1), Date(2019,1,1)),  # test 2018, includes Q4 2018 drawdown
    ("W7b", Date(2014,1,1), Date(2019,1,1), Date(2019,1,1), Date(2020,1,1)),  # test 2019, trade-war volatility (overlaps W1)
];

Random.seed!(SEED);

println("="^70)
println("  Item 15: 7th walk-forward fold(s) for CHMM-N")
println("  Folds: ", [f[1] for f in FOLDS])
println("="^70)

train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
df_train = train_dataset[TICKER];
df_oos   = oos_dataset[TICKER];
df_full = [df_train; df_oos];
df_full = df_full[sortperm(df_full.timestamp), :];
println("[data] $TICKER full timeline: $(df_full[1, :timestamp]) -> $(df_full[end, :timestamp]) ($(nrow(df_full)) rows)");

function _slice_log_returns(df::DataFrame, t_start::Date, t_end_excl::Date; Δt::Float64=DT)
    ts = Date.(df.timestamp);
    mask = (ts .>= t_start) .& (ts .< t_end_excl);
    sub = df[mask, :];
    P = sub.volume_weighted_average_price;
    if length(P) < 2; return Float64[]; end
    R = (1 / Δt) .* (log.(P[2:end] ./ P[1:end-1])) .- 0.0;
    return Vector{Float64}(R);
end

function eval_fold(R_test::Vector{Float64}, sim::Matrix{Float64}; L::Int=L_LAGS, α::Float64=ALPHA_KS)
    np = size(sim, 2); n_o = length(R_test);
    L_use = min(L, n_o - 1);
    acf_o = autocor(abs.(R_test), 1:L_use);
    ks_pass = 0; kurt = 0.0; acf_mae = 0.0;
    for p in 1:np
        s = sim[:, p];
        ks_p = pvalue(ApproximateTwoSampleKSTest(s, R_test));
        ks_pass += (ks_p ≥ α) ? 1 : 0;
        kurt += kurtosis(s);
        acf_mae += mean(abs.(autocor(abs.(s), 1:L_use) .- acf_o));
    end
    return (ks_rate = 100 * ks_pass / np, kurt = kurt / np, acf_mae = acf_mae / np);
end

results = NamedTuple[];

for (fid, ts, te, vs, ve) in FOLDS
    println("\n" * "="^60);
    println("  Fold $fid:  train [$ts, $te)  test [$vs, $ve)");
    println("="^60);
    R_train = _slice_log_returns(df_full, ts, te);
    R_test  = _slice_log_returns(df_full, vs, ve);
    @printf("  T_train = %d   T_test = %d\n", length(R_train), length(R_test));
    if length(R_test) < 30
        @warn "Test slice < 30 days, skipping fold $fid";
        continue;
    end
    for K in K_VALUES
        Random.seed!(SEED + 7 * K);
        m = build(MyContinuousHiddenMarkovModel,
            (observations=R_train, number_of_states=K, max_iter=MAX_ITER));
        Random.seed!(SEED + 7 * K + 100);
        sim = simulate_returns(m, length(R_test); n_paths=N_PATHS);
        metr = eval_fold(R_test, sim);
        @printf("  CHMM-N (K=%d): KS = %5.1f%%   kurt_sim = %6.3f   ACF-MAE |G| = %.4f\n",
            K, metr.ks_rate, metr.kurt, metr.acf_mae);
        push!(results, (fold=fid, K=K, T_train=length(R_train), T_test=length(R_test),
                        ks_rate=metr.ks_rate, kurt=metr.kurt, acf_mae=metr.acf_mae));
    end
end

open(joinpath(OUT_DIR, "walkforward_w7.csv"), "w") do io
    write(io, "fold,K,T_train,T_test,KS_rate_pct,kurt_sim,acf_mae_abs\n");
    for r in results
        write(io, @sprintf("%s,%d,%d,%d,%.2f,%.4f,%.6f\n",
            r.fold, r.K, r.T_train, r.T_test, r.ks_rate, r.kurt, r.acf_mae));
    end
end

open(joinpath(OUT_DIR, "walkforward_w7.txt"), "w") do io
    println(io, "Item 15: seventh walk-forward fold(s) for CHMM-N at K = $K_VALUES");
    println(io, "Same protocol as run_walkforward_oos.jl: 5y train / 1y test, $N_PATHS paths/fold");
    println(io, "="^70);
    for K in K_VALUES
        rows = [r for r in results if r.K == K];
        if isempty(rows); continue; end
        @printf(io, "\nCHMM-N at K = %d:\n", K);
        @printf(io, "  %-5s %-9s %-9s %-9s %-9s %-9s\n",
            "fold", "T_train", "T_test", "KS (%)", "kurt", "ACF-MAE");
        for r in rows
            @printf(io, "  %-5s %-9d %-9d %-9.1f %-9.3f %-9.4f\n",
                r.fold, r.T_train, r.T_test, r.ks_rate, r.kurt, r.acf_mae);
        end
    end
    println(io);
    println(io, "Reading: W7a covers the 2018 calendar year including the Q4 2018 drawdown;");
    println(io, "W7b covers the 2019 calendar year including trade-war volatility (overlaps");
    println(io, "the existing W1 fold by construction; included as a sensitivity row).");
    println(io);
    println(io, "Combined with the existing W1-W6 panel (results/walkforward/walkforward_summary.txt)");
    println(io, "this gives a 7-fold panel for the body framing of Section 4.3.");
end

println("\n[done] Output: $OUT_DIR (walkforward_w7.{csv,txt})");
