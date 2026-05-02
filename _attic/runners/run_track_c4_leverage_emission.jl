# ========================================================================================= #
# run_track_c4_leverage_emission.jl
#
# Track C4: leverage-emission ablation. Replace per-state Gaussian emission with
#     r_t = μ_k + ρ_k * min(r_{t-1}, 0) + σ_k * ε_t,   ε_t ~ N(0,1)
# where ρ_k is a per-state coefficient on the previous-day negative-return indicator
# (r_{t-1}^- = min(r_{t-1}, 0)). Under leverage effect ρ_k should be negative for at least
# some states.
#
# Plug-in estimator: fit flat CHMM-N at K=18 (reuse existing), Viterbi-decode on R_is, then
# for each state regress r_t on r_{t-1}^- over the within-state (t-1, t) pairs. Simulate
# with the leverage recurrence.
#
# Outputs:
#   results/track_c4/leverage_emission_ablation.txt
#   results/track_c4/Table-4-Extended-Metrics-C4.txt
#   results/track_c4/VaR_LR_tests_c4.txt
#   results/track_c4/Track-C4-summary.txt
# ========================================================================================= #

const _PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."));
using Pkg; Pkg.activate(_PROJECT_ROOT);
include(joinpath(_PROJECT_ROOT, "Include.jl"));

using Random
const SEED = 20260422;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const N_PATHS   = 1000;
const K_MAIN    = 18;
const MAX_ITER  = 60;
const WINDOW_LEN = 20;
const N_WINDOWS  = 500;
const MAX_LAG_LEV = 20;

const TRACK_A_DIR = joinpath(_ROOT, "results", "track_a");
const TRACK_C_DIR = joinpath(_ROOT, "results", "track_c4");
mkpath(TRACK_C_DIR);

println("="^72)
println("  Track C4: leverage-emission ablation (CHMM-N-Lev)")
println("="^72)

# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY IS + OoS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
n_is = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
n_oos = length(R_oos);
println("  IS $n_is, OoS $n_oos");

# --------------------------------------------------------------------------------------- #
# Fit flat CHMM-N (reuse pattern from track_a)
# --------------------------------------------------------------------------------------- #
println("\n[fit] Fitting flat CHMM-N at K=$K_MAIN...");
Random.seed!(SEED + 500);
chmm_n = build(MyContinuousHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
println("  converged in $(length(chmm_n.log_likelihood_history)) iters");

decoded = viterbi(R_is, chmm_n);

# --------------------------------------------------------------------------------------- #
# Per-state ρ_k, μ_k, σ_k fit with leverage feature r_{t-1}^- = min(r_{t-1}, 0)
# --------------------------------------------------------------------------------------- #
println("\n[fit] Per-state leverage-emission regression...");

μ_k = Vector{Float64}(undef, K_MAIN);
ρ_k = Vector{Float64}(undef, K_MAIN);
σ_k = Vector{Float64}(undef, K_MAIN);
for k in 1:K_MAIN
    idx = [t for t in 2:n_is if decoded[t] == k];
    if length(idx) < 5
        μ_k[k] = mean(chmm_n.emission[k]);
        ρ_k[k] = 0.0;
        σ_k[k] = std(chmm_n.emission[k]);
        continue;
    end
    y  = R_is[idx];
    x  = [min(R_is[t-1], 0.0) for t in idx];
    # OLS with intercept
    X = hcat(ones(length(idx)), x);
    β = X \ y;
    resid = y .- X * β;
    μ_k[k] = β[1];
    ρ_k[k] = β[2];
    σ_k[k] = max(std(resid), 1e-4);
end

println("  per-state ρ summary: min=$(round(minimum(ρ_k), digits=4)) median=$(round(median(ρ_k), digits=4)) max=$(round(maximum(ρ_k), digits=4))");
println("  negative-ρ states: ", count(ρ_k .< 0), " / $K_MAIN (expected negative = leverage effect)");

# --------------------------------------------------------------------------------------- #
# Simulate N_PATHS paths IS + OoS using leverage-emission recursion
# --------------------------------------------------------------------------------------- #
println("\n[sim] Simulating $N_PATHS paths IS + OoS with leverage-emission recursion...");

# Stationary start distribution
T_mat = zeros(K_MAIN, K_MAIN);
for i in 1:K_MAIN; T_mat[i, :] = probs(chmm_n.transition[i]); end
π_stat = (T_mat ^ 1000)[1, :];
π_stat .= max.(π_stat, 1e-12); π_stat ./= sum(π_stat);
start_dist = Categorical(π_stat);

function _simulate_lev(n_steps::Int)
    out = Vector{Float64}(undef, n_steps);
    s = rand(start_dist);
    chain = chmm_n(s, n_steps);
    y_prev = mean(chmm_n.emission[chain[1]]);
    for t in 1:n_steps
        k = chain[t];
        y_prev_neg = min(y_prev, 0.0);
        out[t] = μ_k[k] + ρ_k[k] * y_prev_neg + σ_k[k] * randn();
        y_prev = out[t];
    end
    return out;
end

Random.seed!(SEED + 510);
lev_is  = Array{Float64,2}(undef, n_is,  N_PATHS);
lev_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    lev_is[:,  i] = _simulate_lev(n_is);
    lev_oos[:, i] = _simulate_lev(n_oos);
end

# --------------------------------------------------------------------------------------- #
# Evaluate: leverage effect, MMD, disc AUC, VaR
# --------------------------------------------------------------------------------------- #
println("\n[eval] Leverage-effect comparison...");
obs_lev_is  = leverage_effect(R_is;  max_lag=MAX_LAG_LEV);
obs_lev_oos = leverage_effect(R_oos; max_lag=MAX_LAG_LEV);

lev_is_samples  = [leverage_effect(lev_is[:, i];  max_lag=MAX_LAG_LEV).avg_neg for i in 1:N_PATHS];
lev_oos_samples = [leverage_effect(lev_oos[:, i]; max_lag=MAX_LAG_LEV).avg_neg for i in 1:N_PATHS];
asym_is_samples  = [leverage_effect(lev_is[:, i];  max_lag=MAX_LAG_LEV).asymmetry for i in 1:N_PATHS];
asym_oos_samples = [leverage_effect(lev_oos[:, i]; max_lag=MAX_LAG_LEV).asymmetry for i in 1:N_PATHS];

println("  observed IS  avg_neg = $(round(obs_lev_is.avg_neg, digits=4))  asym = $(round(obs_lev_is.asymmetry, digits=5))");
println("  CHMM-N-Lev IS  avg_neg = $(round(median(lev_is_samples), digits=4))  asym = $(round(median(asym_is_samples), digits=5))");
println("  p-value avg_neg (IS):  $(round(sim_pvalue(obs_lev_is.avg_neg, lev_is_samples), digits=3))");
println("  p-value avg_neg (OoS): $(round(sim_pvalue(obs_lev_oos.avg_neg, lev_oos_samples), digits=3))");

println("\n[eval] MMD + disc AUC...");
Random.seed!(SEED + 520);
obs_win_is = let
    W_all = windowize(R_is, WINDOW_LEN; stride=max(1, (n_is - WINDOW_LEN) ÷ N_WINDOWS));
    idx = randperm(size(W_all, 2))[1:min(N_WINDOWS, size(W_all, 2))];
    W_all[:, idx];
end;
syn_win_is = sample_windows_from_archive(lev_is, WINDOW_LEN, size(obs_win_is, 2); rng=MersenneTwister(SEED + 521));
mmd_is = mmd2_rbf(obs_win_is, syn_win_is; rng=MersenneTwister(SEED + 522));
sig_is = sig_mmd2(obs_win_is, syn_win_is; depth=3, rng=MersenneTwister(SEED + 523));
auc = discriminator_auc(R_is, lev_is; window=WINDOW_LEN, n_windows=N_WINDOWS, rng=MersenneTwister(SEED + 524));
println("  CHMM-N-Lev  MMD IS=$(round(mmd_is, digits=5))  sig-MMD IS=$(round(sig_is, digits=5))  disc AUC IS=$(round(auc.auc, digits=3))");

println("\n[eval] Unconditional VaR LR tests...");
v01 = quantile(vec(lev_is), 0.01); v05 = quantile(vec(lev_is), 0.05);
br01 = R_oos .<= v01; br05 = R_oos .<= v05;
k01 = kupiec_lr(br01, 0.01); k05 = kupiec_lr(br05, 0.05);
c01 = christoffersen_lr(br01); c05 = christoffersen_lr(br05);
println("  1 % VaR: LR_uc=$(round(k01.LR, digits=2)) br=$(round(100*k01.breach_rate, digits=1))% LR_ind=$(round(c01.LR, digits=2))");
println("  5 % VaR: LR_uc=$(round(k05.LR, digits=2)) br=$(round(100*k05.breach_rate, digits=1))% LR_ind=$(round(c05.LR, digits=2))");

# --------------------------------------------------------------------------------------- #
# Compare vs flat CHMM-N from Track A
# --------------------------------------------------------------------------------------- #
cache_path = joinpath(TRACK_A_DIR, "sim_archive_cache.jld2");
base_archive = load(cache_path)["archive"];
flat_n_is  = base_archive["CHMM-N"].is;
flat_n_oos = base_archive["CHMM-N"].oos;

flat_lev_is_samples  = [leverage_effect(flat_n_is[:, i];  max_lag=MAX_LAG_LEV).avg_neg for i in axes(flat_n_is, 2)];
flat_lev_oos_samples = [leverage_effect(flat_n_oos[:, i]; max_lag=MAX_LAG_LEV).avg_neg for i in axes(flat_n_oos, 2)];
flat_asym_is = [leverage_effect(flat_n_is[:, i]; max_lag=MAX_LAG_LEV).asymmetry for i in axes(flat_n_is, 2)];

flat_pv_is  = sim_pvalue(obs_lev_is.avg_neg, flat_lev_is_samples);
flat_pv_oos = sim_pvalue(obs_lev_oos.avg_neg, flat_lev_oos_samples);
lev_pv_is  = sim_pvalue(obs_lev_is.avg_neg, lev_is_samples);
lev_pv_oos = sim_pvalue(obs_lev_oos.avg_neg, lev_oos_samples);

# --------------------------------------------------------------------------------------- #
# Write outputs
# --------------------------------------------------------------------------------------- #
open(joinpath(TRACK_C_DIR, "leverage_emission_ablation.txt"), "w") do io
    println(io, "="^95);
    println(io, "Track C4. Leverage-emission ablation (CHMM-N-Lev) vs flat CHMM-N.");
    println(io, "="^95);
    println(io, "");
    println(io, "Emission  : r_t = μ_k + ρ_k * min(r_{t-1}, 0) + σ_k * ε_t,  ε_t ~ N(0,1)");
    println(io, "Estimator : Viterbi decode on flat CHMM-N at K=$K_MAIN, then per-state OLS on (r_{t-1}^-, r_t).");
    println(io, "Seeds     : $SEED.");
    println(io, "");
    println(io, "Per-state ρ_k distribution:");
    println(io, "  min=$(round(minimum(ρ_k), digits=4))  median=$(round(median(ρ_k), digits=4))  max=$(round(maximum(ρ_k), digits=4))");
    println(io, "  negative-ρ states: $(count(ρ_k .< 0)) / $K_MAIN");
    println(io, "");
    println(io, "Observed leverage metric (IS):  avg_neg = $(round(obs_lev_is.avg_neg, digits=4)),  asym = $(round(obs_lev_is.asymmetry, digits=5))");
    println(io, "Observed leverage metric (OoS): avg_neg = $(round(obs_lev_oos.avg_neg, digits=4)), asym = $(round(obs_lev_oos.asymmetry, digits=5))");
    println(io, "");
    println(io, rpad("Model", 14), " | avg_neg IS  | asym IS    | pv IS     | avg_neg OoS | asym OoS   | pv OoS");
    println(io, "-"^95);
    println(io, rpad("CHMM-N",      14), " | ",
                rpad(round(median(flat_lev_is_samples),  digits=4), 10), " | ",
                rpad(round(median(flat_asym_is),         digits=4), 9),  " | ",
                rpad(round(flat_pv_is, digits=3), 9), " | ",
                rpad(round(median(flat_lev_oos_samples), digits=4), 11), " | ",
                rpad("-", 10), " | ", rpad(round(flat_pv_oos, digits=3), 9));
    println(io, rpad("CHMM-N-Lev",  14), " | ",
                rpad(round(median(lev_is_samples), digits=4), 10), " | ",
                rpad(round(median(asym_is_samples), digits=4), 9),  " | ",
                rpad(round(lev_pv_is, digits=3), 9), " | ",
                rpad(round(median(lev_oos_samples), digits=4), 11), " | ",
                rpad(round(median(asym_oos_samples), digits=4), 10), " | ",
                rpad(round(lev_pv_oos, digits=3), 9));
    println(io, "="^95);
    println(io, "");
    println(io, "Interpretation: a negative avg_neg indicates the leverage effect (negative past returns");
    println(io, "predict larger subsequent squared returns). pv near 0.5 means the model covers observed well.");
end

# Summary
open(joinpath(TRACK_C_DIR, "Track-C4-summary.txt"), "w") do io
    println(io, "Track C4 summary: leverage-emission ablation (CHMM-N-Lev)");
    println(io, "="^70);
    println(io, "");
    println(io, "Fit: per-state ρ_k regression, $(count(ρ_k .< 0)) of $K_MAIN states negative (expected).");
    println(io, "     ρ range [$(round(minimum(ρ_k), digits=4)), $(round(maximum(ρ_k), digits=4))], median $(round(median(ρ_k), digits=4)).");
    println(io, "");
    println(io, "Leverage-effect metric:");
    println(io, "  Observed IS avg_neg = $(round(obs_lev_is.avg_neg, digits=4))");
    println(io, "  Flat CHMM-N  IS median avg_neg = $(round(median(flat_lev_is_samples), digits=4))  pv = $(round(flat_pv_is, digits=3))");
    println(io, "  CHMM-N-Lev   IS median avg_neg = $(round(median(lev_is_samples),   digits=4))  pv = $(round(lev_pv_is,  digits=3))");
    println(io, "  Observed OoS avg_neg = $(round(obs_lev_oos.avg_neg, digits=4))");
    println(io, "  Flat CHMM-N  OoS median avg_neg = $(round(median(flat_lev_oos_samples), digits=4))  pv = $(round(flat_pv_oos, digits=3))");
    println(io, "  CHMM-N-Lev   OoS median avg_neg = $(round(median(lev_oos_samples),   digits=4))  pv = $(round(lev_pv_oos,  digits=3))");
    println(io, "");
    println(io, "MMD IS = $(round(mmd_is, digits=5)) (flat CHMM-N was 0.00015 in Track A)");
    println(io, "disc AUC IS = $(round(auc.auc, digits=3)) (flat CHMM-N was 0.646)");
    println(io, "");
    println(io, "1 % VaR: LR_uc=$(round(k01.LR, digits=2)) (flat 1.58), breach=$(round(100*k01.breach_rate, digits=1))%");
    println(io, "5 % VaR: LR_uc=$(round(k05.LR, digits=2)) (flat 3.83), breach=$(round(100*k05.breach_rate, digits=1))%");
end

println("\n" * "="^72);
println("  Track C4 complete.");
println("  Results: $TRACK_C_DIR");
println("="^72);
