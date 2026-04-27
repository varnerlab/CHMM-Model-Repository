# ========================================================================================= #
# run_non_us_asset.jl
#
#
# Adds GLD (SPDR Gold Trust ETF, commodity asset class with non-equity underlying)
# to the existing six-ticker universe (SPY, NVDA, JNJ, JPM, AAPL, QQQ) and reports:
#
#   1. Pipeline-A univariate fidelity for GLD across all three CHMM emission families:
#      KS IS/OoS, simulated kurtosis vs observed, ACF-MAE.
#   2. Pipeline-B Student-t copula off-diagonal MAE on the 7-ticker universe IS and OoS.
#   3. Per-pair off-diagonal MAE breakdown so the IS-to-OoS degradation noted in
#      results.tex can be localised to specific pairs.
#
# Inputs : standard CHMM-SP500-Train-10yr.jld2 + CHMM-SP500-OoS-Remainder.jld2 (both
#          already contain GLD; verified via diagnostic).
# Outputs: results/non_us_asset/Non_US_Asset.txt
#          results/non_us_asset/Per_Pair_OffDiag_MAE.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random, Statistics, Printf
const SEED = 20260422;
Random.seed!(SEED);

const RISK_FREE     = 0.0;
const DT            = 1/252;
const N_PATHS       = 200;          # match Pipeline B convention
const K_MAIN        = 18;
const MAX_ITER      = 60;
const ASSETS_BASE   = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];
const NEW_ASSET     = "GLD";
const ASSETS        = vcat(ASSETS_BASE, [NEW_ASSET]);
const MARKET        = "SPY";

const OUT_DIR = joinpath(_ROOT, "results", "non_us_asset");
mkpath(OUT_DIR);
const OUT_PATH      = joinpath(OUT_DIR, "Non_US_Asset.txt");
const PAIRS_PATH    = joinpath(OUT_DIR, "Per_Pair_OffDiag_MAE.txt");

println("="^88);
println("  Non-US asset class extension (commodity: $NEW_ASSET).");
println("  Universe: $ASSETS");
println("  Seed: $SEED");
println("="^88);

# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading IS + OoS for the augmented universe...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];

# Trim training set to common max-day rows (mirrors run_cross_asset_sim_copula.jl)
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
available = [t for t in ASSETS if haskey(dataset, t) && haskey(oos_dataset, t)];
println("  available: ", available);
@assert NEW_ASSET in available "GLD missing from one of the standard datasets";

R_is_list  = [log_growth_matrix(dataset, t; Δt=DT, risk_free_rate=RISK_FREE) for t in available];
R_oos_list = [log_growth_matrix(oos_dataset, t; Δt=DT, risk_free_rate=RISK_FREE) for t in available];

n_is  = minimum(length.(R_is_list));
n_oos = minimum(length.(R_oos_list));
R_is  = hcat([r[1:n_is]  for r in R_is_list]...);
R_oos = hcat([r[1:n_oos] for r in R_oos_list]...);
d     = length(available);
idx_m = findfirst(x -> x == MARKET, available);
idx_g = findfirst(x -> x == NEW_ASSET, available);
println("  IS matrix: $(size(R_is)), OoS matrix: $(size(R_oos))");

# --------------------------------------------------------------------------------------- #
# Pipeline-A univariate fidelity for GLD across the three emission families
# --------------------------------------------------------------------------------------- #
function _stationary_start_dist(model, K::Int)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    π_stat = (T_mat^1000)[1, :];
    return Categorical(π_stat);
end

function _simulate_chmm_paths(model, start_dist, n_is::Int, n_oos::Int, np::Int)
    sim_is  = Array{Float64,2}(undef, n_is,  np);
    sim_oos = Array{Float64,2}(undef, n_oos, np);
    for i in 1:np
        s0 = rand(start_dist); st = model(s0, n_is);
        for j in 1:n_is;  sim_is[j, i]  = rand(model.emission[st[j]]); end
        s0 = rand(start_dist); st = model(s0, n_oos);
        for j in 1:n_oos; sim_oos[j, i] = rand(model.emission[st[j]]); end
    end
    return sim_is, sim_oos;
end

function _ks_pass_rate(R_obs::AbstractVector, R_sim::AbstractMatrix; α::Float64=0.05)
    np = size(R_sim, 2);
    pass = 0;
    for i in 1:np
        ks = HypothesisTests.ApproximateTwoSampleKSTest(R_obs, R_sim[:, i]);
        if pvalue(ks) >= α; pass += 1; end
    end
    return 100.0 * pass / np;
end

function _mean_excess_kurt(R_sim::AbstractMatrix)
    np = size(R_sim, 2);
    s = 0.0;
    for i in 1:np; s += kurtosis(R_sim[:, i]); end
    return s / np;
end

function _acf_mae(R_obs::AbstractVector, R_sim::AbstractMatrix; L::Int=252)
    obs_acf = autocor(abs.(R_obs), 1:L);
    np = size(R_sim, 2);
    sim_acf_mean = zeros(L);
    for i in 1:np
        sim_acf_mean .+= autocor(abs.(R_sim[:, i]), 1:L);
    end
    sim_acf_mean ./= np;
    return mean(abs.(sim_acf_mean .- obs_acf));
end

R_is_gld  = R_is[:, idx_g];
R_oos_gld = R_oos[:, idx_g];

println("\n[A] Pipeline-A univariate fidelity for $NEW_ASSET (K=$K_MAIN, $N_PATHS paths each)");

println("  CHMM-N...");
Random.seed!(SEED + 11);
chmm_n_gld = build(MyContinuousHiddenMarkovModel, (
    observations=R_is_gld, number_of_states=K_MAIN, max_iter=MAX_ITER));
sd_n = _stationary_start_dist(chmm_n_gld, K_MAIN);
n_is_sim, n_oos_sim = _simulate_chmm_paths(chmm_n_gld, sd_n, n_is, n_oos, N_PATHS);
gld_n = (
    is_ks    = _ks_pass_rate(R_is_gld,  n_is_sim),
    oos_ks   = _ks_pass_rate(R_oos_gld, n_oos_sim),
    is_kurt  = _mean_excess_kurt(n_is_sim),
    oos_kurt = _mean_excess_kurt(n_oos_sim),
    acf_mae  = _acf_mae(R_is_gld, n_is_sim),
);

println("  CHMM-t...");
Random.seed!(SEED + 12);
chmm_t_gld = build(MyStudentTHiddenMarkovModel, (
    observations=R_is_gld, number_of_states=K_MAIN, max_iter=MAX_ITER));
sd_t = _stationary_start_dist(chmm_t_gld, K_MAIN);
t_is_sim, t_oos_sim = _simulate_chmm_paths(chmm_t_gld, sd_t, n_is, n_oos, N_PATHS);
gld_t = (
    is_ks    = _ks_pass_rate(R_is_gld,  t_is_sim),
    oos_ks   = _ks_pass_rate(R_oos_gld, t_oos_sim),
    is_kurt  = _mean_excess_kurt(t_is_sim),
    oos_kurt = _mean_excess_kurt(t_oos_sim),
    acf_mae  = _acf_mae(R_is_gld, t_is_sim),
);

println("  CHMM-L...");
Random.seed!(SEED + 13);
chmm_l_gld = build(MyLaplaceHiddenMarkovModel, (
    observations=R_is_gld, number_of_states=K_MAIN, max_iter=MAX_ITER));
sd_l = _stationary_start_dist(chmm_l_gld, K_MAIN);
l_is_sim, l_oos_sim = _simulate_chmm_paths(chmm_l_gld, sd_l, n_is, n_oos, N_PATHS);
gld_l = (
    is_ks    = _ks_pass_rate(R_is_gld,  l_is_sim),
    oos_ks   = _ks_pass_rate(R_oos_gld, l_oos_sim),
    is_kurt  = _mean_excess_kurt(l_is_sim),
    oos_kurt = _mean_excess_kurt(l_oos_sim),
    acf_mae  = _acf_mae(R_is_gld, l_is_sim),
);

obs_is_kurt  = kurtosis(R_is_gld);
obs_oos_kurt = kurtosis(R_oos_gld);

# --------------------------------------------------------------------------------------- #
# Pipeline-B: 7-ticker Student-t copula off-diag MAE
# --------------------------------------------------------------------------------------- #
println("\n[B] Pipeline-B Student-t copula on 7-ticker universe (K=$K_MAIN marginals)");

# Fit per-asset CHMM-N marginals for all 7 tickers
println("  Fitting per-asset CHMM-N marginals...");
chmms = Vector{AbstractMarkovModel}(undef, d);
for j in 1:d
    print("    $(available[j])...");
    Random.seed!(SEED + 100 + j);
    chmms[j] = build(MyContinuousHiddenMarkovModel, (
        observations=R_is[:, j], number_of_states=K_MAIN, max_iter=MAX_ITER));
    println(" done.");
end

println("  Fitting Student-t copula on 7-ticker universe...");
Random.seed!(SEED + 200);
t_copula_7 = build(MyStudentTCopulaModel, (
    returns=R_is, tickers=available, marginals=chmms));
println("    ν* = ", t_copula_7.nu);

println("  Simulating Student-t copula paths ($N_PATHS paths)...");
Random.seed!(SEED + 201);
t_paths_is_7  = simulate(t_copula_7, n_is,  N_PATHS);
Random.seed!(SEED + 202);
t_paths_oos_7 = simulate(t_copula_7, n_oos, N_PATHS);

# Off-diagonal MAE on 7-ticker universe
function _offdiag_mae(Σ_a::AbstractMatrix, Σ_b::AbstractMatrix)
    n = size(Σ_a, 1);
    s = 0.0; c = 0;
    for i in 1:n, j in 1:n
        if i != j
            s += abs(Σ_a[i, j] - Σ_b[i, j]); c += 1;
        end
    end
    return s / c;
end

Σ_obs_is  = cor(R_is);
Σ_obs_oos = cor(R_oos);

mae_is_7  = mean([_offdiag_mae(cor(t_paths_is_7[:, :, p]),  Σ_obs_is)  for p in 1:N_PATHS]);
mae_oos_7 = mean([_offdiag_mae(cor(t_paths_oos_7[:, :, p]), Σ_obs_oos) for p in 1:N_PATHS]);

# Repeat for 6-ticker (no GLD) baseline as comparison
println("  Refitting Student-t copula on 6-ticker baseline (no $NEW_ASSET) for comparison...");
base_idx = [j for j in 1:d if available[j] != NEW_ASSET];
R_is_6   = R_is[:, base_idx];
R_oos_6  = R_oos[:, base_idx];
chmms_6  = chmms[base_idx];
tickers_6 = available[base_idx];

Random.seed!(SEED + 300);
t_copula_6 = build(MyStudentTCopulaModel, (
    returns=R_is_6, tickers=tickers_6, marginals=chmms_6));
Random.seed!(SEED + 301);
t_paths_is_6  = simulate(t_copula_6, n_is,  N_PATHS);
Random.seed!(SEED + 302);
t_paths_oos_6 = simulate(t_copula_6, n_oos, N_PATHS);

Σ_obs_is_6  = cor(R_is_6);
Σ_obs_oos_6 = cor(R_oos_6);
mae_is_6  = mean([_offdiag_mae(cor(t_paths_is_6[:, :, p]),  Σ_obs_is_6)  for p in 1:N_PATHS]);
mae_oos_6 = mean([_offdiag_mae(cor(t_paths_oos_6[:, :, p]), Σ_obs_oos_6) for p in 1:N_PATHS]);

# --------------------------------------------------------------------------------------- #
# Per-pair off-diag MAE on the 7-ticker universe (IS and OoS)
# --------------------------------------------------------------------------------------- #
println("\n[C] Per-pair off-diagonal MAE breakdown (7-ticker universe)");

# Mean simulated correlation matrix across paths
Σ_sim_is_avg  = zeros(d, d);
Σ_sim_oos_avg = zeros(d, d);
for p in 1:N_PATHS
    Σ_sim_is_avg  .+= cor(t_paths_is_7[:,  :, p]);
    Σ_sim_oos_avg .+= cor(t_paths_oos_7[:, :, p]);
end
Σ_sim_is_avg  ./= N_PATHS;
Σ_sim_oos_avg ./= N_PATHS;

# --------------------------------------------------------------------------------------- #
# Write report
# --------------------------------------------------------------------------------------- #
println("\n[write] $OUT_PATH");

open(OUT_PATH, "w") do io
    println(io, "="^110);
    println(io, "Non-US asset class extension (commodity: $NEW_ASSET).");
    println(io, "="^110);
    println(io);
    println(io, "Setup     : seed=$SEED; $N_PATHS simulated paths per generator; K=$K_MAIN; α=0.05 for KS.");
    println(io, "Universe  : $(join(available, ", "))");
    println(io, "New asset : $NEW_ASSET (SPDR Gold Trust ETF). Asset class = commodity (gold).");
    println(io, "Underlying: physically-backed gold; non-equity asset class held in US-listed wrapper.");
    println(io, "IS / OoS  : T_IS = $n_is, T_OoS = $n_oos.");
    println(io);
    println(io, "-"^110);
    println(io, "1. Pipeline-A univariate fidelity for $NEW_ASSET");
    println(io, "-"^110);
    println(io);
    @printf(io, "%-12s | %-7s | %-7s | %-9s | %-9s | %-9s\n",
            "Family", "IS KS%", "OoS KS%", "Sim IS κ", "Sim OoS κ", "ACF-MAE");
    @printf(io, "%-12s | %-7s | %-7s | %-9.3f | %-9.3f | %-9s\n",
            "Observed",  "--",   "--",   obs_is_kurt, obs_oos_kurt, "--");
    @printf(io, "%-12s | %-7.1f | %-7.1f | %-9.3f | %-9.3f | %-9.4f\n",
            "CHMM-N",  gld_n.is_ks, gld_n.oos_ks, gld_n.is_kurt, gld_n.oos_kurt, gld_n.acf_mae);
    @printf(io, "%-12s | %-7.1f | %-7.1f | %-9.3f | %-9.3f | %-9.4f\n",
            "CHMM-t",  gld_t.is_ks, gld_t.oos_ks, gld_t.is_kurt, gld_t.oos_kurt, gld_t.acf_mae);
    @printf(io, "%-12s | %-7.1f | %-7.1f | %-9.3f | %-9.3f | %-9.4f\n",
            "CHMM-L",  gld_l.is_ks, gld_l.oos_ks, gld_l.is_kurt, gld_l.oos_kurt, gld_l.acf_mae);
    println(io);
    println(io, "-"^110);
    println(io, "2. Pipeline-B Student-t copula off-diagonal MAE: 7-ticker (with $NEW_ASSET) vs 6-ticker baseline");
    println(io, "-"^110);
    println(io);
    @printf(io, "%-25s | %-12s | %-12s | ν*\n", "Universe", "IS off-MAE", "OoS off-MAE");
    @printf(io, "%-25s | %-12.4f | %-12.4f | %d\n",
            "6-ticker baseline",      mae_is_6, mae_oos_6, Int(t_copula_6.nu));
    @printf(io, "%-25s | %-12.4f | %-12.4f | %d\n",
            "7-ticker (with $NEW_ASSET)", mae_is_7, mae_oos_7, Int(t_copula_7.nu));
    println(io);
    println(io, "Reading:");
    println(io, "  - The 6-ticker IS off-MAE here re-derives the headline 0.027 number from results.tex");
    println(io, "    (small numerical drift expected because this run uses an independent seed offset).");
    println(io, "  - The 7-ticker number quantifies how the dependence layer scales when a non-US asset");
    println(io, "    class is added to the universe. Equity-bias diagnostic: if 7-ticker << 6-ticker on");
    println(io, "    OoS, the OoS correlation gap is partly an equity-cluster artefact.");
    println(io);
    println(io, "="^110);
end

println("\n[write] $PAIRS_PATH");
open(PAIRS_PATH, "w") do io
    println(io, "="^110);
    println(io, "Per-pair off-diagonal MAE on the 7-ticker universe.");
    println(io, "="^110);
    println(io);
    println(io, "For each ticker pair (i, j), reports |Σ_sim_avg[i,j] − Σ_obs[i,j]| separately for IS and OoS.");
    println(io, "Σ_sim_avg is the path-mean correlation matrix from the Student-t copula on CHMM-N marginals.");
    println(io);
    @printf(io, "%-6s | %-6s | %-9s | %-9s | %-9s | %-9s | %-9s\n",
            "Asset i", "Asset j", "IS obs", "IS sim", "IS |Δ|", "OoS obs", "OoS |Δ|");
    println(io, "-"^88);
    for i in 1:d, j in (i+1):d
        @printf(io, "%-6s | %-6s | %-9.4f | %-9.4f | %-9.4f | %-9.4f | %-9.4f\n",
                available[i], available[j],
                Σ_obs_is[i, j],  Σ_sim_is_avg[i, j],
                abs(Σ_sim_is_avg[i, j]  - Σ_obs_is[i, j]),
                Σ_obs_oos[i, j],
                abs(Σ_sim_oos_avg[i, j] - Σ_obs_oos[i, j]));
    end
    println(io);
    println(io, "-"^88);
    println(io);
    println(io, "Pairs ranked by OoS |Δ| (largest IS-to-OoS degradation first):");
    pairs = NamedTuple[];
    for i in 1:d, j in (i+1):d
        push!(pairs, (i=i, j=j,
            is_delta  = abs(Σ_sim_is_avg[i, j]  - Σ_obs_is[i, j]),
            oos_delta = abs(Σ_sim_oos_avg[i, j] - Σ_obs_oos[i, j])));
    end
    sort!(pairs, by = p -> -p.oos_delta);
    println(io);
    @printf(io, "%-15s | %-9s | %-9s | %-12s\n", "Pair", "IS |Δ|", "OoS |Δ|", "OoS - IS");
    println(io, "-"^60);
    for p in pairs
        @printf(io, "%-15s | %-9.4f | %-9.4f | %-12.4f\n",
                "$(available[p.i])-$(available[p.j])",
                p.is_delta, p.oos_delta, p.oos_delta - p.is_delta);
    end
    println(io);
    println(io, "="^110);
end

println("\n[done] Non-US asset run complete.");
println("       Headline:    $OUT_PATH");
println("       Per-pair:    $PAIRS_PATH");
