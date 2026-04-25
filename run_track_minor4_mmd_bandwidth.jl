# ========================================================================================= #
# run_track_minor4_mmd_bandwidth.jl
#
# Track Minor 4 (revision response): the existing MMD calculation uses median-heuristic
# bandwidth γ computed from the pooled (observed + simulated) window matrix, which makes
# γ a function of the generator under evaluation and therefore makes the cross-generator
# MMD ranking unstable. The referee asks to fix γ to the median-heuristic on the
# observed-only sample so the kernel scale is identical across all generators.
#
# This script: (i) computes the fixed γ_obs from 500 observed 20-day windows; (ii)
# simulates each generator's IS path archive; (iii) recomputes MMD against observed under
# the fixed γ_obs; (iv) reports the comparison and the cross-generator ranking under the
# fixed bandwidth.
#
# Output:
#   results/track_minor4/MMD_Bandwidth_Fix.txt
#   ../CHMM-paper/results/revision/minor4_mmd_bandwidth.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using Printf
const SEED = 20260422;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const N_PATHS   = 500;
const K_MAIN    = 18;
const MAX_ITER  = 60;
const W         = 20;        # window length
const N_WINDOWS = 500;       # MMD sample size per side

const TRACK_DIR        = joinpath(_ROOT, "results", "track_minor4");
const PAPER_REVISION_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "revision"));
mkpath(TRACK_DIR); mkpath(PAPER_REVISION_DIR);

println("="^72)
println("  Track Minor 4: fixed observed-sample MMD bandwidth (referee Minor 4)")
println("="^72)

println("\n[data] Loading SPY IS...");
train = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train["AAPL"]);
ds = Dict{String,DataFrame}();
for (t, data) in train; if nrow(data) == max_days; ds[t] = data; end; end
all_tickers = keys(ds) |> collect |> sort;
all_R = log_growth_matrix(ds, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
n_is = length(R_is);
println("  IS $n_is");

# Observed 20-day windows + standardisation (z-score per window)
function _standardize_columns!(M::Matrix{Float64})
    for j in 1:size(M, 2)
        col = M[:, j];
        μ = mean(col); s = std(col);
        if s > 1e-12; M[:, j] = (col .- μ) ./ s; end
    end
    return M;
end

println("\n[bandwidth] computing fixed γ from observed-only $N_WINDOWS windows of length $W...");
Random.seed!(SEED + 50);
obs_windows_full = windowize(R_is, W; stride=1);
n_full = size(obs_windows_full, 2);
sel_obs = sort(rand(1:n_full, N_WINDOWS));
obs_W = obs_windows_full[:, sel_obs];
_standardize_columns!(obs_W);
γ_fixed = median_bandwidth(obs_W; rng=Random.GLOBAL_RNG);
println("  γ_fixed = $(round(γ_fixed, digits=4))   (observed-only median heuristic, $N_WINDOWS windows of length $W)");

# Pooled-archive windows for each generator: standardise, compute MMD against obs_W under γ_fixed.
function _gen_windows(archive::Matrix{Float64})
    Random.seed!(SEED + 60 + size(archive, 2));
    sim_W = sample_windows_from_archive(archive, W, N_WINDOWS);
    _standardize_columns!(sim_W);
    return sim_W;
end

# --------------------------------------------------------------------------------------- #
# Fit + simulate each generator (mirrors M7)
# --------------------------------------------------------------------------------------- #
println("\n[generators] fitting and simulating IS archives...");

println("  CHMM-N..."); Random.seed!(SEED + 1);
chmm_n = build(MyContinuousHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
println("  CHMM-t..."); Random.seed!(SEED + 2);
chmm_t = build(MyStudentTHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
println("  CHMM-L..."); Random.seed!(SEED + 3);
chmm_l = build(MyLaplaceHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
println("  GARCH(1,1)..."); Random.seed!(SEED + 4);
garch = build(MyGARCHModel, (observations=R_is,));
println("  EGARCH..."); Random.seed!(SEED + 5); fit_eg = fit_egarch11(R_is);
println("  GJR-GARCH..."); Random.seed!(SEED + 6); fit_gjr = fit_gjr11(R_is);
println("  GARCH-t..."); Random.seed!(SEED + 7); fit_gt = fit_garcht11(R_is);
println("  HAR-RV..."); Random.seed!(SEED + 8); fit_har = fit_harrv(R_is);
println("  MS-GARCH K=2..."); Random.seed!(SEED + 9); ms2 = fit_msgarch_k2(R_is);
println("  MS-GARCH K=3..."); Random.seed!(SEED + 10); ms3 = fit_msgarch_k3(R_is);

function _simulate_paths(sim_fn, T::Int, n_paths::Int)
    paths = Matrix{Float64}(undef, T, n_paths);
    for p in 1:n_paths; paths[:, p] = sim_fn(); end
    return paths;
end

println("\n[simulate] $N_PATHS IS paths per generator...");
Random.seed!(SEED + 200);
archives = Dict{String, Matrix{Float64}}();
archives["CHMM-N"]      = _simulate_paths(() -> simulate_returns(chmm_n, n_is), n_is, N_PATHS);
archives["CHMM-t"]      = _simulate_paths(() -> simulate_returns(chmm_t, n_is), n_is, N_PATHS);
archives["CHMM-L"]      = _simulate_paths(() -> simulate_returns(chmm_l, n_is), n_is, N_PATHS);
archives["GARCH(1,1)"]  = _simulate_paths(() -> simulate_garch(garch, n_is), n_is, N_PATHS);
archives["EGARCH"]      = _simulate_paths(() -> simulate_egarch(fit_eg, n_is), n_is, N_PATHS);
archives["GJR-GARCH"]   = _simulate_paths(() -> simulate_gjr(fit_gjr, n_is), n_is, N_PATHS);
archives["GARCH-t"]     = _simulate_paths(() -> simulate_garcht(fit_gt, n_is), n_is, N_PATHS);
archives["HAR-RV"]      = _simulate_paths(() -> simulate_harrv(fit_har, n_is), n_is, N_PATHS);
archives["MS-GARCH K=2"]= _simulate_paths(() -> simulate_msgarch(ms2, n_is), n_is, N_PATHS);
archives["MS-GARCH K=3"]= _simulate_paths(() -> simulate_msgarch(ms3, n_is), n_is, N_PATHS);

# --------------------------------------------------------------------------------------- #
# MMD: per-generator bandwidth (legacy) vs fixed observed-sample bandwidth (Minor 4 fix)
# --------------------------------------------------------------------------------------- #
println("\n[MMD] computing legacy (per-generator) and fixed (Minor 4) MMDs...");

results = NamedTuple[];
for name in ["CHMM-N", "CHMM-t", "CHMM-L", "GARCH(1,1)", "EGARCH", "GJR-GARCH",
             "GARCH-t", "HAR-RV", "MS-GARCH K=2", "MS-GARCH K=3"]
    sim_W = _gen_windows(archives[name]);
    mmd_legacy = mmd2_rbf(obs_W, sim_W);                    # γ from pooled (legacy)
    mmd_fixed  = mmd2_rbf(obs_W, sim_W; γ=γ_fixed);          # γ_obs (Minor 4 fix)
    push!(results, (name=name, mmd_legacy=mmd_legacy, mmd_fixed=mmd_fixed));
    println("  $(rpad(name, 14))  legacy MMD = $(@sprintf("%.4e", mmd_legacy))   fixed MMD = $(@sprintf("%.4e", mmd_fixed))");
end

# Sort under fixed bandwidth for ranking comparison
order_fixed = sortperm([r.mmd_fixed for r in results]);
println("\n[ranking] under fixed bandwidth (smaller = closer to observed):");
for (rank, idx) in enumerate(order_fixed)
    r = results[idx];
    println("  $(rpad(rank, 2)) $(rpad(r.name, 14))  fixed MMD $(@sprintf("%.4e", r.mmd_fixed))   legacy $(@sprintf("%.4e", r.mmd_legacy))");
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
open(joinpath(TRACK_DIR, "MMD_Bandwidth_Fix.txt"), "w") do io
    println(io, "="^130);
    println(io, "Track Minor 4. Fixed observed-sample MMD bandwidth (referee Minor 4 response).");
    println(io, "="^130);
    println(io, "");
    println(io, "Setup    : $N_WINDOWS standardised 20-day windows per generator. Fixed γ computed from observed-only median heuristic.");
    println(io, "γ_fixed  = $(round(γ_fixed, digits=4))   (legacy γ varies per generator since it pools observed and simulated windows).");
    println(io, "");
    println(io, rpad("Generator", 14), " | ", rpad("legacy MMD (per-gen γ)", 22), " | ", rpad("fixed MMD (γ_obs)", 18), " | rank under fixed");
    println(io, "-"^120);
    fixed_ranks = sortperm([r.mmd_fixed for r in results]);
    rank_of = Dict(results[fixed_ranks[i]].name => i for i in 1:length(fixed_ranks));
    for r in results
        println(io, rpad(r.name, 14), " | ",
                    rpad(@sprintf("%.4e", r.mmd_legacy), 22), " | ",
                    rpad(@sprintf("%.4e", r.mmd_fixed), 18), " | ",
                    "rank $(rank_of[r.name])");
    end
    println(io, "="^130);
    println(io, "");
    println(io, "Reading: legacy MMD numbers depend on the per-generator pooled bandwidth, so they are not directly comparable across rows.");
    println(io, "         The fixed-γ column uses the observed-only bandwidth and is the cross-generator-comparable metric the referee asked for.");
end

open(joinpath(PAPER_REVISION_DIR, "minor4_mmd_bandwidth.csv"), "w") do io
    println(io, "model,mmd_legacy,mmd_fixed,gamma_fixed");
    for r in results
        println(io, "$(r.name),$(@sprintf("%.6e", r.mmd_legacy)),$(@sprintf("%.6e", r.mmd_fixed)),$(round(γ_fixed, digits=6))");
    end
end

println("\n" * "="^72);
println("  Track Minor 4 complete.");
println("  Text: $(joinpath(TRACK_DIR, "MMD_Bandwidth_Fix.txt"))");
println("  CSV : $(joinpath(PAPER_REVISION_DIR, "minor4_mmd_bandwidth.csv"))");
println("="^72);
