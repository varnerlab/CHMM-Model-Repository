# ========================================================================================= #
# run_ks_block_bootstrap.jl
#
# two complementary fixes for the
# headline KS pass-rate metric.
#
#   (a) Mean two-sample KS p-value distribution (5/25/50/75/95 quantiles) per generator,
#       in addition to the binary pass-rate at α = 0.05. This addresses the referee's
#       "report the p-value distribution rather than only a pass rate" sub-ask.
#
#   (b) Block-bootstrap recalibration: empirical 95% critical value of the KS statistic
#       under stationary block bootstrap of the IS series (mean block length L ∈ {5, 10,
#       20}), then a "block-aware" pass rate per generator that compares each simulated
#       path's KS statistic to the block-bootstrap critical value rather than the
#       asymptotic Kolmogorov critical value. Addresses the referee's volatility-clustering
#       concern on the asymptotic test.
#
# Generators covered: CHMM-N / -t / -L at K = 18; GARCH(1,1) Gaussian; i.i.d. bootstrap.
# All numbers under the revision seed 20260422 with N_paths = 500 (down from 1,000 for
# tractability across the bootstrap recalibration; mean p-value distribution is stable
# at this sample size).
#
# Outputs:
#   results/ks_block_bootstrap/KS_Bootstrap_Recalibration.txt
#   ../CHMM-paper/results/robustness/ks_block_bootstrap.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random
using Statistics
using HypothesisTests
const SEED = 20260422;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const K_MAIN    = 18;
const MAX_ITER  = 60;
const N_PATHS   = 500;
const B_BOOT    = 1000;
const BLOCK_LENGTHS = [5, 10, 20];

const OUT_DIR       = joinpath(_ROOT, "results", "ks_block_bootstrap");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^72)
println("  KS p-value distribution + block-bootstrap recalibration")
println("  Seed $SEED, K=$K_MAIN, N_paths=$N_PATHS, B_boot=$B_BOOT, L=$BLOCK_LENGTHS")
println("="^72)

# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY IS...");
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
println("  IS $n_is");

# --------------------------------------------------------------------------------------- #
# Stationary block bootstrap (Politis-Romano 1994)
# --------------------------------------------------------------------------------------- #
"""
    stationary_block_bootstrap(x, T, L) -> Vector{Float64}

Returns a length-T stationary-block-bootstrap resample of the input series x with mean
block length L. Block lengths are geometric(1/L); the starting index is uniform on 1..n;
wrap-around indexing.
"""
function stationary_block_bootstrap(x::Vector{Float64}, T::Int, L::Int)::Vector{Float64}
    n = length(x);
    out = Vector{Float64}(undef, T);
    p = 1.0 / L;
    t = 1;
    while t <= T
        i = rand(1:n);
        # Geometric block length with mean L
        block_len = 1;
        while rand() > p; block_len += 1; end
        for _ in 1:block_len
            if t > T; break; end
            out[t] = x[((i - 1) % n) + 1];
            i += 1;
            t += 1;
        end
    end
    return out;
end

"""
    ks_statistic(a, b) -> Float64

Two-sample Kolmogorov-Smirnov statistic between samples a and b.
"""
function ks_statistic(a::AbstractVector{Float64}, b::AbstractVector{Float64})::Float64
    return ApproximateTwoSampleKSTest(a, b).δ;
end

# --------------------------------------------------------------------------------------- #
# Block-bootstrap KS critical value
# --------------------------------------------------------------------------------------- #
println("\n[block-boot] computing stationary block-bootstrap KS critical values at L = $BLOCK_LENGTHS...");
crit_values = Dict{Int, Float64}();
all_null_ks = Dict{Int, Vector{Float64}}();
for L in BLOCK_LENGTHS
    Random.seed!(SEED + 1000 + L);
    null_stats = Vector{Float64}(undef, B_BOOT);
    for b in 1:B_BOOT
        boot = stationary_block_bootstrap(R_is, n_is, L);
        null_stats[b] = ks_statistic(R_is, boot);
    end
    crit_values[L] = quantile(null_stats, 0.95);
    all_null_ks[L] = null_stats;
    println("  L = $L : 95% critical value = $(round(crit_values[L], digits=4))   (asymptotic two-sample $(round(1.36 * sqrt(2 / n_is), digits=4)))");
end
asymp_crit = 1.36 * sqrt(2 / n_is);

# --------------------------------------------------------------------------------------- #
# Generators: simulate N_paths IS paths and score
# --------------------------------------------------------------------------------------- #
println("\n[generators] fitting and simulating N_paths = $N_PATHS for each generator...");

# Fit
println("  CHMM-N..."); Random.seed!(SEED + 1);
chmm_n = build(MyContinuousHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
println("  CHMM-t..."); Random.seed!(SEED + 2);
chmm_t = build(MyStudentTHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
println("  CHMM-L..."); Random.seed!(SEED + 3);
chmm_l = build(MyLaplaceHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
println("  CHMM-GED..."); Random.seed!(SEED + 5);
chmm_ged = build(MyGEDHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
println("  GARCH(1,1)..."); Random.seed!(SEED + 4);
garch = build(MyGARCHModel, (observations=R_is,));

# Simulate
function _simulate_paths_chmm(model, T::Int, n_paths::Int)::Matrix{Float64}
    paths = Matrix{Float64}(undef, T, n_paths);
    for p in 1:n_paths
        paths[:, p] = simulate_returns(model, T);
    end
    return paths;
end
function _simulate_paths_garch(model, T::Int, n_paths::Int)::Matrix{Float64}
    paths = Matrix{Float64}(undef, T, n_paths);
    for p in 1:n_paths
        paths[:, p] = simulate_garch(model, T);
    end
    return paths;
end
function _simulate_paths_iid_boot(R_obs::Vector{Float64}, T::Int, n_paths::Int)::Matrix{Float64}
    n = length(R_obs);
    paths = Matrix{Float64}(undef, T, n_paths);
    for p in 1:n_paths
        for t in 1:T; paths[t, p] = R_obs[rand(1:n)]; end
    end
    return paths;
end

Random.seed!(SEED + 100); sims = Dict{String, Matrix{Float64}}();
println("  simulating CHMM-N..."); sims["CHMM-N"] = _simulate_paths_chmm(chmm_n, n_is, N_PATHS);
println("  simulating CHMM-t..."); sims["CHMM-t"] = _simulate_paths_chmm(chmm_t, n_is, N_PATHS);
println("  simulating CHMM-L..."); sims["CHMM-L"] = _simulate_paths_chmm(chmm_l, n_is, N_PATHS);
println("  simulating CHMM-GED..."); sims["CHMM-GED"] = _simulate_paths_chmm(chmm_ged, n_is, N_PATHS);
println("  simulating GARCH(1,1)..."); sims["GARCH(1,1)"] = _simulate_paths_garch(garch, n_is, N_PATHS);
println("  simulating i.i.d. bootstrap..."); sims["iid bootstrap"] = _simulate_paths_iid_boot(R_is, n_is, N_PATHS);

# --------------------------------------------------------------------------------------- #
# Score: per-path KS statistic + p-value, then summarise
# --------------------------------------------------------------------------------------- #
function score_generator(name::String, sim::Matrix{Float64}, R_obs::Vector{Float64})
    n_p = size(sim, 2);
    ks = Vector{Float64}(undef, n_p);
    pv = Vector{Float64}(undef, n_p);
    for p in 1:n_p
        test = ApproximateTwoSampleKSTest(R_obs, sim[:, p]);
        ks[p] = test.δ;
        pv[p] = pvalue(test);
    end
    asymp_pass = mean(pv .>= 0.05);
    block_pass = Dict(L => mean(ks .< crit_values[L]) for L in BLOCK_LENGTHS);
    return (
        name=name,
        asymp_pass_rate=asymp_pass,
        mean_pv=mean(pv), median_pv=median(pv),
        pv_q05=quantile(pv, 0.05), pv_q25=quantile(pv, 0.25),
        pv_q75=quantile(pv, 0.75), pv_q95=quantile(pv, 0.95),
        ks_q05=quantile(ks, 0.05), ks_q50=median(ks), ks_q95=quantile(ks, 0.95),
        block_pass=block_pass,
    );
end

println("\n[score] per-generator KS p-value distribution + block-bootstrap pass rates...");
panel = NamedTuple[];
for name in ["iid bootstrap", "GARCH(1,1)", "CHMM-N", "CHMM-t", "CHMM-L", "CHMM-GED"]
    r = score_generator(name, sims[name], R_is);
    push!(panel, r);
    println("  $(rpad(name, 14))  asymp_pass $(round(100*r.asymp_pass_rate, digits=1))%  mean_pv $(round(r.mean_pv, digits=3))  pv_q05 $(round(r.pv_q05, digits=3))  pv_q95 $(round(r.pv_q95, digits=3))   block5 $(round(100*r.block_pass[5], digits=1))%  block10 $(round(100*r.block_pass[10], digits=1))%  block20 $(round(100*r.block_pass[20], digits=1))%");
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
open(joinpath(OUT_DIR, "KS_Bootstrap_Recalibration.txt"), "w") do io
    println(io, "="^150);
    println(io, "KS p-value distribution + block-bootstrap recalibration.");
    println(io, "="^150);
    println(io, "");
    println(io, "Setup     : SPY IS ($n_is observations); $N_PATHS simulated paths per generator under seed $SEED.");
    println(io, "Block boot: stationary block bootstrap (Politis-Romano 1994) of IS at mean block length L ∈ $BLOCK_LENGTHS, B = $B_BOOT.");
    println(io, "");
    println(io, "Block-bootstrap KS critical values (95% null quantile):");
    for L in BLOCK_LENGTHS
        println(io, "  L = $(rpad(L, 4)) crit = $(round(crit_values[L], digits=4))");
    end
    println(io, "  asymp.   crit = $(round(asymp_crit, digits=4))   (1.36 * sqrt(2 / n) for two-sample KS at α = 0.05)");
    println(io, "");
    println(io, rpad("Generator", 14), " | ",
                rpad("asymp pass%", 11), " | ",
                rpad("mean pv", 7), " | ", rpad("med pv", 6), " | ",
                rpad("pv q05", 6), " | ", rpad("pv q25", 6), " | ", rpad("pv q75", 6), " | ", rpad("pv q95", 6), " | ",
                rpad("KS q05", 6), " | ", rpad("KS q50", 6), " | ", rpad("KS q95", 6), " | ",
                rpad("blk5%", 5), " | ", rpad("blk10%", 6), " | ", rpad("blk20%", 6));
    println(io, "-"^150);
    for r in panel
        println(io, rpad(r.name, 14), " | ",
                    rpad(round(100*r.asymp_pass_rate, digits=1), 11), " | ",
                    rpad(round(r.mean_pv, digits=3), 7), " | ",
                    rpad(round(r.median_pv, digits=3), 6), " | ",
                    rpad(round(r.pv_q05, digits=3), 6), " | ",
                    rpad(round(r.pv_q25, digits=3), 6), " | ",
                    rpad(round(r.pv_q75, digits=3), 6), " | ",
                    rpad(round(r.pv_q95, digits=3), 6), " | ",
                    rpad(round(r.ks_q05, digits=3), 6), " | ",
                    rpad(round(r.ks_q50, digits=3), 6), " | ",
                    rpad(round(r.ks_q95, digits=3), 6), " | ",
                    rpad(round(100*r.block_pass[5], digits=1), 5), " | ",
                    rpad(round(100*r.block_pass[10], digits=1), 6), " | ",
                    rpad(round(100*r.block_pass[20], digits=1), 6));
    end
    println(io, "="^150);
    println(io, "");
    println(io, "Reading: 'asymp pass%' is the original Table 3 metric (fraction of paths with two-sample KS p-value ≥ 0.05).");
    println(io, "         'mean pv' and the q05/q25/q75/q95 are the p-value distribution across the $N_PATHS simulated paths.");
    println(io, "         'blkL%' is the block-bootstrap pass rate at mean block length L: fraction of paths whose KS");
    println(io, "         statistic D < block-bootstrap 95% null quantile (which accounts for clustering in the IS).");
end

open(joinpath(PAPER_ROBUSTNESS_DIR, "ks_block_bootstrap.csv"), "w") do io
    println(io, "generator,asymp_pass_pct,mean_pv,median_pv,pv_q05,pv_q25,pv_q75,pv_q95,ks_q05,ks_q50,ks_q95,block5_pass_pct,block10_pass_pct,block20_pass_pct");
    for r in panel
        println(io, "$(r.name),$(round(100*r.asymp_pass_rate, digits=2)),$(round(r.mean_pv, digits=4)),$(round(r.median_pv, digits=4)),$(round(r.pv_q05, digits=4)),$(round(r.pv_q25, digits=4)),$(round(r.pv_q75, digits=4)),$(round(r.pv_q95, digits=4)),$(round(r.ks_q05, digits=4)),$(round(r.ks_q50, digits=4)),$(round(r.ks_q95, digits=4)),$(round(100*r.block_pass[5], digits=2)),$(round(100*r.block_pass[10], digits=2)),$(round(100*r.block_pass[20], digits=2))");
    end
    println(io, "");
    println(io, "block_length,KS_crit_95");
    for L in BLOCK_LENGTHS
        println(io, "$L,$(round(crit_values[L], digits=5))");
    end
    println(io, "asymptotic,$(round(asymp_crit, digits=5))");
end

println("\n" * "="^72);
println("  KS block bootstrap complete.");
println("  Text report : $(joinpath(OUT_DIR, "KS_Bootstrap_Recalibration.txt"))");
println("  Paper CSV   : $(joinpath(PAPER_ROBUSTNESS_DIR, "ks_block_bootstrap.csv"))");
println("="^72);
