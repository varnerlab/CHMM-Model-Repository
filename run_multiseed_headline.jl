# ========================================================================================= #
# run_multiseed_headline.jl
#
# Track Minor 10 (revision response): the paper reports headline statistics under a single
# global seed. The referee asks for a Monte Carlo robustness summary across 10-20 seed
# replicates for the headline pass-rate and VaR-LR statistics.
#
# This script re-runs the headline IS/OoS pipeline (CHMM-N / -t / -L at K = 18) under 10
# alternative seeds and reports mean ± std across seeds for: IS KS pass rate, OoS KS pass
# rate, simulated IS excess kurtosis, ACF-MAE on |r|, and Kupiec LR_uc on the unconditional
# pooled-archive VaR at α ∈ {0.01, 0.05}.
#
# Output:
#   results/track_minor10/MultiSeed.txt
#   ../CHMM-paper/results/robustness/multiseed_headline.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
using Printf

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const N_PATHS   = 250;     # smaller per-seed sample for tractability across 10 seeds × 3 models
const K_MAIN    = 18;
const MAX_ITER  = 60;

const BASE_SEED = 20260422;
const SEED_GRID = [BASE_SEED + 100*k for k in 1:10];

const TRACK_DIR        = joinpath(_ROOT, "results", "track_minor10");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(TRACK_DIR); mkpath(PAPER_ROBUSTNESS_DIR);

println("="^72)
println("  Track Minor 10: 10-seed Monte Carlo on headline metrics (referee Minor 10)")
println("="^72)

println("\n[data] Loading SPY IS + OoS...");
train = MyPortfolioDataSet() |> x -> x["dataset"];
oos = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train["AAPL"]);
ds = Dict{String,DataFrame}();
for (t, data) in train; if nrow(data) == max_days; ds[t] = data; end; end
all_tickers = keys(ds) |> collect |> sort;
all_R = log_growth_matrix(ds, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
n_is = length(R_is);
R_oos = Vector{Float64}(log_growth_matrix(oos, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
n_oos = length(R_oos);
println("  IS $n_is, OoS $n_oos");

# --------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------- #
function ks_pass_rate(R_ref, sim; α::Float64=0.05)
    n_p = size(sim, 2); n_pass = 0;
    for p in 1:n_p
        pv = pvalue(ApproximateTwoSampleKSTest(R_ref, sim[:, p]));
        if pv >= α; n_pass += 1; end
    end
    return n_pass / n_p;
end
function acf_mae(R_obs, sim; max_lag::Int=252)
    obs = autocor(abs.(R_obs), 1:max_lag);
    n_p = size(sim, 2);
    sm = zeros(max_lag);
    for p in 1:n_p; sm .+= autocor(abs.(sim[:, p]), 1:max_lag); end
    sm ./= n_p;
    return mean(abs.(obs .- sm));
end
function var_calibration(R_oos, sim_oos)
    pooled = vec(sim_oos);
    out = Dict{Float64, NamedTuple}();
    for α in [0.01, 0.05]
        v = quantile(pooled, α);
        br = R_oos .<= v;
        k = kupiec_lr(br, α);
        out[α] = (br_rate=k.breach_rate, LR_uc=k.LR);
    end
    return out;
end
function _simulate_paths(sim_fn, T::Int, n_paths::Int)
    paths = Matrix{Float64}(undef, T, n_paths);
    for p in 1:n_paths; paths[:, p] = sim_fn(); end
    return paths;
end

# --------------------------------------------------------------------------------------- #
# Single-seed evaluation: fit + simulate + score
# --------------------------------------------------------------------------------------- #
function eval_one_seed(seed::Int, family::Symbol)
    Random.seed!(seed);
    if family == :n
        m = build(MyContinuousHiddenMarkovModel,
            (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
    elseif family == :t
        m = build(MyStudentTHiddenMarkovModel,
            (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
    else
        m = build(MyLaplaceHiddenMarkovModel,
            (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
    end
    Random.seed!(seed + 1000);
    sim_is  = _simulate_paths(() -> simulate_returns(m, n_is), n_is, N_PATHS);
    Random.seed!(seed + 2000);
    sim_oos = _simulate_paths(() -> simulate_returns(m, n_oos), n_oos, N_PATHS);
    return (
        is_ks=ks_pass_rate(R_is, sim_is),
        oos_ks=ks_pass_rate(R_oos, sim_oos),
        sim_kurt=mean([kurtosis(sim_is[:, p]) for p in 1:N_PATHS]),
        acf_mae=acf_mae(R_is, sim_is),
        var01=var_calibration(R_oos, sim_oos)[0.01],
        var05=var_calibration(R_oos, sim_oos)[0.05],
    );
end

# --------------------------------------------------------------------------------------- #
# Multi-seed sweep
# --------------------------------------------------------------------------------------- #
println("\n[sweep] $(length(SEED_GRID)) seeds × 3 emission families...");

all_results = Dict{Symbol, Vector{NamedTuple}}();
for fam in [:n, :t, :l]
    all_results[fam] = NamedTuple[];
end

for (i, seed) in enumerate(SEED_GRID)
    println("\n  seed $i = $seed");
    for fam in [:n, :t, :l]
        r = eval_one_seed(seed, fam);
        push!(all_results[fam], r);
        fname = fam == :n ? "CHMM-N" : fam == :t ? "CHMM-t" : "CHMM-L";
        println("    $fname  IS_KS $(round(100*r.is_ks, digits=1))%  OoS_KS $(round(100*r.oos_ks, digits=1))%  kurt $(round(r.sim_kurt, digits=2))  ACF $(round(r.acf_mae, digits=4))  LR_uc01 $(round(r.var01.LR_uc, digits=2))  LR_uc05 $(round(r.var05.LR_uc, digits=2))");
    end
end

# --------------------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------------------- #
function _mean_std(xs)
    return (mean=mean(xs), std=std(xs), min=minimum(xs), max=maximum(xs));
end

agg = Dict{Symbol, NamedTuple}();
for (fam, results) in all_results
    is_ks    = [r.is_ks    for r in results];
    oos_ks   = [r.oos_ks   for r in results];
    kurt     = [r.sim_kurt for r in results];
    acf      = [r.acf_mae  for r in results];
    LRuc01   = [r.var01.LR_uc for r in results];
    br01     = [r.var01.br_rate for r in results];
    LRuc05   = [r.var05.LR_uc for r in results];
    br05     = [r.var05.br_rate for r in results];
    agg[fam] = (
        is_ks=_mean_std(is_ks),
        oos_ks=_mean_std(oos_ks),
        kurt=_mean_std(kurt),
        acf=_mean_std(acf),
        LRuc01=_mean_std(LRuc01),
        br01=_mean_std(br01),
        LRuc05=_mean_std(LRuc05),
        br05=_mean_std(br05),
    );
end

println("\n[aggregate] mean ± std across $(length(SEED_GRID)) seeds:");
for fam in [:n, :t, :l]
    fname = fam == :n ? "CHMM-N" : fam == :t ? "CHMM-t" : "CHMM-L";
    a = agg[fam];
    println("  $fname  IS_KS $(round(100*a.is_ks.mean, digits=1))% ± $(round(100*a.is_ks.std, digits=1))%   OoS_KS $(round(100*a.oos_ks.mean, digits=1))% ± $(round(100*a.oos_ks.std, digits=1))%   kurt $(round(a.kurt.mean, digits=2)) ± $(round(a.kurt.std, digits=2))   ACF $(round(a.acf.mean, digits=4)) ± $(round(a.acf.std, digits=4))   LR_uc01 $(round(a.LRuc01.mean, digits=2)) ± $(round(a.LRuc01.std, digits=2))   LR_uc05 $(round(a.LRuc05.mean, digits=2)) ± $(round(a.LRuc05.std, digits=2))");
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
open(joinpath(TRACK_DIR, "MultiSeed.txt"), "w") do io
    println(io, "="^150);
    println(io, "Track Minor 10. Multi-seed Monte Carlo on headline metrics (referee Minor 10 response).");
    println(io, "="^150);
    println(io, "");
    println(io, "Setup : $(length(SEED_GRID)) global seeds (base $BASE_SEED, increment 100), N_paths = $N_PATHS per seed, K = $K_MAIN.");
    println(io, "Note  : the $N_PATHS paths per seed is reduced from the paper's 1000 to keep total compute tractable; the seed-to-seed std reported here is therefore an upper bound on the std the headline panel would show under N_paths = 1000.");
    println(io, "");
    for fam in [:n, :t, :l]
        fname = fam == :n ? "CHMM-N" : fam == :t ? "CHMM-t" : "CHMM-L";
        a = agg[fam];
        println(io, "$fname:");
        println(io, "  IS  KS pass rate : $(round(100*a.is_ks.mean, digits=2))% ± $(round(100*a.is_ks.std, digits=2))%   range [$(round(100*a.is_ks.min, digits=1))%, $(round(100*a.is_ks.max, digits=1))%]");
        println(io, "  OoS KS pass rate : $(round(100*a.oos_ks.mean, digits=2))% ± $(round(100*a.oos_ks.std, digits=2))%   range [$(round(100*a.oos_ks.min, digits=1))%, $(round(100*a.oos_ks.max, digits=1))%]");
        println(io, "  sim IS kurt      : $(round(a.kurt.mean, digits=3)) ± $(round(a.kurt.std, digits=3))             range [$(round(a.kurt.min, digits=2)), $(round(a.kurt.max, digits=2))]");
        println(io, "  ACF-MAE          : $(round(a.acf.mean, digits=4)) ± $(round(a.acf.std, digits=4))");
        println(io, "  Kupiec LR_uc 1%  : $(round(a.LRuc01.mean, digits=2)) ± $(round(a.LRuc01.std, digits=2))");
        println(io, "  Kupiec LR_uc 5%  : $(round(a.LRuc05.mean, digits=2)) ± $(round(a.LRuc05.std, digits=2))");
        println(io, "");
    end
    println(io, "="^150);
end

open(joinpath(PAPER_ROBUSTNESS_DIR, "multiseed_headline.csv"), "w") do io
    println(io, "model,seed,IS_KS_pct,OoS_KS_pct,sim_kurt,ACF_MAE,br01_pct,LRuc01,br05_pct,LRuc05");
    for fam in [:n, :t, :l]
        fname = fam == :n ? "CHMM-N" : fam == :t ? "CHMM-t" : "CHMM-L";
        for (i, r) in enumerate(all_results[fam])
            println(io, "$fname,$(SEED_GRID[i]),$(round(100*r.is_ks, digits=2)),$(round(100*r.oos_ks, digits=2)),$(round(r.sim_kurt, digits=3)),$(round(r.acf_mae, digits=5)),$(round(100*r.var01.br_rate, digits=2)),$(round(r.var01.LR_uc, digits=3)),$(round(100*r.var05.br_rate, digits=2)),$(round(r.var05.LR_uc, digits=3))");
        end
    end
    println(io, "");
    println(io, "model,IS_KS_mean,IS_KS_std,OoS_KS_mean,OoS_KS_std,kurt_mean,kurt_std,ACF_mean,ACF_std,LRuc01_mean,LRuc01_std,LRuc05_mean,LRuc05_std");
    for fam in [:n, :t, :l]
        fname = fam == :n ? "CHMM-N" : fam == :t ? "CHMM-t" : "CHMM-L";
        a = agg[fam];
        println(io, "$fname,$(round(100*a.is_ks.mean, digits=3)),$(round(100*a.is_ks.std, digits=3)),$(round(100*a.oos_ks.mean, digits=3)),$(round(100*a.oos_ks.std, digits=3)),$(round(a.kurt.mean, digits=4)),$(round(a.kurt.std, digits=4)),$(round(a.acf.mean, digits=5)),$(round(a.acf.std, digits=5)),$(round(a.LRuc01.mean, digits=3)),$(round(a.LRuc01.std, digits=3)),$(round(a.LRuc05.mean, digits=3)),$(round(a.LRuc05.std, digits=3))");
    end
end

println("\n" * "="^72);
println("  Track Minor 10 complete.");
println("  Text: $(joinpath(TRACK_DIR, "MultiSeed.txt"))");
println("  CSV : $(joinpath(PAPER_ROBUSTNESS_DIR, "multiseed_headline.csv"))");
println("="^72);
