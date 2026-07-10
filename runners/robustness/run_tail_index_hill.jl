# =========================================================================== #
# run_tail_index_hill.jl
#
# Credible heavy-tail characterization for the CHMM: a Hill tail-index estimate
# on observed SPY vs the shared-nu CHMM-t (K*=3) simulated paths.
#
# Motivation. The deck argues the heavy-tail stylized fact through excess
# kurtosis ("narrows but does not close the gap, 4.68 vs 7.68"). Per Cont (2001)
# heavy-tailedness is a tail-index (power-law) property, P(|G|>x) ~ x^-alpha with
# alpha in (2,5); excess kurtosis of a Student-t is 6/(nu-4), a near-singular
# functional when the true tail index is ~3-4 (as observed equity returns are).
# This runner estimates alpha directly (Hill) so the claim can be stated as a
# tail index rather than a kurtosis gap. The Student-t emission has tail index
# alpha = nu analytically, so the shared-nu K*=3 model (nu = 5.81) has alpha ~ 5.8.
#
# Hill estimator (ported verbatim from Paper I,
# HMM-w-jumps-paper/code/downstream-evaluation/src/Metrics.jl): for the upper
# tail, xi_hat = (1/(k-1)) sum_{i<k} log(X_(i)/X_(k)) with k = tail_frac * n
# largest order statistics; for a Pareto-tailed law xi_hat ~ 1/alpha.
#
# Reuses the saved shared-nu K=3 simulated paths (results/chmm_t_shared_nu/
# sims_K3.jld2, written by run_chmm_t_shared_nu.jl); no refit is performed. If
# that artefact is missing, run run_chmm_t_shared_nu.jl first.
#
# Output: results/diagnostics/tail_index_hill.{txt,csv}
# Usage:  julia --project=. runners/robustness/run_tail_index_hill.jl
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, StatsBase, Printf, JLD2;

const SEED      = 20260420;
const DT        = 1/252;
const RISK_FREE = 0.0;
const TAIL_FRACS = [0.02, 0.05, 0.10];
const L_BLOCK   = 10;          # stationary-block-bootstrap mean block length
const B_BOOT    = 2000;        # bootstrap replicates for the observed-tail CI
const OUT_DIR   = joinpath(_ROOT, "results", "diagnostics");
mkpath(OUT_DIR);
Random.seed!(SEED);

# --------------------------------------------------------------------------- #
# Hill upper-tail index (verbatim port of Paper I hill_index). Returns xi = 1/alpha.
function hill_xi(x::AbstractVector{<:Real}; tail_frac::Float64 = 0.05)
    @assert 0.0 < tail_frac < 1.0 "tail_frac must lie in (0, 1)"
    sx = sort(x; rev = true)
    k  = max(2, floor(Int, tail_frac * length(sx)))
    @assert sx[k] > 0.0 "tail order statistic must be positive for Hill estimator"
    s = 0.0
    for i in 1:(k - 1)
        s += log(sx[i] / sx[k])
    end
    return s / (k - 1)
end
hill_alpha(x; tail_frac = 0.05) = 1.0 / hill_xi(x; tail_frac = tail_frac)

_exkurt(x) = sum(((x .- mean(x)) ./ std(x)).^4) / length(x) - 3.0;

# Stationary block bootstrap (Politis-Romano 1994), mean block length L.
function _sbb_idx(n::Int, L::Real, rng::AbstractRNG)
    p = 1 / L
    idx = Vector{Int}(undef, n)
    idx[1] = rand(rng, 1:n)
    for t in 2:n
        idx[t] = rand(rng) < p ? rand(rng, 1:n) : (idx[t-1] == n ? 1 : idx[t-1] + 1)
    end
    return idx
end

# 95% CI on the observed upper-tail Hill xi via block bootstrap of the raw series.
function _boot_xi_ci(g::AbstractVector{<:Real}; tail_frac = 0.05, L = L_BLOCK, B = B_BOOT, seed = SEED)
    rng = Random.MersenneTwister(seed)
    n = length(g)
    out = Vector{Float64}(undef, B)
    for b in 1:B
        gb = g[_sbb_idx(n, L, rng)]
        out[b] = hill_xi(abs.(gb); tail_frac = tail_frac)
    end
    return (median(out), quantile(out, 0.025), quantile(out, 0.975))
end

# Per-path Hill on a simulated (T x n_paths) matrix: magnitude upper tail.
function _sim_hill(sim::AbstractMatrix; tail_frac = 0.05)
    np = size(sim, 2)
    xis = [hill_xi(abs.(@view sim[:, p]); tail_frac = tail_frac) for p in 1:np]
    return xis
end

# --------------------------------------------------------------------------- #
println("[load] SPY IS / OoS excess log growth (VWAP) ...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt = DT, risk_free_rate = RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is  = all_R[:, idx_spy];
R_oos = log_growth_matrix(MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"], "SPY"; Δt = DT, risk_free_rate = RISK_FREE);
println("  IS = $(length(R_is))  OoS = $(length(R_oos))")

sims_path = joinpath(_ROOT, "results", "chmm_t_shared_nu", "sims_K3.jld2");
isfile(sims_path) || error("Missing $sims_path; run runners/headline/run_chmm_t_shared_nu.jl first.");
sims = JLD2.load(sims_path);
SIM_IS  = sims["is"];    # T_is  x N_PATHS
SIM_OOS = sims["oos"];   # T_oos x N_PATHS
println("  loaded shared-nu K=3 sims: IS $(size(SIM_IS)), OoS $(size(SIM_OOS))")

# Left-tail (losses) helper: magnitudes of negative returns.
_losses(g) = abs.(filter(<(0.0), g));

# --------------------------------------------------------------------------- #
# Observed magnitude Hill across tail fractions, plus left-tail at 5%.
obs_rows = NamedTuple[];
for tf in TAIL_FRACS
    xi_is  = hill_xi(abs.(R_is);  tail_frac = tf)
    xi_oos = hill_xi(abs.(R_oos); tail_frac = tf)
    push!(obs_rows, (tf = tf, xi_is = xi_is, a_is = 1/xi_is, xi_oos = xi_oos, a_oos = 1/xi_oos))
end
xi_is_left  = hill_xi(_losses(R_is);  tail_frac = 0.05);
xi_oos_left = hill_xi(_losses(R_oos); tail_frac = 0.05);

# Bootstrap CI on observed magnitude Hill at 5%.
med_is, lo_is, hi_is    = _boot_xi_ci(R_is;  tail_frac = 0.05, seed = SEED);
med_oos, lo_oos, hi_oos = _boot_xi_ci(R_oos; tail_frac = 0.05, seed = SEED + 1);

# Simulated (shared-nu K=3) per-path Hill at each tail fraction.
sim_rows = NamedTuple[];
for tf in TAIL_FRACS
    xis_is  = _sim_hill(SIM_IS;  tail_frac = tf)
    xis_oos = _sim_hill(SIM_OOS; tail_frac = tf)
    push!(sim_rows, (tf = tf,
        xi_is = mean(xis_is), a_is = 1/mean(xis_is),
        xi_is_lo = quantile(xis_is, 0.025), xi_is_hi = quantile(xis_is, 0.975),
        xi_oos = mean(xis_oos), a_oos = 1/mean(xis_oos),
        xi_oos_lo = quantile(xis_oos, 0.025), xi_oos_hi = quantile(xis_oos, 0.975)))
end

# --------------------------------------------------------------------------- #
function _emit(io)
    println(io, "="^96)
    println(io, "Hill tail-index estimate: observed SPY vs shared-nu CHMM-t (K*=3)")
    println(io, "="^96)
    println(io)
    println(io, "Estimator: upper-tail Hill on |G_t| (Paper I port). xi_hat ~ 1/alpha for a Pareto tail.")
    println(io, "Observed excess kurtosis: IS = $(round(_exkurt(R_is), digits=2)), OoS = $(round(_exkurt(R_oos), digits=2)).")
    println(io, "Shared-nu CHMM-t K*=3: nu = 5.8076 (so the emission tail index is alpha = nu ~ 5.81).")
    println(io, "Cont (2001) empirical band for daily equity returns: alpha in (2, 5), often 3-4.")
    println(io)
    @printf(io, "%-6s | %-26s | %-26s\n", "tail", "OBSERVED  xi (alpha)", "CHMM-t sim  xi (alpha)")
    @printf(io, "%-6s | %-12s %-12s | %-12s %-12s\n", "frac", "IS", "OoS", "IS", "OoS")
    println(io, "-"^70)
    for (o, s) in zip(obs_rows, sim_rows)
        @printf(io, "%-6.2f | %5.3f (%4.2f)  %5.3f (%4.2f) | %5.3f (%4.2f)  %5.3f (%4.2f)\n",
                o.tf, o.xi_is, o.a_is, o.xi_oos, o.a_oos, s.xi_is, s.a_is, s.xi_oos, s.a_oos)
    end
    println(io)
    @printf(io, "Observed magnitude Hill at 5%%, block-bootstrap (L=%d, B=%d) 95%% CI on xi:\n", L_BLOCK, B_BOOT)
    @printf(io, "  IS : xi = %5.3f  95%% CI [%5.3f, %5.3f]  -> alpha ~ %.2f  [%.2f, %.2f]\n",
            med_is, lo_is, hi_is, 1/med_is, 1/hi_is, 1/lo_is)
    @printf(io, "  OoS: xi = %5.3f  95%% CI [%5.3f, %5.3f]  -> alpha ~ %.2f  [%.2f, %.2f]\n",
            med_oos, lo_oos, hi_oos, 1/med_oos, 1/hi_oos, 1/lo_oos)
    println(io)
    @printf(io, "Simulated magnitude Hill at 5%%, across-path 95%% band on xi:\n")
    @printf(io, "  IS : xi = %5.3f  [%5.3f, %5.3f]  -> alpha ~ %.2f\n",
            sim_rows[2].xi_is, sim_rows[2].xi_is_lo, sim_rows[2].xi_is_hi, sim_rows[2].a_is)
    @printf(io, "  OoS: xi = %5.3f  [%5.3f, %5.3f]  -> alpha ~ %.2f\n",
            sim_rows[2].xi_oos, sim_rows[2].xi_oos_lo, sim_rows[2].xi_oos_hi, sim_rows[2].a_oos)
    println(io)
    @printf(io, "Left tail only (losses) Hill at 5%%: observed IS alpha ~ %.2f, OoS alpha ~ %.2f\n",
            1/xi_is_left, 1/xi_oos_left)
    println(io)
    println(io, "Reading. The observed tail index sits in Cont's (2,5) band; the shared-nu CHMM-t")
    println(io, "produces a genuine finite power-law tail (alpha near nu = 5.8), slightly THINNER in")
    println(io, "the extreme tail than observed. Excess kurtosis (6/(nu-4)) exaggerates this gap")
    println(io, "because it is a near-singular 4th-moment functional; the tail index is the stable read.")
end

_emit(stdout)
open(joinpath(OUT_DIR, "tail_index_hill.txt"), "w") do io; _emit(io); end

open(joinpath(OUT_DIR, "tail_index_hill.csv"), "w") do io
    println(io, "series,window,tail_frac,xi,alpha")
    for o in obs_rows
        @printf(io, "observed,IS,%.2f,%.4f,%.4f\n",  o.tf, o.xi_is,  o.a_is)
        @printf(io, "observed,OoS,%.2f,%.4f,%.4f\n", o.tf, o.xi_oos, o.a_oos)
    end
    for s in sim_rows
        @printf(io, "chmm_t_sim,IS,%.2f,%.4f,%.4f\n",  s.tf, s.xi_is,  s.a_is)
        @printf(io, "chmm_t_sim,OoS,%.2f,%.4f,%.4f\n", s.tf, s.xi_oos, s.a_oos)
    end
    @printf(io, "observed_left,IS,0.05,%.4f,%.4f\n",  xi_is_left,  1/xi_is_left)
    @printf(io, "observed_left,OoS,0.05,%.4f,%.4f\n", xi_oos_left, 1/xi_oos_left)
end

println("\n[done] $(joinpath(OUT_DIR, "tail_index_hill.txt"))")
