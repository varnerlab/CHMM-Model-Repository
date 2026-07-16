# =========================================================================== #
# run_kurtosis_bootstrap.jl
#
# Peer-review item 9 (R3 W7-RE4): stationary block bootstrap CIs on the
# observed (and simulated) IS / OoS excess kurtosis on SPY. Settles whether
# the IS-vs-OoS kurtosis disagreement (7.68 vs 5.29) is itself statistically
# distinguishable, and whether per-variant simulated kurtosis differences
# survive a CI-based comparison.
#
# Output: results/diagnostics/kurtosis_bootstrap.txt
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, Printf, StatsBase, Distributions;

const SEED = 20260420;
const OUT  = joinpath(_ROOT, "results", "diagnostics", "kurtosis_bootstrap.txt");
mkpath(dirname(OUT));
Random.seed!(SEED);

# Excess kurtosis: kurt - 3.
function _excess_kurtosis(x::AbstractVector{<:Real})
    return StatsBase.kurtosis(x);  # already returns excess kurtosis
end

# Stationary block bootstrap (Politis-Romano 1994) with mean block length L.
function _stationary_block_bootstrap(x::AbstractVector{<:Real}, L::Real, B::Int;
                                     rng::AbstractRNG = Random.GLOBAL_RNG)
    n = length(x);
    p = 1 / L;
    out = Vector{Float64}(undef, B);
    @inbounds for b in 1:B
        idx = Vector{Int}(undef, n);
        idx[1] = rand(rng, 1:n);
        for t in 2:n
            if rand(rng) < p
                idx[t] = rand(rng, 1:n);
            else
                prev = idx[t-1];
                idx[t] = prev == n ? 1 : prev + 1;
            end
        end
        out[b] = _excess_kurtosis(@view x[idx]);
    end
    return out;
end

# --- Load IS and OoS SPY series ---
ds_is  = MyPortfolioDataSet()["dataset"]["SPY"];
ds_oos = MyOutOfSamplePortfolioDataSet()["dataset"]["SPY"];

# Concatenate IS+OoS to compute the matched series, then split into windows
# matching the paper convention.
DT  = 1/252;
RF  = 0.0;

function _series_log_growth(df::DataFrame; Δt = DT, rf = RF)
    p = df.volume_weighted_average_price;
    g = (1/Δt) .* (log.(p[2:end] ./ p[1:end-1])) .- rf;
    return g;
end

g_is  = _series_log_growth(ds_is);
g_oos = _series_log_growth(ds_oos);

println("[load] T_IS  = $(length(g_is))");
println("[load] T_OoS = $(length(g_oos))");
println("[load] observed excess kurtosis: IS = $(round(_excess_kurtosis(g_is), digits=3)), OoS = $(round(_excess_kurtosis(g_oos), digits=3))");

# --- Stationary block bootstrap CIs at L = 5, 10, 20, 50 (mean block length) ---
const B    = 5_000;
const Ls   = [5, 10, 20, 50];

results_is  = Dict{Int, Vector{Float64}}();
results_oos = Dict{Int, Vector{Float64}}();
for L in Ls
    rng = Random.MersenneTwister(SEED + L);
    boot_is  = _stationary_block_bootstrap(g_is,  L, B; rng = rng);
    boot_oos = _stationary_block_bootstrap(g_oos, L, B; rng = rng);
    results_is[L]  = boot_is;
    results_oos[L] = boot_oos;
    @printf("  L = %2d  IS:  median = %5.3f, 95%% CI = [%5.3f, %5.3f]   OoS:  median = %5.3f, 95%% CI = [%5.3f, %5.3f]\n",
            L,
            median(boot_is), quantile(boot_is, 0.025), quantile(boot_is, 0.975),
            median(boot_oos), quantile(boot_oos, 0.025), quantile(boot_oos, 0.975));
end

# --- Descriptive bootstrap probability: fraction of (IS_b - OoS_b) > 0. ---
# NOT a p-value: the two windows are resampled around their own observed kurtosis
# levels, so no equal-kurtosis null is imposed (fourth-review item 3). The
# inferential object is the DIFFERENCE distribution below.
function _diff_p(b_is::Vector{Float64}, b_oos::Vector{Float64})
    return mean(b_is .> b_oos);
end

# --- Bootstrap distribution of the difference (IS_b - OoS_b) with percentile
# and basic CIs. The IS and OoS resamples are independent, so the pairwise
# differences approximate the sampling distribution of kurt_IS - kurt_OoS. ---
function _diff_ci(b_is::Vector{Float64}, b_oos::Vector{Float64}, obs_diff::Float64)
    d = b_is .- b_oos;
    pct = (quantile(d, 0.025), quantile(d, 0.975));
    bas = (2*obs_diff - quantile(d, 0.975), 2*obs_diff - quantile(d, 0.025));
    return (pct=pct, basic=bas, med=median(d));
end

# --- Write summary ---
open(OUT, "w") do io
    println(io, "Stationary block bootstrap CIs on excess kurtosis (peer-review item 9 / R3 W7-RE4)");
    println(io, "Source data: SPY IS (T = $(length(g_is))) and SPY OoS (T = $(length(g_oos))) excess log growth");
    println(io, "Bootstrap construction: Politis-Romano 1994, mean block length L, $B replicates per L.");
    println(io, "="^88);
    println(io);
    @printf(io, "Observed excess kurtosis: IS = %.3f   OoS = %.3f   IS - OoS = %.3f\n",
            _excess_kurtosis(g_is), _excess_kurtosis(g_oos), _excess_kurtosis(g_is) - _excess_kurtosis(g_oos));
    println(io);
    println(io, "Bootstrap distributions:");
    println(io, "L     | IS median | IS 95% CI         | OoS median | OoS 95% CI         | Pr(IS > OoS)");
    println(io, "-"^88);
    for L in Ls
        boot_is  = results_is[L];
        boot_oos = results_oos[L];
        @printf(io, "%-5d | %9.3f | [%5.3f, %5.3f]   | %10.3f | [%5.3f, %5.3f]   | %.3f\n",
                L, median(boot_is), quantile(boot_is, 0.025), quantile(boot_is, 0.975),
                median(boot_oos), quantile(boot_oos, 0.025), quantile(boot_oos, 0.975),
                _diff_p(boot_is, boot_oos));
    end
    println(io);
    obs_diff = _excess_kurtosis(g_is) - _excess_kurtosis(g_oos);
    println(io, "Bootstrap distribution of the DIFFERENCE (IS - OoS excess kurtosis) - the");
    println(io, "inferential object. The difference CI, not the overlap of the two marginal CIs,");
    println(io, "is what tests whether the windows differ:");
    println(io, "L     | diff median | percentile 95% CI   | basic 95% CI        | covers 0?");
    println(io, "-"^88);
    for L in Ls
        ci = _diff_ci(results_is[L], results_oos[L], obs_diff);
        covers = (ci.pct[1] <= 0.0 <= ci.pct[2]) ? "yes" : "no";
        @printf(io, "%-5d | %11.3f | [%6.3f, %6.3f]   | [%6.3f, %6.3f]   | %s\n",
                L, ci.med, ci.pct[1], ci.pct[2], ci.basic[1], ci.basic[2], covers);
    end
    println(io);
    println(io, "Reading:");
    for L in Ls
        ci = _diff_ci(results_is[L], results_oos[L], obs_diff);
        if !(ci.pct[1] <= 0.0 <= ci.pct[2])
            @printf(io, "  - At L = %d the percentile 95%% CI for the difference excludes zero.\n", L);
        end
    end
    ci10 = _diff_ci(results_is[10], results_oos[10], obs_diff);
    if ci10.pct[1] <= 0.0 <= ci10.pct[2]
        @printf(io, "  - At L = 10 the percentile 95%% CI for the IS - OoS difference is [%.3f, %.3f],\n",
                ci10.pct[1], ci10.pct[2]);
        println(io, "    which covers zero: the data do not exclude equal IS and OoS excess kurtosis");
        println(io, "    at the 5% level under this bootstrap. This is a non-rejection, not evidence");
        println(io, "    of equality.");
    else
        @printf(io, "  - At L = 10 the percentile 95%% CI for the IS - OoS difference is [%.3f, %.3f],\n",
                ci10.pct[1], ci10.pct[2]);
        println(io, "    which excludes zero: the windows' excess kurtosis levels differ at the 5%");
        println(io, "    level under this bootstrap.");
    end
    println(io);
    @printf(io, "  - Pr(IS > OoS) at L = 10: %.3f. This is a DESCRIPTIVE bootstrap probability -\n",
            _diff_p(results_is[10], results_oos[10]));
    println(io,  "    the fraction of independent resample pairs in which the IS draw exceeds the");
    println(io,  "    OoS draw. It is NOT a p-value: each window is resampled around its own");
    println(io,  "    observed kurtosis, so no equal-kurtosis null is imposed (fourth-review item 3).");
end

println("[done] $OUT");
