# =========================================================================== #
# run_kurtosis_bootstrap.jl
#
# Stationary block bootstrap displays on the observed IS / OoS excess kurtosis
# on SPY, plus a tail-robust inferential target.
#
# Two layers (see CHANGELOG.md):
#   [descriptive] Raw excess kurtosis. The paper's own Hill estimate
#     (α̂ ≈ 3.15 < 4) says the population fourth moment is plausibly
#     infinite, in which case raw excess kurtosis has no stable population
#     value to estimate and the n-out-of-n percentile bootstrap is not a
#     calibrated 95% interval for a tail functional in this regime. The raw
#     tables below are therefore SENSITIVITY DISPLAYS, not inference.
#   [inferential] Winsorized excess kurtosis at trim q (each tail clamped to
#     its q / 1-q quantile before the fourth-moment computation). The
#     winsorized functional has finite moments of all orders for ANY tail
#     index, is a smooth functional of the marginal distribution, and the
#     stationary block bootstrap is a standard tool for such functionals
#     under mixing. The IS - OoS difference CI on the winsorized target is
#     the inferential statement this artifact supports.
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

# Winsorized excess kurtosis at symmetric trim q: clamp each tail at its own
# q / (1-q) empirical quantile, then compute excess kurtosis. The thresholds
# are recomputed inside every resample, so the bootstrapped object is the
# functional itself (distribution -> winsorized kurtosis), not a fixed clamp.
function _winsorized_excess_kurtosis(x::AbstractVector{<:Real}, q::Float64)
    lo = quantile(x, q);
    hi = quantile(x, 1 - q);
    return _excess_kurtosis(clamp.(x, lo, hi));
end

# Stationary block bootstrap (Politis-Romano 1994) with mean block length L,
# applied to an arbitrary functional f of the resampled series.
function _stationary_block_bootstrap(x::AbstractVector{<:Real}, L::Real, B::Int;
                                     rng::AbstractRNG = Random.GLOBAL_RNG,
                                     f::Function = _excess_kurtosis)
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
        out[b] = f(x[idx]);
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

# --- Tail-robust inferential target: winsorized excess kurtosis at trim q ---
const QS = [0.01, 0.05];
wins_is  = Dict{Tuple{Float64,Int}, Vector{Float64}}();
wins_oos = Dict{Tuple{Float64,Int}, Vector{Float64}}();
for q in QS, L in Ls
    rng = Random.MersenneTwister(SEED + L + round(Int, 10_000q));
    wins_is[(q, L)]  = _stationary_block_bootstrap(g_is,  L, B; rng = rng,
                            f = x -> _winsorized_excess_kurtosis(x, q));
    wins_oos[(q, L)] = _stationary_block_bootstrap(g_oos, L, B; rng = rng,
                            f = x -> _winsorized_excess_kurtosis(x, q));
end
wobs_is  = Dict(q => _winsorized_excess_kurtosis(g_is,  q) for q in QS);
wobs_oos = Dict(q => _winsorized_excess_kurtosis(g_oos, q) for q in QS);
for q in QS
    @printf("  winsorized q = %.2f: observed IS = %.3f, OoS = %.3f\n", q, wobs_is[q], wobs_oos[q]);
end

# --- Descriptive bootstrap probability: fraction of (IS_b - OoS_b) > 0. ---
# NOT a p-value: the two windows are resampled around their own observed kurtosis
# levels, so no equal-kurtosis null is imposed (see CHANGELOG.md). The
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
    println(io, "Stationary block bootstrap displays on excess kurtosis + tail-robust inference");
    println(io, "Source data: SPY IS (T = $(length(g_is))) and SPY OoS (T = $(length(g_oos))) excess log growth");
    println(io, "Bootstrap construction: Politis-Romano 1994, mean block length L, $B replicates per L.");
    println(io, "="^88);
    println(io);
    println(io, "SCOPE. The observed top-5% Hill estimate on this series is ~3.15 < 4, so the");
    println(io, "population fourth moment is plausibly infinite. In that regime raw excess kurtosis");
    println(io, "has no stable population value and the n-out-of-n percentile bootstrap is not a");
    println(io, "calibrated interval for it. Sections [D1]-[D2] below are therefore DESCRIPTIVE");
    println(io, "sensitivity displays. Section [I] is the inferential statement: winsorized excess");
    println(io, "kurtosis at trim q (tails clamped at the q / 1-q empirical quantiles, thresholds");
    println(io, "recomputed inside every resample) has finite moments of all orders for any tail");
    println(io, "index, and the stationary block bootstrap is standard for such smooth marginal");
    println(io, "functionals under mixing.");
    println(io);
    @printf(io, "Observed raw excess kurtosis: IS = %.3f   OoS = %.3f   IS - OoS = %.3f\n",
            _excess_kurtosis(g_is), _excess_kurtosis(g_oos), _excess_kurtosis(g_is) - _excess_kurtosis(g_oos));
    println(io);
    println(io, "[D1] Raw excess kurtosis bootstrap distributions (DESCRIPTIVE):");
    println(io, "L     | IS median | IS 95% band       | OoS median | OoS 95% band       | Pr(IS > OoS)");
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
    println(io, "[D2] Raw difference (IS - OoS) resampling bands (DESCRIPTIVE - see SCOPE; these are");
    println(io, "sensitivity displays of resampling spread, not calibrated 95% confidence intervals):");
    println(io, "L     | diff median | percentile 95% band | basic 95% band      | covers 0?");
    println(io, "-"^88);
    for L in Ls
        ci = _diff_ci(results_is[L], results_oos[L], obs_diff);
        covers = (ci.pct[1] <= 0.0 <= ci.pct[2]) ? "yes" : "no";
        @printf(io, "%-5d | %11.3f | [%6.3f, %6.3f]   | [%6.3f, %6.3f]   | %s\n",
                L, ci.med, ci.pct[1], ci.pct[2], ci.basic[1], ci.basic[2], covers);
    end
    println(io);
    println(io, "[I] Winsorized excess kurtosis (INFERENTIAL target; trim q per tail):");
    for q in QS
        wdiff_obs = wobs_is[q] - wobs_oos[q];
        @printf(io, "  q = %.2f  observed winsorized: IS = %.3f   OoS = %.3f   IS - OoS = %.3f\n",
                q, wobs_is[q], wobs_oos[q], wdiff_obs);
        println(io, "  L     | IS 95% CI         | OoS 95% CI        | diff percentile 95% CI | covers 0?");
        println(io, "  ", "-"^84);
        for L in Ls
            bi = wins_is[(q, L)]; bo = wins_oos[(q, L)];
            ci = _diff_ci(bi, bo, wdiff_obs);
            covers = (ci.pct[1] <= 0.0 <= ci.pct[2]) ? "yes" : "no";
            @printf(io, "  %-5d | [%5.3f, %5.3f]   | [%5.3f, %5.3f]   | [%6.3f, %6.3f]      | %s\n",
                    L, quantile(bi, 0.025), quantile(bi, 0.975),
                    quantile(bo, 0.025), quantile(bo, 0.975), ci.pct[1], ci.pct[2], covers);
        end
        println(io);
    end
    println(io, "Reading:");
    ci10w = _diff_ci(wins_is[(0.01, 10)], wins_oos[(0.01, 10)], wobs_is[0.01] - wobs_oos[0.01]);
    if ci10w.pct[1] <= 0.0 <= ci10w.pct[2]
        @printf(io, "  - Inference (winsorized q = 0.01, L = 10): the 95%% percentile CI for the IS - OoS\n");
        @printf(io, "    difference is [%.3f, %.3f], which covers zero: the data do not exclude equal\n",
                ci10w.pct[1], ci10w.pct[2]);
        println(io, "    IS and OoS winsorized excess kurtosis at the 5% level. Non-rejection, not");
        println(io, "    equality.");
    else
        @printf(io, "  - Inference (winsorized q = 0.01, L = 10): the 95%% percentile CI for the IS - OoS\n");
        @printf(io, "    difference is [%.3f, %.3f], which excludes zero: the windows' winsorized excess\n",
                ci10w.pct[1], ci10w.pct[2]);
        println(io, "    kurtosis levels differ at the 5% level under this bootstrap.");
    end
    println(io, "  - The raw-kurtosis displays [D1]-[D2] are descriptive only; under the paper's own");
    println(io, "    heavy-tail diagnosis (Hill < 4) they cannot be read as calibrated 95% intervals.");
    @printf(io, "  - Pr(IS > OoS) at L = 10 (raw): %.3f. This is a DESCRIPTIVE bootstrap probability -\n",
            _diff_p(results_is[10], results_oos[10]));
    println(io,  "    the fraction of independent resample pairs in which the IS draw exceeds the");
    println(io,  "    OoS draw. It is NOT a p-value: each window is resampled around its own");
    println(io,  "    observed kurtosis, so no equal-kurtosis null is imposed.");
end

println("[done] $OUT");
