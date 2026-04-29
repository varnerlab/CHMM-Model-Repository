# run_cross_ticker_anova.jl
# Within-sector vs between-sector ANOVA on the 30-ticker OoS KS distribution
# (peer-review item 18 / R2 Q5).
# Operates on the existing results/sector_panel/sector_panel_summary.csv
# without re-running the cross-ticker fits.

using CSV, DataFrames, Statistics, Printf, Distributions, Random;

const REPO_ROOT = @__DIR__;
const PANEL_CSV = joinpath(REPO_ROOT, "results", "sector_panel", "sector_panel_summary.csv");
const OUT_TXT   = joinpath(REPO_ROOT, "results", "sector_panel", "anova_oos_ks.txt");

df = CSV.read(PANEL_CSV, DataFrame);

# F-statistic for one-way ANOVA on ks_oos_pct grouped by sector.
function one_way_anova(values::Vector{<:Real}, groups::Vector{<:AbstractString})
    grand = mean(values);
    n = length(values);
    g = unique(groups);
    K = length(g);
    ss_between = 0.0;
    ss_within  = 0.0;
    for s in g
        mask = groups .== s;
        n_s = sum(mask);
        m_s = mean(values[mask]);
        ss_between += n_s * (m_s - grand)^2;
        ss_within  += sum((values[mask] .- m_s).^2);
    end
    df_between = K - 1;
    df_within  = n - K;
    ms_between = ss_between / df_between;
    ms_within  = ss_within  / df_within;
    F = ms_between / ms_within;
    eta_sq = ss_between / (ss_between + ss_within);
    return (F = F, df_between = df_between, df_within = df_within,
            ss_between = ss_between, ss_within = ss_within,
            ms_between = ms_between, ms_within = ms_within,
            eta_sq = eta_sq, K = K, n = n, grand = grand);
end

vals = Float64.(df.ks_oos_pct);
sectors = String.(df.sector);
result = one_way_anova(vals, sectors);

p = ccdf(FDist(result.df_between, result.df_within), result.F);

# Permutation p-value as a robustness check (5000 permutations).
Random.seed!(20260420);
B = 5000;
F_perm = zeros(B);
for b in 1:B
    perm = sectors[randperm(length(sectors))];
    r = one_way_anova(vals, perm);
    F_perm[b] = r.F;
end
p_perm = mean(F_perm .>= result.F);

open(OUT_TXT, "w") do io
    println(io, "Cross-ticker OoS KS ANOVA: within-sector vs between-sector");
    println(io, "Source: results/sector_panel/sector_panel_summary.csv (30 tickers, 10 sectors x 3 reps)");
    println(io, "Peer-review item 18 / R2 Q5.");
    println(io, repeat("=", 78));
    println(io);
    @printf(io, "n = %d  K (sectors) = %d  grand mean OoS KS = %.2f%%\n",
            result.n, result.K, result.grand);
    println(io);
    println(io, "One-way ANOVA on OoS KS pass rate by sector:");
    @printf(io, "  SS between = %10.2f   df = %d   MS = %10.2f\n",
            result.ss_between, result.df_between, result.ms_between);
    @printf(io, "  SS within  = %10.2f   df = %d   MS = %10.2f\n",
            result.ss_within, result.df_within, result.ms_within);
    @printf(io, "  F(%d, %d)   = %.3f\n",
            result.df_between, result.df_within, result.F);
    @printf(io, "  p (F-dist) = %.4f\n", p);
    @printf(io, "  p (perm, B=%d) = %.4f\n", B, p_perm);
    @printf(io, "  eta^2      = %.3f\n", result.eta_sq);
    println(io);
    println(io, "Per-sector means and within-sector standard deviations:");
    g = unique(sectors);
    for s in sort(g)
        mask = sectors .== s;
        v = vals[mask];
        @printf(io, "  %-25s n=%d   mean = %5.2f%%   sd = %5.2f%%   range = [%5.2f, %5.2f]%%\n",
                s, length(v), mean(v), Statistics.std(v), minimum(v), maximum(v));
    end
    println(io);
    println(io, "Reading:");
    if p < 0.05
        println(io, "  ANOVA F-test rejects null of equal sector means at alpha=0.05.");
        println(io, "  Sector membership explains a non-trivial share of OoS KS variance");
        @printf(io,  "  (eta^2 = %.1f%%): the cross-ticker failures are sector-stratified, not\n", 100 * result.eta_sq);
        println(io, "  uniformly distributed across the universe.");
    else
        println(io, "  ANOVA F-test fails to reject null of equal sector means at alpha=0.05.");
        println(io, "  Within-sector variance dominates between-sector variance: the cross-ticker");
        @printf(io,  "  failures are explained more by within-sector heterogeneity (eta^2 = %.1f%%)\n", 100 * result.eta_sq);
        println(io, "  than by sector-level effects.");
    end
end

println("Wrote $OUT_TXT");
println();
println("Results:");
println(read(OUT_TXT, String));
