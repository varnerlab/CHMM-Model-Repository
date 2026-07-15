# ========================================================================================= #
# run_feed_boundary_check.jl
#
# Replaces the retired run_vendor_stitch_check.jl (third-audit item 1), whose
# "Polygon vs Alpaca 323-session overlap" was CIRCULAR: its Polygon side was
# MyOutOfSamplePortfolioDataSet(), the stitched OoS file whose 2025--2026 rows ARE the
# Alpaca extension rows appended by build_new_train_oos.jl, so the check compared the
# Alpaca rows with themselves and necessarily reported exact equality. The raw vendor
# files are date-disjoint -- Polygon/Massive ends 2024-12-31 and the Alpaca/IEX
# extension begins 2025-01-03 -- so no genuine cross-vendor overlap exists in this
# repository and no cross-vendor equality check is possible from the data on disk.
#
# This runner instead characterizes the feed boundary INSIDE the 573-session OoS window
# (2024-01-04 .. 2026-04-20), where the switch from Polygon consolidated aggregates to
# Alpaca IEX-only bars falls between 2024-12-31 and 2025-01-03:
#   1. per-segment growth-rate summary stats (Polygon-sourced 2024 returns vs
#      Alpaca-sourced 2025--2026 returns), from the vendor VWAP field;
#   2. two-sample KS between the two segments (NOTE: confounded with genuine time
#      variation -- this is a descriptive break diagnostic, not a vendor-equality test);
#   3. the same per-segment stats from the close column (price-field robustness);
#   4. the single cross-boundary return whose two endpoint prices come from different
#      feeds;
#   5. CHMM-N K = 3 filter-conditional VaR breaches by segment, from the stored
#      conditional_var_series_chmmN_k3.csv.
#
# Output: results/diagnostics/feed_boundary_check.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Dates, Statistics, Printf, HypothesisTests

const BOUNDARY  = Date(2025, 1, 3);   # first Alpaca/IEX session
const DT        = 1/252;
const TICKER    = "SPY";
const VWAP_COL  = :volume_weighted_average_price;
const OUT_PATH  = joinpath(_ROOT, "results", "diagnostics", "feed_boundary_check.txt");
const VAR_CSV   = joinpath(_ROOT, "results", "conditional_var_all_families",
                           "conditional_var_series_chmmN_k3.csv");

_excess_kurt(x) = begin
    m = mean(x); s2 = mean((x .- m).^2);
    mean((x .- m).^4) / s2^2 - 3.0;
end

_g(p) = log.(p[2:end] ./ p[1:(end-1)]) ./ DT;

function _seg_stats(g::AbstractVector)
    (n=length(g), mu=mean(g), sd=std(g), kurt=_excess_kurt(g));
end

println("="^88);
println("  OoS feed-boundary diagnostic (replaces the circular vendor-stitch check)");
println("="^88);

oos = MyOutOfSamplePortfolioDataSet()["dataset"][TICKER];
dates = Date.(oos.timestamp);
println("[setup] OoS $TICKER rows: $(length(dates))  span $(minimum(dates)) .. $(maximum(dates))");
println("[setup] Feed boundary: Polygon consolidated through $(maximum(dates[dates .< BOUNDARY])),");
println("        Alpaca/IEX from $BOUNDARY (raw vendor files are date-disjoint).");

g_vwap  = _g(oos[!, VWAP_COL]);
g_close = _g(oos[!, :close]);
gdates  = dates[2:end];                      # each return dated by its ending session

pre_m   = gdates .< BOUNDARY;                # both endpoint prices Polygon
post_m  = gdates .> BOUNDARY;                # both endpoint prices Alpaca
cross_m = gdates .== BOUNDARY;               # spans the feed switch

results = Dict{String,Any}();
for (label, g) in (("VWAP field", g_vwap), ("close column", g_close))
    pre  = _seg_stats(g[pre_m]);
    post = _seg_stats(g[post_m]);
    ks   = HypothesisTests.ApproximateTwoSampleKSTest(g[pre_m], g[post_m]);
    results[label] = (pre=pre, post=post, D=ks.δ, p=pvalue(ks), cross=g[cross_m][1]);
    @printf("\n[%s]\n", label);
    @printf("  Polygon segment (2024)      : n=%3d  mean=%+.3f  sd=%.3f  exc.kurt=%+.2f\n",
            pre.n, pre.mu, pre.sd, pre.kurt);
    @printf("  Alpaca segment  (2025-2026) : n=%3d  mean=%+.3f  sd=%.3f  exc.kurt=%+.2f\n",
            post.n, post.mu, post.sd, post.kurt);
    @printf("  two-sample KS (pre vs post) : D=%.4f  p=%.4f   (time-confounded)\n",
            ks.δ, pvalue(ks));
    @printf("  cross-boundary return       : %+.3f  (annualised; endpoints from different feeds)\n",
            g[cross_m][1]);
end

# ----------------------------------------------------------------------------------------- #
# VaR breaches by segment (stored CHMM-N K = 3 series; row i <-> OoS session i)
# ----------------------------------------------------------------------------------------- #
lines = readlines(VAR_CSV);
hdr = split(lines[1], ",");
ci = Dict(String(h) => i for (i, h) in enumerate(hdr));
n_rows = length(lines) - 1;
@assert n_rows == length(dates) "VaR series rows ($n_rows) != OoS sessions ($(length(dates)))";
b05 = zeros(Int, 2); b01 = zeros(Int, 2); nseg = zeros(Int, 2);
for r in 2:length(lines)
    f = split(lines[r], ",");
    idx = parse(Int, f[ci["idx"]]);
    seg = dates[idx] < BOUNDARY ? 1 : 2;     # boundary-day forecast counted in Alpaca segment
    nseg[seg] += 1;
    b05[seg] += parse(Int, f[ci["breach_05"]]);
    b01[seg] += parse(Int, f[ci["breach_01"]]);
end
println("\n[VaR] CHMM-N K = 3 filter-conditional VaR breaches by feed segment:");
@printf("  Polygon segment: n=%3d  breaches(5%%)=%2d (%.2f%%, exp %.1f)  breaches(1%%)=%d (%.2f%%, exp %.1f)\n",
        nseg[1], b05[1], 100b05[1]/nseg[1], 0.05nseg[1], b01[1], 100b01[1]/nseg[1], 0.01nseg[1]);
@printf("  Alpaca segment : n=%3d  breaches(5%%)=%2d (%.2f%%, exp %.1f)  breaches(1%%)=%d (%.2f%%, exp %.1f)\n",
        nseg[2], b05[2], 100b05[2]/nseg[2], 0.05nseg[2], b01[2], 100b01[2]/nseg[2], 0.01nseg[2]);

# ----------------------------------------------------------------------------------------- #
open(OUT_PATH, "w") do io
    println(io, "="^92);
    println(io, "OoS feed-boundary diagnostic  (third-audit item 1; replaces vendor_stitch_check.txt)");
    println(io, "="^92);
    println(io, "Provenance: the raw vendor files are date-disjoint. Polygon/Massive consolidated");
    println(io, "aggregates cover 2014-01-03 .. 2024-12-31; the Alpaca Markets IEX-feed extension");
    println(io, "covers 2025-01-03 .. 2026-04-20. NO cross-vendor overlap exists in data/, so no");
    println(io, "vendor-equality check is possible from the repository data. The retired");
    println(io, "vendor_stitch_check compared the stitched OoS rows with the Alpaca file they were");
    println(io, "copied from (circular; exact equality by construction) and is superseded here.");
    println(io);
    println(io, "The feed switch falls INSIDE the 573-session OoS window (2024-01-04 .. 2026-04-20).");
    println(io, "Segments: Polygon-sourced returns end on sessions < $(BOUNDARY); Alpaca-sourced");
    println(io, "returns end on sessions > $(BOUNDARY); the single cross-boundary return spans both.");
    println(io, "The pre-vs-post KS is CONFOUNDED with genuine time variation; it is a descriptive");
    println(io, "break diagnostic, not a vendor-equality test.");
    println(io);
    for label in ("VWAP field", "close column")
        r = results[label];
        println(io, "[$label]");
        @printf(io, "  Polygon segment (2024)      : n=%3d  mean=%+.3f  sd=%.3f  exc.kurt=%+.2f\n",
                r.pre.n, r.pre.mu, r.pre.sd, r.pre.kurt);
        @printf(io, "  Alpaca segment  (2025-2026) : n=%3d  mean=%+.3f  sd=%.3f  exc.kurt=%+.2f\n",
                r.post.n, r.post.mu, r.post.sd, r.post.kurt);
        @printf(io, "  two-sample KS (pre vs post) : D=%.4f  p=%.4f\n", r.D, r.p);
        @printf(io, "  cross-boundary return       : %+.3f (annualised)\n", r.cross);
        println(io);
    end
    println(io, "[VaR] CHMM-N K = 3 filter-conditional VaR breaches by feed segment");
    @printf(io, "  Polygon segment: n=%3d  breaches(5%%)=%2d (%.2f%%, exp %.1f)  breaches(1%%)=%d (%.2f%%, exp %.1f)\n",
            nseg[1], b05[1], 100b05[1]/nseg[1], 0.05nseg[1], b01[1], 100b01[1]/nseg[1], 0.01nseg[1]);
    @printf(io, "  Alpaca segment : n=%3d  breaches(5%%)=%2d (%.2f%%, exp %.1f)  breaches(1%%)=%d (%.2f%%, exp %.1f)\n",
            nseg[2], b05[2], 100b05[2]/nseg[2], 0.05nseg[2], b01[2], 100b01[2]/nseg[2], 0.01nseg[2]);
end
println("\n[done] Wrote $OUT_PATH");
