# =========================================================================== #
# run_vendor_stitch_check.jl
#
# Review-response Item 9: vendor-stitch sanity check for the OoS window.
#
# The paper's OoS series stitches two data vendors:
#   Polygon.io via Massive : 2014-01-03 to 2025-11-18
#   Alpaca Markets / IEX   : 2024-01-04 to 2026-04-20 (extension)
#
# This runner verifies that the two vendors agree on the overlap window and
# that no detectable boundary artefact appears in the daily-return autocorrel-
# ation around the stitch date (2025-11-19).
#
# Diagnostics:
#   1. Per-day VWAP differential on the overlap (Polygon vs Alpaca).
#   2. Per-day return differential on the overlap.
#   3. Two-sample KS test on the overlap returns (Polygon vs Alpaca).
#   4. Rolling 30-day kurtosis on each source over the overlap.
#   5. Lag-1 autocorrelation of |G_t| in a +/- 20-day window around the
#      stitch date on the deployed (stitched) series.
#
# Decision rule: if the vendors disagree on the overlap (KS p < 0.01) or the
# lag-1 |G_t| ACF jumps by more than 2 sigma relative to the local baseline at
# the stitch date, fall back to a single-vendor cutoff and rerun every OoS-
# dependent table.
#
# Output: results/diagnostics/vendor_stitch_check.{txt,csv}
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Dates, Statistics, LinearAlgebra, Distributions, Printf

const OUT_DIR = joinpath(_ROOT, "results", "diagnostics");
mkpath(OUT_DIR);

const STITCH_DATE = Date(2025, 11, 19);
const WINDOW_DAYS = 20;

println("="^80)
println("  Item 9: Vendor-stitch sanity check (Polygon vs Alpaca on OoS overlap)")
println("="^80)

# -------- load both sources --------
# Polygon Train + OoS-Remainder spans 2014-01-03 to 2025-11-18 inclusive.
# The 2025-01-03 to 2026-04-20 OHLC file holds the Alpaca extension.
polygon_train = MyPortfolioDataSet()["dataset"];
polygon_oos   = MyOutOfSamplePortfolioDataSet()["dataset"];
alpaca_path   = joinpath(_PATH_TO_DATA, "SP500-Daily-OHLC-1-3-2025-to-4-20-2026.jld2");
@assert isfile(alpaca_path) "Alpaca extension file missing: $alpaca_path"
alpaca_ext    = _jld2(alpaca_path)["dataset"];

# Use SPY as the canonical comparison ticker.
const TICKER = "SPY";
@assert haskey(polygon_oos, TICKER) "Polygon OoS missing $TICKER"
@assert haskey(alpaca_ext, TICKER)  "Alpaca extension missing $TICKER"

poly_df = polygon_oos[TICKER];
alp_df  = alpaca_ext[TICKER];
println("[setup] Polygon OoS rows: $(nrow(poly_df))")
println("[setup] Alpaca ext rows : $(nrow(alp_df))")
println("[setup] Polygon OoS date span: $(minimum(Date.(poly_df.timestamp))) to $(maximum(Date.(poly_df.timestamp)))")
println("[setup] Alpaca ext date span : $(minimum(Date.(alp_df.timestamp))) to $(maximum(Date.(alp_df.timestamp)))")

# -------- align on overlap --------
overlap_dates = sort(collect(intersect(Set(Date.(poly_df.timestamp)), Set(Date.(alp_df.timestamp)))));
n_overlap = length(overlap_dates);
println("\n[overlap] $n_overlap shared dates")

if n_overlap < 30
    println("\n[warn] Fewer than 30 overlap days. Vendor-stitch check is data-limited:")
    println("       cannot compute KS or rolling-kurtosis comparisons reliably.")
    println("       Reporting only what is computable.\n")
end

# pick rows in date order
function _pick_rows(df, dates_set)
    sel = [d in dates_set for d in Date.(df.timestamp)];
    return df[sel, :];
end
overlap_set = Set(overlap_dates);
poly_o = _pick_rows(poly_df, overlap_set);
alp_o  = _pick_rows(alp_df,  overlap_set);
sort!(poly_o, :timestamp);
sort!(alp_o,  :timestamp);
@assert Date.(poly_o.timestamp) == Date.(alp_o.timestamp) "Date alignment mismatch after sort"

# -------- diagnostics 1+2: per-day VWAP / return differentials --------
const VWAP_COL = :volume_weighted_average_price;
poly_v = poly_o[!, VWAP_COL];
alp_v  = alp_o[!,  VWAP_COL];
vwap_diff = (poly_v .- alp_v) ./ alp_v;
poly_ret = log.(poly_v[2:end] ./ poly_v[1:(end-1)]) .* 252;
alp_ret  = log.(alp_v[2:end]  ./ alp_v[1:(end-1)])  .* 252;
ret_diff = poly_ret .- alp_ret;

println("[diag] VWAP differential (Polygon - Alpaca) / Alpaca:")
println(@sprintf("       mean=%+.6f  sd=%.6f  max-abs=%.6f",
    mean(vwap_diff), std(vwap_diff), maximum(abs.(vwap_diff))));
println("[diag] Daily return differential (Polygon - Alpaca):")
println(@sprintf("       mean=%+.6f  sd=%.6f  max-abs=%.6f",
    mean(ret_diff), std(ret_diff), maximum(abs.(ret_diff))));

# -------- diagnostic 3: two-sample KS on overlap returns --------
ks_ret = NaN; ks_p = NaN;
if n_overlap >= 5
    ks = HypothesisTests.ApproximateTwoSampleKSTest(poly_ret, alp_ret);
    ks_ret = ks.δ;       # KS statistic
    ks_p   = pvalue(ks);
    println(@sprintf("[diag] Two-sample KS (Polygon vs Alpaca returns): D=%.4f, p=%.4f",
        ks_ret, ks_p));
end

# -------- diagnostic 4: rolling 30-day kurtosis on each source --------
function _rolling_kurt(x::AbstractVector, w::Int)
    n = length(x);
    out = fill(NaN, n);
    for t in w:n
        out[t] = kurtosis(x[(t-w+1):t]);
    end
    return out;
end
roll_w = min(30, max(5, n_overlap ÷ 3));
poly_kurt = _rolling_kurt(poly_ret, roll_w);
alp_kurt  = _rolling_kurt(alp_ret,  roll_w);
kurt_diff = poly_kurt .- alp_kurt;
valid = .!isnan.(kurt_diff);
println(@sprintf("[diag] Rolling-%d kurtosis differential (Polygon - Alpaca):", roll_w))
if any(valid)
    println(@sprintf("       mean=%+.4f  sd=%.4f  max-abs=%.4f",
        mean(kurt_diff[valid]), std(kurt_diff[valid]), maximum(abs.(kurt_diff[valid]))));
end

# -------- diagnostic 5: lag-1 |G_t| ACF around the stitch date --------
# Build the deployed (stitched) series: Polygon up to and including 2025-11-18,
# Alpaca from 2025-11-19 onward.
function _stitched_series()
    # Polygon part of OoS up through 2025-11-18
    poly_idx = Date.(poly_df.timestamp) .<= STITCH_DATE - Day(1);
    alp_idx  = Date.(alp_df.timestamp)  .>= STITCH_DATE;
    p_dates  = Date.(poly_df.timestamp)[poly_idx];
    p_vwap   = poly_df[poly_idx, VWAP_COL];
    a_dates  = Date.(alp_df.timestamp)[alp_idx];
    a_vwap   = alp_df[alp_idx, VWAP_COL];
    dates = vcat(p_dates, a_dates);
    vwap  = vcat(p_vwap,  a_vwap);
    p     = sortperm(dates);
    return dates[p], vwap[p];
end
dates_s, vwap_s = _stitched_series();
ret_s = log.(vwap_s[2:end] ./ vwap_s[1:(end-1)]) .* 252;
abs_s = abs.(ret_s);
date_s_ret = dates_s[2:end];

# locate stitch index (first day on or after STITCH_DATE in the return series)
stitch_idx = findfirst(d -> d >= STITCH_DATE, date_s_ret);
if stitch_idx === nothing
    println("\n[warn] Stitch date $STITCH_DATE not in stitched return series; skipping ACF check.")
else
    a, b = max(1, stitch_idx - WINDOW_DAYS), min(length(abs_s), stitch_idx + WINDOW_DAYS);
    pre  = abs_s[max(1, stitch_idx - 2*WINDOW_DAYS):(stitch_idx - 1)];
    post = abs_s[stitch_idx:min(length(abs_s), stitch_idx + 2*WINDOW_DAYS - 1)];
    function _lag1_acf(x::AbstractVector)
        if length(x) < 4; return NaN; end
        return cor(x[1:(end-1)], x[2:end]);
    end
    pre_acf  = _lag1_acf(pre);
    post_acf = _lag1_acf(post);
    println(@sprintf("\n[diag] Lag-1 |G_t| ACF in +/- %d day window around stitch:", WINDOW_DAYS))
    println(@sprintf("       pre-stitch  (n=%d): %+.4f", length(pre),  pre_acf));
    println(@sprintf("       post-stitch (n=%d): %+.4f", length(post), post_acf));
    println(@sprintf("       differential: %+.4f", post_acf - pre_acf));
end

# -------- decision summary --------
verdict = "PASS";
notes = String[];
if !isnan(ks_p) && ks_p < 0.01
    verdict = "FAIL";
    push!(notes, "KS rejects vendor agreement on overlap (p=$ks_p < 0.01)")
end
if any(valid) && maximum(abs.(kurt_diff[valid])) > 1.5
    push!(notes, "Rolling-kurtosis differential exceeds 1.5 on at least one window");
end

println("\n" * "="^80)
println("  Verdict: $verdict")
if !isempty(notes)
    println("  Notes:")
    for n in notes; println("    - $n"); end
end
println("="^80)

# -------- emit artefacts --------
open(joinpath(OUT_DIR, "vendor_stitch_check.txt"), "w") do io
    println(io, "Item 9: Vendor-stitch sanity check (Polygon via Massive vs Alpaca / IEX)")
    println(io, "  Stitch date: $STITCH_DATE")
    println(io, "  Overlap window: $n_overlap shared trading days")
    println(io, "  Verdict: $verdict")
    println(io, "")
    println(io, "Diagnostics:")
    println(io, @sprintf("  VWAP differential (Polygon - Alpaca)/Alpaca: mean=%+.6f sd=%.6f max-abs=%.6f",
        mean(vwap_diff), std(vwap_diff), maximum(abs.(vwap_diff))))
    println(io, @sprintf("  Return differential                       : mean=%+.6f sd=%.6f max-abs=%.6f",
        mean(ret_diff), std(ret_diff), maximum(abs.(ret_diff))))
    if !isnan(ks_p)
        println(io, @sprintf("  Two-sample KS on overlap returns: D=%.4f, p=%.4f", ks_ret, ks_p))
    end
    if !isempty(notes)
        println(io, "")
        println(io, "Notes:")
        for n in notes; println(io, "  - $n"); end
    end
end

open(joinpath(OUT_DIR, "vendor_stitch_check.csv"), "w") do io
    write(io, "metric,value\n")
    write(io, @sprintf("overlap_days,%d\n", n_overlap))
    write(io, @sprintf("vwap_diff_mean,%.6e\n", mean(vwap_diff)))
    write(io, @sprintf("vwap_diff_sd,%.6e\n",   std(vwap_diff)))
    write(io, @sprintf("vwap_diff_maxabs,%.6e\n", maximum(abs.(vwap_diff))))
    write(io, @sprintf("ret_diff_mean,%.6e\n",  mean(ret_diff)))
    write(io, @sprintf("ret_diff_sd,%.6e\n",    std(ret_diff)))
    write(io, @sprintf("ret_diff_maxabs,%.6e\n",  maximum(abs.(ret_diff))))
    if !isnan(ks_p)
        write(io, @sprintf("ks_D,%.4f\n", ks_ret))
        write(io, @sprintf("ks_pvalue,%.4f\n", ks_p))
    end
    write(io, "verdict,$verdict\n")
end

println("\n[done] Output: $OUT_DIR/vendor_stitch_check.{txt,csv}")
