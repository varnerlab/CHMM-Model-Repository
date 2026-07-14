# run_k_selection_hac.jl
#
# Reviewer Round 2 / Item B2 (peer-review.md R2#1, R2#req-1).
#
# The body k-fold K-selection inference reports |z| under independent-fold
# pooled SEs. Rolling-origin folds inherit autocorrelation from the underlying
# series, so the independence assumption understates the |z| (and the proper
# inference may shift the substantive read).
#
# This runner re-computes the |z| under a Diebold-Mariano-style Newey-West HAC
# variance over the paired diff series across folds, for both fold designs
# (4-fold full-year cadence and 6-fold half-year cadence). Output is a single
# CSV / TXT pair suitable for citing in the body and for a one-row appendix
# subsection.
#
# Inputs are the per-fold val_ll/obs values reported in:
#   results/k_selection_validation/K_Selection_Kfold_Pre2020.txt        (4 folds)
#   results/k_selection_validation/K_Selection_Kfold_H12y_Pre2020.txt   (6 folds)
#
# Output:
#   results/k_selection_hac/k_selection_hac.csv
#   results/k_selection_hac/k_selection_hac.txt
#
# This is a pure post-processing operation; no model fits are re-run.

using Statistics
using Printf

const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "k_selection_hac")
isdir(OUTDIR) || mkpath(OUTDIR)

# Per-fold val_ll/obs from the cached k-fold runs (VWAP re-run, 2026-07-14) ----------------------------
# 4-fold full-year cadence (Appendix sec:k_selection_kfold_pre2020)
val_ll_4fold = Dict(
     3 => [-1.8535, -1.3792, -2.0483, -1.8912],
     6 => [-1.9642, -1.3414, -2.0306, -1.8626],
     9 => [-2.0266, -1.4777, -2.0431, -1.8726],
    12 => [-2.1151, -1.5387, -2.1031, -1.9137],
    18 => [-2.2862, -1.8010, -2.1716, -1.9867],
)

# 6-fold half-year cadence (Appendix sec:k_selection_kfold_pre2020 robustness)
val_ll_6fold = Dict(
     3 => [-1.3974, -1.3238, -2.1121, -1.9998, -1.9484, -1.8203],
     6 => [-1.3888, -1.2639, -2.0828, -1.9773, -1.9268, -1.7992],
     9 => [-1.5171, -1.3566, -2.0942, -2.0040, -1.9552, -1.7869],
    12 => [-1.5474, -1.4629, -2.1427, -2.0319, -1.9654, -1.8418],
    18 => [-1.8430, -1.7047, -2.1905, -2.1328, -2.0741, -1.8414],
)

"""
    nw_lrv(d, h)

Newey--West Bartlett-kernel long-run variance estimator on series `d` with
truncation lag `h`.
"""
function nw_lrv(d::AbstractVector, h::Int)
    n = length(d)
    dbar = mean(d)
    e = d .- dbar
    γ0 = sum(abs2, e) / n  # divide by n (not n-1) per NW
    s = γ0
    for k in 1:h
        γk = sum(e[k+1:n] .* e[1:n-k]) / n
        w = 1.0 - k / (h + 1)
        s += 2 * w * γk
    end
    return max(s, 0.0)  # truncate to non-negative
end

"""
    hac_dm(d)

Diebold--Mariano-style HAC test on a paired diff series. Returns
(mean, n, h_NW, indep_SE, hac_SE, z_indep, z_hac).
"""
function hac_dm(d::AbstractVector)
    n = length(d)
    h_nw = max(1, floor(Int, n^(1/3)))
    indep_var = var(d)                      # 1/(n-1) sample variance
    indep_se = sqrt(indep_var / n)
    lrv = nw_lrv(d, h_nw)
    hac_se = sqrt(lrv / n)
    dbar = mean(d)
    z_indep = dbar / indep_se
    z_hac = dbar / hac_se
    return (dbar=dbar, n=n, h_nw=h_nw, indep_se=indep_se, hac_se=hac_se,
            z_indep=z_indep, z_hac=z_hac)
end

# Compute paired diffs and HAC stats -----------------------------------------
function compute_panel(label::String, val_ll::Dict)
    rows = NamedTuple[]
    for (Ka, Kb) in [(6, 3), (18, 6), (18, 3)]
        d = val_ll[Ka] .- val_ll[Kb]
        r = hac_dm(d)
        push!(rows, (cadence=label, comparison="K=$Ka vs K=$Kb",
                     dbar=r.dbar, n=r.n, h_nw=r.h_nw,
                     indep_se=r.indep_se, hac_se=r.hac_se,
                     z_indep=r.z_indep, z_hac=r.z_hac))
    end
    return rows
end

rows_4 = compute_panel("4-fold full-year", val_ll_4fold)
rows_6 = compute_panel("6-fold half-year", val_ll_6fold)
all_rows = vcat(rows_4, rows_6)

# CSV ------------------------------------------------------------------------
csv_path = joinpath(OUTDIR, "k_selection_hac.csv")
open(csv_path, "w") do io
    println(io, "cadence,comparison,dbar,n,h_nw,indep_se,hac_se,z_indep,z_hac")
    for r in all_rows
        @printf(io, "%s,%s,%.6f,%d,%d,%.6f,%.6f,%.4f,%.4f\n",
                r.cadence, r.comparison, r.dbar, r.n, r.h_nw,
                r.indep_se, r.hac_se, r.z_indep, r.z_hac)
    end
end
println("wrote ", csv_path)

# TXT ------------------------------------------------------------------------
txt_path = joinpath(OUTDIR, "k_selection_hac.txt")
open(txt_path, "w") do io
    println(io, repeat("=", 110))
    println(io, "HAC-corrected K-selection inference  (Reviewer Round 2 / Item B2)")
    println(io, repeat("=", 110))
    println(io)
    println(io, "Paired diff series across rolling-origin folds; Diebold-Mariano-style")
    println(io, "Newey-West Bartlett-kernel long-run variance with truncation lag h = floor(n^(1/3)).")
    println(io, "Source: cached per-fold val_ll/obs in K_Selection_Kfold[_H12y]_Pre2020.txt.")
    println(io)
    @printf(io, "  %-20s %-15s  %8s  %3s  %3s  %10s  %10s  %8s  %8s\n",
            "cadence", "comparison", "dbar", "n", "h", "indep_SE", "HAC_SE", "z_indep", "z_HAC")
    println(io, "  ", repeat("-", 105))
    for r in all_rows
        @printf(io, "  %-20s %-15s  %+8.4f  %3d  %3d  %10.4f  %10.4f  %+8.4f  %+8.4f\n",
                r.cadence, r.comparison, r.dbar, r.n, r.h_nw,
                r.indep_se, r.hac_se, r.z_indep, r.z_hac)
    end
    println(io)
    println(io, "Reading (computed from the rows above).")
    for cadence_rows in (rows_4, rows_6)
        r63 = first(filter(r -> r.comparison == "K=6 vs K=3", cadence_rows))
        sig = abs(r63.z_hac) >= 1.96 ? "exceeds" : "stays below"
        dir = r63.dbar > 0 ? "K=6 above K=3" : "K=3 above K=6"
        @printf(io, "  %s, K=6 vs K=3: dbar=%+.4f (%s), z_HAC=%+.2f %s 1.96 in magnitude.\n",
                r63.cadence, r63.dbar, dir, r63.z_hac, sig)
    end
    println(io, "  The sign of the K=6 vs K=3 gap differs between the two fold cadences, and with")
    println(io, "  only n = 4 or n = 6 fold differences both the normal and HAC approximations are")
    println(io, "  too small-sample to support formal significance claims; treat every z above as a")
    println(io, "  descriptive check. K=18 rows quantify the magnitude of the large-K deterioration")
    println(io, "  under the same caveat.")
end
println("wrote ", txt_path)

println("\ndone.")
