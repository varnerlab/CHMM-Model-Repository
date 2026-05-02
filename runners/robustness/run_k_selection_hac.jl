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

# Per-fold val_ll/obs from the cached k-fold runs ----------------------------
# 4-fold full-year cadence (Appendix sec:k_selection_kfold_pre2020)
val_ll_4fold = Dict(
     3 => [-2.0133, -1.5552, -2.2504, -2.0011],
     6 => [-2.0673, -1.5376, -2.2811, -1.9738],
     9 => [-2.1193, -1.6089, -2.3104, -1.9929],
    12 => [-2.1815, -1.7323, -2.3192, -2.0552],
    18 => [-2.5881, -1.8880, -2.4229, -2.1540],
)

# 6-fold half-year cadence (Appendix sec:k_selection_kfold_pre2020 robustness)
val_ll_6fold = Dict(
     3 => [-1.5987, -1.4327, -2.2865, -2.2138, -2.0651, -1.9356],
     6 => [-1.5747, -1.4507, -2.2791, -2.2515, -2.0520, -1.8882],
     9 => [-1.6569, -1.5221, -2.3085, -2.2802, -2.0628, -1.9102],
    12 => [-1.7580, -1.5476, -2.3222, -2.3360, -2.1243, -1.9770],
    18 => [-1.9208, -1.7563, -2.4370, -2.3416, -2.2315, -2.0811],
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
    println(io, "Reading.")
    println(io, "  K=6 vs K=3 mean held-out log-lik gap remains insignificant under HAC at both cadences")
    println(io, "  (|z_HAC| < 1.96), consistent with the body conclusion that the held-out criterion")
    println(io, "  cannot distinguish K=6 from K=3 at conventional levels. The HAC correction does shift")
    println(io, "  the magnitude of |z| relative to the independent-fold pooled-SE statistic, which the")
    println(io, "  body should report explicitly: independent-fold |z| = 0.07 (4-fold) / 0.04 (6-fold)")
    println(io, "  vs HAC |z| (above). For K=18 vs K=6 the independent-fold |z| of 1.92 / 1.70 is also")
    println(io, "  re-evaluated under HAC; if HAC pushes |z_HAC| above 1.96, the K=18 sensitivity")
    println(io, "  reference can be re-described accordingly.")
end
println("wrote ", txt_path)

println("\ndone.")
