# ========================================================================================= #
# run_bh_fdr.jl
#
# Benjamini-Hochberg false-discovery-rate adjustment for the VaR back-test panels
# (2026-07 technical-review response, finding 3: the main text promised a BH
# correction in the appendix that did not exist).
#
# Pure post-processing over the cached artifacts; no model fits are re-run.
# Three panels, each adjusted separately (the BH family is the panel):
#   1. Walk-forward Christoffersen-cc, 24 rows
#      (results/walkforward/walkforward_conditional_var.csv, column p_cc)
#   2. Four-family main-window Christoffersen-cc, 16 rows
#      (results/conditional_var_all_families/conditional_var_panel.csv, p_cc)
#   3. Four-family main-window Engle-Manganelli DQ, 16 rows
#      (results/diagnostics/engle_manganelli_dq_all_families.csv, DQ_p)
#
# Output: results/diagnostics/bh_fdr/bh_fdr.{csv,txt}
# ========================================================================================= #

using Printf
using DelimitedFiles

const ROOT = joinpath(@__DIR__, "..", "..");
const OUT_DIR = joinpath(ROOT, "results", "diagnostics", "bh_fdr");
mkpath(OUT_DIR);
const Q_LEVEL = 0.05;

function bh_qvalues(p::Vector{Float64})
    m = length(p);
    order = sortperm(p);
    q = fill(NaN, m);
    prev = 1.0;
    for (rank, idx) in Iterators.reverse(collect(enumerate(order)))
        val = min(prev, p[idx] * m / rank);
        q[idx] = val;
        prev = val;
    end
    return q;
end

function read_panel(path::String, label_cols::Vector{Int}, p_col::Int)
    raw, header = readdlm(path, ',', Any; header=true);
    labels = [join(string.(raw[i, label_cols]), " ") for i in 1:size(raw, 1)];
    p = Float64.(raw[:, p_col]);
    return labels, p;
end

panels = [
    ("walkforward cc", joinpath(ROOT, "results", "walkforward", "walkforward_conditional_var.csv"), [1, 2, 3], 14),
    ("four-family cc", joinpath(ROOT, "results", "conditional_var_all_families", "conditional_var_panel.csv"), [1, 2, 3], 12),
    ("four-family DQ", joinpath(ROOT, "results", "diagnostics", "engle_manganelli_dq_all_families.csv"), [1, 2, 3], 9),
];

rows = NamedTuple[];
for (panel, path, label_cols, p_col) in panels
    labels, p = read_panel(path, label_cols, p_col);
    q = bh_qvalues(p);
    for i in eachindex(p)
        push!(rows, (panel=panel, label=labels[i], p=p[i], q=q[i],
                     raw_reject=p[i] < 0.05, bh_reject=q[i] < Q_LEVEL));
    end
end

csv_path = joinpath(OUT_DIR, "bh_fdr.csv");
open(csv_path, "w") do io
    println(io, "panel,label,p,q_bh,raw_reject_5pct,bh_reject_q05");
    for r in rows
        @printf(io, "%s,%s,%.4f,%.4f,%d,%d\n", r.panel, r.label, r.p, r.q,
                r.raw_reject ? 1 : 0, r.bh_reject ? 1 : 0);
    end
end

txt_path = joinpath(OUT_DIR, "bh_fdr.txt");
open(txt_path, "w") do io
    println(io, "="^100);
    println(io, "Benjamini-Hochberg FDR adjustment for the VaR back-test panels (2026-07 review, finding 3)");
    println(io, "="^100);
    println(io);
    println(io, "Each panel is one BH family; q-values by the step-up procedure; FDR level q = $Q_LEVEL.");
    println(io);
    for (panel, _, _, _) in panels
        sub = filter(r -> r.panel == panel, rows);
        n_raw = count(r -> r.raw_reject, sub);
        n_bh = count(r -> r.bh_reject, sub);
        @printf(io, "%-16s: %2d rows, %2d raw rejections at 5%%, %2d BH rejections at q = %.2f\n",
                panel, length(sub), n_raw, n_bh, Q_LEVEL);
        for r in sub
            if r.raw_reject || r.bh_reject
                @printf(io, "    %-28s p = %.4f   q = %.4f   %s\n", r.label, r.p, r.q,
                        r.bh_reject ? "BH-reject" : "raw-only (not BH-significant)");
            end
        end
        println(io);
    end
    println(io, "Reading (computed): rejections whose q-value stays below $Q_LEVEL survive the");
    println(io, "multiplicity correction within their panel; raw-only rows do not.");
end
println("wrote ", csv_path);
println("wrote ", txt_path);
