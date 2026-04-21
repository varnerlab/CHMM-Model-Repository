# ========================================================================================= #
# run_v7_full_rebuild.jl
#
# Master driver that re-runs every v7-relevant pipeline against the new
# 10-year training / OoS-remainder split. Produces every figure, metrics
# file, and table referenced by Paper_v7.tex.
#
# Stages (each is an independent top-level include; failure in one does
# not abort the others, but a warning is printed):
#   1. run_all_analysis.jl             (stylized facts + per-K internals, SPY)
#   2. run_multi_emission_analysis.jl  (K x family sensitivity)
#   3. run_baselines_and_cross_asset.jl(Table 2 + Table T2)
#   4. run_cross_asset_sim_copula.jl   (SIM + Gaussian/Student-t copulas)
#   5. run_v7_revisions.jl             (utility, nu diagnostics, ...)
#   6. run_v7_gru.jl                   (deep-generative baseline)
#   7. run_v7_figures.jl               (K=18 main-body figures)
# ========================================================================================= #

using Pkg; Pkg.activate(".");
using Dates

println("="^72);
println("  Paper v7 — full rebuild against new train/OoS split");
println("  Start: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"));
println("="^72);

const SCRIPTS = [
    "run_all_analysis.jl",
    "run_multi_emission_analysis.jl",
    "run_baselines_and_cross_asset.jl",
    "run_cross_asset_sim_copula.jl",
    "run_v7_revisions.jl",
    "run_v7_gru.jl",
    "run_v7_figures.jl",
];

stage_times = Dict{String, Float64}();
stage_status = Dict{String, Symbol}();

for (i, script) in enumerate(SCRIPTS)
    println("\n" * "="^72);
    println("  [$i/$(length(SCRIPTS))] $script");
    println("  Start: ", Dates.format(now(), "HH:MM:SS"));
    println("="^72);
    t0 = time();
    try
        Main.include(joinpath(@__DIR__, script));
        stage_status[script] = :ok;
    catch err
        @warn "Stage failed: $script" exception=(err, catch_backtrace())
        stage_status[script] = :error;
    end
    dt = time() - t0;
    stage_times[script] = dt;
    println("\n  [stage $i] finished in $(round(dt, digits=1))s  status=$(stage_status[script])");
end

println("\n" * "="^72);
println("  FULL REBUILD COMPLETE");
println("  End:   ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"));
println("  Per-stage timings (s):");
for s in SCRIPTS
    println("    $(rpad(s, 38)) $(round(stage_times[s], digits=1)) s   $(stage_status[s])");
end
println("="^72);
