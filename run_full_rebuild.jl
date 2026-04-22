# ========================================================================================= #
# run_full_rebuild.jl
#
# Master driver that re-runs every analysis pipeline end-to-end. Produces every
# figure, metrics file, and table referenced by the paper.
#
# Stages (each is an independent top-level include; failure in one does not abort
# the others, but a warning is printed):
#   1.  run_all_analysis.jl               (stylized facts + per-K internals, SPY)
#   2.  run_multi_emission_analysis.jl    (K x family sensitivity)
#   3.  run_baselines_and_cross_asset.jl  (Table 2 + Table T2, Pipeline A)
#   4.  run_cross_asset_sim_copula.jl     (Table T3, Pipeline B)
#   5.  run_diagnostics.jl                (utility, nu diagnostics, walk-forward, ...)
#   6.  run_gru_baseline.jl               (deep-generative baseline)
#   7.  run_equity_price_sim.jl           (price fans + terminal distributions)
#   8.  run_track_a_metrics.jl            (A1, A2, A3, A6, A7, A8, A9)
#   9.  run_track_a_utility.jl            (A4, A5)
#   10. run_track_b4_msgarch.jl           (B4 MS-GARCH baseline)
#   11. run_track_c1_smchmm.jl            (C1 semi-Markov extension)
#   12. run_track_c3_conditional_var.jl   (C3a conditional VaR)
#   13. run_track_c3_time_varying_transition.jl (C3 time-varying transitions)
#   14. run_track_c4_leverage_emission.jl (C4 leverage-emission ablation)
#   15. run_figures.jl                    (K=18 main-body figures)
# ========================================================================================= #

using Pkg; Pkg.activate(".");
using Dates

println("="^72);
println("  Full rebuild against the current train/OoS split");
println("  Start: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"));
println("="^72);

const SCRIPTS = [
    "run_all_analysis.jl",
    "run_multi_emission_analysis.jl",
    "run_baselines_and_cross_asset.jl",
    "run_cross_asset_sim_copula.jl",
    "run_diagnostics.jl",
    "run_gru_baseline.jl",
    "run_equity_price_sim.jl",
    "run_track_a_metrics.jl",
    "run_track_a_utility.jl",
    "run_track_b4_msgarch.jl",
    "run_track_c1_smchmm.jl",
    "run_track_c3_conditional_var.jl",
    "run_track_c3_time_varying_transition.jl",
    "run_track_c4_leverage_emission.jl",
    "run_figures.jl",
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
