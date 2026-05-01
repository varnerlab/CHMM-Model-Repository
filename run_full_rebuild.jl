# ========================================================================================= #
# run_full_rebuild.jl
#
# Master driver that re-runs every analysis pipeline end-to-end. Produces every
# figure, metrics file, and table referenced by the arXiv preprint.
#
# Stages (each is an independent top-level include; failure in one does not abort
# the others, but a warning is printed):
#   1.  run_all_analysis.jl                  (stylized facts + per-K internals, SPY)
#   2.  run_multi_emission_analysis.jl       (K x family sensitivity)
#   3.  run_baselines_and_cross_asset.jl     (headline Pipeline A panel + per-ticker)
#   4.  run_cross_asset_sim_copula.jl        (Pipeline B Student-t copula)
#   5.  run_diagnostics.jl                   (utility, nu diagnostics, K-selection, ...)
#   6.  run_quantgan_baseline.jl             (QuantGAN deep-generative baseline row)
#   7.  run_msgarch_baselines.jl             (MS-GARCH K=2/3 rows in extended panel)
#   8.  run_smchmm_baseline.jl               (SM-CHMM rows in extended panel)
#   9.  run_figures.jl                       (K=18 main-body figures)
#
# Earlier journal-revision stages (Track A extended evaluation, Track B3 diffusion,
# Track C3 conditional-VaR variants, Track C4 leverage emission, GRU baseline) and
# superseded one-shot runners (run_copula_profile_ci.jl, run_hsmm_ml.jl,
# run_multiseed_headline.jl, regen_var_es_fig.jl, et al.) are archived under
# _attic_v10/runners/. Descriptive runners cited only as standalone exploratory
# runs are invoked manually outside this dispatcher.
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
    # "run_quantgan_baseline.jl",   # excluded: slowest stage by far, deterministic
                                    # with the global seed. Run standalone via
                                    #   julia --project=. run_quantgan_baseline.jl
                                    # only when the QuantGAN row in the extended
                                    # panel needs refreshing (architecture, training
                                    # data, or seed change). Re-include here only
                                    # if you want a single-command end-to-end
                                    # rebuild including the deep-generative row.
    "run_msgarch_baselines.jl",
    "run_smchmm_baseline.jl",
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
