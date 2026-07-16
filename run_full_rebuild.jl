# ========================================================================================= #
# run_full_rebuild.jl
#
# HEADLINE rebuild driver: re-runs the headline analysis stages listed below
# in one pass (nine enumerated; stage 6, QuantGAN, is excluded from the default
# stage list, leaving eight executable). It does NOT regenerate every artefact in the paper --
# QuantGAN (excluded below), the reference Bayesian MSGARCH row (R/renv,
# standalone), and the standalone VaR back-test, walk-forward, state-selection,
# spectral, cross-asset-uncertainty, and robustness runners are invoked
# individually; see RUNNERS.md for the complete artefact-by-artefact map.
# Exits with status 1 if any stage fails (each stage is an independent
# top-level include; failure in one does not abort the others, but it is
# recorded and reported in the final summary):
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
# _attic/runners/. Descriptive runners cited only as standalone exploratory
# runs are invoked manually outside this dispatcher.
# ========================================================================================= #

using Pkg; Pkg.activate(".");
using Dates

println("="^72);
println("  Full rebuild against the current train/OoS split");
println("  Start: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"));
println("="^72);

const SCRIPTS = [
    "runners/headline/run_all_analysis.jl",
    "runners/headline/run_multi_emission_analysis.jl",
    "runners/headline/run_baselines_and_cross_asset.jl",
    "runners/headline/run_cross_asset_sim_copula.jl",
    "runners/diagnostics/run_diagnostics.jl",
    # "runners/headline/run_quantgan_baseline.jl",   # excluded: slowest stage by far,
                                    # deterministic with the global seed. Run standalone via
                                    #   julia --project=. runners/headline/run_quantgan_baseline.jl
                                    # only when the QuantGAN row in the extended
                                    # panel needs refreshing (architecture, training
                                    # data, or seed change). Re-include here only
                                    # if you want a single-command end-to-end
                                    # rebuild including the deep-generative row.
    "runners/headline/run_msgarch_baselines.jl",
    "runners/headline/run_smchmm_baseline.jl",
    "runners/headline/run_figures.jl",
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

failed = [s for s in SCRIPTS if stage_status[s] != :ok];
println("\n" * "="^72);
if isempty(failed)
    println("  HEADLINE REBUILD COMPLETE ($(length(SCRIPTS))/$(length(SCRIPTS)) stages ok)");
else
    println("  HEADLINE REBUILD COMPLETED WITH $(length(failed)) FAILED STAGE(S):");
    for s in failed
        println("    FAILED: $s");
    end
end
println("  End:   ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"));
println("  Per-stage timings (s):");
for s in SCRIPTS
    println("    $(rpad(s, 38)) $(round(stage_times[s], digits=1)) s   $(stage_status[s])");
end
println("="^72);
isempty(failed) || exit(1);
