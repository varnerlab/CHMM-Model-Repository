# ========================================================================================= #
# run_paper_artifact_check.jl
#
# Paper/artifact consistency gate (third-review Phase A). Each check names a value that
# must appear BOTH in a model-repo artifact file and in the paper .tex source that prints
# it. When a runner regenerates an artifact and the value moves, the artifact-side check
# fails until the paper is updated (and vice versa), so displayed numbers can no longer
# drift silently away from their sources.
#
# The runner-to-artifact provenance map lives in results/artifacts_manifest.csv. This
# script holds the value-level assertions; extend CHECKS whenever a paper-facing number
# is added or regenerated.
#
# Usage:  julia --project=. runners/diagnostics/run_paper_artifact_check.jl
# Exit code 0 iff every check passes on both sides.
# ========================================================================================= #

const _ROOT      = abspath(joinpath(@__DIR__, "..", ".."));
const PAPER_ROOT = abspath(joinpath(_ROOT, "..", "CHMM-Paper-Repository"));

# (name, artifact path relative to model repo — file or directory,
#  tex path relative to paper repo, or "" to check the artifact side only,
#  value string that must appear verbatim in both)
const CHECKS = [
    ("chmm_n_acf_mae",        "results/kstar3_headline",                                   "sections/results.tex",              "0.0462"),
    ("chmm_n_crps_oos",       "results/kstar3_headline",                                   "sections/results.tex",              "1.0393"),
    ("chmm_l_crps_oos",       "results/kstar3_headline",                                   "sections/results.tex",              "1.0432"),
    ("chmm_t_pen_crps_oos",   "results/kstar3_headline",                                   "sections/results.tex",              "1.0382"),
    ("chmm_t_pen_is_kurt",    "results/kstar3_headline",                                   "sections/results.tex",              "17.67"),
    ("garcht_acf_pp_is",      "results/garch_suite",                                       "",                                  "0.048"),
    ("msgarch3_acf_pp_is",    "results/garch_suite",                                       "sections/results.tex",              "0.0408"),
    ("garch_pooled_acf",      "results/garch_suite",                                       "sections/baselines_appendix.tex",   "0.0309"),
    ("garch_canonical_pp",    "results/garch_canonical",                                   "sections/results.tex",              "0.0490"),
    ("cross_asset_oos_mae",   "results/cross_asset/Table-T3-Cross-Asset-Dependence.txt",   "sections/results.tex",              "0.209"),
    ("cross_asset_is_mae",    "results/cross_asset/Table-T3-Cross-Asset-Dependence.txt",   "sections/results.tex",              "0.027"),
    ("rolling_copula_paired", "results/cross_asset/Rolling_Copula_OoS.txt",                "sections/cross_asset_appendix.tex", "0.223"),
    ("dq_chmm_n_k3_a01_p",    "results/diagnostics/engle_manganelli_dq.txt",               "sections/sensitivity_appendix.tex", "0.056"),
    ("dq_boot_chmm_n_k3",     "results/var_inference_upgrade/var_inference_upgrade.txt",   "sections/results.tex",              "0.17"),
    ("msgarch_k4_medvar",     "results/msgarch_conditional_var/msgarch_conditional_var.csv", "sections/results.tex",            "-4.16"),
    ("hill_observed_top5",    "results/diagnostics/tail_index_families.txt",               "sections/results.tex",              "3.15"),
    ("spectral_median_share", "results/diagnostics/spectral_rank_cross_ticker.txt",        "sections/sensitivity_appendix.tex", "0.785"),
    ("spectral_nem_min",      "results/diagnostics/spectral_rank_cross_ticker.txt",        "sections/sensitivity_appendix.tex", "0.333"),
    ("spectral_spy_dom",      "results/diagnostics/spectral_rank.txt",                     "sections/sensitivity_appendix.tex", "0.939"),
    ("acf_halflife",          "results/acf_horizon/acf_horizon.txt",                       "sections/results.tex",              "14.5"),
    ("feed_sd_polygon",       "results/diagnostics/feed_boundary_check.txt",               "sections/supplementary.tex",        "1.640"),
    ("feed_sd_alpaca",        "results/diagnostics/feed_boundary_check.txt",               "sections/supplementary.tex",        "2.215"),
    ("walkforward_k3_median", "results/walkforward/walkforward_summary.txt",               "sections/supplementary.tex",        "62.1"),
    ("shared_nu_crps_oos",    "results/chmm_t_shared_nu/crps.txt",                         "sections/results.tex",              "1.0406"),
];

_read_all(path) = begin
    if isdir(path)
        buf = IOBuffer();
        for (root, _, files) in walkdir(path), f in files
            endswith(f, r"\.(txt|csv)") || continue;
            write(buf, read(joinpath(root, f), String));
        end
        String(take!(buf));
    else
        isfile(path) ? read(path, String) : "";
    end
end

n_fail = 0;
println("="^96);
println("  Paper/artifact consistency check   (model repo <-> $(PAPER_ROOT))");
println("="^96);
for (name, art_rel, tex_rel, value) in CHECKS
    art_path = joinpath(_ROOT, art_rel);
    art_text = _read_all(art_path);
    art_ok = occursin(value, art_text);
    art_msg = isempty(art_text) ? "MISSING FILE" : (art_ok ? "ok" : "value absent");

    if isempty(tex_rel)
        tex_ok, tex_msg = true, "(artifact-only)";
    else
        tex_path = joinpath(PAPER_ROOT, tex_rel);
        tex_text = isfile(tex_path) ? read(tex_path, String) : "";
        tex_ok = occursin(value, tex_text);
        tex_msg = isempty(tex_text) ? "MISSING FILE" : (tex_ok ? "ok" : "value absent");
    end

    status = art_ok && tex_ok ? "PASS" : "FAIL";
    status == "FAIL" && (global n_fail += 1);
    println(rpad(status, 5), rpad(name, 24), rpad(value, 10),
            "artifact: ", rpad(art_msg, 14), "paper: ", tex_msg);
end
println("-"^96);
println(n_fail == 0 ? "ALL CHECKS PASS" : "$n_fail CHECK(S) FAILED");
exit(n_fail == 0 ? 0 : 1);
