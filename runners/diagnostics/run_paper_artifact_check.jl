# ========================================================================================= #
# run_paper_artifact_check.jl
#
# Paper/artifact consistency gate (third-review Phase A; upgraded fourth review, item 9).
# Three layers:
#
#   [1] MANIFEST STATUS GATE - results/artifacts_manifest.csv is the runner-to-artifact
#       provenance map. Any row whose status field contains STALE / UNTRACED / pending /
#       defect fails the gate, so the manifest can no longer read green while its own
#       metadata says otherwise. ("unverified" is allowed: it marks rows not yet
#       spot-checked, not rows known to be wrong.)
#
#   [2] SUBSTRING CHECKS - a value string must appear verbatim BOTH in a model-repo
#       artifact (file or directory) and in the paper .tex source that prints it. Used
#       where the artifact and the paper print the value at the same precision.
#
#   [3] KEYED CHECKS - for drift-prone numbers where the artifact carries more digits
#       than the paper: a regex (with one capture group) extracts the numeric value from
#       a specific artifact file; the value is rounded to the paper's precision and the
#       formatted string must (a) equal the expected paper value and (b) appear in the
#       paper .tex file. This pins the number to its exact source row/column rather than
#       to an accidental substring elsewhere in the file.
#
# Usage:  julia --project=. runners/diagnostics/run_paper_artifact_check.jl
# Exit code 0 iff every layer passes.
# ========================================================================================= #

using Printf

const _ROOT      = abspath(joinpath(@__DIR__, "..", ".."));
const PAPER_ROOT = abspath(joinpath(_ROOT, "..", "CHMM-Paper-Repository"));
const MANIFEST   = joinpath(_ROOT, "results", "artifacts_manifest.csv");

n_fail = 0;

# ----------------------------------------------------------------------------------------- #
# [1] Manifest status gate
# ----------------------------------------------------------------------------------------- #
println("="^96);
println("  [1] Manifest status gate   ($MANIFEST)");
println("="^96);
const BAD_STATUS = [r"STALE"i, r"UNTRACED"i, r"pending"i, r"defect"i];
if isfile(MANIFEST)
    for (i, line) in enumerate(readlines(MANIFEST))
        i == 1 && continue;
        isempty(strip(line)) && continue;
        status = strip(split(line, ",")[end]);
        hits = [pat for pat in BAD_STATUS if occursin(pat, line)];
        if !isempty(hits)
            global n_fail += 1;
            println("FAIL  manifest row $i: status flags ", join([string(p.pattern) for p in hits], ", "),
                    "  ->  ", first(line, 90), "...");
        end
    end
    n_fail == 0 && println("ok    no STALE/UNTRACED/pending/defect statuses in $(length(readlines(MANIFEST)) - 1) rows");
else
    global n_fail += 1;
    println("FAIL  manifest missing");
end

# ----------------------------------------------------------------------------------------- #
# [2] Substring checks
# (name, artifact path relative to model repo - file or directory,
#  tex path relative to paper repo, or "" to check the artifact side only,
#  value string that must appear verbatim in both)
# ----------------------------------------------------------------------------------------- #
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
    ("rolling_copula_roll_t", "results/cross_asset/Rolling_Copula_OoS.txt",                "sections/cross_asset_appendix.tex", "0.186"),
    ("dq_chmm_n_k3_a01_p",    "results/diagnostics/engle_manganelli_dq.txt",               "sections/sensitivity_appendix.tex", "0.056"),
    ("msgarch_k4_medvar",     "results/msgarch_conditional_var/msgarch_conditional_var.csv", "sections/results.tex",            "-4.16"),
    ("hill_observed_top5",    "results/diagnostics/tail_index_families.txt",               "sections/results.tex",              "3.15"),
    ("spectral_median_share", "results/diagnostics/spectral_rank_cross_ticker.txt",        "sections/sensitivity_appendix.tex", "0.785"),
    ("spectral_nem_min",      "results/diagnostics/spectral_rank_cross_ticker.txt",        "sections/sensitivity_appendix.tex", "0.333"),
    ("spectral_spy_dom",      "results/diagnostics/spectral_rank.txt",                     "sections/sensitivity_appendix.tex", "0.939"),
    ("spectral_dom_int_k18",  "results/diagnostics/spectral_rank_cross_ticker.txt",        "sections/sensitivity_appendix.tex", "0.935"),
    ("spectral_acf_k3_near",  "results/diagnostics/spectral_rank_cross_ticker.txt",        "sections/sensitivity_appendix.tex", "0.0543"),
    ("spectral_acf_k18_near", "results/diagnostics/spectral_rank_cross_ticker.txt",        "sections/sensitivity_appendix.tex", "0.0697"),
    ("xticker_refit_median",  "results/sector_panel/sector_panel_quarterly_refit.txt",     "sections/sensitivity_appendix.tex", "84.6"),
    ("acf_halflife",          "results/acf_horizon/acf_horizon.txt",                       "sections/results.tex",              "14.5"),
    ("acf_band_1_5",          "results/acf_horizon/acf_horizon.txt",                       "",                                  "0.0416"),
    ("acf_band_64_252",       "results/acf_horizon/acf_horizon.txt",                       "",                                  "0.0410"),
    ("feed_sd_polygon",       "results/diagnostics/feed_boundary_check.txt",               "sections/supplementary.tex",        "1.640"),
    ("feed_sd_alpaca",        "results/diagnostics/feed_boundary_check.txt",               "sections/supplementary.tex",        "2.215"),
    ("walkforward_k3_median", "results/walkforward/walkforward_summary.txt",               "sections/supplementary.tex",        "62.1"),
    ("shared_nu_crps_oos",    "results/chmm_t_shared_nu/crps.txt",                         "sections/results.tex",              "1.0406"),
    ("shared_nu_oos_ks",      "results/chmm_t_shared_nu/chmm_t_shared_nu.csv",             "sections/results.tex",              "82.1"),
    ("ks2s_crit_is_L20",      "results/ks_block_bootstrap/KS_Bootstrap_Recalibration.txt", "sections/baselines_appendix.tex",   "0.0497"),
    ("ks2s_crit_oos_L20",     "results/ks_block_bootstrap/KS_Bootstrap_Recalibration_OoS.txt", "sections/baselines_appendix.tex", "0.0927"),
    ("kurt_diff_ci_L10",      "results/diagnostics/kurtosis_bootstrap.txt",                "sections/sensitivity_appendix.tex", "-4.24"),
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

println();
println("="^96);
println("  [2] Substring checks   (model repo <-> $(PAPER_ROOT))");
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

# ----------------------------------------------------------------------------------------- #
# [3] Keyed checks
# (name, artifact path, regex with ONE capture group extracting the numeric value,
#  rounding digits, expected paper-precision string, tex path)
# ----------------------------------------------------------------------------------------- #
const KEYED = [
    ("dq_boot_chmm_n_k3",  "results/var_inference_upgrade/var_inference_upgrade.txt",
        r"CHMM-N K=3\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", 2, "0.19", "sections/results.tex"),
    ("dq_boot_msgarch_k4", "results/var_inference_upgrade/var_inference_upgrade.txt",
        r"MS-GARCH K=4\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", 2, "0.10", "sections/results.tex"),
    ("crps_dm_tpen_vs_l_praw", "../CHMM-Paper-Repository/results/robustness/crps_dm_kstar3.csv",
        r"chmm_t_pen_vs_chmm_l,[-\d.]+,[-\d.]+,[-\d.]+,[-\d.]+,([\d.]+)", 3, "0.042", "sections/results.tex"),
    ("copula_stat_t_armmean", "../CHMM-Paper-Repository/results/robustness/rolling_copula_oos.csv",
        r"arm_means_complete,+6\.0,,([\d.]+)", 3, "0.264", "sections/cross_asset_appendix.tex"),
    ("copula_refit_t_mean", "../CHMM-Paper-Repository/results/robustness/rolling_copula_oos.csv",
        r"refit_effect_Student_t_static_rolling_,+mean=([\d.]+)", 3, "0.077", "sections/cross_asset_appendix.tex"),
    ("copula_refit_g_mean", "../CHMM-Paper-Repository/results/robustness/rolling_copula_oos.csv",
        r"refit_effect_Gaussian_static_rolling_,+mean=([\d.]+)", 3, "0.082", "sections/cross_asset_appendix.tex"),
    ("copula_family_roll_mean", "../CHMM-Paper-Repository/results/robustness/rolling_copula_oos.csv",
        r"family_effect_rolling_t_Gaussian_,+mean=([\d.]+)", 3, "0.007", "sections/cross_asset_appendix.tex"),
    ("wins_kurt_diff_lo", "results/diagnostics/kurtosis_bootstrap.txt",
        r"difference is \[(-?[\d.]+),", 2, "-1.26", "sections/sensitivity_appendix.tex"),
    ("wins_kurt_diff_hi", "results/diagnostics/kurtosis_bootstrap.txt",
        r"difference is \[-?[\d.]+, ([\d.]+)\]", 2, "1.83", "sections/sensitivity_appendix.tex"),
];

println();
println("="^96);
println("  [3] Keyed checks   (regex-pinned artifact value, rounded to paper precision)");
println("="^96);
for (name, art_rel, pat, digits, expect, tex_rel) in KEYED
    art_path = joinpath(_ROOT, art_rel);
    art_text = isfile(art_path) ? read(art_path, String) : "";
    m = match(pat, art_text);
    if m === nothing
        global n_fail += 1;
        println(rpad("FAIL", 5), rpad(name, 26), "artifact: ",
                isempty(art_text) ? "MISSING FILE" : "regex not matched");
        continue;
    end
    raw = parse(Float64, m.captures[1]);
    formatted = string(round(raw, digits=digits));
    # normalise trailing zeros to the expected string's precision
    want = parse(Float64, expect);
    art_ok = isapprox(round(raw, digits=digits), want; atol=10.0^(-digits)/2);
    tex_path = joinpath(PAPER_ROOT, tex_rel);
    tex_text = isfile(tex_path) ? read(tex_path, String) : "";
    tex_ok = occursin(expect, tex_text);
    status = art_ok && tex_ok ? "PASS" : "FAIL";
    status == "FAIL" && (global n_fail += 1);
    println(rpad(status, 5), rpad(name, 26), rpad("raw=$(m.captures[1])", 16),
            rpad("expect=$expect", 14),
            "artifact: ", rpad(art_ok ? "ok" : "rounds to $formatted", 22),
            "paper: ", tex_ok ? "ok" : "value absent");
end

println("-"^96);
println(n_fail == 0 ? "ALL CHECKS PASS" : "$n_fail CHECK(S) FAILED");
exit(n_fail == 0 ? 0 : 1);
