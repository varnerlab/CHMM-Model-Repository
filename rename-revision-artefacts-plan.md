# CHMM-Model rename plan (sister to CHMM-paper revision-artefact rename)

Goal: align this repo with the CHMM-paper rename so that re-running any
referee-revision analysis writes to the new descriptive paths instead of the
deprecated `results/revision/M*_*.csv` layout, and rename the M-coded
scripts themselves to match the descriptive style.

## Scope

- **Layer A (essential).** Update each of the 12 scripts that write CSVs into
  `CHMM-paper/results/revision/`. Without this, re-running any analysis either
  fails (target dir gone) or re-creates the old layout and re-leaks the
  M-codes the paper just scrubbed.
- **Layer B (consistency).** Rename the 11 M-coded / minor-coded script files
  themselves to descriptive names. `run_track_c3_filter_var.jl` is **out of
  scope** for Layer B — its `c3` is an existing Track C convention, not an
  M-code. Layer A still applies to it (it writes `M3_filter_var_backtest.csv`).

## CSV-write mapping (Layer A)

The constant in each script is `PAPER_REVISION_DIR`. It needs to become
`PAPER_ROBUSTNESS_DIR` and resolve to `.../CHMM-paper/results/robustness/`.
Each script also has a header comment naming its example output path and
1–2 `open(joinpath(PAPER_REVISION_DIR, "<old>.csv"))` call sites.

| Script | Old CSV name | New CSV name |
|---|---|---|
| run_track_c3_filter_var.jl | M3_filter_var_backtest.csv | filter_var_backtest.csv |
| run_track_m2_ks_bootstrap.jl | M2_ks_bootstrap.csv | ks_block_bootstrap.csv |
| run_track_m4_rolling_and_weekly.jl | M4_rolling_origin.csv, M4_weekly.csv | rolling_origin_oos.csv, weekly_frequency.csv |
| run_track_m5_lr_ind_null.jl | M5_lr_ind_null.csv | lr_ind_bootstrap_null.csv |
| run_track_m6_var_ci.jl | M6_var_ci.csv | kupiec_mc_ci.csv |
| run_track_m7_garch_suite.jl | M7_garch_suite.csv | garch_suite.csv |
| run_track_m8_k_selection.jl | M8_k_selection.csv | k_selection_validation.csv |
| run_track_m9_skew_emissions.jl | M9_skew_emissions.csv | skew_emissions_ablation.csv |
| run_track_m10_nu_shrinkage.jl | M10_nu_shrinkage.csv | nu_shrinkage_sweep.csv |
| run_track_minor4_mmd_bandwidth.jl | minor4_mmd_bandwidth.csv | mmd_fixed_bandwidth.csv |
| run_track_minor6_kdisc13_centroid.jl | minor6_kdisc13_centroid.csv | kdisc13_centroid_ablation.csv |
| run_track_minor10_multiseed.jl | minor10_multiseed.csv | multiseed_headline.csv |

## Script-file rename mapping (Layer B)

| Old script | New script |
|---|---|
| run_track_m2_ks_bootstrap.jl | run_ks_block_bootstrap.jl |
| run_track_m4_rolling_and_weekly.jl | run_rolling_and_weekly.jl |
| run_track_m5_lr_ind_null.jl | run_lr_ind_bootstrap_null.jl |
| run_track_m6_var_ci.jl | run_kupiec_mc_ci.jl |
| run_track_m7_garch_suite.jl | run_garch_suite.jl |
| run_track_m8_k_selection.jl | run_k_selection_validation.jl |
| run_track_m9_skew_emissions.jl | run_skew_emissions_ablation.jl |
| run_track_m10_nu_shrinkage.jl | run_nu_shrinkage_sweep.jl |
| run_track_minor4_mmd_bandwidth.jl | run_mmd_fixed_bandwidth.jl |
| run_track_minor6_kdisc13_centroid.jl | run_kdisc13_centroid_ablation.jl |
| run_track_minor10_multiseed.jl | run_multiseed_headline.jl |

## Cross-repo follow-up

The CHMM-paper devnotes (`revision-code-todo.md`, `revision-plan-JoFE.md`)
mention the old script names. After Layer B those references become stale —
need a small additional pass to update them too.

## Execution steps

1. **Write this plan** — done.
2. **Final pre-edit grep** — confirm zero external references to the M-coded
   scripts/CSVs anywhere in CHMM-Model (CLAUDE.md, README.md, SPECIFICATION.md,
   user-comments.md, planning/, docs/, src/, test/, Notebooks/, other run_*.jl).
3. **Layer A** — for each of the 12 scripts, edit:
   - header-comment example output path
   - `const PAPER_REVISION_DIR = ... "revision" ...` → `const PAPER_ROBUSTNESS_DIR = ... "robustness" ...`
   - `mkpath(PAPER_REVISION_DIR)` → `mkpath(PAPER_ROBUSTNESS_DIR)`
   - all `joinpath(PAPER_REVISION_DIR, "M*.csv" or "minor*.csv")` → new descriptive name
   - any `println` referencing the path
4. **Layer B** — `git mv` 11 script files; update each script's header comment
   that names itself.
5. **Cross-repo follow-up** — update CHMM-paper devnotes (`revision-code-todo.md`,
   `revision-plan-JoFE.md`) to reference new script names.
6. **Final grep verification** — confirm zero references to old script names
   or `results/revision` paths anywhere in CHMM-Model or CHMM-paper.

## Status

- [x] Step 1 — Plan written.
- [x] Step 2 — Pre-edit grep: only the 12 scripts themselves contain old refs; no README, dispatcher, src/, test/, Notebooks/, or docs/ cross-references. Layer B rename is fully insulated.
- [x] Step 3 — Layer A: all 12 scripts now write to `results/robustness/<descriptive>.csv`. Constant renamed `PAPER_REVISION_DIR` → `PAPER_ROBUSTNESS_DIR`. Verified by grep. Self-naming `# run_track_minor*` comments at line 2 of three scripts will be handled when those files are renamed in Layer B.
- [x] Step 4 — Layer B: 11 scripts `git mv`'d to descriptive names; line-2 self-naming comments updated; one stray forward reference (`run_track_m4_indices.jl` planned script in run_rolling_and_weekly.jl L18) renamed to `run_indices_panel.jl` for consistency.
- [x] Step 5 — 22 script-name refs in CHMM-paper devnotes updated (11 names × 2 files, via `replace_all`).
- [x] Step 6 — Final cross-repo grep: zero matches in CHMM-Model and zero in CHMM-paper for old script names, `PAPER_REVISION_DIR`, or `results/revision/` paths.

## Deviations / notes

- Layer A also caught a stray comment at L277 of `run_track_c3_filter_var.jl` (now `run_track_c3_filter_var.jl` — unchanged in Layer B because `c3` is Track C, not an M-code) referencing the old `results/revision/` directory; updated to `results/robustness/`.
- Layer B caught a forward reference to a planned-but-undelivered script `run_track_m4_indices.jl` inside `run_rolling_and_weekly.jl`; renamed to `run_indices_panel.jl` to keep the future name aligned with the new convention.
- "Track M2 / Track M3 / ..." descriptive prose at line 4 of each script header preserved as historical documentation, mirroring the treatment of M-code prose in CHMM-paper devnotes. Only the line-2 self-naming filename comments were updated.
- `run_track_c3_filter_var.jl` deliberately not renamed in Layer B: `c3` is part of the existing Track C convention (alongside `run_track_c1_smchmm.jl`, `run_track_c2_large_universe.jl`, `run_track_c3_conditional_var.jl`, etc.), not an M-code.
