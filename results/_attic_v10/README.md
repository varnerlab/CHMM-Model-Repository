# Attic: v10 journal-revision result directories

This directory holds result subdirectories that were produced during the
v10 (journal-revision) phase of the paper but are not consumed by the
arXiv preprint.

The arXiv preprint sources its numbers from:
- `results/SPY/`, `results/cross_asset/`, `results/cross_asset_large/`,
  `results/diagnostics/`, `results/equity_price_sim/` (headline panels).
- `results/track_b1/` (QuantGAN row in the extended baseline panel).
- `results/track_b4/` (MS-GARCH rows in the extended baseline panel).
- `results/track_c1/` (SM-CHMM rows in the extended baseline panel).
- `../CHMM-paper/results/robustness/` (per-CSV outputs from the
  descriptive `run_*.jl` runners after the M-script rename pass).

What lives here:

| Subdir | Origin | Why archived |
|---|---|---|
| `track_a/` | Track A "extended evaluation" runner | Leverage / agg-kurtosis / joint-pv tables not cited by arXiv body or appendix |
| `track_b3/` | Track B3 diffusion baseline | Not in arXiv panel |
| `track_c3/` | Track C3 conditional-VaR variants | Conditional-VaR experiments deferred to a companion paper |
| `track_c4/` | Track C4 leverage-emission ablation | Not in arXiv panel |
| `track_m2/` ... `track_m12/`, `track_minor4/`, `track_minor6/`, `track_minor10/` | M-coded referee-comment runners | After the script-rename pass these directories hold only stale metadata; live outputs go to `CHMM-paper/results/robustness/` |

History: see `../../rename-revision-artefacts-plan.md` for the previous
M-script rename pass, and `../../../CHMM-paper/arxiv-prep-review.md` for
the arXiv-prep audit that flagged these as orphan.
