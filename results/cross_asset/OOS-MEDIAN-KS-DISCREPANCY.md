# Discrepancy: manuscript "Median OoS KS" column vs. raw per-asset data

**Status:** open manuscript fix, flagged 2026-07. Data in this repo is correct; the manuscript
summary table has one stale column. No code or data change is needed here, only a manuscript edit.

## Summary

The cross-asset summary table in the Paper II manuscript,
`tab:cross_asset_supp_summary` (`chapters/papers/paper-two/supplementary.tex`, table at lines
334-348), reports a **Median OoS KS** column that does not match the authoritative per-asset data in
this repo (`results/cross_asset/Table-T3-Cross-Asset-Dependence.txt`). The **Median IS KS** column
and **both off-diagonal MAE** columns reconcile exactly; only the OoS-median column is wrong.

Use the raw per-asset values in `Table-T3-Cross-Asset-Dependence.txt` (section 1) as authoritative.

## The two versions

Manuscript `tab:cross_asset_supp_summary` (supplementary.tex:342-345), "Median OoS KS" column:

| Model | Median OoS KS (manuscript) |
| --- | --- |
| Single Index Model | 87.0 |
| Gaussian copula | 87.5 |
| Student-t copula (nu*=6) | 85.8 |
| Truncated C-vine | 86.0 |

Raw per-asset OoS KS and their true medians, from
`Table-T3-Cross-Asset-Dependence.txt` section 1 (per-asset rows lines 33-38, Median row line 40):

| Model | Per-asset OoS KS (SPY, NVDA, JNJ, JPM, AAPL, QQQ) | True median |
| --- | --- | --- |
| Single Index Model | 78.5, 84.0, 98.0, 24.5, 86.0, 84.5 | **84.2** |
| Gaussian copula | 84.5, 61.5, 96.5, 64.0, 87.0, 84.0 | **84.2** |
| Student-t copula (nu*=6) | 77.5, 62.5, 94.0, 57.5, 86.0, 85.5 | **81.5** |
| Truncated C-vine | 77.5, 64.0, 93.5, 57.5, 84.5, 88.5 | **81.0** |

The medians in the raw dump's own Median row (line 40: SIM 84.2, Gaussian 84.2, Student-t 81.5,
C-vine 81.0) match a hand recomputation from the per-asset rows.

### Worked check (Student-t)

Sorted Student-t OoS values: {57.5, 62.5, 77.5, 85.5, 86.0, 94.0}. Median = mean of the 3rd and 4th
= (77.5 + 85.5) / 2 = **81.5**, not 85.8. Every model's manuscript OoS-median is off by 2 to 5
points in the same direction, consistent with the column having been produced by a different or
older computation (likely a different fold set, path seed, or a mean/median mix-up), not by
median-ing these six per-asset values.

## What DOES reconcile (no action needed)

- **Median IS KS** (supplementary.tex:342-345): 71.0 / 88.8 / 89.5 / 90.0 matches
  `Table-T3` section 1 Median-IS row exactly.
- **Off-diag MAE IS** and **Off-diag MAE OoS** (supplementary.tex:342-345): 0.077/0.029/0.027/0.068
  and 0.252/0.204/0.209/0.236 match `Table-T3` section 2 (lines 45-48) exactly. The
  per-asset per-model KS in `tab:cross_asset_p2` (Student-t column) also matches.
- Frobenius per model (`Table-T3` section 2, lines 45-48: SIM 0.531/1.762, Gaussian 0.196/1.472,
  Student-t 0.182/1.507, C-vine 0.463/1.662 IS/OoS) is computed and stored but not rendered in the
  manuscript.

## Recommendation

1. In the manuscript, either recompute the **Median OoS KS** column from the per-asset OoS values
   (SIM 84.2, Gaussian 84.2, Student-t 81.5, C-vine 81.0) or drop the median-KS columns and cite the
   per-asset detail in the cross-asset appendix. The table caption's qualitative ordering is
   unaffected (the dependence-quality claims rest on the MAE columns, which are correct).
2. The M-exam defense deck's cross-asset table (`figures/tables/tbl-p2-copula.png`, built by
   `M-exam-presentation/render_tables.py`) already uses the raw per-asset values from this repo, not
   the manuscript's OoS-median column, so no deck change is required.

## Sources

- Manuscript table: `chapters/papers/paper-two/supplementary.tex:334-348` (label
  `tab:cross_asset_supp_summary`).
- Authoritative data: `results/cross_asset/Table-T3-Cross-Asset-Dependence.txt` (per-asset KS section
  1, lines 30-40; correlation reproduction section 2, lines 42-48). K*=3, 200 paths, seed
  `20260422`.
