# Results Generation: Single-Index vs. Cross-Asset CHMM

## Purpose

Document which of the two CHMM pipelines was used to produce each result in the
study, flag the inconsistencies this created, and propose a single canonical
approach to carry through every section for the paper.

Main tickers in scope: **SPY, NVDA, JNJ, JPM, AAPL, QQQ**.

---

## Current state: two pipelines, both live

### Pipeline A. Single-index trained CHMM (per-ticker independent)

Each ticker's return series is fit with its own CHMM (K=18) independently. No
cross-asset coupling is introduced. Evaluation is purely marginal (per-asset KS,
AD, kurtosis, ACF-MAE, W1, Hellinger, coverage).

Scripts:

- `run_baselines_and_cross_asset.jl`
- `run_multi_emission_analysis.jl` (SPY-only K-sweep and family sweep)
- `run_equity_price_sim.jl`
- `run_full_rebuild.jl` (walk-forward + VaR/ES + block-bootstrap power + nu diagnostics, via `run_diagnostics.jl`)

Outputs currently used in the paper:

| Result                                                   | File                                                         | Scope                       |
| -------------------------------------------------------- | ------------------------------------------------------------ | --------------------------- |
| Table 2 (baselines vs. CHMM)                             | `results/SPY/Table-2-Baselines.txt`                          | SPY only                    |
| Table T1 (state-resolution sweep)                        | `results/SPY/Table-T1-State-Resolution-Sensitivity.txt`      | SPY only                    |
| Table T1 (multi-emission family sweep)                   | `results/SPY/Table-T1-Multi-Emission.txt`                    | SPY only                    |
| **Table T2 (per-ticker CHMM-N / t / L)**                 | `results/SPY/Table-T2-Per-Ticker-Emission-Families.txt`                       | 6 tickers, single-index     |
| Per-K / per-family figures (Convergence, IS, OoS, Trans) | `results/SPY/K*/...`, `results/SPY/multi_emission/K18/*/...` | SPY only                    |
| Price fans + terminal distributions                      | `results/equity_price_sim/Fig-{TICKER}-*`                    | 6 tickers, single-index     |
| VaR / ES back-test                                       | `results/diagnostics/utility/VaR_ES_Backtest.txt`            | SPY only                    |

### Pipeline B. Cross-asset extension (SIM + Gaussian copula + Student-t copula)

Marginals are still per-ticker CHMM (K=18, Gaussian emissions only), but
dependence across the six tickers is injected via (i) SIM with SPY as market
factor, (ii) Gaussian copula on CHMM marginals, (iii) Student-t copula with
profile-MLE ν* = 6.0. Evaluation adds cross-asset correlation reproduction on
top of per-asset KS.

Script: `run_cross_asset_sim_copula.jl`.

Outputs currently used in the paper:

| Result                                                 | File                                                     | Scope                   |
| ------------------------------------------------------ | -------------------------------------------------------- | ----------------------- |
| **Table T2 (SIM + Gaussian + t-copula)**               | `results/cross_asset/Table-T3-Cross-Asset-Dependence.txt`| 6 tickers, cross-asset  |
| Fig 6 (cross-asset correlation heatmap)                | `results/cross_asset/Fig-Cross-Asset-Correlation.*`      | 6 tickers, cross-asset  |
| Per-asset IS KS bar chart (SIM / Gauss / t-cop)        | `results/cross_asset/Fig-Cross-Asset-KS-Dist.*`          | 6 tickers, cross-asset  |
| Copula profile-MLE ν search                            | `results/diagnostics/copula_profile/*`                   | 6 tickers, cross-asset  |

### The collision

Two files are both tagged "Table T2" and both cover the same six tickers, yet
they evaluate different objects:

- `results/SPY/Table-T2-Per-Ticker-Emission-Families.txt` answers: *"Does the CHMM reproduce each
  ticker's marginal distribution, and does this conclusion survive switching the
  emission family (N / t / L)?"*
- `results/cross_asset/Table-T3-Cross-Asset-Dependence.txt` answers: *"Given
  per-ticker CHMM marginals, how well does each dependence mechanism (SIM vs.
  Gaussian copula vs. Student-t copula) reproduce the empirical cross-asset
  correlation structure?"*

The `Main-Body-Selection-Note.md` currently designates the SIM+copula version
as the main-body Table T2, which drops the three-emission-family per-ticker
result out of the main body entirely.

---

## Recommendation: single-index as the canonical pipeline, cross-asset as one self-contained extension section

### Summary

Make **single-index trained CHMM (Pipeline A) the canonical per-asset
evaluation pipeline for every section of the paper**. Reserve **cross-asset
(Pipeline B) for exactly one dedicated section and one table/figure pair**.
Rename to eliminate the "two Table T2" collision.

### Why this is the better academic choice

1. **The central claim is a marginal claim.** The paper's main contribution,
   per CLAUDE.md and the Main-Body-Selection-Note, is that a continuous HMM at
   small K reproduces the three stylized facts (heavy tails, negligible linear
   ACF, persistent volatility clustering in |G_t|) **without** jumps. These are
   per-asset properties. Evaluating them with a per-asset CHMM is the direct
   test; layering SIM or copula dependence on top only dilutes the evidence
   because a reviewer cannot tell whether a per-asset KS result was helped or
   hurt by the copula step.

2. **Separation of contributions is what reviewers reward.** The prior paper
   (arXiv:2603.10202) this work mirrors treats the single-asset model and the
   cross-asset extension as distinct sections. Keeping that separation here
   produces a cleaner narrative: Section N establishes CHMM as a marginal
   generator (single-index); Section N+1 extends it to a portfolio generator
   (cross-asset). Each section evaluates exactly one claim.

3. **Price fans, VaR/ES, walk-forward, and the emission-family robustness
   check are all inherently per-asset.** Forcing them through a cross-asset
   pipeline would require a defensible treatment of joint path draws, which is
   neither needed for the argument nor standard in this literature. Single-
   index is also the only choice that matches what `run_equity_price_sim.jl`
   and `run_full_rebuild.jl` already produce.

4. **Cross-asset is where the SIM/copula extension genuinely earns its space.**
   It adds a **different** metric (cross-asset correlation reproduction via
   Frobenius norm and off-diagonal MAE) that is invisible under any single-
   asset analysis. That is the right place, and the only right place, for
   Pipeline B.

5. **Avoids the "two Table T2" reviewer trap.** A Gemini / Claude review round
   already flagged confusion around this naming. One canonical Table T2, one
   Table T3, resolves it permanently.

### Proposed canonical table / figure layout

| Label     | Content                                                                      | Pipeline                        | Current source file                                            |
| --------- | ---------------------------------------------------------------------------- | ------------------------------- | -------------------------------------------------------------- |
| Table 1   | Descriptive stats (IS / OoS), six tickers                                    | Data only                       | (generate if not already present)                              |
| Table 2   | Seven-way SPY comparison (Bootstrap, Gaussian, Laplace, Disc NJ, Disc WJ, GARCH, CHMM-N) | A (SPY)          | `results/SPY/Table-2-Baselines.txt`                            |
| Table T1a | State-resolution sweep (K = 3..21), SPY, CHMM-N                              | A (SPY)                         | `results/SPY/Table-T1-State-Resolution-Sensitivity.txt`        |
| Table T1b | Emission-family sweep (CHMM-N / t / L × K) on SPY                            | A (SPY)                         | `results/SPY/Table-T1-Multi-Emission.txt`                      |
| **Table T2** | **Per-ticker marginal fidelity, six tickers × three emission families**  | **A (single-index, 6 tickers)** | `results/SPY/Table-T2-Per-Ticker-Emission-Families.txt`                         |
| **Table T3** | **Cross-asset dependence: SIM vs. Gaussian copula vs. Student-t copula (correlation reproduction + per-asset KS sanity check)** | **B (cross-asset, 6 tickers)** | `results/cross_asset/Table-T3-Cross-Asset-Dependence.txt`      |
| Fig 1     | Stylized facts (SPY)                                                         | A                               | existing SPY stylized-facts figures                            |
| Fig 2     | Baum-Welch convergence (SPY, K=18, CHMM-N)                                   | A                               | `results/SPY/K18/Fig-Convergence*` (or multi_emission/K18/N/)  |
| Fig 3     | IS comparison panel (SPY, K=18, CHMM-N)                                      | A                               | `results/SPY/K18/Fig-3-IS-Comparison*`                         |
| Fig 4     | OoS validation panel (SPY, K=18, CHMM-N)                                     | A                               | `results/SPY/K18/Fig-4-OoS-Validation*`                        |
| Fig 5     | Emission PDFs + transition heatmap (SPY, K=18, CHMM-N)                       | A                               | `results/SPY/multi_emission/K18/N/Fig-Emission-PDFs*`          |
| Fig 6     | Price fans (6 tickers × 3 families)                                          | A                               | `results/equity_price_sim/Fig-*-PriceFan-*`                    |
| **Fig 7** | **Cross-asset correlation heatmap (Observed / SIM / Gaussian / Student-t)**  | **B**                           | `results/cross_asset/Fig-Cross-Asset-Correlation*`             |
| Appendix  | VaR/ES back-test, walk-forward, block-bootstrap power, copula profile, nu diagnostics, GRU comparison, Ryden K=2 | Mixed (A + B) | `results/diagnostics/...`                                      |

### Renaming action items (safe, mechanical)

1. Copy `results/SPY/Table-T2-Per-Ticker-Emission-Families.txt` to `results/SPY/Table-T2-Per-Ticker-Emission-Families.txt` so the filename advertises its actual content. Keep the old name as a symlink or a one-line pointer file for backward compatibility until the paper is final.
2. Copy `results/cross_asset/Table-T3-Cross-Asset-Dependence.txt` to `results/cross_asset/Table-T3-Cross-Asset-Dependence.txt`.
3. Update `results/SPY/Main-Body-Selection-Note.md` Fig 6 + Table T2 references: Fig 6 becomes Fig 7 and points to Pipeline B; Table T2 points to Pipeline A; Table T3 is added for Pipeline B.
4. In the paper (`Paper_v*.tex`), rewrite the two section-opening sentences so Section "Per-ticker generalization" cites Table T2 (A) and Section "Cross-asset extension" cites Table T3 (B). Do not use em dashes.
5. Leave the underlying scripts unchanged so re-running them reproduces the same artifacts. Only rename outputs.

### What each main ticker looks like under the canonical plan

- **SPY**: appears in Table 2, Table T1a, Table T1b, Table T2 (A), Table T3 (B), Fig 1-5, Fig 6, Fig 7, all appendix diagnostics. Canonical training: single-index CHMM-N at K=18.
- **NVDA, JNJ, JPM, AAPL, QQQ**: appear in Table T2 (A), Table T3 (B), Fig 6 price fans, cross-asset correlation heatmap (Fig 7). Canonical training per ticker: single-index CHMM (all three families for Table T2; Gaussian only for Table T3 marginals because that is what `run_cross_asset_sim_copula.jl` uses, and the point of Table T3 is dependence, not family comparison).
- Walk-forward (JNJ, JPM) and VaR/ES (SPY) stay in the appendix unchanged.

### What to stop doing

- Stop treating `results/cross_asset/Table-T3-Cross-Asset-Dependence.txt` as the main-body Table T2. It is a cross-asset dependence table, not a per-asset fidelity table, and the two answer different questions. Move it to Table T3.
- Do not introduce cross-asset training into equity price fans, VaR/ES, walk-forward, K-sweep, or the emission-family sweep. None of those sections need joint dynamics; adding them would raise questions without strengthening the claims.
- Do not fit the SIM/copula on top of CHMM-t or CHMM-L marginals just for symmetry. The Student-t copula already carries the tail-dependence story; adding nine family-by-dependence combinations inflates the table with little marginal evidence.

---

## Open questions for the user

1. Should Table T3 also report a **tail-dependence coefficient** (λ_U, λ_L), given Student-t copula is already selected by profile MLE? This is a small addition to `run_cross_asset_sim_copula.jl` and makes the extension section more defensible against a reviewer who asks "why a copula rather than a multivariate Gaussian".
2. For Fig 6 (price fans), should the paper show all six tickers × three families (18 panels) or restrict to the three main tickers × three families (9 panels) to save pages? The current full set is 36 figures on disk.
3. Should the walk-forward section be expanded from two defensives (JNJ, JPM) to all six tickers before submission? Right now it reads as a spot-check rather than a systematic robustness result.

