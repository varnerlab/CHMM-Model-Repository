# ICAIF 2026 Revision Plan (Paper_v8-icaif → v9)

**Status:** Weak draft. Plan supersedes `claude-icaif-review.md`.
**Venue:** ICAIF 2026, ACM sigconf, 8-page hard limit, anonymous, no supplement.
**Current file:** `paper/Paper_v8-icaif.tex` (builds, 8 pages).
**Target file:** `paper/Paper_v9-icaif.tex` (fresh copy, not an edit-in-place).

---

## 1. Unified thesis for the revision

> A continuous HMM at $K = 18$ is a high-fidelity, easily-trained synthetic data generator for equity returns. It reproduces all three canonical stylized facts (heavy tails, negligible linear ACF, persistent volatility clustering) on SPY in-sample and five-ticker generalization, and beats every classical and deep-generative baseline in the literature (GARCH, discrete-HMM with Poisson jumps, i.i.d. bootstrap, GRU+Gaussian-head) on the joint KS+ACF fidelity frontier. On top of per-asset CHMM marginals, a Student-t copula ($\nu^* = 6$) reproduces cross-asset correlation better than a single-index baseline.

Every paragraph in the revision must serve this thesis. Reviewer asks that cannot be cut at 8 pages either (a) reinforce the thesis as stated or (b) get parked in the full-paper companion.

## 2. Scope decisions (what is IN / OUT of v9-icaif)

**Dropped from v9-icaif (per user direction):**
- Online / streaming CHMM (Capp'e-Moulines, online EM, mini-batch EM, KL-warmup, distillation, etc.).
- Rolling-window / walk-forward deployment as a *methodology*. The JPM walk-forward number is also dropped from the main text. OoS degradation on NVDA / JPM is acknowledged once as a stationarity scope note, nothing more.
- Section 4.6 "Online and rolling-window deployment" of the current Paper_v8-icaif.tex — deleted entirely.
- Any future-work mention of refit regimes.
- Derivative pricing, option pricing, IV calibration (already out of scope; keep the one-line Scope note).

**Kept / strengthened:**
- Pipeline A (univariate) + Pipeline B (cross-asset) separation.
- Three-emission ablation (CHMM-N, CHMM-t, CHMM-L) at identical $K = 18$.
- Head-to-head against discrete NJ / WJ, GARCH, bootstrap, Gaussian i.i.d.
- Cross-asset SIM vs Gaussian vs Student-t copula.

**Newly added to match abstract and close remaining reviewer holes:**
- GRU+Gaussian-head row in the Pipeline A comparison table. The current abstract promises "recurrent deep-generative baselines" but the body does not deliver — either add the row (preferred) or strike the phrase. We add the row; space is freed by deleting the online section.
- Novelty-positioning paragraph in Section 1 against the continuous-HMM-for-finance literature (Hamilton 1989, Bulla & Bulla 2006, Nystrup et al. 2017/2018).
- Confidence-interval column (Wilson-score at 95%) on every pass rate in Tables 1 and 3.
- Computational-cost sentence in Section 3 (one full EM fit at $K = 18$, $T = 2{,}516$ under 10 s CHMM-N / 30 s CHMM-t on commodity laptop).
- Per-state $\nu_k$ histogram reference (as a one-line inline statement, not a figure — 8 pages cannot afford a third figure).

## 3. Reviewer-item disposition

Mapping of each item from `claude-icaif-review.md` to what v9-icaif does.

| Review item | Action in v9-icaif |
|---|---|
| 4.1 Novelty positioning vs continuous-HMM finance lit | ADDRESS. Half-paragraph in Section 1. |
| 4.2 Confidence intervals on pass rates | ADDRESS. Wilson-score CI column in Tables 1 and 3. |
| 4.3 Six-ticker universe too small | ADDRESS BY SOFTENING, not by scaling. Language change: "on the six-ticker universe" throughout. Scaling to 30 tickers is parked for the full paper. |
| 4.4 ACF-MAE hides structure | ADDRESS. Add a small inset panel to Figure 1(d) overlaying observed vs simulated median ACF. Already have the image in `figs/` for the full paper. |
| 4.5 Computational cost absent | ADDRESS. One-sentence wall-clock note in Section 3 methods. |
| 4.6 Walk-forward as second headline | REVERSE COURSE. Drop walk-forward from main text entirely. Keep only a 1-line stationarity scope acknowledgement. |
| 4.7 "deep-generative baselines" underspecified | ADDRESS. Add GRU row in Table 1 with IS/OoS KS, AD, kurtosis, ACF-MAE. |
| 5 Online / streaming CHMM extension | DROP per user direction. |
| 6.1 Figure numbering | VERIFY. Visual check on compiled PDF. |
| 6.2 $T$ symbol overloaded | ADDRESS. Rename transition matrix to $\mathbf{A}$ in Eq 1 and downstream; keep $T$ for series length. |
| 6.3 Annualization factor surprise | ADDRESS. One sentence after Eq 2 stating variance scales by $252^2$. |
| 6.4 Golden-section bracket | ADDRESS. One sentence reporting that $\hat\nu_k$ histogram has the bulk at the upper bracket with two states at the lower edge. |
| 6.5 Missing GRU baseline | RESOLVED by 4.7 action. |

## 4. Section-by-section plan with page budget (8 pages total, two-column)

Target column-inches are approximate but add up to 8 pages.

### Section 1 — Introduction (0.75 page, currently 0.75 page)
- Para 1 (keep, tighten): three stylized facts; generator-for-downstream-use framing; Ryd'en 1998 limitation.
- Para 2 (**new, replaces 4.1 gap**): one paragraph that explicitly names Hamilton 1989, Bulla & Bulla 2006, Nystrup et al. 2017/2018 as continuous-emission HMM predecessors in finance, and states exactly what is new here: (i) per-state $\nu_k$ ECM, (ii) closed-form weighted-MAD Laplace M-step, (iii) head-to-head with a jump-augmented discrete baseline, (iv) Pipeline A / Pipeline B framing with cross-asset copula composition. Two sentences, no more.
- Para 3 (keep): Alswaidan et al. 2026 discrete HMM as the anchored baseline and what we remove (quantization, jump hyperparameters).
- Para 4 (new, 2 sentences): digital-twin framing — the generator's job is to produce paths that a downstream risk-or-scenario-consumer can treat as data.
- Para 5 (keep): Pipeline A / Pipeline B introduction.

### Section 2 — Methods (1.5 pages, currently 1.5 pages)
- 2.1 CHMM definition + three emission families. Rename transition matrix to $\mathbf{A}$ (reviewer 6.2). Keep Equations 1-3 with $\mathbf{A}$.
- 2.1 add one sentence after defining $G_t$: "Variance in these units is annualized-squared; the 7.68 observed excess kurtosis is unitless."
- 2.1 add one sentence after the M-step description: "One full EM fit at $K = 18$ on $T = 2{,}516$ runs in approximately 10 s (CHMM-N), 30 s (CHMM-t with golden-section $\nu_k$ search), and 11 s (CHMM-L) in Julia on a 2024-era laptop." (reviewer 4.5)
- 2.2 Cross-asset dependence (SIM, Gaussian / Student-t copula): keep verbatim.
- 2.3 Evaluation: keep but add one sentence that pass-rate Wilson-score half-widths at $n = 1{,}000$ are $\approx 1.4$ pp at $p = 0.95$ and $\approx 2.5$ pp at $p = 0.80$, so that Table 1 CIs can be read without recomputation.

### Section 3 — Empirical study (4 pages, currently 3.5 pages plus 0.5 page stripped from online)
- 3.1 Data (0.15 page, keep).
- 3.2 Model selection: $K = 18$ (0.2 page, keep).
- 3.3 SPY seven-model comparison (Pipeline A, Table 1, ~1.0 page):
  - Add GRU row. Numbers from full paper Table~2: IS KS 18.1%, OoS KS 52.5%, IS AD 13.7%, OoS AD 55.0%, Kurt 2.85, ACF-MAE 0.0518, $W_1$ 0.196.
  - Add Wilson-score CI column on IS KS.
  - Keep the single sentence about the GRU diagnostic: "the auto-regressive Gaussian head produces a near-Gaussian unconditional marginal even though the recurrence captures clustering."
  - Rename section "SPY model comparison" to drop "deep-generative" unless the row is in. Now it is, so the label stays.
- 3.4 Six-ticker generalization (Pipeline A, Table 2, 0.5 page). Soften language to "on the six-ticker universe" in one place. Delete the paragraph "The OoS gap on NVDA and JPM motivates the rolling-window deployment analysis of \S sec:online" — replace with "The OoS gap on NVDA and JPM under the 2024-2026 regime is discussed in the Scope paragraph."
- 3.5 Cross-asset dependence (Pipeline B, Table 3, 0.8 page). Add Wilson CIs to per-asset IS KS. Keep Figure 2 (correlation heatmap 4-panel) verbatim.
- **3.6 DELETED — "Online and rolling-window deployment."** All 0.5 page reclaimed.
- 3.6 (new, short, 0.2 page) Stationarity scope. One paragraph: "The 2024-2026 OoS KS drop on NVDA (55.8%) and JPM (49.8%) reflects a stationarity violation, not a model-class failure. The same six-ticker panel passes at $\geq 90\%$ on four tickers. A full treatment of this effect is deferred; the scope of the present paper is the generator itself under the fixed-IS protocol."

### Section 4 — Discussion and conclusion (~0.7 page, currently ~0.6 page)
- Resolving Ryd'en at moderate $K$: keep, one short paragraph.
- Pipeline B ordering (Student-t > Gaussian > SIM): keep, one short paragraph.
- Emission-family practical guide: two sentences. "CHMM-L is the cheapest, highest AD. CHMM-t has the smallest $W_1$/H. CHMM-N has the tightest ACF-MAE."
- Scope: option pricing out; six-ticker universe; stationarity.
- Ethics & data: one line.

### Figures (2 total — hard ceiling at 8 pages, acmart two-column)
- Figure 1 — SPY stylized-facts panel (a-d). Unchanged.
  - Add inset to panel (d): simulated median ACF overlay (CHMM-N) vs observed, lags 1-252. One extra line, answers reviewer 4.4.
- Figure 2 — Cross-asset correlation heatmap (4-panel: observed, SIM, Gaussian copula, Student-t copula). Unchanged.

### Tables (3 total)
- Table 1 — SPY model comparison. Current 8 rows (Bootstrap, Gaussian i.i.d., Discrete NJ, Discrete WJ, GARCH, CHMM-N, CHMM-t, CHMM-L) + **new GRU row** = 9 rows. Add Wilson CI column on IS KS.
- Table 2 — Per-ticker CHMM-N generalization. Unchanged (6 rows + median).
- Table 3 — Pipeline B cross-asset. Unchanged. Add Wilson CI on per-asset IS KS.

### References (trimmed)
- Drop Capp'e 2011, Capp'e-Moulines 2009, Delyon-Lavielle-Moulines 1999 (no longer cited after online section removed).
- Add Hamilton 1989, Bulla & Bulla 2006, Nystrup et al. 2017 and/or 2018 (used in novelty paragraph).
- Keep all others.

## 5. Abstract rewrite

Current abstract ends with: "outperforming GARCH(1,1), discrete-HMM, and recurrent deep-generative baselines across seven distributional and autocorrelation metrics."
After adding GRU row, this sentence becomes accurate. Keep verbatim.

Replace the single-sentence copula paragraph if needed so the 250-word cap is respected.

## 6. Edit sequence (suggested)

1. Copy `Paper_v8-icaif.tex` → `Paper_v9-icaif.tex`; copy `References_v8-icaif.bib` → `References_v9-icaif.bib`.
2. Delete Section 4.6 (online deployment) and its references from `References_v9-icaif.bib` (Capp'e, Delyon).
3. Add Hamilton, Bulla-Bulla, Nystrup to bib.
4. Rewrite Section 1 with new Para 2 (novelty positioning) and new Para 4 (digital-twin framing).
5. Rename transition matrix $T \to \mathbf{A}$ globally.
6. Add computational-cost sentence to Section 2.1.
7. Add Wilson-CI sentence to Section 2.3.
8. Add GRU row to Table 1. Ensure abstract sentence is supported.
9. Add Wilson-CI column to Table 1 and Table 3 IS KS columns.
10. Soften "six tickers is sufficient" wording to "on the six-ticker universe."
11. Strip the walk-forward paragraph from Section 3.4; replace with one-line stationarity scope.
12. Add inset to Figure 1 panel (d) showing simulated-median ACF.
13. Compile. Verify 8 pages. Adjust caption lengths if overfull.
14. Strip `Paper_v8-icaif*` build artifacts once v9 builds cleanly.

## 7. Explicit non-goals for v9-icaif

- No online / streaming / rolling / walk-forward material.
- No option pricing / IV surface.
- No 30-ticker scaling — deferred to full paper companion.
- No vine / factor copula — deferred to full paper companion.
- No additional emission families (skew-t, $\alpha$-stable) — deferred to full paper companion.
- No Bayesian / variational treatment.

## 8. Success criteria

- Builds cleanly at exactly 8 pages in sigconf anonymous.
- Every claim in the abstract is supported by a number in the body.
- Every pass rate in Tables 1 and 3 carries a Wilson-score 95% CI half-width (either inline or as a column).
- Every reviewer item from `claude-icaif-review.md` is either ADDRESSED or explicitly listed as Scope / Limitation.
- Thesis readable in one sentence after the first paragraph of Section 1.
