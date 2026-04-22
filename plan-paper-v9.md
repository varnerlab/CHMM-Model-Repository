# Full-Paper Revision Plan (Paper_v8 → v9)

**Current file:** `paper/Paper_v8.tex` (59 pages, non-ACM, three authors Alswaidan/Jin/Varner).
**Target file:** `paper/Paper_v9.tex` (same class, same author block, same journal-length format).
**Purpose:** Unify the paper around a single, defensible framing.

---

## 1. Unified thesis for v9

> The continuous HMM is a high-fidelity digital twin for equity returns: easy to train, requires no custom hyperparameters, and reproduces the stylized facts of equity returns while beating classical and deep-generative benchmarks on the joint distributional + temporal fidelity frontier. The same EM scaffold extends to three emission families (Gaussian, Student-t, Laplace) and composes cleanly with copula dependence for multi-asset synthesis. Two downstream consumers (VaR / ES backtest and multi-horizon price simulation) verify that the fidelity on the return series translates into calibrated risk numbers and calibrated price fans.

Three framing nouns the paper must hit consistently:
1. **Digital twin** — the thing we build.
2. **High fidelity** — proven on seven metrics on returns + on VaR/ES on the risk side + on $90\%$ envelope coverage on the price side.
3. **Practical to train** — 20-40 EM iterations, 10-30 s at $K = 18$, one closed-form M-step (CHMM-N, CHMM-L) or one golden-section per state (CHMM-t), no jump hyperparameters, no grid search.

Every section must reinforce at least one of those three. Sections that cannot should be cut.

## 2. Scope decisions (IN / OUT)

**IN:**
- Three-emission-family CHMM (N, t, L) at $K = 18$ on SPY + 5 additional tickers.
- Seven-metric evaluation panel.
- Head-to-head against: bootstrap, block-bootstrap, Gaussian i.i.d., Laplace i.i.d., discrete NJ, discrete WJ, Bin-T NJ, GARCH(1,1), GRU+Gaussian-head.
- Cross-asset extension: SIM + Gaussian copula + Student-t copula on six tickers.
- VaR / ES back-test as utility diagnostic for the univariate digital twin.
- Price-fan simulation (terminal + path-level CRPS, $\overline{C}_{0.9}$, MAPE) as utility diagnostic.
- State-resolution sensitivity $K \in \{3, 6, 9, 12, 15, 18, 21\}$.
- Ryd'en replication at $K = 2$ (quantile vs random init) to show the Ryd'en limitation is a distributional low-$K$ artefact.
- Bin-T NJ (discrete HMM with bin-conditional Student-t emissions) to isolate quantization-vs-emission effects.
- $\nu_k$ bracket-sensitivity study for CHMM-t.

**OUT (per user direction):**
- Online EM / streaming CHMM / Capp'e-Moulines.
- Rolling-window / walk-forward as a *methodology*. JPM walk-forward stays as a one-paragraph stationarity check, not a headline.
- Derivative / option pricing, implied-volatility surface.
- Generalized hyperbolic / $\alpha$-stable / skew-t / skew-Laplace emissions (mentioned only in future work).
- Full-universe scaling via vine / factor copulas (mentioned only in future work).
- Bayesian / variational CHMM.

## 3. Structural changes vs v8

v8 works section-by-section but the story drifts: it opens as "continuous HMM for equity returns," pivots to "emission-family ablation," pivots again to "cross-asset," adds "VaR/ES utility," then adds "price-fan simulation," and ends with a walk-forward recovery note on JPM. v9 reorders the narrative so the reader hits the same evidence in the order the thesis requires.

**v9 section skeleton (same chapters as v8, different emphasis inside each):**

1. **Introduction** — Digital-twin framing first. State the concrete claim in one sentence. Position against: (a) Ryd'en-style low-$K$ Gaussian HMMs, (b) continuous-emission finance HMMs (Hamilton 1989, Bulla & Bulla 2006, Nystrup et al. 2017/2018, Rossi & Gallo 2006), (c) discrete-HMM-with-jumps of Alswaidan et al. 2026, (d) deep-generative baselines (TimeGAN, Quant GAN) at an elevation that acknowledges them without pretending the GRU here is a frontier architecture. End with the Pipeline A / Pipeline B orientation and the scope list.
2. **Related work** — Organize around the three competing paradigms the thesis beats: (i) single-regime parametric (GARCH), (ii) discrete HMMs with jump augmentation, (iii) deep-generative. For each, name the representative papers and the reported metric they fail on. Keep HSMM and continuous-emission HMM as a fourth bucket that the present paper extends rather than rivals.
3. **Methods** — CHMM (three families, shared scaffold, quantile init, log-space FB); cross-asset (SIM, Gaussian / Student-t copula); benchmarks (bootstrap, block bootstrap, Gaussian / Laplace i.i.d., discrete NJ / WJ / Bin-T NJ, GARCH, GRU). Evaluation protocol with seven metrics + VaR/ES utility + price-level utility. Add explicit computational-cost table (per-family EM wall-clock at $K = 18$).
4. **Empirical study** — Reorganized into three narrative blocks aligned to the thesis:
   - **Block A: Univariate fidelity on SPY (stylized facts)** — descriptive stats → model selection ($K=18$) → twelve-model comparison (Table 2) → $K$-sensitivity → IS/OoS fidelity at $K = 18$. End with the one-row decision guide (Table: variant choice).
   - **Block B: Cross-asset digital twin** — Pipeline A ticker generalization → Pipeline B SIM vs Gaussian copula vs Student-t copula → cross-asset correlation reproduction.
   - **Block C: Downstream utility of the digital twin** — VaR / ES bracket check → multi-horizon price-fan coverage and CRPS → interpretation. Both demonstrate that fidelity on returns translates into calibrated downstream numbers.
5. **Discussion** — Ryd'en resolution; why jumps are unnecessary; temporal-distributional tradeoff; why copulas beat SIM; emission-family ablation takeaways. Keep the $\nu_k$ overshoot diagnostic. **Remove** the walk-forward discussion from here; move a 1-paragraph stationarity note to Limitations.
6. **Conclusion** — Three-sentence thesis restatement + three-bullet future work (skew-heavy-tailed emissions, full-universe vine/factor copulas, principled $K$ selection by held-out LL). Remove the online / Bayesian / streaming bullets.

## 4. Section-by-section edit list

### 4.1 Introduction (`sections/introduction_v9.tex`)
- Opening: stylized facts → digital-twin framing. New first sentence: "A synthetic generator of equity returns is useful as a digital twin only if it reproduces the three canonical stylized facts simultaneously."
- Ryd'en limitation as the motivating gap in paragraph 2.
- Paragraph 3 (new): position against the continuous-HMM finance literature (Hamilton 1989, Bulla & Bulla 2006, Nystrup et al. 2017/2018). State what is new here: (i) per-state Student-t $\nu_k$ ECM; (ii) weighted-MAD Laplace M-step; (iii) three-family ablation; (iv) Pipeline A / Pipeline B + copula composition. This paragraph is the single biggest reviewer-gap fix from the ICAIF peer review.
- Paragraph 4: discrete-HMM-with-jumps baseline anchor.
- Paragraph 5: Pipeline A / Pipeline B orientation.
- Paragraph 6: scope — generators and generators-for-downstream-risk only; option pricing explicitly out.
- Strip sentence: "A quarterly walk-forward re-estimation on JPM recovers about 15 percentage points of OoS KS..." — moved to Limitations.

### 4.2 Related Work (`sections/related_v9.tex`)
- Reorganize around the three paradigms (single-regime, discrete-HMM-with-jumps, deep generative) + continuous-emission finance HMMs.
- Add Hamilton 1989 and Nystrup 2017/2018 explicitly.
- Shorten TimeGAN / Quant GAN discussion; keep the variance-collapse failure reference.

### 4.3 Methods (`sections/methods_v9.tex`)
- Keep 3.1 (data + $G_t$), 3.2 (CHMM), 3.3 (benchmarks), 3.4 (cross-asset), 3.5 (metrics) structure.
- Add 3.2 computational-cost micro-table: per-family EM wall-clock, convergence iteration count, and memory footprint at $K = 18$, $T = 2{,}516$.
- Add one sentence in 3.5 stating Wilson-score pass-rate CI half-widths at the paper's simulation budget ($n = 1{,}000$ paths).
- Strip every forward reference to "online EM" / "rolling window."
- Rename the transition matrix symbol from $T$ to $\mathbf{A}$ to disambiguate from series length $T$.

### 4.4 Empirical study (`sections/results_v9.tex`)
- Keep every subsection that currently lives in v8 *except* subsection 5.7 (walk-forward result on JPM / JNJ). Move the JPM walk-forward recovery into Discussion→Limitations as a one-paragraph stationarity diagnostic, not a headline.
- Reorganize the subsection order (no rewriting of table content needed):
  1. Descriptive statistics and stylized facts
  2. Model selection ($K = 18$)
  3. Twelve-model SPY comparison (Pipeline A)
  4. $K$-sensitivity
  5. Out-of-sample evaluation at $K = 18$
  6. Cross-asset univariate generalization (Pipeline A)
  7. Cross-asset dependence (Pipeline B)
  8. Utility — VaR / ES
  9. Utility — OoS price simulation
- In subsection 3 add Wilson-score CI column on IS KS for Table 2.
- In subsection 7 add Wilson-score CI on per-asset IS KS for Table T3.
- In every subsection, add one sentence that restates the thesis noun: "This confirms the univariate digital twin is [fidelity claim]," "This is the cross-asset digital twin composition," etc.

### 4.5 Discussion (`sections/discussion_v9.tex`)
- Keep: Ryd'en resolution, Bin-T NJ interpretation, $\nu_k$ overshoot diagnostic, why-jumps-unnecessary.
- Replace the "Cross-Asset Robustness and OoS Degradation on NVDA and JPM" paragraph: keep the OoS degradation observation but drop the rolling-window lead. Replace with a limitation noting that the CHMM is fit once under the stationarity assumption and that non-stationary periods (2024-2026 for NVDA, JPM) degrade OoS KS; the walk-forward one-liner number becomes a 1-sentence diagnostic (not a methodology): "A single-quarter refit on JPM recovers 15 pp of OoS KS, supporting the stationarity explanation; a full treatment of time-varying dynamics is left to future work."
- Strip: all future-work bullets on rolling windows, Bayesian online, Capp'e-Moulines.
- Limitations list: condense to five items — residual tail gap, within-regime symmetry, stationarity, small cross-asset universe, IC-vs-held-out model selection.

### 4.6 Conclusion (`sections/conclusion_v9.tex`)
- Three-sentence thesis restatement: digital twin, fidelity numbers (one line), comparison result (one line).
- Future work: skew-heavy-tailed emissions; full-universe vine / factor copulas; principled $K$ via held-out log-likelihood.
- Remove: regime-aware dynamic asset allocation bullet (stretches the paper's stated scope).
- Keep: Data & Code Availability block (seeds, repo URLs). Keep CoI and author contributions as-is.

### 4.7 Supplemental (`sections/supplemental_v9.tex`)
- Keep all appendices that defend the thesis: Baum-Welch recursion, metric definitions, $K$-sweep sensitivity, $\nu_k$ diagnostics, KS power calibration, per-ticker tails, Ryd'en $K = 2$ replication, per-ticker price fans.
- Keep the JPM walk-forward table as an appendix stationarity diagnostic, but demote it from subsection to single paragraph.
- Drop any appendix discussion of online EM or Bayesian updates.

## 5. Figures and tables (total count and purpose)

**Main-body figures (keep at 10, down from ~15):**
1. Stylized-facts four-panel on SPY IS.
2. Fitted CHMM internals (emission densities + transition heatmap).
3. IS comparison figure at $K = 18$.
4. OoS validation figure at $K = 18$.
5. Cross-asset correlation heatmap (4-panel).
6. Cross-asset IS KS grouped bar chart.
7. VaR / ES calibration error bars.
8. SPY OoS price fan (keep all 3 families since the fan is the direct digital-twin visual).
9. NVDA OoS price fan (same three-family panel).
10. SPY + NVDA terminal-price histograms.

**Drop / move to supplement:** the five-panel price fan set for JNJ, JPM, AAPL, QQQ.

**Main-body tables (keep at 6):**
1. Descriptive stats.
2. Twelve-model SPY comparison — add GRU row Wilson CI column.
3. Per-ticker generalization (three families × six tickers).
4. Cross-asset dependence (SIM vs Gaussian copula vs Student-t copula).
5. OoS terminal-price coverage.
6. Path-level price metrics ($\overline{C}_{0.9}$, MAPE, CRPS).

Variant-choice guide remains as a tiny table inside Section 3.

## 6. Tone and language changes

- Every appearance of "online," "streaming," "rolling," "refit," or "walk-forward" in the main text must be reviewed. They survive only in (a) the one-sentence Limitations note on stationarity and (b) the appendix walk-forward diagnostic paragraph.
- Every appearance of "option pricing," "derivative," "IV surface" stays out of the main text except in the explicit Scope / out-of-scope sentence.
- Every subsection should name its evidence back to the digital-twin thesis in one sentence. Example: "The three emission families share the same scaffold, so the practitioner gets a tunable digital twin from a single codebase."

## 7. Edit sequence

1. Copy the `sections/*_v8.tex` files to `sections/*_v9.tex` and switch `Paper_v9.tex` to include the v9 section files.
2. Copy `References_v8.bib` → `References_v9.bib`; remove unused entries (Capp'e, Delyon, etc.); add Hamilton 1989, Bulla & Bulla 2006, Nystrup 2017/2018 if missing.
3. Introduction rewrite first — gates every downstream section.
4. Related work rewrite.
5. Methods: add computational-cost table; rename $T \to \mathbf{A}$; add Wilson-CI sentence.
6. Results: reorder subsections; add Wilson-CI columns; delete the walk-forward subsection; add thesis-hook sentences.
7. Discussion: rewrite stationarity paragraph; strip future-work bullets on online/Bayesian; condense Limitations.
8. Conclusion rewrite.
9. Supplemental: demote walk-forward to paragraph; drop online EM references.
10. Compile. Target page count: 35-45 pages (was 59 in v8). Bulk of savings comes from (i) tightened methods, (ii) reduced figure count in results, (iii) the walk-forward demotion.

## 8. Success criteria

- Thesis restated in three nouns (digital twin / high fidelity / easy to train) in Section 1 paragraph 1, Section 4 opening sentence, and Conclusion paragraph 1.
- Every empirical subsection names which of the three nouns it defends.
- Total page count: 35-45 journal pages, down from 59.
- No main-text mention of online EM, streaming, rolling-window-as-method, or options pricing.
- Every table carries the CI column where applicable.
- Compiles cleanly. Tables and figures are in one-to-one correspondence with the figure list in Section 5 above.

## 9. Non-goals

- Not redesigning the evaluation suite.
- Not re-running experiments. All tables and numbers already exist from v8.
- Not re-generating figures from scratch. The few edits (Wilson-CI columns, reordered panels, dropped subfigures) are caption/table edits.
- Not changing any hyperparameter, seed, or simulation budget.
