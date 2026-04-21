# Reviewer Assessment: Paper V6

**Journal / Special Issue.** ACM Journal of Data and Information Quality (JDIQ) — Special Issue on *Quality of Synthetic Data* (submission deadline 2026-03-01; technical papers up to 23 pages).

**Paper title.** *Continuous Hidden Markov Models for Equity and Volatility Index Dynamics: Gaussian, Student-t, and Laplace Emissions Trained by EM, with Cross-Asset SIM and Copula Extensions.*

**Authors.** A. Alswaidan, C. Jin, J.D. Varner (Cornell University).

**Submission format.** Technical paper, 43 pages (A4, 11pt), main body ~22 pages, appendix ~20 pages. Within page-cap policy if the appendix is treated as supplementary material; borderline if page count is strict for the main article. See §12.

---

## 1. Summary

The paper introduces a family of continuous hidden Markov models (CHMM-N, CHMM-t, CHMM-L) for generating synthetic financial time series. All three variants share a log-space forward-backward scaffold and quantile-based initialization; the M-step is the classical Baum-Welch update for Gaussian emissions, an ECM-plus-golden-section sequence for Student-t emissions, and closed-form weighted-median / weighted-MAD estimators for Laplace emissions. The authors argue (i) that a moderate state resolution ($K=18$) is sufficient to reproduce the three Cont (2001) stylized facts without jump augmentation, contradicting Rydén et al. (1998); (ii) that heavy-tailed emissions close the residual kurtosis gap of the Gaussian CHMM; and (iii) that the CHMM marginals compose cleanly with Gaussian / Student-t copulas for multi-asset synthesis, outperforming a Single Index Model baseline. Results are reported on ten years of SPY plus NVDA, JNJ, JPM, AAPL, QQQ, with a VIX/VXX extension.

The evaluation uses a seven-metric panel (KS and AD pass rates, excess kurtosis, ACF-MAE on $|G_t|$, Wasserstein-1, Hellinger, quantile coverage) evaluated per path over 1,000 simulated paths. A direct head-to-head against the prior discrete-HMM work of Alswaidan et al. (2026) and a GARCH(1,1) benchmark is included. The appendix contains detailed algorithmic pseudocode, full state-resolution sensitivity tables across all three emission families, and derivations for the SIM and copula constructions.

## 2. Fit to the Special Issue

The special issue explicitly welcomes "innovative quality metrics, relevant case studies, and novel SDG and DA methods that include quality guarantees by design," with emphasis on the **fidelity / utility / privacy** triad and on bridging theoretical quality measures and practical deployment. This paper fits that scope well on two of the three dimensions:

- **Fidelity.** The seven-metric panel covers univariate similarity (KS, AD, $W_1$, Hellinger, quantile coverage), tail behaviour (excess kurtosis), and longitudinal similarity (ACF-MAE over 252 lags). The cross-asset extension adds bivariate and multivariate fidelity (Pearson correlation reproduction, Frobenius distance, off-diagonal MAE). This is a well-motivated panel that maps cleanly onto the special issue's "univariate / bivariate / multivariate / longitudinal similarity" taxonomy.
- **Utility.** Partially addressed: the paper positions the generator as a backtesting/risk-management artefact but does not run a downstream utility benchmark (e.g. VaR or ES back-test, portfolio optimization TRTS, or trading-strategy performance). This is the main scope gap for JDIQ — see §4.
- **Privacy.** Not applicable by construction: SPY/VIX/VXX prices are public, so privacy is out of scope. The authors should state this explicitly in the Scope paragraph to align with the special issue's framing.

**Verdict on fit.** In scope. The SDG + quality-metric-panel framing is a clean match. The reviewer recommends a brief Scope paragraph that (i) acknowledges the JDIQ fidelity/utility/privacy triad, (ii) states that privacy is out-of-scope for public market data, and (iii) positions the seven-metric panel as the fidelity contribution.

## 3. Strengths

1. **Methodological completeness.** The three emission families share the same EM scaffold, making the comparison internally consistent. The ECM + golden-section search for $\nu_k$ (CHMM-t) and the weighted-median M-step (CHMM-L) are correctly derived and are non-trivial refinements of the classical Baum-Welch algorithm. Full pseudocode for all three (Algorithms 1, 1b, 1c in the appendix) is an asset.
2. **Evaluation rigour.** The seven-metric panel is well chosen: KS and AD cover the centre and tails of the marginal, $W_1$ and Hellinger are quantile-weighted distances, excess kurtosis diagnoses the Gaussian-mixture bound, ACF-MAE on $|G_t|$ is the direct Rydén-et-al. diagnostic, and quantile coverage reports envelope calibration. The paper reports every metric on both IS (2014–2024, $T=2{,}766$) and OoS (2025, $T=219$) windows and averages over 1,000 paths per model, which is appropriate for the scale of the problem.
3. **Head-to-head with the discrete baseline.** The paper reproduces the prior paper's discrete HMM (NJ and WJ) inside the same evaluation pipeline, making the "continuous emissions vs. quantized emissions" contrast quantitative and direct. The 95.9-percentage-point swing in IS KS pass rate (0% $\to$ 95.9%) is the cleanest demonstration of why quantisation is costly for distributional fidelity, and is a useful empirical lesson beyond this specific paper.
4. **Ablation across emission families.** Table~\ref{tab:sensitivity_multi} reports the full $K \in \{3,6,9,12,15,18,21\}$ sweep for all three emission families. This is a 21-cell ablation table, and it shows convincingly that the Rydén failure is not an artefact of emission choice and that the plateau in distributional pass rates ($K \in \{15,18,21\}$) is stable across families.
5. **Cross-asset extension with honest limitations.** The SIM-versus-copula comparison is clean: SIM distorts non-market marginals by construction (JPM IS KS collapses to 36.0% under SIM vs. 99.5% under the Student-t copula), while rank-reordering copulas preserve marginals exactly. The Student-t copula's profile-MLE selection of $\nu^*=6$ is a meaningful quantitative finding.
6. **Reproducibility.** Code repository is cited (`https://github.com/altashly1/CHMM-Model`), with algorithmic pseudocode, hyperparameters ($\texttt{tol}=10^{-4}$, $\texttt{max\_iter}=60$), and data provenance (SPDR SPY daily 2014-01-03 to 2024-12-31) explicitly documented. This meets JDIQ's practical-deployment criterion.
7. **Writing and organization.** The three-layer structure (main body → appendix methods → appendix figures/tables) is appropriate, and cross-referencing between the body and the supplemental is disciplined. The figures are clean and accurate.

## 4. Major Concerns

### 4.1 Downstream utility evaluation is absent

The JDIQ special issue explicitly distinguishes "fidelity" (distributional similarity) from "utility" (performance on a specific downstream task). The paper covers fidelity comprehensively but not utility. Concretely, a synthetic-data generator for financial returns should demonstrate *at least one* of:

- **Risk-metric pass-through.** Compute 1% / 5% VaR and ES on simulated paths and compare to historical VaR/ES on the observed series. Are the backtest-on-simulation risk numbers calibrated to the backtest-on-observed numbers? The existing quantile-coverage metric gets halfway there but does not report a VaR/ES figure.
- **Train-synthetic, test-real.** Fit any supervised model (e.g. a volatility forecaster, a trading signal, a classifier for regime) on synthetic paths and evaluate on held-out real data. Report accuracy degradation vs. training on real data.
- **Portfolio optimisation TSTR.** Run a mean-variance or Kelly optimizer on CHMM-simulated cross-asset paths and compare the resulting portfolio's Sharpe / MDD against real-path optimization.

Even one such experiment (Section 4.8 in results, 1–2 pages) would move the paper from "fidelity benchmark" to "quality of synthetic data" in the JDIQ sense, and would substantially strengthen the fit to the special issue.

### 4.2 Excess-kurtosis over-correction under CHMM-t is under-explained

Table 2 reports CHMM-t IS excess kurtosis of 12.60 against an observed value of 7.71 — a 63% over-shoot — while the cross-asset panel shows even larger over-corrections (78.00 for JNJ, 53.28 for NVDA). The authors attribute this to "the ECM search assigns small $\nu_k$ to a subset of tail states," but:

- The lower ECM bracket is $\nu_{\min}=2.1$, which is close to the 2-moment frontier. If several tail states land at $\nu_k \approx 2.1$, the in-sample mixture kurtosis is unbounded by construction. **A per-state $\nu_k$ histogram for the $K=18$ fit would make the diagnosis concrete** and would answer whether the overshoot is (a) a genuine fit to conditional heavy tails or (b) an artefact of the lower bracket.
- Reporting OoS excess kurtosis of 6.42 (vs. observed 6.24) is encouraging but does not rule out bracket-limit behaviour. A sensitivity study over $\nu_{\min} \in \{2.5, 3.0, 4.0, \infty\}$ would be convincing.
- The authors recommend CHMM-t "for applications that prioritize tail-weighted distributional metrics." If CHMM-t systematically over-states tail heaviness by 60–90%, a consumer using CHMM-t for VaR or ES computation would systematically overstate risk. This is the utility consequence of the observed over-shoot and should be flagged.

### 4.3 KS/AD test power at OoS length

$T_{\text{OoS}}=219$ is short enough that the KS two-sample test has low power. The paper acknowledges this for the Gaussian i.i.d. and discrete HMM rows ("artificially inflated 74.5% OoS KS"), but does not acknowledge the symmetric concern for the CHMM OoS rows (94.7–95.2%). **Quantifying the OoS KS power — e.g. a simulation calibration where a known-correct generator's OoS pass rate is computed — would strengthen every OoS claim in the paper.** This is a standard diagnostic and is approximately one page of extra results.

### 4.4 The Rydén-et-al. counterclaim deserves a targeted demonstration

The paper's headline narrative contradiction with Rydén et al. (1998) rests on (a) higher $K$ and (b) quantile-based initialization. The Rydén paper used $K=2$ with random initialization. A direct replication — fit the Rydén $K=2$ Gaussian HMM with *both* random and quantile-based initialization on SPY — would be definitive. Currently the $K=3$ row of the sensitivity table (IS KS 88.7%, ACF-MAE 0.0453) is offered as the key evidence, but the contrast with random initialization is never made explicit on the $K=2$ setting that Rydén actually evaluated.

### 4.5 The JPM/JNJ OoS "stationarity" explanation is under-supported

The OoS KS pass rates on JNJ (68.4%) and JPM (72.0%) are 26–30 percentage points below the IS values, and the paper attributes this to the stationarity assumption. Several alternative explanations deserve to be ruled out:

- Is 2025 simply anomalous for defensives (sector-specific macroeconomic or policy shock) rather than a stationarity break? A subsample analysis on 2014–2019 / 2020–2024 / 2025 IS-vs-OoS triangle would help distinguish.
- Is the IS fit overfitting short-horizon volatility that does not persist out of sample? A walk-forward re-estimation — refit CHMM every quarter in the OoS window — is the natural test and is alluded to in the "Future work" list but never run. Running even a single rolling-window baseline (one additional row in Table T2) would materially strengthen the interpretation.

### 4.6 Copula-MLE detail omitted

The Student-t copula is selected by profile MLE over a discrete grid $\nu \in \{2,3,4,5,6,8,10,15,20,30\}$, but the paper does not report the profile log-likelihood values, the gap between adjacent $\nu$ points, or a confidence interval for $\hat{\nu}^{*} = 6$. Without these, the choice of $\nu^{*}=6$ is qualitatively plausible but statistically unanchored. Adding a small profile log-likelihood plot in the supplemental would resolve this in one figure.

## 5. Moderate Concerns

### 5.1 Model selection criteria for $K$

The "joint" state-count selection (four information criteria + seven-metric fidelity score) is sensible but ad hoc. Because AIC, BIC, HQC, and CAIC are ordinally close, there are cases where the IC-minimizing $K$ would be different from the fidelity-maximizing $K$; the paper does not say how ties/conflicts are adjudicated. The sensitivity analysis at $K = 15, 18, 21$ is flat, which actually makes the point that fine model selection doesn't matter in this regime — but this should be stated as the conclusion, rather than the "$K=18$ wins on every metric" language currently in the results (which is not quite true at Table T1-Multi-Emission: $K=21$ ties or beats $K=18$ on CHMM-N IS KS, IS AD, and OoS KS).

### 5.2 Bootstrap block length for the temporal metric

The paper's bootstrap row resamples returns i.i.d. and is explicitly presented as "destroying temporal structure." For ACF-MAE this is the right negative control. But for the distributional metrics (KS/AD/$W_1$/Hellinger), a block-bootstrap benchmark with a realistic block length (e.g. 10–20 trading days) would be a more informative non-parametric ceiling, because it preserves clustering while matching the marginal exactly. This would also address the reviewer's concern (§4.3) about the KS i.i.d. assumption.

### 5.3 The VIX/VXX extension is claimed but not quantified in the main text

The abstract and introduction both state that the CHMM passes the seven-metric battery on VIX and VXX. The main body contains only a brief narrative paragraph (§6.7); the actual numerical tables for VIX/VXX are one small paragraph in the supplemental (§A.19). Either the claim should be promoted to the main text (with a table) or softened (it is currently presented as a headline contribution but is not demonstrated in the main body).

### 5.4 Missing comparisons to deep generative baselines

The paper cites TimeGAN / GAN-based generators in the related-work section but does not benchmark against any of them. Given that JDIQ places synthetic data in a broader context that includes deep generative models, at least one deep baseline (TimeGAN, Fin-GAN, or a simple LSTM autoregressive model trained on SPY) would strengthen the "CHMM is competitive" narrative. If computationally heavy, a single OoS KS + ACF-MAE number for TimeGAN on SPY cited from literature would suffice.

## 6. Minor Concerns

### 6.1 Abstract

- Line 56 of Paper_v6.tex: "The heavy-tailed CHMM-t and CHMM-L variants close the kurtosis gap of the Gaussian CHMM (mean simulated excess kurtosis rises from 5.04 to 6.34 under CHMM-L and 12.60 under CHMM-t against the observed 7.71)." — The 12.60 number is an *overshoot*, not a "close of the kurtosis gap." Reword to be precise: CHMM-L brings the simulated kurtosis from 5.04 to 6.34 (closer to 7.71); CHMM-t overshoots to 12.60, with an OoS kurtosis of 6.42 that aligns with the observed OoS 6.24.
- "Six alternatives" in the abstract — actually seven models are compared (bootstrap, Gaussian, Laplace, Discrete NJ, Discrete WJ, GARCH, plus three CHMM variants, total 9). Consider "against six alternative generators" explicitly clarifies the reference category, but also confirm the count.

### 6.2 Table 2 column title

"Discrete NJ" and "Discrete WJ" — define NJ / WJ at first mention in the results section (currently only in the table caption).

### 6.3 Cross-asset AD OoS column

Table T2 (cross-asset) no longer has an AD OoS column (correctly, since the output file does not emit it). The "AD IS" column header and the removed "AD OoS" column should be acknowledged in the caption so the reader does not wonder why OoS AD is missing while OoS KS is present.

### 6.4 Algorithm placement

Algorithms 1 (Baum-Welch Gaussian), 1b (ECM Student-t), and 1c (EM Laplace) are 1.5-page each. Consider shortening to a unified algorithm with a "family-specific M-step" switch so the three are presented on a single 1.5-page block, following the narrative in §A.2. This would also make the "shared scaffold + family-specific M-step" structure visually obvious.

### 6.5 Figure legibility

- Figures 3–4 in the K=18 panels use small sub-panel titles. On a printed JDIQ page, the KS pass-rate annotation is borderline legible. Consider increasing the sub-panel font size to 10pt.
- The transition matrix heatmaps (Figs. supplemental) use `log10` color scale but the color bar is missing in some subplots; check that all three family panels include a color bar.

### 6.6 Notation

$\gamma_t(k)$ is defined twice (eq. 3 in §A.2, and again in eq. around M-step). Keep one definition.

### 6.7 The introduction claims "92.0% OoS KS under CHMM-t at $K=3$"

The appendix Table T1-multi shows CHMM-t IS KS 92.0% at $K = 3$ and OoS KS 93.0%. The introduction references these numbers but does not explain *why* CHMM-t already beats CHMM-N at $K = 3$ (CHMM-N IS KS is 88.7% at $K=3$). The explanation — that heavy tails compensate for under-resolved regime structure — is latent in the paper and should be stated explicitly.

## 7. Reproducibility

The paper scores high on reproducibility: code repository, pseudocode for all four core procedures (Baum-Welch Gaussian, ECM Student-t, EM Laplace, copula rank-reordering, SIM), data provenance, hyperparameters, random-seed discipline (implicit — should be explicit), and a seven-metric evaluation script. Two small gaps:

1. **Random seed.** Is the 1,000-path Monte Carlo deterministic (seeded) for the reported numbers? The paper should state the seed convention, because "run the code and reproduce the table" only works if the pipeline is deterministic.
2. **Software environment.** The README documents Julia 1.12+, but the `Manifest.toml` pin is essential for exact reproducibility and should be referenced in the paper (or an "Artifact" appendix should state the Julia version and package versions).

## 8. Relation to Prior Work (Alswaidan et al. 2026)

The paper's positioning with respect to the prior discrete-HMM paper is unusually candid: the prior paper is reproduced as Discrete NJ/WJ *inside* this paper's pipeline, and the prior paper's copula ordering is re-confirmed with continuous marginals. Two observations:

- The claim that "the jump mechanism becomes unnecessary at moderate $K$" is well supported for the Gaussian case but is less sharply demonstrated for the discrete-with-Student-t-emissions configuration used by Alswaidan et al. (2026). The discrete pipeline fits bin-conditional Student-t emissions that are not reproduced here. A direct "discrete + bin-Student-t + no jumps" row in Table 2 would be the fairest discrete counterpart, and would show whether quantization per se (not emission choice) is the KS-killer.
- Nit: the in-text description on page 10 ("re-fitted on the SPY in-sample series under the same pipeline") could be confused with "re-implemented" — state plainly whether the code used for Discrete NJ/WJ is the original JumpHMM.jl code or a reproduction.

## 9. Clarity and Quality of Writing

Generally high. The narrative is well-organized, the methods section is precise, and the figures are clean. Specific suggestions:

- The "Summary of the three CHMM variants" paragraph (results §4.3) is useful but repeats content from each variant's own paragraph. Consider consolidating into a one-row decision table:

| Use case                                       | Variant   |
|------------------------------------------------|-----------|
| Best KS + minimal tail overshoot               | CHMM-N    |
| Tightest $W_1$, $H$; OoS kurtosis alignment    | CHMM-t    |
| Cleanest IS kurtosis match; best AD; cheapest  | CHMM-L    |

- The limitations list (§5 in discussion) is exemplary: seven concrete items with reasonable mitigation plans. This section is a model for other SDG papers.

## 10. Review Criteria Scoring

Against JDIQ's review criteria for technical papers:

| Criterion                                   | Score (1–5) | Comment                                                   |
|---------------------------------------------|:-----------:|-----------------------------------------------------------|
| Novelty of contribution                     | 4           | New EM-scaffold family; Rydén contradiction; fidelity panel |
| Technical soundness                         | 4           | ECM + weighted-median derivations correct; ablations sound |
| Quality of experimental evaluation          | 4           | Seven-metric panel, ten-year data, cross-asset, 1K paths  |
| Relevance to special issue                  | 3.5         | Fidelity strong; utility missing; privacy NA               |
| Clarity and presentation                    | 4.5         | Well-organized; figures clean; appendix complete           |
| Reproducibility                             | 4           | Code public; seed/Manifest pin should be explicit          |
| Comparison with state of the art            | 3           | Missing deep baselines; discrete-Student-t comparison      |

**Overall rating (1–5).** 4.0 — *accept with minor revision* (conditional on the utility experiment in §4.1 and the $\nu_k$ bracket analysis in §4.2).

## 11. Recommendation

**Minor revision.** The paper is a serious, well-executed technical contribution on SDG for financial time series with a strong fidelity-evaluation panel and clean multi-family ablations. Two revisions would materially elevate it within JDIQ's scope:

1. **Add at least one utility experiment** (a VaR/ES back-test calibration or a TSTR downstream task) — 1–2 pages, §4.1 above.
2. **Add a per-state $\nu_k$ histogram and a bracket-sensitivity analysis for CHMM-t** — 0.5 page, §4.2 above.

The other concerns are either addressable in a short revision letter or are minor copy-edits. In their current state the main contributions (the three-family EM framework, the Rydén contradiction, and the copula ordering on continuous marginals) are sound and publishable.

## 12. Page-Count Compliance Note

The technical-paper cap for JDIQ is 23 pages. Paper_v6.pdf is 43 pages (A4, 11pt). The main body is approximately 22 pages and the appendix 20 pages. JDIQ typically accepts supplementary material beyond the page cap, but the authors should confirm the exact policy with the guest editors and either (i) move non-essential appendix content to a separate supplementary PDF or (ii) re-format to a tighter column/font pattern. The current algorithm panels and the multi-emission sensitivity table would be the two most natural candidates for the supplementary PDF.

## 13. Short Summary for the Meta-Reviewer

A well-motivated, methodologically careful SDG paper with an unusually complete fidelity evaluation across three emission families and two cross-asset dependence constructions. The one clear revision needed is a downstream utility benchmark to match JDIQ's fidelity/utility framing. The heavy-tailed emission over-correction under CHMM-t needs a short diagnostic, and a discrete-with-Student-t comparison would close the prior-paper loop cleanly. With these additions, the paper would be a strong fit for the special issue.

---

*End of review. Filed against `Paper_v6.pdf` (43 pages, 3.21 MB, compiled 2026-04-20 19:48 UTC).*
