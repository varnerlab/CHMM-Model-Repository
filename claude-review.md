**Editorial Decision:** Revise and Resubmit (Major Revisions)

Thank you for submitting your manuscript, "Continuous Gaussian Hidden Markov Models for Equity and Volatility Index Dynamics," to the *ACM Journal of Data and Information Quality* (JDIQ) Special Issue on Synthetic Data.

The manuscript offers a careful empirical re-examination of the Rydén, Teräsvirta & Åsbrink (1998) result — that Gaussian-emission HMMs cannot reproduce the slow decay of ACF(|r|) — and argues that the pessimism of the 1990s literature was driven as much by initialization and state-count choices as by the geometric-dwell-time limitation of the Markov kernel itself. The companion framing as a *synthetic-data-quality* benchmark and the full pipeline from univariate CHMM through a Single-Index Model (SIM) and Gaussian / Student-t copulas to a multi-asset generator is exactly the kind of end-to-end study that JDIQ readers will appreciate. I also note the prior discrete + Poisson-jumps baseline (Alswaidan & Varner, 2026, arXiv:2603.10202) is reproduced here under identical evaluation metrics, which is the right way to anchor a methodological contribution.

That said, the paper as submitted reads more like a thorough empirical appendix than a self-contained journal article. My review below focuses on six structural issues; the accompanying reviewer from my panel has separately flagged the Student-t emission and downstream-utility gaps, and I concur with those recommendations without repeating them here.

### 1. Paper length and "one model per section" sprawl

The current manuscript devotes a full sub-section with figures to each of K ∈ {3, 6, 9, 11, 13} and in several places carries the same metric table through both the main body and an appendix. At roughly [length], this is at least 30% longer than the JDIQ target and the redundancy dilutes the central claim.

* **Editor's Recommendation:** Select a single canonical model — based on a transparent criterion such as ACF-MAE or Anderson–Darling pass rate on the SPY in-sample set — and demote the full K sweep to a single sensitivity table (Table T1 in your Appendix is already a good template). The main body should show one convergence plot, one emission PDF panel, one transition heatmap, and one IS/OoS validation figure, all for the chosen K. The other K values belong in supplementary material. If reviewers insist on the sweep in-body, it should fit on one page as a line plot of metrics versus K.

### 2. Choice of K should be principled, not illustrative

Related to (1): you sweep K ∈ {3, 6, 9, 11, 13} without stating why those values. A sensitivity curve is useful, but the reader then expects an argument for picking one. Information-criterion (BIC/AIC) trajectories, or held-out log-likelihood via walk-forward, would convert an "illustration" into a "recommendation."

* **Editor's Recommendation:** (i) Extend the grid on a coarser, evenly spaced mesh — multiples of 3 up to K=21 would be natural, given that above ~20 the Gaussian mixture ceases to be identifiable from ~2,500 daily observations; (ii) report BIC and held-out LL along with the stylized-fact metrics; (iii) explicitly state the selection rule used in the main body. I note that you have 424 candidate assets in the training universe, which is more than enough to perform this without leakage.

### 3. Mimicking the prior paper's comparison scaffolding

I appreciate that Table 2 (Bootstrap / Gaussian / Laplace / GARCH(1,1) / CHMM) and Table T2 (NVDA, JNJ, JPM cross-asset) are deliberately aligned with the tables of arXiv:2603.10202. This parallelism is a strength — reviewers should be able to read the two papers side-by-side. However, the alignment is currently incomplete:

* The discrete + Poisson-jumps model is referenced in the introduction but is **not** shown in Table 2 of the present submission. For a paper whose core claim is *"the continuous model does not need jumps,"* the discrete + jumps row is the most important row in the table.
* The multi-asset Table T2 in the discrete paper spans six tickers (SPY, NVDA, JNJ, JPM, AAPL, QQQ); the present submission reports only three (NVDA, JNJ, JPM) in the CHMM-only Table T2, although the SIM/copula table correctly spans all six. Align the two cross-asset tables so the reader sees the same universe throughout.
* The Frobenius-norm correlation reproduction and Kendall-τ Σ-estimation analysis are excellent but are only present in the SIM/copula extension. A single row in the main Table 2 giving the average off-diagonal correlation error for each benchmark would let the reader see the univariate-vs-dependence trade-off immediately.

* **Editor's Recommendation:** Restructure Tables 2 and T2 so every benchmark appears on the same ticker universe (SPY + NVDA + JNJ + JPM + AAPL + QQQ) under the same metrics (KS, AD, kurtosis, ACF-MAE, W1, Hellinger, 90% coverage, Frobenius-Σ). The discrete + jumps model from arXiv:2603.10202 belongs in Table 2.

### 4. The "JNJ drop" is understated

Your OoS KS pass rate for JNJ falls from 99.8% IS to 68.7% OoS at K=13 — a 31-point drop, larger than any other cross-asset result in the paper. You attribute this to *"idiosyncratic regime shifts in 2025"* but do not quantify it. A drop of this magnitude is either (i) a finite-sample artefact of the short 2025 OoS window, (ii) genuine non-stationarity that should be addressed with rolling re-estimation, or (iii) a model-selection issue (K was tuned on the full universe, not per-asset).

* **Editor's Recommendation:** Run a rolling-window CHMM on JNJ with a 1-year window and report OoS KS pass rate as a function of window size. If the drop persists, it is a real result and the paper is stronger for stating so; if it recovers, the stationary assumption is fine and only the global-K choice needs revisiting. Either outcome is publishable.

### 5. Statistical multiplicity in the KS / AD tables

You report KS and AD pass rates across 1,000 simulated paths and declare success whenever the per-path p-value exceeds 0.05. This is a reasonable *goodness-of-fit coverage* diagnostic but it is not a calibrated hypothesis test: under the null (simulated = observed), the expected pass rate is 95%, not 100%. The manuscript occasionally treats a 94.4% pass rate as a "pass" and a 74.0% Gaussian rate as a "fail" without stating the null expectation.

* **Editor's Recommendation:** Add a single sentence in Section 3.2 clarifying that 95% is the theoretical ceiling under correct specification and that values above ~90% are consistent with a well-calibrated generator. Optionally, run a Kolmogorov-Smirnov goodness-of-fit test on the *distribution of p-values* (it should be ≈ Uniform[0,1] under the null) and report that as a single summary statistic per model.

### 6. Minor but cumulative issues

* **Reproducibility of Baum-Welch.** Quantile-based initialization is deterministic given the data but the EM update is not — floating-point tie-breaking and the stopping criterion (tol=1e-4, max_iter=60) can move metrics by 1-2% between runs. Report metrics as mean ± std over ≥ 10 independent EM starts for the main table.
* **GARCH benchmark.** Your GARCH(1,1) log-likelihood is fit by Nelder-Mead from a 4×4 grid; the `ARCHModels.jl` or `rugarch` maximum-likelihood fit is the field standard and should at minimum be cross-checked. As it stands, GARCH's very low 13% in-sample KS pass is a surprising result that deserves either validation or a narrative.
* **Notation.** The "excess growth rate" G_t = (1/Δt) log(P_t/P_{t-1}) − r_f is non-standard. Either (i) state that G_t is annualized log-return at Δt = 1/252 and r_f = 0 on first use, or (ii) use r_t for the daily log-return and reserve G for the annualized scale. Currently both conventions appear in the draft.
* **References to the prior paper.** The discrete + jumps baseline is both your own prior work and the most direct comparator. A single sub-section contrasting the two models' parameter counts, free hyperparameters (ε, λ, bin count, K), and wall-clock training time would anchor the "simpler and comparable" claim that the title promises.

### 7. Summary

This paper is one of the more careful HMM-for-finance manuscripts I have reviewed in the last year. The core claim — that a continuous Gaussian HMM with quantile-initialized Baum-Welch reproduces all three canonical stylized facts at small-to-moderate K without requiring a jump mechanism — is, to my knowledge, novel in the form and rigor in which it is presented here. The extension from a single-asset CHMM to a 424-asset SIM/copula system, validated on true 2025 hold-out data, is exactly the kind of end-to-end synthetic-data-generation pipeline that JDIQ wants to showcase.

The path to acceptance is, in my view, almost entirely structural: (a) cut the paper by ~30% by demoting the K sweep to a sensitivity table and picking one canonical model; (b) align Tables 2 and T2 with the prior paper on the same ticker universe and put the discrete + jumps row back into Table 2; (c) clarify the statistical interpretation of per-path KS/AD coverage; and (d) address the JNJ OoS degradation with a short rolling-window experiment. With these changes I would expect to recommend *Accept with Minor Revisions* on the next round.

*— Editor, JDIQ Special Issue on Synthetic Data*
