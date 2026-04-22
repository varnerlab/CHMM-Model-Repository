# VIX CHMM Integration Plan (companion to plan-paper-v9.md and plan-icaif-v9.md)

**Status:** Proposal. Supersedes nothing in the existing v9 plans; adds a new scope item to the full paper, leaves ICAIF v9 body unchanged.
**Purpose:** Resolve the "Cade's VIX contribution" scope question without breaking the existing thesis.
**Decision requested from advisor:** Approve Option A (standalone Block D in full paper, out of ICAIF body) before any .tex editing.

---

## 1. Decision recommendation

**Option A (recommended).** Add a standalone univariate CHMM on log(VIX) as a new "Block D: Volatility-index robustness" in the full paper v9 only. Keep the equity and VIX models decoupled. No joint training, no shared state space, no copula coupling in v9. Frame as generalization-of-scaffold, not as a second thesis.

**Option B (rejected).** Joint CHMM over (return, VIX) or concatenated two-observable emissions. Already tested by the first author and degrades equity simulation quality. The literature says this should fail because the two observables have incompatible stylized-fact profiles (see Section 3). Do not pursue.

**Option C (rejected for ICAIF v9-icaif only).** Add VIX block to the 8-page ICAIF submission. Rejected because every column-inch in the 8-page budget currently defends one of the three thesis nouns (digital twin / high fidelity / easy to train on equity returns). Adding VIX either dilutes the thesis or forces a cut of Pipeline B / GRU / stylized-facts figure, all of which serve the equity thesis more directly.

Option A keeps the professor's request satisfied, keeps the ICAIF submission on-thesis, and uses the existing EM scaffold rather than any new machinery.

## 2. Why VIX must be modeled differently from equity

The project's current framing ("annualized excess log returns $G_t$, mean near zero, symmetric") does not apply to the VIX. Three load-bearing differences:

1. **Stationarity.** The VIX is stationary in levels (augmented Dickey-Fuller rejects the unit root on daily VIX over 2014 to 2026, at conventional significance). Equity prices are non-stationary; only returns are stationary. So the transform changes: equity uses first log differences, VIX uses log levels (or log-level first differences).
2. **Support and central tendency.** VIX is mechanically bounded above zero and mean-reverts to ~15 to 20. A Gaussian CHMM trained on raw VIX would waste state capacity encoding the bounded support.
3. **Autocorrelation structure.** The canonical equity fact is "ACF of returns is near zero; ACF of squared/absolute returns decays slowly." The canonical VIX fact is "ACF of **levels** decays slowly," with fractional memory exponent d in 0.3 to 0.5 reported across the volatility-index literature (Ghosh 2022; Chen, Huang, and coauthors).

Consequence: the equity CHMM emission family is applied to `r_t = log(p_t / p_{t-1}) / dt - r_f`. The VIX CHMM emission family should be applied to one of `y_t = log(VIX_t)` (levels) or `y_t = log(VIX_t) - log(VIX_{t-1})` (increments). Modeling raw VIX levels as Gaussian is not defensible.

## 3. Stylized facts a volatility-index generator must reproduce

Literature-grounded checklist. This replaces the three-stylized-facts panel (heavy tails, near-zero return ACF, persistent ACF of absolute returns) used for equity.

1. **Mean reversion of levels** with finite long-run mean (15 to 20 on VIX). Diagnostic: ADF p-value, half-life to mean estimated as $-\log(2)/\log(\phi)$ from AR(1) on log levels. Reference models: mean-reverting log-OU (Kaeck and Alexander 2013), log-Heston (Papanicolaou and Sircar 2014).
2. **Slow decay of ACF of levels** (not squared increments). Diagnostic: ACF(log VIX) at lags 1 to 252, compared to simulated median with bootstrap band. Plot analogous to the equity ACF(|r|) panel but applied to log levels.
3. **Positive skewness and right-tail jumps.** VIX does not crash; it spikes. Diagnostic: sample skewness on log-level increments; Hill estimator of right-tail index; QQ-plot of simulated vs observed upper-tail quantiles. Reference: jump-diffusion VIX of Dotsis, Psychoyios, Skiadopoulos (2007).
4. **Leptokurtosis of increments.** Diagnostic: sample kurtosis on log-level increments. Expect excess kurtosis well above Gaussian. Reference models: any volatility-jump process.
5. **Leverage effect** (cross-asset; optional). Negative correlation between equity returns and VIX changes, asymmetric on down days. Univariate VIX CHMM does not address this; it is parked as cross-asset future work and not tested in v9.
6. **Volatility of volatility clustering** (optional, VVIX literature). Skip in v9 body; one-line acknowledgement in Future Work.

## 4. Model specification (what to fit)

Use the existing `MyContinuousHiddenMarkovModel` scaffold from `src/Types.jl`, `src/Factory.jl`, `src/Compute.jl`. No new type. The only difference from the equity pipeline is the observable.

- **Observable.** Default: $y_t = \log(\text{VIX}_t)$ (levels). Alternative: $\Delta y_t = \log(\text{VIX}_t) - \log(\text{VIX}_{t-1})$ (increments). Report both; pick one for the main Block D table based on which better reproduces the Section 3 checklist.
- **Emission families.** Same three: Gaussian, Student-t with per-state $\nu_k$, Laplace. Same quantile-based init, same golden-section $\nu_k$ update, same log-space FB.
- **State count $K$.** Start at $K = 18$ to match equity for like-for-like comparison. Add a short $K \in \{3, 6, 9, 12, 15, 18, 21\}$ sensitivity in an appendix only if Block D needs it.
- **Training window.** Match the equity 10-year training window (2014-01-03 to 2023-12-29) to keep the data regime aligned with the SPY CHMM.
- **OoS window.** 2024-01-02 to most recent available (matches `SP500-Daily-OHLC-1-3-2025-to-4-20-2026.jld2`).
- **Data source.** CBOE ^VIX close-to-close, same date grid as SPY. Fetch via same pipeline used for `build_new_train_oos.jl`; add `build_vix_train_oos.jl` as a parallel script.

## 5. Evaluation protocol for Block D

Adapted from the equity seven-metric panel. Not identical, because the stylized facts are different.

| Metric | What it tests | Form |
|---|---|---|
| ADF p-value | stationarity of levels | pass if p < 0.05 on observed and simulated |
| AR(1) half-life | mean-reversion speed | match within 25% of observed half-life |
| ACF of log levels, lags 1-252 | long-memory of levels | MAE of simulated-median ACF vs observed |
| KS on log-level increments | marginal distribution | same threshold interpretation as equity KS |
| Skewness of increments | asymmetric spikes | signed match |
| Kurtosis of increments | heavy tails of increments | ratio match |
| Right-tail Hill estimator (top 5%) | spike tail | within-band match |

No VaR/ES utility for Block D (options-pricing extension is explicitly out of scope per professor). No price-fan analogue (VIX is not a price). The utility framing for this block is "proof of generative generalization," not downstream risk.

## 6. Where Block D lives in the full paper v9

Inserted after the existing Block C (downstream utility). Self-contained, does not alter any other block.

**Section 4, new subblock ordering:**

- Block A: Univariate equity fidelity on SPY (unchanged).
- Block B: Cross-asset equity digital twin (unchanged).
- Block C: Downstream utility, VaR/ES and price-fan (unchanged).
- **Block D (new): Volatility-index generalization on log(VIX).**

### Block D subsections (target 1.5 pages main body)

1. Motivation and scope (1 paragraph). State that the CHMM scaffold is not equity-specific and generalizes to any univariate stationary observable with heavy tails and mean reversion. Cite Rossi and Gallo 2006 (HMM on realized volatility), Papanicolaou and Sircar 2014 (regime-switching Heston for VIX + SPX joint calibration), Dotsis et al. 2007 (jump-diffusion VIX).
2. Data and transform (1 paragraph). log(VIX) on the 2014-2023 window. Descriptive stats: mean, std, skewness, kurtosis of levels and of increments; ADF on levels; AR(1) half-life.
3. Fit and single-table result (1 table, "Table D1"). Rows: CHMM-N, CHMM-t, CHMM-L, all at K=18, plus one AR(1) on log-levels baseline and one Gaussian-iid-on-increments baseline. Columns: ADF-pass, half-life match, ACF-MAE, KS on increments, skew, kurtosis ratio, Hill tail index.
4. One figure ("Figure D1"). Four panels: observed vs simulated log(VIX) path overlay; observed vs simulated increments histogram; observed vs simulated ACF(log VIX); QQ-plot of increments.
5. Paragraph interpreting result in thesis language: "The same scaffold that fits equity returns also fits the volatility-index observable, reproducing its distinct stylized-fact profile (stationarity, long-memory of levels, right-tail jumps) without any model-specific machinery. This supports the digital-twin framing at the algorithmic-scaffold level rather than the return-series level."

### What Block D does NOT claim

- Does not claim a VIX-based options pricing result. Keep the existing out-of-scope sentence in the Introduction.
- Does not claim joint equity-VIX dynamics are captured. The two CHMMs are decoupled.
- Does not claim a new best VIX model. Compares only to two trivial baselines (AR(1), Gaussian iid on increments) because the goal is generalization of scaffold, not a VIX horse race.
- Does not propose a leverage-effect or VVIX extension in v9.

## 7. Diffs to plan-paper-v9.md

Minimal, surgical diffs to the existing full-paper plan:

- **Section 2 (Scope decisions), IN list:** Add one bullet, "Univariate volatility-index generalization: CHMM on log(VIX) at K=18, three emission families, 2014-2023 train."
- **Section 2 (Scope decisions), OUT list:** Reinforce "joint equity-VIX modeling, leverage effect, VIX options pricing" all remain OUT.
- **Section 3 (Structural changes), Section 4 skeleton:** Add Block D after Block C in subsection 4.
- **Section 4.4 (Empirical study):** Add a new subsection 10 "Volatility-index generalization" after existing subsection 9.
- **Section 5 (Figures and tables), Main-body tables:** Add Table D1 (new Table 7).
- **Section 5 (Figures and tables), Main-body figures:** Add Figure D1 (new Figure 11). This pushes the headcount to 11 figures; acceptable at journal length.
- **Section 7 (Edit sequence):** Insert a new step between existing steps 6 and 7: "6b. Fit VIX CHMM for three emission families on log(VIX); generate Table D1 and Figure D1; draft Block D subsection prose."
- **Section 8 (Success criteria):** Add one criterion, "Block D reproduces stationarity, slow ACF of log levels, and right-tail skew on the VIX without any VIX-specific code changes."
- **Section 9 (Non-goals):** Reinforce "not redesigning the scaffold for VIX."

## 8. Diffs to plan-icaif-v9.md

Zero body diffs. One footnote-level change:

- **Section 1 (Thesis):** unchanged.
- **Section 2 (Scope):** Add one bullet to the Dropped list, "Volatility-index (VIX) generalization is deferred to the companion full paper; one-sentence reference only."
- **Section 4 (Section-by-section), Section 1 Introduction:** Add one half-sentence in the existing digital-twin framing paragraph, along the lines of: "The same scaffold extends to other univariate financial observables, discussed in the companion full paper." No new citation in the 8-page bib.
- **Section 4 (Section-by-section), Section 4 Discussion:** No change.

The 8-page page budget and figure/table count are unchanged. No reviewer item mapping changes.

## 9. Implementation plan (Julia, concrete)

### 9.1 Code additions (no new types)

- **New file `build_vix_train_oos.jl`.** Same structure as `build_new_train_oos.jl` but downloads and cleans ^VIX series instead of SPY. Saves `data/VIX-Train-2014-2023.jld2` and `data/VIX-OoS-2024-Onward.jld2`. Transform: store raw VIX, compute `log_vix = log.(vix)` and `dlog_vix = diff(log_vix)` in memory; the JLD2 stores raw + date vectors.
- **New file `run_vix_generalization.jl`.** Mirrors the structure of the equity fidelity script. Loads the VIX dataframe, chooses observable (default log(VIX)), fits three CHMM families at K=18, saves fitted models and simulations to `results/vix/`.
- **New file `run_vix_diagnostics.jl`.** Computes the Block D metric panel (Section 5 above) and emits `results/vix/Table-D1.csv` and `figs/Fig-VIX-BlockD-Panel.svg`.

### 9.2 No changes to `src/*.jl`

The existing CHMM scaffold is observable-agnostic. No new type, no new factory, no new simulate method. If this assumption breaks during implementation, stop and flag before introducing new code paths.

### 9.3 Data hygiene

- Align the VIX date grid to the SPY trading calendar (same NYSE holidays).
- If VIX has more dates than SPY on a given window, subset to the SPY intersection so the training window is exactly comparable.
- Sanity-check the ADF p-value and the AR(1) half-life on the 2014-2023 training window before running EM.

## 10. Risks and open questions

| Risk | Mitigation |
|---|---|
| Reviewer reads Block D as a second thesis and asks for cross-asset equity-VIX coupling. | Explicit "scaffold generalization, not second thesis" language at the top of Block D; one-sentence scope paragraph parking cross-asset coupling as future work. |
| Transform choice (levels vs increments) changes the result. | Report both in an appendix; pick the main-body variant based on which reproduces more Section 3 facts. Both are literature-supported. |
| K=18 may not be the right resolution for VIX. | Run a 30-minute K-sweep, report in appendix only if the main-body K=18 result looks obviously wrong. Do not re-open model selection in the main body. |
| Professor requests options-pricing extension later. | Keep Block D's scope tight; if required, add a follow-up short-paper, not a v9 scope creep. |
| Block D displaces a stronger piece of evidence in v9. | It does not; it is additive. The 1.5 pages are net-new text, not a rewrite. |

## 11. Out-of-scope for this plan

- Joint equity-VIX dynamics.
- Leverage effect and VVIX.
- VIX options pricing, IV surface.
- Any change to the ICAIF 8-page body.
- Any change to the existing equity Block A / B / C prose or tables.
- Bayesian / regime-switching Heston comparison (mentioned in the literature paragraph but not implemented).

## 12. Success criteria for Block D

1. Block D reproduces stationarity (ADF p < 0.05 on simulated), slow ACF of log levels (ACF-MAE within the equity benchmark's band), and right-tail skew on the VIX, using the same CHMM codebase as equity.
2. Block D fits in 1.5 pages of body + 1 table + 1 figure in the full paper v9.
3. No changes to `src/*.jl` required to fit the VIX. If changes are needed, stop and revisit the scaffold generalization claim before writing.
4. The existing three equity thesis nouns (digital twin, high fidelity, easy to train) survive unchanged in the introduction and conclusion.
5. Block D is explicitly out of the ICAIF v9-icaif body.
