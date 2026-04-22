# Claude ICAIF 2026 Review of Paper_v8-icaif

**Reviewer:** Independent peer review (simulated)
**Venue:** ICAIF 2026 (sigconf, 8-page, double-blind)
**Paper:** "A Continuous Hidden Markov Model for Synthetic Equity Returns: Emission-Family Ablation and Rank-Reordering Copula Extensions"
**Date:** 2026-04-21
**Recommendation:** Weak Accept (with revisions). Conditional on: (a) clearer novelty positioning vs continuous-state financial HMM literature, (b) statistical rigor on pass-rate comparisons, (c) an online / rolling-window extension promoted from diagnostic to methodology.

## 1. Summary

The paper introduces a family of continuous-emission hidden Markov models (CHMM-N, CHMM-t, CHMM-L) trained by log-space EM / ECM on raw annualized excess log returns, and argues that at a single moderate state resolution ($K = 18$) these models reproduce the three canonical stylized facts of equity returns without the return-space quantization or Poisson jump augmentation of the author's (anonymized) prior discrete-HMM framework. The evaluation is staged as Pipeline A (per-ticker univariate fit across six equities) and Pipeline B (Single Index Model plus Gaussian and Student-t rank-reordering copulas on top of Pipeline A marginals). Headline numeric results: 94-95% IS KS pass rate on SPY across all three CHMM variants; 80-84% OoS; Student-t copula reduces off-diagonal correlation MAE to 0.027 against 0.076 for the SIM baseline on the six-ticker universe.

## 2. Relevance to ICAIF

High. Synthetic financial time-series generators are an active area at ICAIF (2021-2025 proceedings contain multiple TimeGAN, Quant GAN, diffusion, and state-space submissions). A generator whose fit is demonstrable through classical distributional tests rather than ad-hoc visual diagnostics is a useful contribution to the community.

## 3. Strengths

### 3.1 Pipeline separation is the paper's cleanest organizational idea.
Decoupling univariate marginal fit from cross-asset dependence allows the copula claim to be read independently of the emission-family claim. Both questions are answered at the right granularity. I would keep this separation even in revision.

### 3.2 The evaluation suite is appropriate and not overfit.
Five-metric scoring (KS, AD, kurtosis, ACF-MAE, Wasserstein-1) plus correlation reproduction for Pipeline B is defensible. The choice to compare against a non-parametric ceiling (i.i.d. bootstrap) surfaces the paper's strongest single finding: the bootstrap wins on marginals but loses on temporal structure (ACF-MAE 0.0627 versus CHMM-N 0.0513). This is the correct framing.

### 3.3 The Student-t copula selection at $\nu^* = 6$ is a credible result.
The range $4 \leq \nu \leq 10$ is consistent with McNeil et al. (2015) and Demarta and McNeil (2005) for equity-index portfolios. The reproduction of this empirical regularity without fitting to it is a useful sanity check.

### 3.4 The discrete-HMM baseline comparison is well-executed.
Reporting 0% IS KS for both discrete NJ and WJ variants frames the methodological gap unambiguously: bin-centroid emissions cannot pass KS on continuous observables. This is the single most persuasive paragraph in the paper.

## 4. Weaknesses

### 4.1 Novelty positioning against continuous-state financial HMM literature is thin.
The related-work discussion anchors on Ryd\'en, Ter\"asvirta, and \"Asbrink (1998) and the recent discrete-HMM framework. Missing: continuous Gaussian-emission HMMs for returns have appeared in finance and econometrics for three decades (Hamilton 1989, Bulla and Bulla 2006, Rossi and Gallo 2006, Nystrup et al. 2017, 2018 for regime-switching portfolio allocation). The paper's true novelty is narrower than "continuous HMM":

- Per-state Student-t $\nu_k$ selected by golden-section ECM inside the M-step, not global $\nu$.
- Weighted-MAD closed-form Laplace updates in an HMM M-step.
- The Pipeline A / Pipeline B framing.
- The specific head-to-head with a jump-augmented discrete-HMM baseline.

Recommend: a half-paragraph in Section 1 explicitly positioning against Nystrup et al. and Bulla et al., stating that continuous-emission HMMs for returns are not themselves new but the three-family ablation at $K = 18$ with the specific M-step schemes are.

### 4.2 No confidence intervals on any pass-rate comparison.
Table 1 reports CHMM-N at 94.7% IS KS, CHMM-L at 95.2%, CHMM-t at 94.7%, GARCH at 23.1%. The CHMM-to-CHMM differences are order-of-magnitude smaller than the CHMM-to-GARCH gap. The reader cannot distinguish whether CHMM-L truly dominates CHMM-N or whether the gap is within sampling noise over 1,000 paths. A Clopper-Pearson or Wilson interval on each pass rate would let the reader tell. Similarly for Table 3: NVDA 96.0% under Gaussian copula versus 93.0% under Student-t copula may or may not be statistically distinguishable.

### 4.3 The six-ticker Pipeline B universe is too small to support general claims.
The off-diagonal correlation MAE comparison (0.027 versus 0.031 versus 0.076) is reported on a $6 \times 6$ correlation matrix with 15 off-diagonal entries. This is a small sample for a ranking claim. The author already acknowledges scaling concerns in the Scope paragraph, but the current presentation overstates generality. Recommend: either extend to 30 tickers (S&P sectors) or soften the language to "on the six-ticker universe".

### 4.4 ACF-MAE as a single number conceals structure.
The ACF-MAE aggregates 252 lag errors into one scalar, which hides whether the error sits at short lags (relevant for daily trading) or long lags (relevant for long-horizon risk). The supplementary ACF curve would resolve this, but the conference version has no supplement. Recommend: add one small inset panel to Figure 1(d) overlaying simulated median ACF against observed.

### 4.5 Computational cost is absent from the paper.
ICAIF practitioners will want to know: for $K = 18$ and $T = 2{,}516$, how long does one EM fit take, in what language, on what hardware? How expensive is the golden-section $\nu_k$ ECM? How does the 1,000-path simulation cost scale? A single sentence and one small table in Section 3 would address this. The underlying Julia implementation is evident from the code repository (per the model codebase) but absent from the paper.

### 4.6 The walk-forward result is buried as a diagnostic when it is arguably the paper's second headline.
Section 4.4 reports a 15-percentage-point OoS KS recovery on JPM under quarterly walk-forward refitting. This is phrased as a diagnostic of stationarity assumption failure. It should instead be presented as a recommended deployment mode, and motivates Section 5 below.

### 4.7 "Recurrent deep-generative baselines" in the abstract is underspecified.
The abstract now states CHMM outperforms "recurrent deep-generative baselines" but the body of the paper only benchmarks against GARCH, discrete-HMM, bootstrap, and Gaussian i.i.d. in Table 1. The GRU + Gaussian-head baseline appears to be in the full Paper_v8 but not in this conference cut. Either include the GRU row in Table 1 or remove "deep-generative" from the abstract, otherwise the claim is unsupported by the body.

## 5. Proposed Major Extension: Online / Streaming CHMM (Author's Idea, Reviewer-Expanded)

The author raised the following question: given the OoS KS degradation on NVDA and JPM under the 2024-2026 regime, would the CHMM be more powerful as an online-learning tool updated daily (or at some acceptable interval) for use in trading-strategy evaluation? The paper already contains the critical evidence: the quarterly walk-forward lifts JPM OoS KS from 49.0% to 64.3%. This section sketches a concrete online extension that I believe would strengthen the paper substantially and open a second contribution.

### 5.1 Why the current fixed-IS protocol understates the model

Current protocol: fit once on 2014-2024, simulate on 2024-2026. Stationarity is assumed over 12 years. This is the wrong evaluation for a generator intended to inform trading.

Correct protocol for a trading application: as of each day $t$, the user has access only to returns up to $t - 1$. The generator must be refit with whatever cadence balances compute against tracking error. The OoS pass rate under a realistic refit cadence is the relevant number.

### 5.2 Four candidate refit regimes

Ordered by increasing theoretical sophistication:

1. **Rolling-window batch refit.** Every $\Delta$ days, drop the oldest $\Delta$ observations, append the newest, rerun full Baum-Welch. Simple, classical signal-processing approach (Nystrup et al. 2018 already do this for regime-switching portfolio allocation). Cost: one full EM per refit.

2. **Expanding-window batch refit.** Keep all history, append new observations, rerun full Baum-Welch. Gives better asymptotic consistency but slower adaptation to regime change. Trade-off choice.

3. **Warm-start batch refit.** Same windowing as (1) or (2), but initialize EM at the previous day's $\hat\theta$. Parameter movement is small day-to-day, so 5-10 EM iterations typically suffice versus the default 60. This alone likely gives a 6-12x speed-up at no loss of accuracy.

4. **Recursive / online EM.** Capp\'e and Moulines (2009), Capp\'e (2011), "Online EM algorithm for latent data models". Maintain a running sufficient-statistic vector $S_t = \gamma_t s(x_t; \theta_{t-1}) + (1 - \gamma_t) S_{t-1}$ with step size $\gamma_t \propto t^{-\alpha}$ for $\alpha \in (1/2, 1]$, and update $\theta_t$ from $S_t$ by the usual M-step. Cost per day: O(K^2) per forward step plus one M-step on accumulated statistics. Provably convergent to a stationary point of the expected log-likelihood under stationarity, with adaptation behavior under slow drift governed by the step-size schedule. This is the streaming analog of SGD for EM.

5. **Stochastic Approximation EM (SAEM).** Delyon, Lavielle, and Moulines (1999). Similar to (4) but with a Monte Carlo approximation of the E-step. Not clearly beneficial for HMMs since the forward-backward E-step is already tractable, but mentioned for completeness.

### 5.3 Techniques from deep-learning / generative-transformer training that transfer

Deep-learning training literature has invested two decades in making noisy iterative optimization reliable. Several ideas map cleanly to HMM training:

**(a) Learning-rate warmup and cosine annealing.** The step size $\gamma_t$ in online EM is directly analogous to an optimizer learning rate. A warmup schedule (linear increase for the first $T_0$ observations while sufficient statistics are sparse) followed by cosine decay has been shown in the SGD literature to stabilize early training. For online-EM HMMs this has not been widely studied but is a free improvement to try.

**(b) Gradient-accumulation analog: mini-batch EM.** Instead of updating sufficient statistics every observation, accumulate over mini-batches of size $B$ (e.g. $B = 20$ trading days). Lowers variance of the running $S_t$ at the cost of delayed response. Directly analogous to gradient accumulation in transformer training.

**(c) Dropout-style regularization on states.** Randomly mask a small fraction of states during each EM iteration (force their posterior to zero) to prevent state collapse, where two or more states converge to near-identical $(\mu_k, \sigma_k)$. This is the closest analog to dropout for a mixture-style model. An alternative is an entropy regularizer on the stationary distribution: penalize $-\sum_k \pi_k \log \pi_k$ below some threshold. State collapse at $K = 18$ is a real and observed failure mode, worth preventing.

**(d) Early stopping on validation log-likelihood.** Hold out the most recent $V$ observations (e.g. last 6 months) from EM training and monitor held-out log-likelihood at each iteration. Stop when it stops improving. Prevents overfitting to the training window and gives a free generalization signal. This is standard practice for deep-learning training but is not standard for HMM EM.

**(e) Fine-tuning / pre-training.** Pre-train one CHMM on a large pooled equity-returns corpus (all S&P constituents, 30 years). Fine-tune per-ticker for few EM iterations. Directly analogous to BERT pre-train then fine-tune. Likely improves per-ticker fit for short-history or illiquid tickers where a from-scratch CHMM has insufficient data.

**(f) Distillation.** Train a small-$K$ CHMM (e.g. $K = 6$) to match the one-step predictive distribution of a large-$K$ CHMM. For deployment in a latency-sensitive trading setting, this gives a cheap student model with teacher-quality behavior. Direct analog of DistilBERT.

**(g) Ensemble / bagging.** Fit $M$ CHMMs with different random initializations on bootstrap resamples of the training data, average their simulated paths. Classical bagging, often underused in HMM work. Provides a cheap uncertainty signal for downstream risk use.

**(h) Regime-change detection as a refit trigger.** Monitor the KL divergence between the current filtered state distribution and the stationary distribution, or the log-likelihood of the last $N$ observations under the current $\hat\theta$. Trigger full refit only when drift exceeds a threshold. Analogous to concept-drift detection in streaming classification. Allows the system to spend compute only when regimes actually change.

**(i) KL-warmup for Bayesian variants.** If the paper were extended to a variational Bayesian CHMM with Dirichlet priors on transition rows and Normal-Inverse-Wishart priors on emissions, a $\beta$-VAE-style KL warmup schedule would balance prior regularization against likelihood fit. Useful for short-history tickers.

### 5.4 What to report in the paper

A one-section "Deployment and Online Extension" would, I believe, move this paper from Weak Accept to Strong Accept. Concretely:

1. One figure showing OoS KS pass rate as a function of refit interval (daily, weekly, monthly, quarterly, never). Expected shape: monotone improvement as interval shrinks, with a knee somewhere.
2. One table showing per-refit wall-clock cost under each regime (full, warm-start, online-EM).
3. A brief theoretical note on which regime is appropriate under which operating assumption: stationary data (never refit), slow drift (warm-start), fast drift or non-stationarity (online EM).

### 5.5 Relation to trading-strategy evaluation

Beyond the abstract of this paper: for trading-strategy evaluation, the generator is being used to produce synthetic return paths for back-testing or RL training. The properties that matter are (a) the joint distribution of paths conditional on the current state (Bayesian filtering, not just generation), and (b) how quickly the generator adapts after a regime break. The current paper reports neither. An online CHMM deployed with Capp\'e-Moulines updates would support both by construction.

## 6. Minor Points

### 6.1 Figure numbering
I see two figures in the icaif version (Fig 1 Stylized Facts, Fig 2 Cross-Asset Correlation). The Gemini review flagged a "Figure 3" header above the Figure 1 caption which I do not reproduce in the current Paper_v8-icaif.tex. Worth double-checking the current compiled PDF against the Gemini report since this may have been fixed already.

### 6.2 Notation
Equation 1 defines $T_{ij} = P(s_{t+1} = j \mid s_t = i)$. $T$ also denotes the length of the return series ($T = 2{,}516$). Two meanings for one symbol. Use $P$ or $A$ for the transition matrix.

### 6.3 "$1/\Delta t$" annualization factor
The $(1/\Delta t)$ factor with $\Delta t = 1/252$ multiplies daily log returns by 252. Convention check: this rescales the marginal variance by a factor of $252^2 = 63{,}504$. Every statistic that reads variance in natural units is on an annualized-squared scale. State this once explicitly, since readers more familiar with annualized volatility (factor $\sqrt{252}$, not 252) will find the 7.68 excess kurtosis surprising.

### 6.4 Golden-section bracket
The bracket $[2.5, 100]$ for $\nu_k$ has a lower endpoint above 2, below which Student-t variance is undefined. Good. The upper endpoint at 100 is effectively Gaussian. If any $\hat\nu_k$ hits the 100 boundary in practice, it signals that the Student-t family is redundant for that state and the CHMM-t result is driven by a small number of genuinely heavy-tailed states. Report the histogram of $\hat\nu_k$ to surface this.

### 6.5 The missing GRU baseline
As noted in 4.7, the abstract mentions deep-generative baselines but Table 1 in the conference version does not. Either include or remove.

## 7. Verdict

- **Novelty:** Moderate. The paper's framing as "first continuous HMM for equity returns" is not defensible; the framing as "three-family emission ablation plus jump-free baseline comparison plus rank-reordering copula composition" is defensible and useful.
- **Rigor:** Moderate. Evaluation suite is good; missing confidence intervals; missing computational cost; missing online / streaming comparison the data itself calls for.
- **Relevance:** High.
- **Clarity:** High. The Pipeline A / Pipeline B split and the tables are clear.
- **Reproducibility:** The paper cites a model codebase implicitly. Anonymous GitHub link in a final-version appendix would address this.

**Decision: Weak Accept, revisions requested.** Priority revisions: (1) novelty positioning (half-paragraph); (2) confidence intervals on pass rates (one table column); (3) online / rolling-window section (new subsection); (4) GRU row in Table 1 or abstract edit.

Without (3), the paper tells half of an important story. With (3), it tells a complete one, and I would upgrade to Strong Accept.

## 8. Reviewer Notes Out of Scope

### 8.1 Relation to the Gemini review
Gemini recommended Strong Accept. I recommend Weak Accept pending the online-learning section. We agree on the core strengths (pipeline separation, copula ordering, discrete-HMM comparison) and on three of four weaknesses (copula scaling, downstream utility, computational cost). We disagree in emphasis on two points:

- Gemini treats the walk-forward result as "consider weaving this into the core methodology." I treat it as: this is the second headline of the paper and should be a full section, because the data itself (OoS degradation on NVDA and JPM) makes the stationarity assumption untenable for any practitioner audience.
- Gemini does not flag the novelty positioning against continuous-state financial HMM literature. I think this is the single most likely reason the paper would be rejected by a well-read ICAIF reviewer.

### 8.2 Suggested citations to add
For an online / rolling-window extension:
- Capp\'e, O. (2011). Online EM Algorithm for Hidden Markov Models. JCGS.
- Capp\'e, O., and Moulines, E. (2009). On-line expectation-maximization algorithm for latent data models. JRSS-B.
- Delyon, B., Lavielle, M., and Moulines, E. (1999). Convergence of a stochastic approximation version of the EM algorithm. Ann. Statist.
- Nystrup, P., Madsen, H., and Lindstr\"om, E. (2017, 2018). Regime-based versus stationary HMMs for financial returns.

For novelty positioning:
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series. Econometrica.
- Bulla, J., and Bulla, I. (2006). Stylized facts of financial time series and hidden semi-Markov models. CSDA.
- Rossi, A., and Gallo, G. M. (2006). Volatility estimation via hidden Markov models. J. Empir. Finance.

## 9. TL;DR for the Author

Three concrete asks if you revise for ICAIF:

1. Add one paragraph in Section 1 that explicitly cites the pre-existing continuous-HMM financial literature and states what is new here (per-state $\nu_k$ ECM, weighted-MAD Laplace, jump-free comparison, Pipeline framing). Do not leave the reader to infer this.

2. Add confidence intervals (Wilson or Clopper-Pearson) to every pass rate in Tables 1-3. Most of the intra-CHMM ranking claims depend on whether these intervals overlap.

3. Write Section 5 of this review (Online / Streaming CHMM) as a new subsection of the paper. Start with warm-start EM as the minimal ask (cheap), include one table of OoS KS versus refit interval (must-have), and gesture at online EM as a future direction (mentions only). This lifts the paper from "useful univariate fit demonstration" to "deployable generator with a defensible update cadence."

Item 3 is the biggest lever. The paper already contains the seed (JPM walk-forward, 15-percentage-point recovery). It just needs to be presented as a methodological commitment rather than a diagnostic remark.
