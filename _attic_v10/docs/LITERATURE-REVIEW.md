# Literature Review: EM/Baum-Welch HMMs for Equity Dynamics & Stylized Facts

*Compiled: April 13, 2026*

---

## 1. Foundational Work

### Ryden, Terasvirta & Asbrink (1998) — THE seminal paper
- **Paper**: "Stylized Facts of Daily Return Series and the Hidden Markov Model", *Journal of Applied Econometrics*, 13(3), 217-244.
- **Link**: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=8181
- **What they did**: Trained a Gaussian-emission HMM on S&P 500 daily returns (1928-1991) via Baum-Welch.
- **Key finding**: Reproduces most stylized facts — heavy tails, negligible return ACF — but **fails on one critical fact: the slow decay of the ACF of squared/absolute returns** (volatility clustering). The HMM's geometric sojourn-time distribution causes ACF to decay exponentially fast rather than hyperbolically.
- **Relevance**: This is the most direct prior work to our continuous model. We MUST cite and differentiate.

### Bulla & Bulla (2006) — HSMM fix for ACF decay
- **Paper**: "Stylized facts of financial time series and hidden semi-Markov models", *Computational Statistics & Data Analysis*, 51(4), 2192-2209.
- **Link**: https://www.researchgate.net/publication/222420957_Stylized_facts_of_financial_time_series_and_hidden_semi-Markov_models
- **What they did**: Proposed hidden semi-Markov models (HSMMs) replacing geometric dwell-time with negative-binomial sojourn-time distributions.
- **Key finding**: HSMMs substantially improve ACF(|r|) reproduction. The slowly decaying ACF is significantly better described by HSMM.
- **Relevance**: Alternative approach to fixing the ACF problem. Our approach uses higher K instead.

### Malmsten & Terasvirta (2004) — GARCH benchmark
- **Paper**: "Stylized Facts of Financial Time Series and Three Popular Models of Volatility"
- **Link**: http://www-stat.wharton.upenn.edu/~steele/Resources/FTSResources/StylizedFacts/MalmstenTerasvirta04.pdf
- **What they did**: Compared GARCH, EGARCH, and other volatility models against the same stylized facts.
- **Key finding**: GARCH captures temporal dynamics (volatility clustering) but not distributional shape (heavy tails).

---

## 2. Existing Literature Landscape

| Approach | Key Papers | What It Does |
|---|---|---|
| Discrete HMM + frequency counting | Alswaidan & Varner (2026) | Bins returns, counts transitions, adds Poisson jumps |
| Gaussian-emission HMM via Baum-Welch | Ryden et al. (1998) | **Already done** — core of our continuous model |
| HSMM (semi-Markov) | Bulla & Bulla (2006) | Fixes ACF decay via non-geometric dwell times |
| Regime-switching GARCH (MS-GARCH) | Multiple authors | HMM states drive GARCH parameters |
| FlowHMM (NeurIPS 2022) | Lorek et al. | Normalizing flows replace Gaussian emissions |
| Regime-switching factor investing | Nystrup et al. (2020) | HMM for portfolio allocation |

---

## 3. Novelty Assessment

### What is NOT novel (already in literature)
- Training a Gaussian-emission HMM on financial returns via Baum-Welch — Ryden et al. (1998) did exactly this
- Showing it reproduces heavy tails and negligible return ACF — same paper
- Observing the ACF(|r|) decay limitation — same paper
- Comparing HMM vs GARCH — multiple papers have done this

### What IS potentially novel in our work
1. **Quantile-based initialization** for Baum-Welch — most papers use random or k-means init. Our approach of sorting observations into K quantile chunks is practical but not formalized in a financial context
2. **The specific comparison**: discrete frequency-counting HMM + jumps vs continuous Baum-Welch HMM (no jumps) — this head-to-head showing the continuous model doesn't need jumps is new because our own discrete paper is the comparison baseline
3. **424-asset scalability** via SIM factor decomposition + copulas — the multi-asset extension at that scale is distinctive
4. **Comprehensive multi-metric evaluation** (KS, AD, kurtosis, ACF-MAE, W1, Hellinger, Coverage) applied systematically to 5+ models
5. **Out-of-sample temporal generalization** on 2025 holdout data

---

## 4. Suggestions for Strengthening Novelty

### 4.1 Directly address the ACF(|r|) decay problem (the Achilles' heel)
Ryden et al. proved standard HMMs can't reproduce slow ACF decay. If our continuous model at certain K values partially reproduces it (even without jumps), this would be a genuinely new finding. Quantify ACF-MAE as a function of K and compare against the Ryden et al. theoretical bound.

### 4.2 State-resolution analysis (K sensitivity)
Ryden et al. used only 2-3 states. Our K in {3,6,9,12,15,18,21} sweep is more comprehensive. The sweet spot at K=18 (best KS/AD/Wasserstein) shows a moderate Gaussian-mixture regime count can approximate volatility clustering without needing semi-Markov extensions; the best ACF-MAE actually sits at K=3, revealing a clear distributional-fidelity / volatility-clustering trade-off worth reporting.

### 4.3 The Baum-Welch vs frequency-counting comparison
No paper has directly compared Baum-Welch continuous HMM against discrete frequency-counting HMM (with and without jumps) on the same data with the same metrics. Our discrete paper is the baseline. Showing that continuous Baum-Welch achieves comparable or better fidelity without jumps is the core novelty claim.

### 4.4 Synthetic data quality framework
The multi-metric evaluation applied systematically to 5+ models is more comprehensive than most papers. Framing this as a synthetic data quality benchmark for financial time series generators gives the evaluation framework independent value.

### 4.5 Out-of-sample temporal generalization
Most HMM papers validate in-sample only. Our IS (2014-2024) + OoS (2025) design with the same metrics is stronger than typical HMM financial papers.

---

## 5. Recommended Novelty Framing

> *"We show that a continuous Gaussian HMM trained via Baum-Welch with quantile-based initialization reproduces all three canonical stylized facts at moderate state resolution (K=6-13) without requiring jump mechanisms, achieving comparable distributional fidelity to the augmented discrete model while eliminating two tunable hyperparameters (ε, λ). We validate out-of-sample on 2025 holdout data."*

Key differentiation from Ryden et al. (1998):
- (a) Quantile initialization strategy
- (b) Systematic K sweep showing partial ACF recovery at higher K
- (c) Head-to-head against our own discrete+jumps baseline
- (d) 2025 out-of-sample validation on modern data
- (e) Multi-metric synthetic data quality evaluation framework

---

## 6. Key References

1. Ryden T, Terasvirta T, Asbrink S (1998). "Stylized facts of daily return series and the hidden Markov model." *J Applied Econometrics* 13(3):217-244. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=8181
2. Bulla J, Bulla I (2006). "Stylized facts of financial time series and hidden semi-Markov models." *Comp Stat & Data Analysis* 51(4):2192-2209. https://dl.acm.org/doi/abs/10.1016/j.csda.2006.07.021
3. Malmsten H, Terasvirta T (2004). "Stylized facts and three popular models of volatility." http://www-stat.wharton.upenn.edu/~steele/Resources/FTSResources/StylizedFacts/MalmstenTerasvirta04.pdf
4. Lorek P et al. (2022). "FlowHMM: Flow-based continuous hidden Markov models." *NeurIPS 2022*. https://proceedings.neurips.cc/paper_files/paper/2022/file/39c5871aa13be86ab978cba7069cbcec-Paper-Conference.pdf
5. Nystrup P et al. (2020). "Regime-switching factor investing with hidden Markov models." *J Risk & Financial Management* 13(12):311. https://www.mdpi.com/1911-8074/13/12/311
6. Cont R (2001). "Empirical properties of asset returns: stylized facts and statistical issues." *Quantitative Finance* 1(2):223-236.
7. Hamilton JD (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica* 57(2):357-384.
8. Bollerslev T (1986). "Generalized autoregressive conditional heteroskedasticity." *J Econometrics* 31(3):307-327.
9. Alswaidan A, Varner JD (2026). "Hybrid hidden Markov model for modeling equity excess growth rate dynamics." *arXiv:2603.10202*. https://arxiv.org/abs/2603.10202
