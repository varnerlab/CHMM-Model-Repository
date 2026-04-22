# Decision Memo: Can `CHMM-Model` + `CHMM-paper` be pushed from v9 "publishable" to "top-tier synthetic-data generator paper"?

_Prepared 2026-04-22, after a full review of `CHMM-Model` source, `CHMM-paper/paper/Paper_v9.tex`, the sibling `CHMM-Vol-Model` / `CHMM-Vol-Paper` repos, and the current literature on synthetic financial time-series generators (QuantGAN, TimeGAN, Sig-WGAN, TimeGrad, neural SDE, vine copulas)._

This memo is the equity-paper companion to `CHMM-Vol-Model/DECISION-MEMO.md`. The upgrade plan that implements the OPEN items below lives in `plan-equity-paper.md` (sibling file)._

---

## Status dashboard (completion tracking)

### Track 0: what already ships in v9 (DONE baseline)

| # | Work item                                                                 | Status |
|---|---------------------------------------------------------------------------|--------|
| 0.1 | CHMM-N / CHMM-t / CHMM-L continuous emissions with per-state ECM on $\nu_k$ | DONE |
| 0.2 | Discrete HMM + Poisson-jump baseline (from `alswaidan2026hybrid`)          | DONE |
| 0.3 | GARCH(1,1) MLE + simulation                                                | DONE |
| 0.4 | GRU + Gaussian head deep-generative baseline                               | DONE |
| 0.5 | i.i.d. bootstrap + stationary block bootstrap nonparametric baselines      | DONE |
| 0.6 | Stylized-fact panel: heavy tails, ACF($r$), ACF($\lvert r\rvert$), Jarque-Bera, Ljung-Box | DONE |
| 0.7 | 7-metric evaluation: KS %, AD %, kurtosis, ACF-MAE, $W_1$, Hellinger, quantile coverage, IS + OoS | DONE |
| 0.8 | VaR / ES utility back-test with 1000-path envelopes                        | DONE |
| 0.9 | Cross-asset: SIM + Gaussian copula + Student-t copula (6 tickers)          | DONE |
| 0.10 | Walk-forward Viterbi regime decoding (rolling 252-step)                   | DONE |
| 0.11 | Price-fan / multi-horizon distribution figures (1d/5d/10d aggregation)    | DONE |
| 0.12 | Paper v9 LaTeX draft with 3-author block, results + discussion + refs     | DONE |
| 0.13 | Rydén (1998) refutation framing (moderate $K$ closes ACF gap)             | DONE |
| 0.14 | BIC/CAIC + multi-metric score for state selection ($K=18$)                | DONE |

### Track A: evaluation rigor (OPEN, highest-ROI first)

| # | Work item                                                                   | Status | Notes |
|---|-----------------------------------------------------------------------------|--------|-------|
| A1 | MMD with RBF + signature kernels (Gretton, Chevyrev-Kormilitzin)           | OPEN   | New `src/Metrics.jl`; wire into `run_baselines_and_cross_asset.jl` |
| A2 | Signature $W_1$ / sig-MMD on truncated path signatures (depth 3 to 5)      | OPEN   | Port or call `iisignature` via `PyCall` |
| A3 | Discriminator AUC (GRU/transformer real-vs-synth, 5-fold CV)               | OPEN   | New `src/Discriminator.jl`; target AUC 0.5 |
| A4 | Train-on-Synthetic Test-on-Real (TSTR) for HAR vol forecaster              | OPEN   | RMSE / QLIKE on 2024 to 2026 OoS |
| A5 | Strategy back-test utility: vol-target + momentum on synth vs real          | OPEN   | Sharpe, max-DD, turnover gap |
| A6 | Leverage-effect metric: $\text{corr}(r_t^2, r_{t-k})$ asymmetry test        | OPEN   | One table row, IS and OoS |
| A7 | Aggregational Gaussianity: kurtosis of 1d/5d/10d/21d aggregated returns    | OPEN   | Standard Cont (2001) stylized fact #10 |
| A8 | Christoffersen + Kupiec LR tests on the existing VaR/ES output              | OPEN   | Ports cleanly from `CHMM-Vol-Model` §6 |
| A9 | Simulation-based p-values on each stylized-fact cell (matches vol paper N) | OPEN   | Joint-coverage summary $\bar{pv}$ |
| A10 | Bump simulation counts to 1000 paths per cell (matches vol paper O)        | OPEN   | Currently mixed; unify at 1000 |

### Track B: one serious deep-generative baseline (OPEN)

| # | Work item                                                                   | Status | Notes |
|---|-----------------------------------------------------------------------------|--------|-------|
| B1 | QuantGAN (TCN + Wasserstein GAN) on standardized log returns               | OPEN   | Python via `PyCall`, or rewrite in Flux.jl |
| B2 | Sig-WGAN (signature-Wasserstein) as path-level competitor                   | OPEN   | One row in Table 4 |
| B3 | Score-based time-series diffusion (TimeGrad-style, depth-2 conditional UNet) | OPEN | One row; acknowledge strong marginals, weaker long-horizon ACF |
| B4 | MS-GARCH (Haas-Mittnik-Paolella 2004) as variance-regime baseline          | OPEN   | Strengthens the discussion, not mandatory |

### Track C: model upgrades that change the paper title (OPEN, priority-ordered)

| # | Work item                                                                   | Status | Notes |
|---|-----------------------------------------------------------------------------|--------|-------|
| C1 | Semi-Markov CHMM with heavy-tailed sojourns (port Yu 2010 FB from vol repo) | OPEN   | Biggest scientific upgrade available; see §4 below |
| C2 | Vine copula cross-asset extension (C-vine or D-vine, Aas 2009) on 50 to 100 assets | OPEN | Scales beyond the current 6 |
| C3 | Time-varying transition matrix $T_{ij}(t)$ via logistic-regression on VIX / realized vol / term spread | OPEN | Already in v9 future-work list |
| C4 | Leverage-effect emission $r_t = \mu_k + \rho_k r_{t-1}^- + \sigma_k \epsilon_t$ (single ablation row) | OPEN | Low-effort, captures one missing fact |

### Track D: nice-to-have (OPEN, skip if time-bound)

| # | Work item                                                                   | Status | Notes |
|---|-----------------------------------------------------------------------------|--------|-------|
| D1 | Skew-t or NIG emissions as a fourth family                                  | OPEN   | Currently deferred |
| D2 | Promote Turing-based Bayesian $\nu_k$ posterior from diagnostic to main     | OPEN   | Already coded, just underused |
| D3 | Identifiability / label-switching appendix (1 page)                         | OPEN   | Matches vol-paper item K |
| D4 | Option pricing / implied-vol surface calibration                            | OUT    | Explicitly deferred in v9 conclusion |

---

## 1. Verdict in one sentence

**Paper v9 is submittable as-is to a mid-tier venue, but it is 60 to 70 percent of the way to a top-tier synthetic-data-generator paper.** The gap is *evaluation breadth* (no MMD, no signature-kernel, no discriminator AUC, no TSTR), *baseline depth* (no serious GAN or diffusion row), and *downstream utility* (only VaR, no strategy back-test). Closing the A-track items and one carefully-chosen B-track baseline is sufficient to move the paper to ACM ICAIF / Journal of Financial Data Science / TMLR. C1 (semi-Markov port from the vol repo) earns a new title and a stronger contribution story; C2 earns the scalable multi-asset headline.

## 2. What v9 already does well

- **Three continuous emission families** (CHMM-N / -t / -L) with a shared EM scaffold and per-state $\nu_k$ via golden-section ECM. This is genuinely novel; the literature treats the Gaussian case almost exclusively.
- **Seven-metric evaluation panel** run IS and OoS, with coverage-envelope style reporting: KS %, AD %, kurtosis, ACF-MAE, $W_1$, Hellinger, quantile coverage.
- **Proper nonparametric baselines**: i.i.d. bootstrap, stationary block bootstrap, which most deep-generative papers skip and regret.
- **Cross-asset dependence**: rank-reordering through Gaussian / Student-t copulas preserves marginals exactly. Table T3 shows the Student-t copula beats SIM on correlation MAE (0.027 vs 0.076).
- **Rydén refutation**: frames the 25-year-old negative result ("Gaussian HMMs cannot produce slow $\lvert r\rvert$ ACF decay") as resolved by moderate $K$ plus quantile initialization. This is an independently publishable insight.
- **Reproducibility**: MIT-licensed, Julia $\geq 1.12$, pinned Manifest.toml, figures regenerated via `run_all_analysis.jl`.

## 3. What the current literature bar requires that v9 does not yet have

### 3.1 Metrics

The community has converged on a small metric set for generator papers since ~2020:

| Metric | v9 status | Source |
|---|---|---|
| KS / AD / Hellinger / $W_1$ on marginals | DONE | Standard |
| ACF-MAE on returns and on $\lvert r\rvert$ | DONE | Cont (2001) |
| **MMD (RBF kernel)** | OPEN | Gretton et al. (2012) |
| **Signature kernel $W_1$ / signature MMD** | OPEN | Chevyrev & Kormilitzin (2016), Ni et al. (2020) |
| **Discriminator AUC** | OPEN | Yoon et al. (2019, TimeGAN); standard since |
| **TSTR predictive utility** | OPEN | Yoon et al. (2019) |
| Quantile coverage / CRPS | DONE | Gneiting & Raftery (2007) |
| Kupiec + Christoffersen VaR LR | OPEN | Christoffersen (1998); already in vol repo |

v9 ships the top block cleanly. The bottom block is what a top-venue reviewer expects.

### 3.2 Baselines

Current SOTA synthetic financial time series generators the paper should name and beat (or at least report):

- QuantGAN (Wiese, Knobloch, Korn, Kretschmer 2020)
- TimeGAN (Yoon, Jarrett, van der Schaar 2019)
- Fin-GAN / Sig-WGAN (Ni, Szpruch, Oberhauser et al. 2020)
- Neural SDE / Sig-Wasserstein GAN (Kidger, Foster, Li, Lyons 2021)
- Score-based time-series diffusion: TimeGrad (Rasul et al. 2021), CSDI (Tashiro et al. 2021), SSSD (Alcaraz & Strodthoff 2022)
- Deep Hedging generator path (Bühler, Gonon, Teichmann, Wood 2019) for utility framing
- MS-GARCH (Haas, Mittnik, Paolella 2004) for variance-regime comparison

v9 only ships a single-layer GRU as a deep baseline. That is not defensible for a top-tier generator paper. One serious GAN and one serious diffusion row is enough; do not run five GANs badly.

### 3.3 Downstream utility

v9's VaR/ES section is good but thin. Reviewers at top venues now expect at least one of:

- Trading-strategy back-test on synthetic paths (Sharpe, max-DD, turnover)
- Train-on-synth Test-on-real predictive task (vol forecasting is the standard)
- Option pricing / implied-vol calibration
- Hedging error on synthetic-calibrated portfolios

The vol paper's short-VIX-futures VaR with Christoffersen LR = 0.03 is exactly the kind of downstream claim the equity paper is missing. Mirror it with a short-SPY or vol-target equity back-test (A5).

## 4. Reuse from `CHMM-Vol-Model` (what to port, what to leave)

Cross-referenced against `CHMM-Vol-Model/src/Compute.jl`:

| Component | Vol-repo location | Equity-repo target | Effort | Value |
|---|---|---|---|---|
| Yu (2010) explicit-duration forward-backward | `Compute.jl:1821-1961` | New `MySemiMarkovContinuousHMM` in `src/Types.jl`, `baum_welch_sm_chmm` in `src/Compute.jl` | Medium | HIGH, enables C1 |
| `_pick_sojourn_family` (NB vs Pareto vs Weibull) | `Compute.jl:1440-1449` | Same | Low | Needed with C1 |
| Sojourn tagging on the model type | `Types.jl:143-194` | Copy shape as `MySemiMarkovContinuousHMM` | Low | Needed with C1 |
| `_runs` run-length encoder | `Compute.jl:1457-1471` | Copy | Trivial | Trivial |
| VaR + Christoffersen + Kupiec LR tests | Vol paper §6, `run_vix_short_vol_var.jl` | Augment `run_diagnostics.jl` VaR block | Low | HIGH for A8 |
| AR(1) residual fit `_fit_ar1_residual_state` | `Compute.jl:1488-1524` | Optional; see §5 of this memo | Low | LOW for equity returns |
| Student-t ECM | already shared | N/A | N/A | N/A |
| Long-memory GPH machinery | vol repo `run_vix_taqqu_verification.jl` | DO NOT PORT | N/A | Not applicable (returns are short-memory) |
| VIX data pipeline | vol repo `Files.jl`, `data/VIX-*.jld2` | DO NOT PORT | N/A | Wrong asset |

**One-line takeaway:** port Yu 2010 + sojourn-family selection (enables C1), port Christoffersen + Kupiec (sharpens A8), leave the long-memory and VIX-specific machinery where it is.

## 5. AR(p) integration: does it have merit for the equity paper?

**Short answer: not on raw returns. Yes on a narrow sub-component if the budget allows.**

Long answer:

- On $r_t$ directly the vol paper's AR(1) works because log-VIX has $\hat{\phi} \approx 0.978$ (half-life 34 days). Equity log returns are empirically white noise ($\hat{\phi} \approx 0.02$); AR(1) on $r_t$ will move none of the current metrics. Adding it would look like methodology-for-its-own-sake and invite the reviewer question "what does this buy you?".
- **Where AR does help for equities**: (i) a Markov-switching GARCH (Haas-Mittnik-Paolella 2004) puts AR(1) on $\sigma_t^2$ inside each regime, which would materially change ACF-MAE on $\lvert r_t\rvert$. That is the right AR story for equity returns but it is a different paper direction. (ii) A leverage-emission ablation $r_t = \mu_k + \rho_k r_{t-1}^- + \sigma_k \epsilon_t$ with $r_{t-1}^- = \min(r_{t-1}, 0)$ captures asymmetric vol response in a single-row change (item C4).

**Recommendation.** Do NOT add AR(p) on returns to v10. Spend that budget on C1 (semi-Markov sojourns, which dominates AR(1) for persistence fidelity anyway) plus the C4 leverage-emission single-ablation row. If a later paper wants an AR direction it should be either in the vol framing (`CHMM-Vol-Paper` already does this well) or a dedicated MS-GARCH-in-HMM follow-up.

## 6. Journal strategy (no Elsevier, ACM / IEEE acceptable)

Ranked by fit to the *synthetic-data-generator* framing.

**Tier 1 (target these):**
1. **ACM ICAIF** (International Conference on AI in Finance). ACM proceedings, exactly this readership, fast turnaround. This paper *is* an ICAIF paper.
2. **Journal of Financial Data Science** (Portfolio Management Research). High prestige in quant, non-Elsevier, values reproducible generator work.
3. **TMLR** (Transactions on Machine Learning Research). Rolling, rigorous, open access; good home for a methods-plus-strong-empirics paper.

**Tier 2 (viable):**
4. ACM Transactions on Modeling and Computer Simulation (TOMACS).
5. Quantitative Finance (Taylor & Francis; not Elsevier).
6. Digital Finance (Springer).
7. IEEE Transactions on Neural Networks and Learning Systems (if B-track baselines land seriously).
8. Risks (MDPI, fast open-access).

**Timestamping via workshops, in parallel with tier 1:** NeurIPS ML for Finance, ICML Time Series, ICAIF side workshops.

**Avoid:** Journal of Econometrics, Journal of Banking and Finance, Finance Research Letters, Pattern Recognition, Expert Systems with Applications (all Elsevier).

## 7. Path forward

The minimum viable upgrade plan for a top-tier submission is **Track A + one B-track row**:

1. A1 + A2 (MMD + signature kernel) as new columns in Table 4.
2. A3 (discriminator AUC) as a single numeric ("AUC = 0.54") in Results.
3. A4 or A5 (TSTR or strategy back-test) as the new "Downstream Utility" subsection, replacing the VaR-only framing.
4. A6 + A7 (leverage + aggregational Gaussianity) as two added rows to the stylized-fact panel.
5. A8 (Kupiec + Christoffersen on existing VaR) as a paragraph in the existing utility section.
6. B1 (QuantGAN) as one row in Table 4. If B3 (TimeGrad-style diffusion) also lands, even better, but B1 is the load-bearing deep baseline reviewers will ask about.

The stretch plan adds **C1 (semi-Markov sojourns)** and re-pitches the paper title as "A semi-Markov regime-switching continuous HMM for equity return synthesis". That earns the paper a narrative upgrade (not just a stylized-facts panel but a structural model), and it ports cleanly from the vol repo since the Yu 2010 machinery is already proven there.

**C2 (vine copula) is the headline multi-asset upgrade** and the main reason a reviewer would pick this paper over any other HMM-based generator: it is the only generator family that scales to 100 assets while keeping exact marginals and interpretable regimes. If only one C-track item lands, choose C1 or C2 based on venue: C1 for methods venues (TMLR, ICAIF methods track), C2 for finance venues (JFDS, Quantitative Finance).

## 8. One-line answer to "can this become top-tier?"

**Yes, within one focused track of work.** Track A alone moves the paper from mid-tier to ICAIF/JFDS quality. Track A + one B row + C1 or C2 makes it a genuinely strong submission to the best non-Elsevier generator venues. Nothing here requires a fundamental rewrite; the v9 scaffold already carries the load, and the missing pieces are standard additions that the vol repo has already proven feasible in Julia at the same scale.

## 9. Related repos

- `CHMM-Vol-Model` and `CHMM-Vol-Paper` (sibling): VIX-only, already draft-complete. Source of Yu 2010 FB, Christoffersen LR, sojourn-family sweep machinery to port.
- `CHMM-paper`: v9 LaTeX draft.
- `CHMM-icaif-Model` / `CHMM-icaif-paper`: older scoped-down ICAIF fork. Not the main vehicle; v9 is.
- `online-CHMM`: prospective online-EM extension; out of scope for this memo.
