# Plan: CHMM-Model + CHMM-paper, upgrade from v9 "publishable" to "top-tier synthetic-data generator paper"

**Status:** v0, 2026-04-22. Companion to `CHMM-Vol-Model/plan-vol-paper.md` and `CHMM-Model/DECISION-MEMO.md` (sibling file in this repo).

**Goal:** move the equity-returns generator paper from mid-tier (v9 as-is is submittable to a specialist finance venue) to top-tier (ACM ICAIF, Journal of Financial Data Science, TMLR). The scaffold stays; what changes is *evaluation breadth*, *one serious deep baseline*, *downstream utility beyond VaR*, and (stretch) *semi-Markov sojourns ported from the vol repo*.

**Scope boundary carried over from v9:** no option pricing, no implied-vol surface calibration, no Hawkes process, no variational Bayes, no particle filters. The one new estimation kernel the stretch plan introduces is Yu (2010) explicit-duration forward-backward, ported directly from `CHMM-Vol-Model/src/Compute.jl:1821-1961`.

---

## Status dashboard (2026-04-22)

### Track 0, v9 baseline, all DONE

1. Continuous HMM with Gaussian / Student-t / Laplace emissions (`src/Compute.jl:278-710`).
2. Per-state $\nu_k$ ECM via golden-section on the Q-function (`src/Compute.jl:449-483`).
3. Discrete HMM + Poisson-jump baseline (`src/Types.jl:47-60`, `src/Compute.jl:33-71`).
4. GARCH(1,1) MLE + simulation (`src/Compute.jl:885-1037`).
5. GRU + Gaussian head baseline (`src/Compute.jl:1158-1326`).
6. i.i.d. bootstrap + Politis-Romano stationary block bootstrap.
7. Stylized-fact panel (`run_all_analysis.jl:66-151`): heavy tails, ACF, vol clustering, JB, LB.
8. 7-metric evaluation table, IS + OoS (KS %, AD %, kurtosis, ACF-MAE, $W_1$, Hellinger, quantile coverage).
9. VaR / ES utility back-test with 1000-path envelopes (`run_diagnostics.jl:186-290`).
10. Cross-asset: SIM + Gaussian copula + Student-t copula on 6 tickers (`src/CrossAsset.jl`).
11. Walk-forward Viterbi regime decoding (rolling 252-step).
12. Price-fan and multi-horizon distribution figures.
13. Paper v9 LaTeX draft (`CHMM-paper/paper/Paper_v9.tex`), 3-author block, no em dashes, Rydén refutation framing, $K=18$ chosen by BIC/CAIC + multi-metric score.

### Track A: evaluation rigor (OPEN)

| Item | Description | Artifact target | Status |
|---|---|---|---|
| A1 | MMD (RBF kernel) on windowed return paths | `src/Metrics.jl::mmd2`, new column in Table 4 | OPEN |
| A2 | Signature-kernel MMD and signature $W_1$ (depth 3 to 5) | `src/Metrics.jl::sig_mmd`, `sig_w1` | OPEN |
| A3 | Discriminator AUC (small GRU classifier, 5-fold CV, real vs synthetic) | `src/Discriminator.jl::disc_auc` | OPEN |
| A4 | TSTR vol-forecaster: train HAR-RV (or small GRU) on synthetic, evaluate RMSE + QLIKE on real OoS | `run_tstr.jl` | OPEN |
| A5 | Strategy back-test: vol-target SPY with $\hat{\sigma}_t$ from synth-trained vs real-trained | `run_strategy_backtest.jl` | OPEN |
| A6 | Leverage-effect metric: $\text{corr}(r_t^2, r_{t-k})$ for $k=1..20$, asymmetry test | New row in Table 4 | OPEN |
| A7 | Aggregational Gaussianity: kurtosis at 1d / 5d / 10d / 21d aggregation | New figure + row | OPEN |
| A8 | Kupiec + Christoffersen LR on existing VaR output (ports from vol repo) | Extend `run_diagnostics.jl:186-290` | OPEN |
| A9 | Simulation-based two-sided p-values on each stylized-fact cell, plus joint $\bar{pv}$ summary | Extends existing evaluation harness | OPEN |
| A10 | Unify simulation count to 1000 paths per cell (matches vol paper) | Cross-file constant audit | OPEN |

### Track B: one serious deep-generative baseline (OPEN)

| Item | Description | Status |
|---|---|---|
| B1 | **QuantGAN** (Wiese et al. 2020): TCN + Wasserstein-GP, on standardized log returns | OPEN |
| B2 | **Sig-WGAN** (Ni et al. 2020): signature-Wasserstein GAN, path-level | OPEN (optional) |
| B3 | **Time-series diffusion** (TimeGrad / SSSD style): conditional denoising on returns | OPEN |
| B4 | MS-GARCH (Haas-Mittnik-Paolella 2004): variance regime comparison | OPEN (optional) |

### Track C: model upgrades that rename the paper (OPEN, priority order)

| Item | Description | Status |
|---|---|---|
| C1 | **Semi-Markov continuous HMM**: port Yu 2010 FB + `_pick_sojourn_family` from vol repo; retrain SM-CHMM-N / -t / -L on SPY; ablate vs flat CHMM | OPEN |
| C2 | **Vine copula cross-asset**: C-vine or D-vine (Aas 2009) scaling to 50 to 100 tickers | OPEN |
| C3 | **Time-varying transition matrix**: $T_{ij}(t)$ via logistic regression on macro features (VIX, realized vol, term spread) | OPEN |
| C4 | **Leverage-effect emission ablation**: $r_t = \mu_k + \rho_k r_{t-1}^- + \sigma_k \epsilon_t$ single row | OPEN |

### Track D: nice-to-have (OPEN)

| Item | Description | Status |
|---|---|---|
| D1 | Skew-t or NIG emissions as a fourth family | OPEN |
| D2 | Promote Turing-based Bayesian $\nu_k$ posterior to main results | OPEN |
| D3 | Identifiability / label-switching appendix | OPEN |
| D4 | Option pricing / implied-vol surface calibration | OUT OF SCOPE |

---

## 1. Thesis

**Primary (committed):** *A regime-switching continuous HMM family serves as a reference synthetic-data generator for equity returns: it reproduces the full Cont (2001) stylized-fact set at moderate state resolution, delivers calibrated downstream utility on both risk (VaR) and strategy (vol-target) back-tests, and scales to cross-asset synthesis via rank-based copulas while preserving marginals exactly.*

This reframes v9's "three emissions reproduce stylized facts" into the synthetic-data-generator framing. The claim now has three legs:

1. **Stylized-fact fidelity**: KS %, AD %, $W_1$, Hellinger, ACF-MAE, kurtosis, leverage, aggregational Gaussianity (DONE + A6 + A7).
2. **Downstream utility**: Christoffersen-calibrated VaR, TSTR vol forecasting, vol-target strategy Sharpe gap (DONE + A4 + A5 + A8).
3. **Cross-asset scalability**: rank-based copula on 6 assets (DONE), upgraded to vine copula on 50 to 100 assets (C2).

**Fallback, if C1 ships:** *A semi-Markov regime-switching continuous HMM for equity synthesis, with heavy-tailed sojourn distributions that provide mean reversion (Markov inner chain) plus crisis-regime persistence (Pareto sojourn tail), ported from the CBOE-VIX model in the companion paper.*

**Rejected:** a deep-learning framing. The paper's strength is transparency, interpretability, and reproducibility; reviewers at ICAIF and JFDS value those over one more GAN variant. B-track baselines are negative-control rows, not claims.

## 2. Data

Unchanged from v9:

- Primary: SPY daily, 2014-01-03 to 2024-01-03 (IS, 2516 obs), 2024-01-04 to 2026-04-20 (OoS, 572 obs).
- Cross-asset: NVDA, JNJ, JPM, AAPL, QQQ at same dates.
- Returns convention: $G_t = (1/dt) \ln(P_t / P_{t-1}) - r_f$ with $dt = 1/252$, $r_f = 0$ (per `CLAUDE.md`).

**Upgrade for C2 only:** expand cross-asset universe to 50 to 100 liquid US equities (S&P 100 or QQQ constituents). Data pull via the forked `alpaca-markets-sdk` already used in `CHMM-Vol-Model` for 5-min RV.

## 3. Observable and transform

Unchanged: annualized excess log returns. No first-difference, no squared return as the modeled observable (those are derived metrics only).

## 4. Stylized-fact checklist

Scoring rubric. Status marks what v9 ships vs what Tracks A through C add.

| # | Fact | v9 | After A |
|---|---|---|---|
| 1 | Heavy tails (excess kurtosis > 3) | DONE | DONE |
| 2 | Negligible linear ACF of $r_t$ | DONE | DONE |
| 3 | Slow decay ACF of $\lvert r_t\rvert$ (volatility clustering) | DONE | DONE |
| 4 | Negative skew | DONE | DONE |
| 5 | Jarque-Bera rejection of normality | DONE | DONE |
| 6 | **Leverage effect**: $\text{corr}(r_t^2, r_{t-k}) < 0$ for small $k$ | MISSING | A6 |
| 7 | **Aggregational Gaussianity**: kurtosis declines toward 3 at longer aggregation | MISSING | A7 |
| 8 | Volume-volatility correlation | OUT (VWAP data available but not modeled) | OUT |
| 9 | Gain-loss asymmetry | PARTIAL (captured via skew) | PARTIAL |

Facts 1 to 5 plus 6 and 7 cover the canonical Cont (2001) set the literature scores generators against.

## 5. Model and methodological contribution

### 5.1 v9 baseline model family (committed)

$r_t \mid s_t = k \sim F_k$ with $F_k$ in $\{\mathcal{N}(\mu_k, \sigma_k^2), t_{\nu_k}(\mu_k, \sigma_k), \text{Laplace}(\mu_k, b_k)\}$, Markov regime $s_t$ with $K \times K$ transition matrix $T$, quantile-based initialization, Baum-Welch in log space.

Three variants: **CHMM-N**, **CHMM-t** (headline), **CHMM-L**.

### 5.2 Stretch upgrade C1: Semi-Markov continuous HMM (SM-CHMM)

Port from `CHMM-Vol-Model`:

- `MySemiMarkovContinuousHMM` type in `src/Types.jl` (shape copied from `CHMM-Vol-Model/src/Types.jl:143-194`).
- `baum_welch_sm_chmm` in `src/Compute.jl` (copy `_sm_forward`, `_sm_backward`, `_sm_estep` from `CHMM-Vol-Model/src/Compute.jl:1821-1961`).
- Per-state sojourn distribution $D_k$: negative-binomial for short / medium regimes, truncated discrete Pareto for the bottom-3 and top-3 tail states. Selection via `_pick_sojourn_family` on decoded run lengths.

**Why semi-Markov for equities, not just for vol:** the bottom-3 tail states (crashes) empirically persist longer than geometric sojourns predict (2008, 2020, 2022 stress episodes). A Pareto tail on sojourns for those states lets the generator produce the clustered-crash morphology that flat CHMM undershoots. Unlike the vol paper, we do *not* invoke Taqqu-Willinger-Sherman here: log returns are short-memory by construction, so the upgrade is about *persistence morphology*, not long-memory $d$.

**Why not AR(p) on returns:** see DECISION-MEMO §5. AR(1) on equity returns is empirically ~0; it moves no metric and would look like methodology-for-its-own-sake. C4 (leverage-emission ablation) is the one place AR-like structure adds real content.

### 5.3 Cross-asset upgrade C2: vine copula

Current (v9): fit a single Student-t copula $C(u_1, ..., u_d)$ with correlation from Kendall's $\tau$, $\nu$ by profile MLE. Rank-reorder per-asset CHMM paths to match the copula sample.

Upgrade: fit a C-vine or D-vine (Aas, Czado, Frigessi, Bakken 2009) via sequential pair-copula decomposition. Each bivariate link in the tree can be Gaussian, Student-t, Clayton, or rotated-Gumbel, selected by AIC per link. Tail asymmetry captured through Clayton lower-tail or rotated-Gumbel upper-tail links. Scales to $d = 100$ without the correlation matrix going near-singular.

**Complexity:** $O(d^2)$ pair-copula fits; negligible compared to the CHMM EM runs.

## 6. Baselines

Required rows in the final Table 4. **Bold** items are new for v10.

- i.i.d. bootstrap (DONE)
- Stationary block bootstrap ($b = 10$, DONE)
- Gaussian i.i.d. (DONE)
- Laplace i.i.d. (DONE)
- GARCH(1,1) (DONE)
- **MS-GARCH** (Haas-Mittnik-Paolella 2004) (B4, optional)
- Discrete HMM no-jump $K = 13$ (DONE, from `alswaidan2026hybrid`)
- Discrete HMM + Poisson jumps $K = 13$ (DONE, from `alswaidan2026hybrid`)
- Bin-T NJ (bin-conditional Student-t) (DONE)
- **CHMM-N / -t / -L** flat (DONE, headline rows)
- **SM-CHMM-N / -t / -L** (C1, ablation rows)
- GRU + Gaussian head (DONE)
- **QuantGAN** (B1, required)
- **Sig-WGAN** (B2, optional)
- **TimeGrad diffusion** (B3, optional but strong)

## 7. Evaluation protocol

Table 4 rows listed above; columns below. Bold columns are new for v10.

| Metric | v9 | After A |
|---|---|---|
| KS pass rate, IS + OoS | DONE | DONE |
| AD pass rate, IS + OoS | DONE | DONE |
| Kurtosis (simulated vs observed) | DONE | DONE |
| **Kurtosis ratio** (sim / obs) | MISSING | DONE (A matches vol paper item P) |
| ACF-MAE on $\lvert r_t\rvert$, lags 1 to 252 | DONE | DONE |
| Wasserstein-1 | DONE | DONE |
| Hellinger distance | DONE | DONE |
| Quantile coverage (90% band) | DONE | DONE |
| **MMD (RBF, $\sigma$ median-heuristic)** | MISSING | A1 |
| **Signature-kernel MMD (depth 3 to 5)** | MISSING | A2 |
| **Discriminator AUC** | MISSING | A3 |
| **Leverage $\text{corr}(r_t^2, r_{t-k})$** | MISSING | A6 |
| **Aggregational kurtosis (1d/5d/10d/21d)** | MISSING | A7 |
| **Simulation-based p-values + $\bar{pv}$** | MISSING | A9 |

Simulation count: 1000 paths per cell (A10).

Block-bootstrap 95 % CI on the observed row (already present in vol repo via `run_vix_main_results.jl`).

## 8. Downstream utility (new subsection, replaces v9 VaR-only framing)

Pick **two** of the following three (one risk, one strategy; drop option pricing entirely, v9 already does):

- **Risk (A8):** existing 1000-path VaR/ES with the Kupiec + Christoffersen LR tests ported from `CHMM-Vol-Model/run_vix_short_vol_var.jl`. Report breach rates and LR stats on IS + OoS and on stress sub-windows (2020 COVID, 2022 rate shock, 2023 to 2026 rest).
- **Predictive (A4):** TSTR. Train a HAR-RV forecaster (Corsi 2009) on *synthetic* SPY return paths, evaluate RMSE + QLIKE against *observed* SPY realized variance on the OoS window. A good generator should produce near-parity RMSE vs the real-trained HAR baseline.
- **Strategy (A5):** vol-target SPY. Position size at time $t$ is $w_t = \sigma_{\text{target}} / \hat{\sigma}_t$ where $\hat{\sigma}_t$ is a rolling-window vol estimate. Train the vol-estimation model on synthetic, apply to real OoS. Compare Sharpe, max-DD, turnover against the real-trained reference strategy.

Strategy (A5) is the one most likely to land the paper a trading-desk readership; TSTR (A4) is the more standard academic choice. Budget-permitting, do both.

## 9. Cross-asset section (expanded per C2)

**Pipeline A (DONE, single asset):** each of 6 tickers gets its own CHMM-t fit and simulated paths; metrics aggregated per-asset.

**Pipeline B v1 (DONE, 6 assets, flat copula):** Student-t copula on ranks, reorder per-asset paths. Correlation MAE = 0.027 in v9 (vs SIM 0.076).

**Pipeline B v2 (C2, 50 to 100 assets, vine copula):** C-vine or D-vine. Report correlation MAE, tail-dependence coefficient $\lambda_L, \lambda_U$ match, portfolio-level VaR on an equal-weight portfolio across the universe.

## 10. Paper structure (target 25 to 30 pages for ACM ICAIF long-paper or JFDS)

Working title (primary thesis): *"A Regime-Switching Continuous HMM as a Reference Synthetic-Data Generator for Equity Returns"*.

Fallback title (if C1 ships): *"Semi-Markov Continuous HMMs for Equity Return Synthesis"*.

| § | Section | Pages | Content |
|---|---|---|---|
| 1 | Introduction and thesis | 2 | Generator framing, Cont (2001) stylized-fact panel, v9 contribution summary, v10 upgrades (A + B + C) |
| 2 | Related work | 2 | Regime-switching (Hamilton 1989, Rydén 1998, Bulla 2006); heavy-tailed HMM (Peel-McLachlan 2000); GAN/diffusion generators (Wiese 2020, Yoon 2019, Ni 2020, Rasul 2021); copulas (Sklar 1959, Aas 2009); stylized facts (Cont 2001) |
| 3 | Data + stylized facts | 2 | SPY + 5 tickers; Fig 1 panel |
| 4 | Model: CHMM-N/-t/-L and (stretch) SM-CHMM | 4 | EM scaffold, per-state $\nu_k$ ECM, semi-Markov extension if C1 |
| 5 | Estimation | 2 | Baum-Welch in log space, quantile init, convergence; Yu 2010 FB if C1 |
| 6 | Baselines | 1 | One-paragraph description of each competitor row |
| 7 | Evaluation protocol | 2 | 7-metric panel + MMD + sig-kernel + discriminator AUC + leverage + aggregational Gaussianity + p-values |
| 8 | Single-asset results | 4 | Table 4, IS + OoS, SPY headline + 5 cross-asset tickers |
| 9 | Downstream utility | 3 | A8 VaR with Kupiec/Christoffersen; A4 TSTR OR A5 strategy (pick two) |
| 10 | Cross-asset results | 3 | Pipeline A + Pipeline B + (C2) vine copula scaling to 50 to 100 assets |
| 11 | Robustness | 2 | $K$-sweep, sojourn-family sweep if C1, sample-period splits, stress-window sub-metrics |
| 12 | Discussion and future work | 2 | MS-GARCH-HMM, Hawkes transitions, online EM (`online-CHMM`), option pricing |
| 13 | References | 1 | |

Total: ~28 pages. Core-minus-utility is 24 pages (drops §9); core-minus-cross-asset is 25 pages (keeps flat copula only). The 28-page target is comfortable for ICAIF long papers and JFDS.

## 11. Work order (timeline-agnostic, gate-based)

Gates mean "do not start the next item until this one passes".

1. **A1, A2** (MMD + signature kernel in `src/Metrics.jl`). *Gate:* new columns appear in Table 4, all existing rows have finite values.
2. **A3** (discriminator AUC). *Gate:* AUC reported for all existing rows; v9 CHMM rows should land near 0.50 to 0.55, GARCH row near 0.55 to 0.60, i.i.d. rows above 0.65.
3. **A6, A7** (leverage + aggregational Gaussianity). *Gate:* two new rows in stylized-fact panel. CHMM-t should match observed within bootstrap CI on both.
4. **A8** (Kupiec + Christoffersen on existing VaR). *Gate:* LR stats added to Table VaR, CHMM-t expected LR < 3.84 on IS (5 % crit value).
5. **A4 or A5** (downstream utility beyond VaR). *Gate:* one subsection drafted. If A4, synth-trained HAR within 10 % RMSE of real-trained. If A5, synth-trained vol-target within 0.3 Sharpe of real-trained on OoS.
6. **A9, A10** (simulation p-values + 1000-path standardization). *Gate:* $\bar{pv}$ column in Table 4, CHMM-t should land above 0.3 joint coverage.
7. **B1** (QuantGAN). *Gate:* one row in Table 4. Honest reporting, even if QuantGAN wins on marginals it should lose on long-horizon ACF and leverage.
8. **Stretch: C1** (semi-Markov port). *Gate:* SM-CHMM-t matches CHMM-t on marginal metrics, beats on ACF tail and on stress-window VaR calibration. If it does not, drop the semi-Markov framing from v10 and revisit.
9. **Stretch: C2** (vine copula) OR **B3** (diffusion). Pick one based on venue target (see §14).
10. **C4** (leverage-emission ablation). *Gate:* one row in Table 4; reviewer-defensible content even if marginal on metrics.
11. **Paper writeup v10.** Main text in `CHMM-paper/paper/Paper_v10.tex` (v9 becomes a frozen reference snapshot per user memory rule on version freezes).

## 12. Reuse map from `CHMM-Vol-Model`

Concrete ports, cross-referenced against existing vol repo paths.

| Component | Vol-repo source | Equity-repo target | Needed for |
|---|---|---|---|
| Yu 2010 explicit-duration FB (`_sm_forward`, `_sm_backward`, `_sm_estep`) | `CHMM-Vol-Model/src/Compute.jl:1821-1961` | New methods in `CHMM-Model/src/Compute.jl` | C1 |
| `MySemiMarkovCHMM` type shape | `CHMM-Vol-Model/src/Types.jl:143-194` | New `MySemiMarkovContinuousHMM` in `CHMM-Model/src/Types.jl` | C1 |
| `_pick_sojourn_family` | `CHMM-Vol-Model/src/Compute.jl:1440-1449` | Same | C1 |
| `_runs` run-length encoder | `CHMM-Vol-Model/src/Compute.jl:1457-1471` | Same | C1 |
| Kupiec + Christoffersen LR test | `CHMM-Vol-Model/run_vix_short_vol_var.jl` | Extend `CHMM-Model/run_diagnostics.jl:186-290` | A8 |
| Stationary bootstrap CI helper | `CHMM-Vol-Model/run_vix_main_results.jl::stationary_bootstrap_ci` | Already in equity repo; confirm parity | A10 |
| Student-t copula | shared already via `CHMM-Model/src/CrossAsset.jl` | N/A | N/A |
| GPH d / long-memory machinery | `CHMM-Vol-Model/run_vix_taqqu_verification.jl` | DO NOT PORT | N/A |

## 13. Risks and open questions

| Risk | Mitigation |
|---|---|
| Discriminator AUC comes out above 0.7 on CHMM-t | This means the model is distinguishable from real; honest reporting is the right move. Frame as "distinguishable on higher-order joint structure, indistinguishable on marginals + ACF". A top venue will accept this with a careful discussion. |
| QuantGAN row beats CHMM-t on marginal MMD | Expected and acceptable. Frame CHMM-t's advantage as interpretability + regime semantics + VaR calibration, not raw marginal fit. |
| Sig-kernel implementation adds a Python dependency | Acceptable as a `PyCall` import, matches the vol repo's approach to `iisignature`. Alternative: the `KernelSignatures.jl` Julia package if it meets numerical parity. |
| Vine copula (C2) non-trivial to implement in Julia | Use `VineCopulas.jl` or call R's `VineCopula` via `RCall` as a pragmatic first pass. Replace with pure-Julia later. |
| SM-CHMM port doesn't close any Table-4 metric beyond CHMM-t | Drop C1 from v10 and keep it as a follow-up. The v10 story is still strong on Track A + B1 alone. |
| Reviewer asks "why not neural SDE?" | Cite Kidger, Foster, Li, Lyons (2021) in related work; frame as complementary (black-box, universal approximator) vs CHMM (interpretable, regime-semantic). Neural SDE can be a B-track row if we have budget. |
| v10 upgrades push the paper past 30 pages | Drop §10 vine copula or §11 robustness into an online appendix. Core paper (Sections 1 to 9) is 22 pp. |

## 14. Venue strategy

See `DECISION-MEMO.md` §6 for the full ranked list. Summary:

- **Primary target: ACM ICAIF.** Fast turnaround, ACM proceedings, exactly this readership. Core paper at 22 to 25 pp fits the long-paper format. Aim for a deadline in the next cycle; Track A alone is sufficient to submit.
- **Secondary: Journal of Financial Data Science** (Portfolio Management Research). Non-Elsevier, high-prestige quant audience, values the full-size journal version (25 to 30 pp) with C2 vine copula included.
- **Backup: TMLR** (Transactions on Machine Learning Research). Rolling, rigorous, open access. Good home if we want to emphasize the methods side (ECM on per-state $\nu_k$, Yu 2010 FB port) over the finance side.

Workshops (in parallel, for timestamping + reviewer feedback): NeurIPS ML for Finance, ICML Time Series, ICAIF side workshops.

Avoid: Journal of Econometrics, Journal of Banking and Finance, Finance Research Letters (all Elsevier per user policy).

## 15. Scope boundaries

- No option pricing or implied-vol surface calibration. Explicitly deferred in v9 conclusion; keep it that way.
- No Hawkes, no variational Bayes, no particle filters. Yu 2010 FB (C1) is the only new estimation kernel.
- No new emission family beyond N / t / L / (optional skew-t in D1).
- No rewrite of existing `src/*.jl` types; C1 adds a new type, does not mutate existing ones.
- No change to paper version freezes: v7 and v8 remain frozen, v9 is live, v10 is the upgrade target (v9 becomes the new frozen reference snapshot when v10 is committed per the user's paper-version-freeze policy).

## 16. First concrete step

**Completed (2026-04-22):** v9 draft, 7-metric panel, VaR utility, cross-asset copula, GRU baseline, Rydén refutation. See DECISION-MEMO §2.

**Next concrete step:** Track A1 + A2 (MMD + signature kernel metrics).

Deliverables, in order:

1. [ ] `src/Metrics.jl` with `mmd2(X, Y; kernel)`, `sig_mmd(X, Y; depth)`, `sig_w1(X, Y; depth)`.
2. [ ] Wire into `run_baselines_and_cross_asset.jl:62-130` as new columns in the 7-metric panel, making it a 10-metric panel.
3. [ ] Regenerate Table 4 and the per-cell CSV outputs.
4. [ ] Add the three-row addition to Table 4 discussion in `CHMM-paper/paper/Paper_v9.tex` (or start `Paper_v10.tex`).
5. [ ] Gate: MMD values should order baselines sensibly. Gaussian i.i.d. and bootstrap should sit mid-pack; CHMM-t should land in the best 3; GARCH should be in the best 3 as well. If ordering is counter-intuitive, investigate kernel bandwidth choice (median heuristic) before proceeding.

After that: A3 (discriminator AUC) is the single biggest reviewer-ask. Then the rest of Track A in dashboard order.

## 17. Out of scope for this plan

- Equity-volatility-joint modeling beyond the flat copula and (planned) vine copula.
- Options surface calibration.
- Online EM (`online-CHMM` sibling repo is the vehicle for that).
- ICAIF short-paper fork (`CHMM-icaif-*` sibling repos); those are on their own track and not coordinated here.
- VIX-specific content; that is `CHMM-Vol-Paper`'s domain.
