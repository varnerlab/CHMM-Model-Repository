> **Scope note.** This is a paper-upgrade planning artifact, not reproducibility documentation. It captures the v9 → v10 scope decision for the companion paper (`CHMM-paper`) and will go stale as that paper advances. For how to reproduce the model and its results, see the repo root `README.md`. For the implementation plan that executes the OPEN items below, see the sibling `plan-equity-paper.md`.

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

### Track A: evaluation rigor (DONE 2026-04-22)

| # | Work item                                                                   | Status | Notes |
|---|-----------------------------------------------------------------------------|--------|-------|
| A1 | MMD with RBF + signature kernels (Gretton, Chevyrev-Kormilitzin)           | **DONE (2026-04-22)** | `src/Metrics.jl::mmd2_rbf`; CHMM-t IS MMD 2.0e-5, best of 9 models |
| A2 | Signature MMD on truncated path signatures (depth 3)                        | **DONE (2026-04-22)** | `src/Metrics.jl::sig_mmd2` + `path_signature`; CHMM-L IS 0.0043 (best continuous) |
| A3 | Discriminator AUC (logistic regression on window features, 5-fold CV)       | **DONE (2026-04-22)** | `src/Metrics.jl::discriminator_auc`; CHMM-t IS AUC 0.607 (closest to 0.5) |
| A4 | Train-on-Synthetic Test-on-Real (TSTR) for HAR(1,5,22) vol forecaster       | **DONE (2026-04-22)** | `run_track_a_utility.jl` §A4; CHMM-N synth-trained QLIKE 1.519 matches real-trained 1.521 |
| A5 | Vol-target strategy back-test using A4 HAR forecaster                       | **DONE (2026-04-22)** | `run_track_a_utility.jl` §A5; GARCH and CHMM-N produce real turnover, i.i.d. baselines produce 0 |
| A6 | Leverage-effect metric: $\text{corr}(r_t^2, r_{t-k})$ and down/up asymmetry | **DONE (2026-04-22)** | `src/Metrics.jl::leverage_effect`; CHMM-N avg -0.038 closest to observed -0.088, pv OoS 0.205 |
| A7 | Aggregational Gaussianity: kurtosis at horizons {1, 5, 10, 21}              | **DONE (2026-04-22)** | `src/Metrics.jl::aggregational_kurtosis`; CHMM-t OoS h=1 5.55 matches observed 5.29 |
| A8 | Kupiec + Christoffersen LR tests on 1 % and 5 % VaR                         | **DONE (2026-04-22)** | `src/Metrics.jl::kupiec_lr,christoffersen_lr`; DiscreteNJ/WJ FAIL 5 % VaR Kupiec (LR 16.81) |
| A9 | Simulation-based p-values on stylized-fact statistics                       | **DONE (2026-04-22)** | `src/Metrics.jl::sim_pvalue`; CHMM-L OoS pv̄ 0.692, CHMM-t 0.661, CHMM-N 0.539 |
| A10 | Unify simulation counts at N_PATHS = 1000 across all Track-A scripts       | **DONE (2026-04-22)** | `run_track_a_metrics.jl`, `run_track_a_utility.jl` |

### Track B: deep-generative + variance-regime baselines (B4 DONE 2026-04-22)

| # | Work item                                                                   | Status | Notes |
|---|-----------------------------------------------------------------------------|--------|-------|
| B1 | QuantGAN (TCN + Wasserstein GAN) on standardized log returns               | OPEN (first-pass Julia-native QuantGAN-style WGAN implemented 2026-04-22)   | `run_track_b1_quantgan.jl`. Real deep baseline row now exists, but is decisively worse than CHMM-t on MMD / sig-MMD / discriminator AUC. |
| B2 | Sig-WGAN (signature-Wasserstein) as path-level competitor                   | OPEN   | One row in Table 4 |
| B3 | Score-based time-series diffusion (TimeGrad-style, depth-2 conditional UNet) | OPEN (first-pass window diffusion implemented 2026-04-22) | `run_track_b3_diffusion.jl`. Strongest deep row on local distributional metrics (MMD, sig-MMD, disc AUC), but still weak on pv̄ and unconditional VaR independence. |
| B4 | MS-GARCH (Haas-Mittnik-Paolella 2004) as variance-regime baseline          | **DONE (2026-04-22)** | `src/MSGARCH.jl`, `run_track_b4_msgarch.jl`. K=2 fit on SPY: calm σ 0.97, stress σ 12.67, p_11=0.914, p_22=0.547. **Best-in-panel unconditional VaR Kupiec** at 1 % (LR_uc 0.01) and 5 % (LR_uc 0.26). Doesn't dominate CHMMs on marginals. |

### Track C: model upgrades that change the paper title (C1 DONE 2026-04-22)

| # | Work item                                                                   | Status | Notes |
|---|-----------------------------------------------------------------------------|--------|-------|
| C1 | Semi-Markov CHMM with heavy-tailed sojourns (port Yu 2010 FB from vol repo) | **DONE (2026-04-22)** | `src/SemiMarkov.jl`, `run_track_c1_smchmm.jl`. 17 of 18 states pick Pareto sojourns. SM variants pass 1 % and 5 % VaR Kupiec cleanly (LR_uc drops from 3.83 to 0.82 for SM-N at 5 %). Trade-off: worse MMD and discriminator AUC on marginals (except SM-CHMM-L which wins MMD at 4.0e-5). |
| C2 | Vine copula cross-asset extension (C-vine or D-vine, Aas 2009) on 50 to 100 assets | OPEN (50-asset large-universe run completed 2026-04-22) | `src/CrossAsset.jl::MyTruncatedCVineCopulaModel`, `run_track_c2_large_universe.jl`. Viable and scalable, but underperforms the flat copulas on both the 6-asset and 50-asset panels. |
| C3a | Regime-conditional VaR via Viterbi decode (first-pass fix for Christoffersen) | **DONE (2026-04-22)** | `run_track_c3_conditional_var.jl`. Flat CHMM-t passes Kupiec AND Christoffersen cleanly at 1 % and 5 % (LR_uc 0.10, LR_ind 0.09 at 1 %). Closes the A8/C1 independence failure. SM-CHMM conditional VaR remains open; needs SM-aware decoder. |
| C3 | Time-varying transition matrix $T_{ij}(t)$ via logistic-regression on VIX / realized vol / term spread | OPEN (lagged-RV + VIX pass implemented 2026-04-22; term spread still open) | `run_track_c3_time_varying_transition.jl`, `run_track_c3_external_covariates.jl`. Viable, but does not beat the flat CHMM-t C3a result; term-spread conditioning remains open. |
| C4 | Leverage-effect emission $r_t = \mu_k + \rho_k r_{t-1}^- + \sigma_k \epsilon_t$ (single ablation row) | **DONE (2026-04-22)** | `run_track_c4_leverage_emission.jl`. Improves OoS leverage coverage pv from 0.205 to 0.308 and lowers IS discriminator AUC from 0.646 to 0.594 vs flat CHMM-N; unconditional VaR worsens, so keep as ablation only. |

### Track D: nice-to-have (OPEN, skip if time-bound)

| # | Work item                                                                   | Status | Notes |
|---|-----------------------------------------------------------------------------|--------|-------|
| D1 | Skew-t or NIG emissions as a fourth family                                  | OPEN   | Currently deferred |
| D2 | Promote Turing-based Bayesian $\nu_k$ posterior from diagnostic to main     | OPEN   | Already coded, just underused |
| D3 | Identifiability / label-switching appendix (1 page)                         | OPEN   | Matches vol-paper item K |
| D4 | Option pricing / implied-vol surface calibration                            | OUT    | Explicitly deferred in v9 conclusion |

### Track V10-Polish: v10 revision pass driven by user review (OPEN, 2026-04-22)

Source: `CHMM-Model/user-comments.md`. All items scoped against `Paper_v10.pdf` and `Paper_v10.tex` + `sections/*_v10.tex`. The v10 first-pass writeup is complete; these items gate a clean submission.

| # | Work item                                                                                                                                                                                                                    | Status | Notes                                                                                                                                         |
|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| V1 | Re-derive DiscreteNJ / WJ $(K_{\text{disc}}, \lambda, \epsilon)$ from `arxiv.org/pdf/2603.10202` (`alswaidan2026hybrid`), re-run `run_baselines_and_cross_asset.jl`, propagate into `methods_v10.tex` and every downstream cell | **DONE 2026-04-22** | Authoritative prior-paper values read from `HMM-withJumps/3-HMM-WithJumps-Simulation-Notebook.ipynb`: $K_{\text{disc}} = 90, \epsilon = 5 \times 10^{-5}, \lambda = 67$. Both `run_baselines_and_cross_asset.jl` and `run_track_a_metrics.jl` re-run. Table 2 rewritten; at the new hyperparameters the discrete NJ/WJ reach $98.3\%$ / $97.9\%$ IS KS, matching CHMM-N/t/L $93.2$--$93.8\%$. Paper narrative pivoted: the CHMM advantage is no longer on coarse KS/AD pass rates, but on kurtosis ($3.5$ vs obs $7.68$), aggregational Gaussianity decay at $h \geq 10$, and $5\%$ VaR Kupiec ($\text{LR}_{\text{uc}} = 16.81$, breach $1.7\%$ vs target $5\%$). See `results/SPY/Table-2-Baselines.txt`, `results/track_a/Table-4-Extended-Metrics.txt`, Table~\ref{tab:model_comparison}, Table~\ref{tab:extended_metrics}, Table~\ref{tab:leverage}, Table~\ref{tab:agg_kurt}, Table~\ref{tab:sim_pvalues}; backups at `*.preV1-backup`. $K$-sensitivity sweep (Table T1) was not re-run and retains slightly stale $K = 18$ entry; flagged for V15. |
| V2 | Add axis labels to Fig 4, Fig 5, and any other v10 figure missing them; audit generator scripts                                                                                                                               | **DONE 2026-04-22** | Shared `_STYLE` kwargs (`guidefontsize=11`, `tickfontsize=10`, `legendfontsize=9`, margin defaults) pushed into `run_figures.jl` and `run_multi_emission_analysis.jl`. Main-body figures regenerated. |
| V3 | Align the two panels of Fig 4 as a single subplot grid, not two stitched standalone figures                                                                                                                                   | **DONE 2026-04-22** | Fig 4 subfigure pair switched from `[b]` valign to `[t]` with inner `\centering`; matched `size=(1500,450)` and identical `_STYLE` kwargs so the panels align on the top baseline. |
| V4 | Rewrite figure and table captions in v10 to be self-contained and informative                                                                                                                                                 | **DONE 2026-04-22** | Eight thin captions rewritten with the "what it shows + headline finding" template: `tab:variant_choice`, `fig:k_sweep_summary`, `tab:var_es`, `fig:var_es`, `tab:sm_sojourn`, `tab:sm_var`, `tab:conditional_var`, `tab:price_sim_oos`, `tab:price_sim_prob`. Long existing captions (Figures 1/2/3/5, Tables 3/12/13) were already top-venue quality and left unchanged. |
| V5 | Remove stale "v6 OoS window" references ($T_{\text{OoS}} = 219$) from v10 section files; check for similar stale-version nuances                                                                                              | **DONE 2026-04-22** | Both surviving v6 cites in `results_v10.tex` rewritten to version-agnostic phrasing (Gaussian OoS paragraph at line 92, OoS generalization paragraph at line 272). Gaussian OoS KS updated from $2.0\%$ to the reran $1.2\%$; CHMM OoS KS band widened to the post-V1 $79$--$86\%$ with explicit cross-reference to the tail/VaR separators. No remaining v6 / $T_{\text{OoS}} = 219$ tokens in any `*_v10.tex`. |
| V6 | Consolidate the random seed (20260422) mention to a single body sentence; drop repeats                                                                                                                                        | **DONE 2026-04-22** | Authoritative seed-policy sentence added at `methods_v10.tex:190` (Evaluation Protocol subsection): global $20260420$, additive sub-seeds $20260421$ (OoS price sim) and $20260422$ (Track A). Duplicate prose in `results_v10.tex:288` and `conclusion_v10.tex:45` reduced to `Section~\ref{sec:metrics}` cross-references; four body-table/figure captions likewise. Supplemental captions keep explicit seeds (appendix scope). |
| V7 | Fix margin overflow on the §4.6 sig-MMD path-lift sentence `We lift each window to a 2D path (t/W, r_t/std(r))`                                                                                                               | **DONE 2026-04-22** | `results_v10.tex:298` inline math promoted to a displayed equation `\eqref{eq:path_lift}` with $\gamma: [0, 1] \to \mathbb{R}^2$. Overfull warning cleared. |
| V8 | Fix margin overflow on the leverage-effect sentence `The observed leverage effect on the IS window, measured as avg_{k=1}^{20} corr(r^2_t, r_{t-k})`                                                                          | **DONE 2026-04-22** | `results_v10.tex:311` inline average promoted to a displayed equation `\eqref{eq:leverage}` with named variable $\bar\rho_{\text{lev}}$. Overfull warning cleared. |
| V9 | Trim unnecessary "reproduced from \ldots" statements in table captions                                                                                                                                                        | **DONE 2026-04-22** | Six caption-end provenance clauses (`tab:leverage`, `tab:agg_kurt`, `tab:sim_pvalues`, `tab:sm_sojourn`, `tab:sm_var`, `tab:conditional_var`) removed. No remaining "reproduced from" tokens in any `*_v10.tex`. |
| V10 | Reconcile §4.3 title ("seven-model comparison", `results_v10.tex:80`) with the actual generator count in Table~\ref{tab:model_comparison}                                                                                    | **DONE 2026-04-22** | Actual count is 12 (Bootstrap, Block-BS, Gaussian, Laplace, DiscreteNJ, DiscreteWJ, Bin-T NJ, GARCH, GRU, CHMM-N/-t/-L). Caption already read "Twelve-model"; prose updated at `results_v10.tex:80`, `discussion_v10.tex:58`, `conclusion_v10.tex:14`. Surviving "seven-metric" cites (seven *metrics* in the panel) are correct and unchanged. |
| V11 | Reconcile Table 3 caption ("comparison to 9 generators") with the actual row count and surrounding prose                                                                                                                     | **DONE 2026-04-22** | Table 3 (`tab:model_comparison`) has 12 rows; Table 4 (`tab:extended_metrics`) has 9. Table 4 caption now self-labels as the nine-row Track A subset of the twelve-row Table 3, explaining the three Track A drops (Block-BS, Bin-T NJ, GRU) on stationary-i.i.d. protocol grounds. V1 follow-up: Table 3 caption's stale $K_{\text{disc}} = 13$, $(\epsilon, \lambda) = (0.01, 3)$ cite replaced with the authoritative $K_{\text{disc}} = 90$, $(5 \times 10^{-5}, 67)$.                                                              |
| Style bundle | Project-macro library and `caption` style setup in `Paper_v10.tex` preamble                                                                                                                                                    | **DONE 2026-04-22** | `\captionsetup{labelfont=bf, labelsep=period, font=small, justification=justified}`; macros `\pct`, `\Kdisc`, `\LRuc`, `\LRind`, `\NPaths`, `\TIS`, `\TOoS`. New prose uses the macros; legacy cells migrate gradually under V16.                                                                                                                                                                    |
| V12 | Add one or two schematic figures (TikZ / SVG) diagramming (i) Pipelines A and B and (ii) the architectures of the generator families                                                                                          | **DONE 2026-04-22 (Pipeline A/B landed; generator-family figure deferred)** | Pipeline A/B TikZ diagram (`fig:pipeline_schematic`) placed at the top of Methods, right after the Data subsection. Pipeline A (top row): observed $\to$ CHMM fit $\to$ 1{,}000 sim paths $\to$ per-asset metrics. Pipeline B (bottom row): $d$ tickers $\to$ per-asset CHMM marginals $\to$ per-asset paths $\to$ copula on ranks $\to$ rank-reordered paths $\to$ cross-asset metrics, with explicit "marginals" / "dependence" labelling. TikZ libraries loaded: `shapes.geometric, arrows.meta, positioning, fit, calc`. Generator-family architecture schematic is deferred: textual description in `sec:benchmarks` already carries the layer and a second schematic would be redundant; add under V15 if reviewers request it. |
| V13 | General figure-quality pass: colorblind-safe palette (max ~6 colors), consistent font / line-width / size, higher-contrast density fans; push defaults into `run_figures.jl`                                                  | **DONE 2026-04-22** | Okabe-Ito colorblind-safe palette (vermillion / blue / bluish-green) adopted across `run_figures.jl` and `run_multi_emission_analysis.jl`; `_OBS_C` / `_SIM_C` / `_MEAN_C` constants + shared `_STYLE` NamedTuple centralise the palette. |
| V14 | Fix Fig 5 panel (b): raise contrast on simulated-path overlay vs observed, add a legend for "Observed" vs "Simulated"; audit other overlay panels                                                                             | **DONE 2026-04-22** | Density-fan alpha raised from $0.05$ to $0.18$; paths switched to Okabe-Ito blue; observed to vermillion; single consolidated legend entry `"CHMM-N simulated (50 paths)"` on the first density call. No other main-body density fans needed the same fix. |
| V15 | Final cross-section nuance sweep: scan all v10 section files for stale version references, margin overflows, unlabeled axes, misleading titles, thin captions                                                                 | **DONE 2026-04-22** | Five residual stale-number cites fixed: CHMM-t IS kurtosis $17.34 \to 14.57$ in four places, overshoot percentage $126\% \to \sim 90\%$, stale "$K_{\text{disc}} = 13$ quantization" explanation at `results_v10.tex:507` rewritten to the correct $\epsilon = 5 \times 10^{-5}$ / $\lambda = 67$ regime. Known residuals: 15 `Overfull \hbox` warnings (non-breaking); appendix-panel inconsistencies from pre-V1 seed runs; generator-family schematic deferred. |
| V16 | `siunitx` migration for numeric tables (Tables 2, 4, leverage, agg-kurt, sim-pvalues, cross-asset, $K$-sweep): per-column `S[table-format=...]` so decimals align                                                              | OPEN   | Invasive (touches every numeric cell); defer until after V10--V15. Pairs with the project-macro library introduced in V10/V11                  |

### Post-v10 external-event items (OPEN, not a v10 submission gate)

These items do not gate the v10 submission. They are queued for a future revision of the paper once an external event fires.

| # | Work item | Status | Trigger |
|---|---|---|---|
| E1 | Cite the semi-Markov / vol companion paper (sibling `SM-CHMM-AR-Paper` / `CHMM-Vol-Paper`) in the equity paper. Replace the current narrative forward references in `discussion_v10.tex` and in `methods_v10.tex` `sec:sm_ablation` with a formal `\cite{...}` entry (bib key to be set once the arXiv ID is known). Soften companion-facing wording in this repo so nothing implies the vol paper is already published: `README.md:122` (SM-CHMM-AR-Model / SM-CHMM-AR-Paper bullet), `run_track_c1_smchmm.jl:534` (Track C1 summary header). | OPEN | External: vol paper posted to arXiv. User will signal when the preprint is live. Decision made 2026-04-22 to defer any citation work until then; the equity paper's semi-Markov content stands on the Yu (2010) citation alone in the interim. |

---

## 1. Verdict in one sentence

**After Track A (DONE 2026-04-22) the paper is at or near top-tier release quality on evaluation rigor.** CHMM-t / -L / -N dominate nearly every new metric: smallest MMD, smallest discriminator AUC (0.607 IS for CHMM-t, vs 0.78-0.80 for i.i.d. baselines), highest OoS joint p-value coverage (CHMM-L 0.692, CHMM-t 0.661, CHMM-N 0.539), and CHMM-N's synth-trained HAR matches the real-trained benchmark out-of-sample on QLIKE (1.519 vs 1.521). Discrete HMM + jumps baseline FAILS 5 % VaR Kupiec (breach rate 1.75 %, LR_uc 16.81): the new tests surface a concrete weakness of the prior-paper baseline that v9 did not expose. Remaining upgrade for full top-tier status: one serious deep-generative baseline (B1 QuantGAN or B3 diffusion) and one C-track model extension (C1 semi-Markov port from vol repo, or C2 vine copula cross-asset). Track A alone already lifts the paper from v9 "publishable" to "defensible" on the evaluation panel.

### Track A headline findings (2026-04-22)

Results written to `results/track_a/`. Seed `20260422`, N_PATHS=1000, K=18 for CHMMs.

- **A1 MMD (RBF, 20-day windows)**: CHMM-t IS 2.0e-5 < CHMM-N 1.5e-4 < CHMM-L 8.9e-4 < i.i.d. baselines 3.5e-3 to 9.4e-3. CHMM-N and CHMM-t also top the OoS MMD panel (6.1e-4 and 2.2e-3 respectively). GARCH IS MMD is numerically 0.0 (bandwidth-heuristic artifact with wide-scale distribution; flagged in paper).
- **A2 Sig-MMD (depth 3 time-lifted path)**: CHMM-L IS 0.0043, CHMM-t 0.0047, CHMM-N 0.0072 dominate. Laplace i.i.d., Bootstrap, Gaussian, GARCH all 0.027 to 0.033, an order of magnitude worse. Discrete NJ is pathologically near-zero due to bin-center quantization; flagged.
- **A3 Discriminator AUC (logistic, 10 window features, 5-fold)**: CHMM-t IS 0.607, CHMM-L 0.623, CHMM-N 0.646. All i.i.d. and GARCH fall in 0.73-0.80. CHMM-t is the single model that comes closest to indistinguishability from real.
- **A4 TSTR HAR vol forecaster**: CHMM-N synth-trained **QLIKE = 1.519 beats the real-trained 1.521** on OoS by a whisker; RMSE ratio 0.988. GARCH synth-trained RMSE 9.98 also beats real-trained (RMSE ratio 0.978). CHMM-t/-L produce useful vol coefficients but smaller in magnitude (regime mixing $\neq$ HAR-linear decay).
- **A5 Vol-target strategy**: only GARCH (turnover 0.57), CHMM-N (0.04), and real-trained (0.20) produce non-trivial position sizing; i.i.d. baselines collapse to the W_MIN clamp. Caveat: the TARGET_VOL=0.15 interacts with the annualized-excess-growth convention so Sharpe values cluster near 1.3; the qualitative signal (GARCH + CHMM-N have real vol dynamics, i.i.d. do not) is the publishable finding.
- **A6 Leverage effect**: observed IS avg $\text{corr}(r_t^2, r_{t-k}) = -0.088$, asymmetry 0.358. CHMM-N avg -0.038, CHMM-t -0.025, CHMM-L -0.024 all capture the sign; CHMM-L asymmetry 0.322 almost matches observed 0.358. GARCH produces essentially zero leverage (-0.0007) by construction. CHMM-N OoS p-value coverage 0.205 is highest.
- **A7 Aggregational Gaussianity**: observed IS 7.68 at h=1 decaying to 2.63 at h=21. CHMM-t 8.46 -> 2.65 -> 2.06 -> 1.35 is the cleanest monotone decay through the observed band. CHMM-L 6.33 -> 1.02 matches h=1 well. GARCH 4.35 and CHMM-N 4.88 undershoot h=1 but produce proper decay. i.i.d. baselines flatten to near 0 at h=21.
- **A8 Kupiec + Christoffersen VaR**: at 1 % VaR all continuous models pass Kupiec (LR_uc < 3.84). At 5 % VaR DiscreteNJ and DiscreteWJ **fail** Kupiec with breach rate 1.75 % vs target 5 % (LR_uc 16.81, p < 0.001); they are overly conservative and waste capital. Laplace and GARCH calibrate best at 5 % (LR_uc 1.23). All generators fail Christoffersen independence (LR_ind 6.47 to 20.87): unconditional VaR cannot capture the clustered-breach structure of the 2024-2026 OoS window; a conditional / Viterbi-decoded VaR is a future-work item.
- **A9 Joint p-value coverage (pv̄ OoS)**: CHMM-L 0.692, CHMM-t 0.661, CHMM-N 0.539 are the top three. GARCH 0.468 and Bootstrap 0.466 follow. Gaussian 0.116, Laplace 0.156, Discrete 0.163-0.164 at the bottom.

The full panel is in `results/track_a/Table-4-Extended-Metrics.txt`, `leverage_effect.txt`, `aggregational_kurtosis.txt`, `sim_pvalues.txt`, `tstr_vol_forecaster.txt`, `vol_target_strategy.txt`, `VaR_LR_tests.txt`.

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

## 6. Release strategy

The release target for the current revision pass is an arXiv preprint. No
specific journal venue has been selected; venue selection is deferred until
after the preprint is posted.

## 7. Path forward

The minimum viable upgrade plan for a top-tier submission is **Track A + one B-track row**:

1. A1 + A2 (MMD + signature kernel) as new columns in Table 4.
2. A3 (discriminator AUC) as a single numeric ("AUC = 0.54") in Results.
3. A4 or A5 (TSTR or strategy back-test) as the new "Downstream Utility" subsection, replacing the VaR-only framing.
4. A6 + A7 (leverage + aggregational Gaussianity) as two added rows to the stylized-fact panel.
5. A8 (Kupiec + Christoffersen on existing VaR) as a paragraph in the existing utility section.
6. B1 (QuantGAN) as one row in Table 4. If B3 (TimeGrad-style diffusion) also lands, even better, but B1 is the load-bearing deep baseline reviewers will ask about.

The stretch plan adds **C1 (semi-Markov sojourns)** and re-pitches the paper title as "A semi-Markov regime-switching continuous HMM for equity return synthesis". That earns the paper a narrative upgrade (not just a stylized-facts panel but a structural model), and it ports cleanly from the vol repo since the Yu 2010 machinery is already proven there.

**C2 (vine copula) is the headline multi-asset upgrade** and the main reason a reviewer would pick this paper over any other HMM-based generator: it is the only generator family that scales to 100 assets while keeping exact marginals and interpretable regimes. If only one C-track item lands, choose C1 for methods framings and C2 for finance framings.

## 8. One-line answer to "can this become top-tier?"

**Yes, within one focused track of work.** Track A alone moves the paper from mid-tier to top-tier quality on evaluation rigor, and is now DONE (2026-04-22). Track A + one B row + C1 or C2 makes it a genuinely strong release. Nothing here requires a fundamental rewrite; the v9 scaffold already carries the load, and the missing pieces are standard additions that the vol repo has already proven feasible in Julia at the same scale.

### Status snapshot (2026-04-22 end-of-session, after B1/B3/C2/C3/C4)

- Track 0 (v9 baseline): DONE.
- Track A (evaluation rigor): **DONE**. Ten new metrics, seven new result files, one new source module (`src/Metrics.jl`), two new run scripts.
- Track B (deep-generative baseline): **B1 first pass implemented, B3 first pass implemented, B4 DONE**. `run_track_b1_quantgan.jl` adds a serious GAN row; outcome is negative-control rather than competitive. `run_track_b3_diffusion.jl` adds a strong diffusion row on local window metrics. The deep-baseline section is now reviewer-defensible.
- Track C (model upgrades): **C1 DONE, C3a DONE, C4 DONE; C2 and C3 tested at meaningful scale but remain non-headline**. New source module (`src/SemiMarkov.jl`), `MySemiMarkovContinuousHMM` type, `run_track_c1_smchmm.jl`, `run_track_c3_conditional_var.jl`, `run_track_c3_time_varying_transition.jl`, `run_track_c3_external_covariates.jl`, `run_track_c4_leverage_emission.jl`, plus `MyTruncatedCVineCopulaModel` in `src/CrossAsset.jl`, `run_track_c2_large_universe.jl`, and result files across `results/track_c1/`, `results/track_c3/`, `results/track_c4/`, `results/cross_asset/`, and `results/cross_asset_large/`. The external-`VIX` C3 pass is now landed; term-spread conditioning remains the main open model item.
- Track D (nice-to-have): OPEN.
- Track V10-Polish (v10 revision pass driven by user review of `Paper_v10.pdf`, `user-comments.md`): **OPEN, 15 items (V1 through V15).** This is the remaining gate on submission. Biggest-impact subset: V1 (correct DiscreteNJ / WJ parameters, re-run downstream tables), V12 (Pipeline A / B + architecture schematics), V7 / V8 (margin overflows), V10 / V11 (model-count reconciliation).
- Paper update: `Paper_v10.tex` scaffolded 2026-04-22 from v9; v9 joins the frozen reference set with v7 / v8. Title set to *"A Regime-Switching Continuous Hidden Markov Model as a Reference Synthetic-Data Generator for Equity Returns: Extended Evaluation, Semi-Markov Ablation, and Regime-Conditional Value-at-Risk"*; abstract rewritten around the three-way operational split; 11 content em dashes inherited from v9 have been removed. The new Extended Evaluation subsection (`sec:extended_evaluation`) has landed with six summary paragraphs and four numerical tables for MMD / sig-MMD / discriminator AUC, leverage effect, aggregational kurtosis, and joint $\bar{pv}$; the Kupiec / Christoffersen call-out against DiscreteNJ / WJ is embedded there as well. The new Semi-Markov Ablation subsection (`sec:sm_ablation`) has landed with the Yu (2010) port, the sojourn-family selection table, the flat-vs-SM VaR LR table, the TSTR HAR tightening summary, and the risk-calibration-not-marginal-fidelity framing. The new Conditional VaR subsection (`sec:conditional_var`) has landed with the Viterbi-decoded conditional-quantile estimator, a flat-vs-conditional LR comparison table that shows flat CHMM-t passing both Kupiec and Christoffersen cleanly at 1 % and 5 % on the 2024-2026 OoS window, the SM-aware-decoder deferral, and the three-way operational-split hand-off. The QuantGAN, window-diffusion, and MS-GARCH baseline paragraphs have also landed in `methods_v10.tex` `sec:benchmarks` with a matching Extended-Evaluation prose paragraph that places their numbers against the CHMM family on MMD / sig-MMD / AUC / $\bar{pv}$ / Kupiec. The three-way operational-split paragraph and a companion Discrete NJ / WJ 5 % VaR Kupiec failure paragraph have landed in `discussion_v10.tex`, and a dedicated `par:discrete_var_failure` callout now sits in `sec:utility` of `results_v10.tex`. All seven v10 writeup deliverables listed in `plan-equity-paper.md` §16 are now complete for the *first pass*. The remaining path to submission is **Track V10-Polish** (V1 through V15 in `plan-equity-paper.md` §status-dashboard and in `user-comments.md`) plus a local LaTeX compile check and a final reading pass.

### Track C3a headline findings (2026-04-22): conditional VaR closes Christoffersen

Results in `results/track_c3/`. Uses existing `viterbi(R_oos, flat_model)` to decode the current state per time step, then takes the α-quantile of that state's conditional emission.

**Flat CHMM passes both tests cleanly:**

| Model | α | Breach % | LR_uc (was A8) | LR_ind (was A8) |
|---|---|---|---|---|
| CHMM-t | 1 % | 0.87 | 0.10 (0.58 unconditional) | **0.09 (15.53 unconditional)** |
| CHMM-t | 5 % | 5.07 | 0.01 (3.83 unconditional) | **0.19 (5.26 unconditional)** |
| CHMM-N | 1 % | 0.70 | 0.58 | 5.73 (marginal; was 18.98) |
| CHMM-N | 5 % | 5.07 | 0.01 | **1.39 (5.26 unconditional)** |
| CHMM-L | 1 % | 0.35 | 3.26 | **0.01 (9.15 unconditional)** |
| CHMM-L | 5 % | 3.32 | 3.83 (marginal) | **2.25 (4.71 unconditional)** |

**CHMM-t with Viterbi-decoded conditional VaR is the headline risk-management result for the paper**: both Kupiec and Christoffersen pass cleanly at both 1 % and 5 % VaR on OoS.

**SM-CHMM conditional VaR is harder.** The existing Viterbi uses the flat CHMM's emission, so the decoded state does not match SM's internal partition after plug-in refitting. Two naive fallbacks (AR-residual scale, stationary within-state σ) both over-breach (9 to 33 % breach rate). Interestingly SM-t under stat-σ achieves clean LR_ind (0.07 at 1 %) but still fails LR_uc. A proper SM-aware Viterbi decoder is future work and not required for the flat-CHMM paper narrative.

### Three-way operational split for the paper v10 narrative

1. **Distributional fidelity**: flat CHMM-t wins (MMD 2.0e-5, discriminator AUC 0.607).
2. **Unconditional VaR calibration**: MS-GARCH (K=2) and SM-CHMM-N / -t tie at the top (1 % Kupiec LR_uc 0.01 each; 5 % MS-GARCH 0.26 best, SM-N 0.82).
3. **Conditional VaR with independent breaches**: flat CHMM-t with Viterbi decode wins (Kupiec 0.10 / 0.01, Christoffersen 0.09 / 0.19 at 1 % / 5 %).

Each addresses a different operational question and the paper can make a concrete recommendation per use case.

### Track B4 headline (MS-GARCH K=2, 2026-04-22)

Fit `src/MSGARCH.jl`: HMP 2004 two-regime recursion, Hamilton filter + Nelder-Mead on 8 params. On SPY IS:

- Regime 1 (calm): ω=0.039, α=0.109, β=0.85, unconditional σ ≈ 0.97.
- Regime 2 (stress): ω=1.098, α=0.251, β=0.742, unconditional σ ≈ 12.67.
- Transition: p_11 = 0.9145 (expected calm sojourn ~12 days), p_22 = 0.5474 (expected stress sojourn ~2 days).

**Unconditional VaR calibration is best-in-panel** for MS-GARCH: 1 % LR_uc = 0.01 (breach rate 1.0 % exactly), 5 % LR_uc = 0.26 (breach rate 4.5 %), 5 % LR_ind = 4.79 (closest to the 3.84 crit value among all unconditional-VaR rows; still marginal fail but much better than 5.26-20.87 for others).

Marginals: MS-GARCH MMD IS = 0.00048 (second-best non-CHMM, behind single GARCH's numerical-zero artifact), disc AUC IS = 0.734 (better than GARCH's 0.766, worse than CHMMs' 0.607-0.646). MS-GARCH gets a defensible Table-4 row but does not challenge CHMM-t for top distributional slot.

**Publishable finding:** MS-GARCH is the correct variance-regime baseline for the paper, and it nearly passes Christoffersen independence at unconditional 5 % VaR (LR_ind 4.79, p 0.029). No other unconditional-VaR generator in the panel comes this close.

### Track C1 headline findings (2026-04-22)

Results in `results/track_c1/`. SM-CHMM variants fit with the plug-in estimator (Viterbi + per-state AR(1) + sojourn-family pick).

**Sojourn structure:** 17 of 18 states choose Pareto sojourns for all three SM variants (one NB each). This matches the vol paper's finding: Pareto-tailed sojourns are empirically dominant even on equity returns at K=18, not just on log-VIX.

**Where SM-CHMM wins cleanly (risk + vol forecasting):**

| Metric | Flat CHMM | SM-CHMM | Delta |
|---|---|---|---|
| 1 % VaR Kupiec LR_uc (N) | 1.58 | **0.01** | -1.57 (to almost exact calibration, breach rate 1.0 %) |
| 1 % VaR Kupiec LR_uc (t) | 0.58 | **0.10** | -0.48 |
| 5 % VaR Kupiec LR_uc (N) | 3.83 | **0.82** | -3.01 (clean pass, breach rate 4.2 %) |
| 5 % VaR Kupiec LR_uc (t) | 3.83 | **1.74** | -2.09 (clean pass) |
| TSTR HAR QLIKE ratio (N) | 0.999 | **1.000** | tie at real-trained |
| TSTR HAR QLIKE ratio (t) | 1.045 | **1.014** | -0.031 |
| TSTR HAR QLIKE ratio (L) | 1.027 | **1.012** | -0.015 |
| MMD IS (L) | 0.00089 | **4.0e-5** | now best in full panel |

**Where SM-CHMM loses (marginal fit):**

| Metric | Flat CHMM | SM-CHMM | Delta |
|---|---|---|---|
| MMD IS (N) | 0.00015 | 0.00384 | +0.00369 (worse) |
| MMD IS (t) | 2.0e-5 | 0.00619 | +0.006 (worse) |
| Disc AUC IS (N) | 0.646 | 0.699 | +0.053 (easier to tell apart) |
| Disc AUC IS (t) | 0.607 | 0.782 | +0.175 (much easier) |
| Disc AUC IS (L) | 0.623 | 0.823 | +0.200 (much easier) |
| pv̄ OoS (t) | 0.661 | 0.384 | -0.277 |

**What did not change in C1:** Christoffersen independence still fails across all nine original rows and all three SM rows (LR_ind 5.9 to 20.9). Unconditional VaR alone cannot capture clustered-breach structure. This gap was closed in Track C3a (see next section): Viterbi-decoded conditional VaR on flat CHMM-t passes both Kupiec and Christoffersen cleanly.

**Publishable finding:** semi-Markov structure is a **risk-calibration upgrade, not a marginal-fidelity upgrade**. The paper narrative now has a clean split: flat CHMM-t for distributional matching (MMD, discriminator, stylized facts), SM-CHMM-N/-t for VaR backtesting and TSTR vol forecasting. This is a better story than "SM is uniformly better" because it gives reviewers a clear operational recommendation.

## 9. Related repos

- `CHMM-Vol-Model` and `CHMM-Vol-Paper` (sibling): VIX-only, already draft-complete. Source of Yu 2010 FB, Christoffersen LR, sojourn-family sweep machinery to port.
- `CHMM-paper`: v9 LaTeX draft.
- `CHMM-icaif-Model` / `CHMM-icaif-paper`: older scoped-down conference-format fork. Not the main vehicle; v9 is.
- `online-CHMM`: prospective online-EM extension; out of scope for this memo.
