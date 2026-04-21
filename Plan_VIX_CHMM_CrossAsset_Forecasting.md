# Plan: VIX-CHMM Regime Forecasting and Cross-Asset Integration

Author: Abdulrahman Alswaidan
Date: 2026-04-21
Status: Proposal

## 1. Objective

Turn the existing VIX CHMM from a descriptive stylized-facts model into a **live, forward-looking volatility-regime forecaster**, and use that forecast to drive the equity CHMM and the cross-asset copula so that joint equity paths are conditioned on the current and projected volatility state of the market.

Concretely we want, at any observation date `t`:

1. A posterior distribution over VIX regimes given all VIX observations up to `t`:  `π^V_t(s) = P(S^V_t = s | y^V_{1:t})`.
2. An `h`-step-ahead forecast posterior `π^V_{t+h|t}` with calibrated uncertainty.
3. A coupling that turns `π^V_{t+h|t}` into a distribution over equity regimes `π^E_{t+h|t}` and a VIX-conditional dependence structure across equities.
4. A joint Monte Carlo simulator that produces equity return paths conditioned on the VIX forecast, usable both for risk statistics (VaR/ES, realized-vol forecasts) and for the `MyCHMMPricingModel` option-pricing engine.

## 2. What we already have (inventory)

From `src/` and the `run_v7_*` driver set:

- `MyContinuousHiddenMarkovModel`, `MyStudentTHiddenMarkovModel`, `MyLaplaceHiddenMarkovModel` trained on VIX log-returns (`run_v7_vix.jl`) with K=18 states.
- Same three emission families trained per-equity on SPY, NVDA, JNJ, JPM, AAPL, QQQ (`run_baselines_and_cross_asset.jl`).
- Viterbi decoding and walk-forward retraining (`Compute.jl::viterbi`, `Compute.jl::walk_forward_regimes`).
- SIM, Gaussian-copula, and Student-t-copula joint samplers with independent CHMM marginals (`src/CrossAsset.jl`).
- `MyCHMMPricingModel` with a **static** VIX-regime-to-equity-vol map built via median(VIX | regime) (`src/Factory.jl:345-378`, `src/Pricing.jl:39-57`).
- In-sample / out-of-sample data splits already constructed (`build_new_train_oos.jl`, `fetch_oos_extended.jl`) through 2026-04-20.

What is **missing**:

- A public forward-filter / smoother that returns `π^V_t` (the Baum-Welch E-step computes it internally but does not expose it).
- An `h`-step forecast routine (`π^V_{t+h|t} = π^V_t · (A^V)^h`).
- Any explicit linkage between `S^V_t` and `S^E_t`. The current pricing linkage is static and one-way; the cross-asset code assumes independent marginals.
- Dynamic (regime-conditional) copula correlation.
- Forecast evaluation harness (CRPS, log-score, Brier on regime identity, ES coverage).

## 3. Core modeling choice

Two viable couplings; we recommend starting with (A) and treating (B) as a stretch goal.

**(A) Empirical regime-coupling matrix (recommended first pass).**
Decode in-sample VIX and equity regime sequences with Viterbi. Estimate a `K^V x K^E` row-stochastic matrix

```
C[s, s'] = P(S^E = s' | S^V = s)
```

directly from co-occurrence counts. Optionally lag by `h` days to capture the empirical "VIX leads equity regime" effect. This stays inside the existing EM-trained marginals, is cheap, interpretable, and fits the paper's framing of VIX as a **conditioning covariate** rather than a new state.

**(B) Factorial / joint HMM (optional upgrade).**
Define a joint state `(S^V, S^E)` with a factored transition

```
P(S^V_{t+1}, S^E_{t+1} | S^V_t, S^E_t)
   = P(S^V_{t+1} | S^V_t) · P(S^E_{t+1} | S^E_t, S^V_t)
```

and fit by EM, reusing the existing forward-backward in log space. This is the rigorous version but roughly `K^V x K^E` more parameters to learn; needs careful initialization (seed with the marginal transitions and `C` from (A)). Worth doing only if (A) produces visible coupling signal on OoS.

## 4. Phased implementation

### Phase 0 -- Scaffolding (0.5 day)

- Extend `AbstractMarkovModel` with a `ForecastHMM` capability trait, or just add methods dispatched on the existing continuous types.
- Decide output type: `RegimeForecast = @NamedTuple{pi::Vector{Float64}, horizon::Int, t::Date}`.

### Phase 1 -- Expose forward filter + h-step forecast (1 day)

Files: `src/Compute.jl`, `src/Types.jl` (no new struct; just a forecast result).

- Implement `forward_filter(hmm, y::Vector{Float64}) -> Matrix{Float64}` returning `alpha_t(s) = P(S_t = s, y_{1:t}) / P(y_{1:t})` in log-space, using the existing `_logsumexp_vec`.
- Implement `regime_forecast(hmm, pi_t, h) = pi_t * A^h` and return a `Vector{Vector{Float64}}` for `h = 1..H`.
- Unit tests against Viterbi on a long history: `argmax(pi_t)` should agree with the Viterbi most-likely state on >~ 85% of days in a stationary stretch.
- Smoke run in `run_v7_vix.jl` printing `pi_T` and the 1/5/20-day ahead regime distributions for the latest OoS date.

### Phase 2 -- Regime-coupling matrix `C` (1 day)

Files: `src/CrossAsset.jl` (new section or `src/CoupledRegimes.jl` if we want it separate), `run_baselines_and_cross_asset.jl`.

- Decode VIX and each equity's regime sequence on the shared IS window.
- Build `C_i[s, s']` per equity `i` (rows normalized, Laplace smoothing with `alpha = 1/K^E`).
- Also build lagged versions `C_i^{(h)}` for `h in {1, 5, 20}` to capture the lead-lag.
- Persist to `results/cross_asset/coupling/C_{ticker}_h{h}.jld2` and write a human-readable `Table-Coupling-Summary.txt` with row-entropy and top-mass state for each VIX regime (lower entropy = stronger coupling).

### Phase 3 -- Joint forecast sampler (1 to 2 days)

Files: `src/CrossAsset.jl` (new struct `MyVIXConditionedJointModel <: AbstractDependenceModel`), `src/Factory.jl`.

Given `pi^V_{t+h|t}` and per-asset coupling `C_i^{(h)}`:

1. Sample `N` VIX regime paths forward from the current filter posterior using `A^V`.
2. For each VIX regime at each step, sample an equity regime from `C_i^{(h)}[s^V, :]` for each asset `i`.
3. Sample equity returns from the per-state emission distribution of each asset's CHMM.
4. Apply the **existing** copula (Gaussian or Student-t) to reorder the simulated innovations so that the target rank correlation is preserved. Parameterize `tau_{ij}` and (for t-copula) `nu` **conditional on the VIX regime**: fit `tau_{ij}(s^V)` on the residuals filtered to days when Viterbi-decoded `S^V = s^V`. Fall back to the unconditional estimate when a regime has too few observations (`< 50` days).
5. Return a `(n_assets, n_steps, n_paths)` array.

This subsumes the current `MyGaussianCopulaModel` / `MyStudentTCopulaModel` as the "VIX-regime-pooled" special case.

### Phase 4 -- Pricing integration (0.5 day)

Files: `src/Pricing.jl`, `src/Factory.jl`.

- Replace the static `volatility_map` in `MyCHMMPricingModel` with a **dynamic** version:
  - Given `pi^V_{t|t}`, compute `E[sigma_t] = sum_s pi^V_t(s) * sigma_map[s]`.
  - Along each MC path, walk the VIX regime using `A^V` and look up `sigma_map[s^V_t]` per step.
- Keep the existing behavior as `pricing_mode = :static` (default for reproducing paper numbers); the new behavior is `pricing_mode = :dynamic`. This way paper results do not regress.
- Sanity: `:dynamic` with the stationary distribution as prior and long `T` should match `:static` within MC error.

### Phase 5 -- OoS forecast evaluation (1 day)

Files: new `run_v7_vix_forecast_eval.jl`, outputs under `results/v7/vix_forecast/`.

On the OoS window (approximately Dec 2024 onward):

- For each `t` in OoS, compute `pi^V_t` by running the forward filter from the start of IS through `t`.
- Produce `h in {1, 5, 20}` forecasts and evaluate:
  - **Regime identity**: Brier score and log-score of `pi^V_{t+h|t}` vs realized Viterbi `S^V_{t+h}` on the extended history.
  - **Realized volatility**: regress 5-day realized equity vol on `E[sigma_{t+h} | pi^V_{t+h|t}]`; report R^2 and slope test vs 1.
  - **VaR/ES**: 1-day and 5-day 1% VaR and ES coverage (Kupiec, Christoffersen) for SPY simulated from the joint sampler vs the independent-marginals baseline.
  - **Option pricing**: price a fixed basket of short-dated SPY options on a rolling basis, compare implied vol produced by `:dynamic` vs `:static` vs market.
- Tables/figures: `Table-T3-Forecast-Scores.txt`, `Fig-VIX-Filter-Posterior-{ticker}.svg`, `Fig-ES-Coverage-{ticker}.svg`.

### Phase 6 -- Factorial HMM upgrade (stretch, 3 to 5 days)

Only pursue if Phase 5 shows that the empirical coupling is strongly regime-dependent (row-entropy of `C` well below `log K^E` for most VIX regimes) AND SPY Brier / ES coverage improves over the independent-marginals baseline.

- Add `MyFactorialHMM` to `Types.jl` (fields: `A_V`, per-equity-regime conditional transition tensors, emission params for each factor).
- Implement joint forward-backward (factorized E-step) and M-step updates; initialize from Phase 2 `C`.
- Add a run driver `run_v7_factorial.jl` and a new results folder `results/v7/factorial/`.
- Compare likelihood on OoS against the empirical-coupling baseline.

## 5. File-level change summary

New:
- `src/CoupledRegimes.jl` (or section in `CrossAsset.jl`): `fit_coupling`, `forecast_joint_regimes`, `MyVIXConditionedJointModel`.
- `run_v7_vix_forecast_eval.jl`: driver for Phase 5.
- `results/v7/vix_forecast/`, `results/cross_asset/coupling/` directories.

Modified:
- `src/Compute.jl`: add `forward_filter`, `regime_forecast`.
- `src/Types.jl`: add forecast result named tuple, `MyVIXConditionedJointModel` struct, optional `MyFactorialHMM`.
- `src/Pricing.jl`, `src/Factory.jl`: add `:dynamic` pricing mode without breaking `:static`.
- `Include.jl`: load order for any new src files (respect the existing Types -> Files -> Factory -> Compute -> Pricing -> Visualize order; put `CoupledRegimes.jl` after `Compute.jl` and before `Pricing.jl`).
- `run_baselines_and_cross_asset.jl`: call Phase 2 fitting and persist `C`.

Untouched:
- Baum-Welch internals, `_logsumexp_vec`, jump coin-flip parameters, tail state ranges (per `CLAUDE.md` "do not change" list).

## 6. Evaluation targets (what a successful outcome looks like)

- VIX regime forecast beats the unconditional stationary distribution on log-score at `h = 1` by a clear margin and at `h = 5` by a meaningful margin; degrades toward the stationary at `h = 20` as expected.
- The empirical coupling matrix `C` is materially non-uniform for crash/boom VIX regimes (row-entropy less than about 70% of `log K^E`).
- VIX-conditioned joint sampler gives better 1-day 1% ES coverage on SPY during OoS stress windows than the independent-marginals copula baseline.
- `MyCHMMPricingModel(:dynamic)` produces short-dated SPY implied-vol term structures that track the VIX forward curve more closely than the static mapping, without degrading long-dated prices.

## 7. Risks and open questions

1. **Regime identifiability across assets.** Viterbi labels are only meaningful within a single model (state 3 of VIX has nothing to do with state 3 of SPY). The coupling `C_i[s^V, s^E]` respects this, but we should sanity-check by relabeling states by `mu_k` rank before reporting.
2. **Lead-lag choice.** Using `h = 0` coupling pools contemporaneous VIX with equity regimes and risks circularity when the user then "forecasts" equities from VIX. Prefer `h >= 1` for anything labeled a forecast.
3. **Small-sample regimes.** Some VIX states may be rarely visited OoS. Use Laplace smoothing and fall back to unconditional transitions when a posterior row is below a minimum sample count.
4. **Parameter drift.** The IS CHMM was trained through 2024; the 2025 OoS includes new market regimes. Decide up front whether the forecast harness is strictly frozen-parameters (paper-clean) or uses `walk_forward_regimes` style refits (more realistic, harder to report).
5. **Copula regime-conditional estimation.** Within-regime samples can be small for state pairs far from the diagonal. A hierarchical shrinkage estimator (toward the pooled `tau_{ij}`) is worth considering.
6. **Scope creep.** Phase 6 (factorial HMM) is tempting but only earns its cost if Phase 5 shows the simpler coupling underperforms. Defer unless the data demands it.

## 8. Suggested execution order

1. Phase 0 and Phase 1 back to back (they are small and unlock everything else).
2. Phase 2 in parallel with drafting the Phase 5 evaluation harness.
3. Phase 3 once `C` and forward filter are both landed.
4. Phase 4 last before evaluation (so Phase 5 can score both pricing modes).
5. Phase 5, then decide on Phase 6 from the numbers.
