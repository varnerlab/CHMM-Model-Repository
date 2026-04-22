# Track C1 results (2026-04-22)

Semi-Markov CHMM port from `CHMM-Vol-Model`. Companion to `../track_a/README.md`.

Produced by `run_track_c1_smchmm.jl`. Seed 20260422, K = 18, N_PATHS = 1000, plug-in estimator (Viterbi + per-state AR(1) + NB vs truncated Pareto sojourn selection).

## Files

| File | Content |
|---|---|
| `Table-4-Extended-Metrics-C1.txt` | Track-A MMD / sig-MMD / disc AUC panel with SM-CHMM-N / -t / -L rows appended |
| `leverage_effect_c1.txt` | Leverage effect panel with SM rows appended |
| `aggregational_kurtosis_c1.txt` | Aggregational kurtosis panel with SM rows appended |
| `sim_pvalues_c1.txt` | Joint p-value coverage with SM rows appended |
| `tstr_vol_forecaster_c1.txt` | TSTR HAR with SM rows appended |
| `vol_target_strategy_c1.txt` | Vol-target strategy with SM rows appended |
| `VaR_LR_tests_c1.txt` | Kupiec + Christoffersen LR with SM rows appended |
| `sojourn_summary.txt` | Per-state sojourn family + mean duration for each SM variant |
| `Track-C1-summary.txt` | Auto-generated flat-vs-SM digest |
| `sim_archive_sm.jld2` | 1000-path simulation archives for SM-CHMM-N / -t / -L (gitignored) |
| `sm_models.jld2` | Fitted SM-CHMM models (gitignored; small but regenerable) |

## Headline findings

1. **17 of 18 states pick Pareto sojourns on SPY** for all three SM variants, matching the vol paper's finding. Heavy-tailed sojourns are empirically dominant at K = 18 even on equity returns.
2. **VaR calibration improves dramatically.** 5 % Kupiec LR_uc drops from 3.83 (flat CHMM-N) to 0.82 (SM-CHMM-N); 1 % LR_uc drops from 1.58 to 0.01. SM-CHMM-t, -N pass both Kupiec tests cleanly.
3. **TSTR HAR improves.** QLIKE ratio tightens toward the real-trained 1.000 for all three SM variants. SM-CHMM-N QLIKE ratio is 1.000 exactly.
4. **Marginal metrics worsen.** MMD and discriminator AUC go up for SM-CHMM-N and SM-CHMM-t because Pareto-sojourn clustering produces paths whose marginals are visibly more structured. Exception: SM-CHMM-L MMD IS is 4.0e-5, the single best value in the full 12-model panel.
5. **Christoffersen independence still fails** across every model (flat and SM). Unconditional VaR alone cannot capture clustered-breach structure in 2024-2026 OoS. This motivates the next step: Track C3 (time-varying transitions) or a conditional-VaR formulation off the Viterbi decode.

## Publishable interpretation

Semi-Markov structure is a **risk-calibration upgrade, not a marginal-fidelity upgrade**. The paper narrative splits cleanly:

- Flat **CHMM-t** for distributional matching (MMD, discriminator, stylized facts).
- **SM-CHMM-N or -t** for VaR back-testing and TSTR vol forecasting.
- **SM-CHMM-L** as a dark-horse candidate that ties both: best-in-panel MMD IS and clean Kupiec passes.

## Reproducing

```julia
using Pkg; Pkg.activate(".")
include("run_track_a_metrics.jl")     # builds the Track A archive (1 to 5 min, cached)
include("run_track_c1_smchmm.jl")     # fits SM-CHMMs and extends the panel (~3 to 5 min)
```
