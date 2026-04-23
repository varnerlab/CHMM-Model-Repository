# Track A results

Extended distributional and tail-fidelity evaluation of the CHMM family against the baseline panel: MMD / signature-MMD / discriminator AUC, leverage-effect profile, aggregational kurtosis across horizons, joint simulation-based p-values, and Kupiec / Christoffersen VaR LR tests.

All tables produced by `run_track_a_metrics.jl` and `run_track_a_utility.jl`. Seed `20260422`, N_PATHS = 1000 paths/model, K = 18 for CHMM-N / -t / -L.

## Files

| File | Items covered |
|---|---|
| `Table-4-Extended-Metrics.txt` | A1 MMD (RBF), A2 sig-MMD (depth-3), A3 discriminator AUC |
| `leverage_effect.txt` | A6 avg corr(r_t^2, r_{t-k}) + down/up asymmetry + sim p-value |
| `aggregational_kurtosis.txt` | A7 kurtosis at horizons {1, 5, 10, 21} |
| `sim_pvalues.txt` | A9 per-stat sim-based p-values and joint pv̄ |
| `tstr_vol_forecaster.txt` | A4 HAR(1,5,22) TSTR RMSE + QLIKE |
| `vol_target_strategy.txt` | A5 vol-target SPY Sharpe / MDD / turnover |
| `VaR_LR_tests.txt` | A8 Kupiec + Christoffersen LR at 1 % and 5 % VaR |
| `sim_archive_cache.jld2` | Cached 1000-path simulation archives for all 9 models |
| `Track-A-summary.txt` | Auto-generated one-page digest |

## Headline findings

1. **CHMM-t wins distributional fidelity.** MMD IS 2.0e-5 (vs 0.009 Gaussian), discriminator AUC IS 0.607 (closest to 0.5 of any model), aggregational kurtosis OoS h=1 5.55 vs observed 5.29.
2. **CHMM-L wins joint OoS coverage.** pv̄ OoS 0.692 (vs CHMM-t 0.661, CHMM-N 0.539, GARCH 0.468).
3. **CHMM-N synth-trained HAR ties real-trained.** QLIKE 1.519 vs 1.521 on OoS (ratio 0.999); RMSE ratio 0.988.
4. **Discrete NJ and Discrete WJ fail 5 % VaR Kupiec.** Breach rate 1.75 % vs target 5 %, LR_uc 16.81, p < 0.001. Materially over-conservative: a concrete negative finding against the prior-paper baseline.
5. **Christoffersen independence fails across all generators.** LR_ind 6.5 to 20.9 at 1 % VaR, 4.7 to 13.2 at 5 % VaR on the 2024-2026 OoS window. Unconditional VaR cannot capture clustered-breach structure. Follow-up work: conditional VaR via Viterbi decode (C1 semi-Markov port makes this natural).

## Reproducing

```julia
using Pkg; Pkg.activate(".")
include("run_track_a_metrics.jl")     # ~5 to 15 minutes, caches sim archives
include("run_track_a_utility.jl")     # ~1 minute, reuses cache
```
