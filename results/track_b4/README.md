# Track B4 results (2026-04-22)

MS-GARCH(1,1) with K = 2 regimes (Haas-Mittnik-Paolella 2004) as the variance-regime baseline. Produced by `run_track_b4_msgarch.jl`.

## Files

| File | Content |
|---|---|
| `Table-4-Extended-Metrics-B4.txt` | Track-A + C1 panel with MS-GARCH row inserted after single-regime GARCH |
| `sim_pvalues_b4.txt` | Joint pv̄ coverage with MS-GARCH row |
| `VaR_LR_tests_b4.txt` | Unconditional VaR Kupiec + Christoffersen with MS-GARCH row |
| `Track-B4-summary.txt` | Auto-generated digest |
| `ms_garch_model.jld2` | Fitted MS-GARCH params (gitignored via pattern) |

## Fit (SPY IS, 2014 to 2024)

| Parameter | Regime 1 (calm) | Regime 2 (stress) |
|---|---|---|
| ω | 0.039 | 1.098 |
| α | 0.109 | 0.251 |
| β | 0.850 | 0.742 |
| α + β | 0.959 | 0.993 |
| Unconditional σ | 0.97 | 12.67 |

Common μ = 0.094. Transition stickiness: p_11 = 0.9145 (expected calm sojourn ~12 days), p_22 = 0.5474 (expected stress sojourn ~2 days). Log-likelihood = -4964.78.

Interpretation: calm regime is near-GARCH behaviour; stress regime has very high unconditional variance (uncond σ²=160, vs calm 0.94) and persistent clustering (α + β = 0.993). Stress is a rare, transient regime.

## Headline results

**Against the 12 other models:**

| Metric | MS-GARCH | Best in panel | Interpretation |
|---|---|---|---|
| MMD IS | 0.00048 | CHMM-N 0.00013 | Better than any non-CHMM row (excluding the GARCH numerical-zero artifact) |
| Sig-MMD IS | 0.031 | CHMM-t 0.0047 | Does not capture short-window path structure |
| Disc AUC IS | 0.734 | CHMM-t 0.607 | Better than single GARCH (0.766) and all i.i.d.; worse than CHMMs |
| pv̄ OoS | 0.487 | CHMM-L 0.692 | Comparable to single GARCH and Bootstrap |
| 1 % VaR LR_uc | **0.01** | tied with Laplace, SM-CHMM-N | **Best-in-panel** Kupiec coverage at 1 % (breach rate exactly 1.0 %) |
| 5 % VaR LR_uc | **0.26** | — | Best-in-panel Kupiec coverage at 5 % |
| 5 % VaR LR_ind | 4.79 | CHMM-L 4.71 | Closest to passing independence among unconditional-VaR rows |

**MS-GARCH is the best unconditional VaR calibrator** on this SPY OoS window. At 1 % VaR its Kupiec LR_uc is 0.01 (perfect coverage) and at 5 % it is 0.26 (best). Christoffersen independence is still marginal (4.79 at 5 %) but lower than any CHMM or GARCH row.

**MS-GARCH does not dominate CHMMs on marginals.** MMD is larger, discriminator AUC is higher (easier to distinguish), pv̄ OoS is lower. This is expected: a two-regime K=2 model can only capture a coarse partition; CHMMs at K=18 model the distribution more finely.

## Publishable interpretation

MS-GARCH earns a spot in Table 4 as the canonical variance-regime baseline. Its best-in-panel unconditional VaR calibration makes it a reviewer-defensible benchmark; its higher MMD and discriminator AUC justify the CHMM family's dominance on distributional fidelity. The story is:

- **Distributional fidelity**: CHMM-t.
- **Unconditional VaR Kupiec (breach-rate calibration)**: MS-GARCH (ties SM-CHMM-N on 1 %).
- **Conditional VaR Kupiec + Christoffersen**: flat CHMM-t with Viterbi decode (from C3a).

## Reproducing

```julia
using Pkg; Pkg.activate(".")
include("run_track_a_metrics.jl")     # builds base archive
include("run_track_c1_smchmm.jl")     # SM rows optional
include("run_track_b4_msgarch.jl")    # ~1 min fit + sim + metrics
```
