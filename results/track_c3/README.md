# Track C3a results (2026-04-22)

Regime-conditional VaR via Viterbi decode. Directly attacks the Christoffersen-independence failure surfaced in Track A (A8) and confirmed in Track C1.

Produced by `run_track_c3_conditional_var.jl`.

## Files

| File | Content |
|---|---|
| `VaR_conditional_LR_tests.txt` | Kupiec LR_uc + Christoffersen LR_ind at 1 % and 5 % VaR for CHMM-N / -t / -L (flat) and three SM variants under two naive conditional-VaR approximations |
| `Track-C3a-summary.txt` | Auto-generated PASS/FAIL digest |

## Method

At each out-of-sample time t:

1. Run the existing `viterbi(R_oos, flat_model)` routine once per CHMM family to get the decoded regime sequence `k_t`.
2. Compute VaR_t(α) as the α-quantile of the state-k_t conditional emission distribution. For flat CHMM this is `quantile(model.emission[k_t], α)`.
3. Breach at t if `R_oos[t] <= VaR_t(α)`. Apply Kupiec + Christoffersen LR tests to the resulting breach sequence.

## Headline results (flat CHMM)

| Model | α | Breach % | LR_uc (was A8) | LR_ind (was A8) |
|---|---|---|---|---|
| CHMM-t | 1 % | 0.87 | 0.10 (was 0.58) | **0.09 (was 15.53)** |
| CHMM-t | 5 % | 5.07 | 0.01 (was 3.83) | **0.19 (was 5.26)** |
| CHMM-N | 1 % | 0.70 | 0.58 (was 1.58) | 5.73 (was 18.98) |
| CHMM-N | 5 % | 5.07 | 0.01 (was 3.83) | **1.39 (was 5.26)** |
| CHMM-L | 1 % | 0.35 | 3.26 (was 3.26) | **0.01 (was 9.15)** |
| CHMM-L | 5 % | 3.32 | 3.83 (was 3.03) | **2.25 (was 4.71)** |

**CHMM-t passes Kupiec AND Christoffersen cleanly at both 1 % and 5 % VaR.** All three flat CHMMs pass Christoffersen at 5 % VaR. This closes the open Christoffersen-independence question from A8 / C1.

## SM-CHMM is harder

Viterbi decoding in this repo uses the flat CHMM's emission model. When the decoded state index is handed to the SM-CHMM (which has different per-state emission parameters after plug-in refitting), the state label does not match SM's internal partition. Two naive fallbacks were tried:

1. **AR-res**: `VaR_t = μ_k + φ_k (y_{t-1} - μ_k) + σ_k * residual-α-quantile`. Breach rate inflates to 15 to 33 %: the AR(1) residual scale is far narrower than the unconditional return scale.
2. **stat-σ**: `VaR_t = μ_k + σ_k / sqrt(1 - φ_k^2) * residual-α-quantile`. Slightly better but still 9 to 30 % breach rate.

Interestingly **SM-t (stat-σ) achieves LR_ind 0.07 at 1 % and 0.44 at 5 %** (clean independence), even though LR_uc fails. The clustering pattern is captured; the breach level is not. This says the Viterbi-decoded state path does carry the right regime information, but the SM state-k emission scale is calibrated to the AR(1) residual rather than the unconditional return. Proper SM conditional VaR requires an SM-aware Viterbi decoder that uses the SM model's own transition + sojourn + AR(1) emission structure, rather than borrowing flat-CHMM Viterbi. That is future work.

## Publishable interpretation

Combine with C1 findings for a clean paper narrative:

- **Marginal-fidelity winner**: flat CHMM-t (best MMD, best discriminator AUC).
- **Unconditional VaR winner**: SM-CHMM-N / -t (clean Kupiec passes at 1 % and 5 %).
- **Conditional VaR winner**: flat CHMM-t with Viterbi decode (passes Kupiec AND Christoffersen cleanly).

Each asks a different operational question and the paper can make a concrete recommendation per use case.

## Reproducing

```julia
using Pkg; Pkg.activate(".")
include("run_track_c1_smchmm.jl")         # builds SM cache if needed
include("run_track_c3_conditional_var.jl") # ~1 min, uses cached SM models
```
