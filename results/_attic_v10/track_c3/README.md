# Track C3 results (2026-04-22)

Regime-conditional VaR via Viterbi decode. Directly attacks the Christoffersen-independence failure surfaced in Track A (A8) and confirmed in Track C1.

Produced by `run_track_c3_conditional_var.jl`, `run_track_c3_time_varying_transition.jl`, and `run_track_c3_external_covariates.jl`.

## Files

| File | Content |
|---|---|
| `VaR_conditional_LR_tests.txt` | Kupiec LR_uc + Christoffersen LR_ind at 1 % and 5 % VaR for CHMM-N / -t / -L (flat) and three SM variants under two naive conditional-VaR approximations |
| `Track-C3a-summary.txt` | Auto-generated PASS/FAIL digest |
| `VaR_time_varying_transition_LR_tests.txt` | Flat-vs-time-varying transition VaR LR tests for CHMM-N / -t / -L |
| `Track-C3-summary.txt` | Short summary of the time-varying transition run |
| `VaR_external_covariate_transition_LR_tests.txt` | Flat-vs-external-covariate transition VaR LR tests for CHMM-N / -t / -L |
| `Track-C3-external-summary.txt` | Short summary of the lagged-RV + VIX run |

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

## C3 first-pass: time-varying transitions

The repo now also includes a first-pass C3 script that replaces the flat
transition row with a lagged-realized-volatility logistic transition model

`P(s_t = j | s_{t-1} = i, x_t) = softmax(a_ij + b_ij x_t)`.

This is intentionally narrower than the full memo target:

- Covariate: standardized lagged 20-day realized volatility only
- No external `VIX` or term-spread data yet
- Emissions stay fixed; only the transition row changes

Headline outcome:

- `CHMM-t TVT` does **not** improve meaningfully on the already-strong `C3a`
  flat conditional-VaR result. At 1 % VaR it remains clean (`LR_uc 0.01`,
  `LR_ind 0.13`), but the flat row was already clean (`0.10`, `0.09`).
- `CHMM-L TVT` materially improves unconditional coverage relative to flat
  `CHMM-L` at both 1 % and 5 % while keeping `LR_ind` below the 3.84 threshold.
- `CHMM-N TVT` worsens independence relative to the flat C3a row.

Interpretation: transition dynamics help some families, but the main paper
headline remains `C3a`: flat `CHMM-t` with Viterbi-decoded conditional VaR is
already the strongest risk-management row.

## C3 external-covariate pass: lagged RV + VIX

The repo now also includes an external-covariate variant that augments the
transition logits with lagged log-VIX from the sibling `CHMM-Vol-Model` repo:

`P(s_t = j | s_{t-1} = i, x_t) = softmax(a_ij + b_ij' x_t)`,

with `x_t = [ standardized lagged RV(20), standardized lagged log(VIX) ]`.

This closes the main implementation gap left by the first-pass C3 script:

- External covariate now exists (`VIX`)
- Still no term-spread covariate
- Emissions remain fixed; only the transition row changes

Headline outcome:

- `CHMM-t XTV` stays clean at both 1 % and 5 % VaR
  (`LR_uc 0.82 / 0.26`, `LR_ind 0.23 / 2.23`), but still does **not** beat the
  flat `C3a` row.
- `CHMM-L XTV` is the strongest mover: it improves 1 % and 5 % unconditional
  coverage materially relative to flat `CHMM-L` while keeping independence
  clean (`LR_ind 0.29 / 1.64`).
- `CHMM-N XTV` gets worse on both 1 % and 5 % independence.

Interpretation: adding an external volatility covariate is viable and useful as
an ablation, but the main paper claim should still center `C3a` flat
`CHMM-t` conditional VaR rather than the time-varying transition extensions.

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
