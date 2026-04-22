# `results/track_c4/`

Track C4: leverage-emission ablation on top of flat CHMM-N.

## Files

| File | Description |
| --- | --- |
| `leverage_emission_ablation.txt` | Main C4 comparison: flat CHMM-N vs CHMM-N-Lev on leverage metrics |
| `Track-C4-summary.txt` | Short narrative summary of the ablation result |

## Headline

`run_track_c4_leverage_emission.jl` adds a per-state negative-return term
`r_t = μ_k + ρ_k * min(r_{t-1}, 0) + σ_k * ε_t` after Viterbi-decoding the flat
CHMM-N state path.

Relative to flat CHMM-N:

- OoS leverage-effect coverage p-value improves from `0.205` to `0.308`.
- IS discriminator AUC improves from `0.646` to `0.594`.
- OoS avg leverage magnitude moves from `-0.0332` to `-0.0359`.
- Unconditional VaR calibration worsens at 1 % (`LR_uc 3.26` vs `1.58`) and is
  only marginal at 5 % (`LR_uc 3.83`).

Interpretation: C4 is useful as a one-row stylized-fact ablation. It improves
leverage fidelity and distinguishability, but it is not a new headline model.
