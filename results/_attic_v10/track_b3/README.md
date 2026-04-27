# `results/track_b3/`

Track B3: window-based diffusion baseline in pure Julia / Flux.

## Files

| File | Description |
| --- | --- |
| `Table-4-Extended-Metrics-B3.txt` | MMD, sig-MMD, discriminator AUC with diffusion row |
| `VaR_LR_tests_b3.txt` | Unconditional VaR Kupiec + Christoffersen with diffusion row |
| `sim_pvalues_b3.txt` | Joint stylized-fact p-value coverage with diffusion row |
| `Track-B3-summary.txt` | Short narrative summary |
| `Loss.svg` | Denoising loss curve |

## Outcome

This first-pass diffusion row is materially stronger than the GAN row on
short-window distributional fidelity:

- `MMD IS = 9.0e-5`
- `sig-MMD IS = 0.0026`
- `disc AUC IS = 0.565`

These beat the current CHMM rows on the window-local metrics panel. But the
same model is weak on joint stylized-fact coverage and unconditional breach
independence:

- `pv̄ OoS = 0.204` vs `CHMM-t 0.661`
- `LR_ind01 = 16.4`, `LR_ind05 = 7.37`

Interpretation: diffusion is now the strongest deep baseline in the repo on
local distributional fidelity, while CHMM-t remains stronger on global
stylized-fact coverage and the risk-management rows.
