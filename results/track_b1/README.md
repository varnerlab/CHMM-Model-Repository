# `results/track_b1/`

Track B1: QuantGAN-style deep-generative baseline in pure Julia / Flux.

## Files

| File | Description |
| --- | --- |
| `Table-4-Extended-Metrics-B1.txt` | MMD, sig-MMD, discriminator AUC with QuantGAN row |
| `VaR_LR_tests_b1.txt` | Unconditional VaR Kupiec + Christoffersen with QuantGAN row |
| `sim_pvalues_b1.txt` | Joint stylized-fact p-value coverage with QuantGAN row |
| `Track-B1-summary.txt` | Short narrative summary |
| `Loss.svg` | Generator / critic training curves |

## Outcome

This first-pass QuantGAN-style baseline is intentionally pragmatic:

- 1D convolutional generator and critic
- Wasserstein loss with critic weight clipping
- rolling-window training on standardized SPY returns
- full-path simulation by stitching generated windows

On the current SPY panel it is **not competitive** with the CHMM family:

- `MMD IS = 0.06373`
- `sig-MMD IS = 0.12877`
- `disc AUC IS = 0.963`

For comparison:

- `CHMM-t`: `0.00019`, `0.00595`, `0.607`
- `CHMM-N`: `0.00013`, `0.00924`, `0.646`
- `CHMM-L`: `0.00099`, `0.00601`, `0.623`

Interpretation: the repo now has a real deep-generative GAN row, but under this
reproducible first-pass setup it behaves as a negative-control baseline rather
than a challenger to CHMM-t.
