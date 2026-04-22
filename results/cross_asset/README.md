# `results/cross_asset/`

Cross-asset dependence results on top of the per-asset CHMM-N marginals.

## Files

| File | Description |
| --- | --- |
| `Table-T3-Cross-Asset-Dependence.txt` | Table T3: SIM, Gaussian copula, Student-t copula, truncated C-vine |
| `Fig-Cross-Asset-Correlation.svg` | Mean simulated correlation matrices against the observed one |
| `Fig-Cross-Asset-KS-Dist.svg` | Per-asset in-sample KS pass-rate sanity check |

## C2 first-pass outcome

The repo now includes a truncated level-1 C-vine copula with edge-wise
Gaussian-vs-Student-t pair selection by AIC.

On the current six-asset panel (`SPY`, `NVDA`, `JNJ`, `JPM`, `AAPL`, `QQQ`):

- The selected root asset is `SPY`.
- All five vine edges pick Student-t pair-copulas.
- The vine preserves marginals well (mean IS KS pass rate `95.8%`).
- It does **not** beat the existing flat Student-t copula on dependence fidelity.

Observed comparison from `Table-T3-Cross-Asset-Dependence.txt`:

- Flat Student-t copula: IS off-diagonal correlation MAE `0.027`, OoS `0.210`
- Truncated C-vine: IS off-diagonal correlation MAE `0.067`, OoS `0.235`

Interpretation: the first-pass vine implementation is viable and scalable, but
for this small six-asset universe the existing flat Student-t copula remains
the stronger dependence model. The vine becomes more interesting as the asset
count grows, where the single global correlation matrix becomes less flexible.
