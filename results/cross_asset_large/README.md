# `results/cross_asset_large/`

Track C2 large-universe dependence-layer scaling on a 50-asset universe.

## Files

| File | Description |
| --- | --- |
| `Table-C2-Large-Universe.txt` | Correlation reproduction and equal-weight portfolio VaR for Gaussian / Student-t / truncated C-vine |
| `Track-C2-Large-summary.txt` | Short narrative summary |

## Outcome

The large-universe C2 run uses the top 50 common tickers by in-sample dollar
volume and isolates the dependence layer by preserving exact empirical
marginals via rank reordering.

Headline result:

- Gaussian copula: `MAE IS 0.035`, `MAE OoS 0.174`
- Student-t copula: `MAE IS 0.037`, `MAE OoS 0.181`
- Truncated C-vine: `MAE IS 0.075`, `MAE OoS 0.189`

So even at 50 assets, the first-pass truncated C-vine does **not** beat the
flat copulas on correlation reproduction. It does improve the equal-weight
portfolio 5 % VaR Kupiec LR relative to the flat copulas (`4.74` vs `8.18`),
but not enough to reverse the main correlation result.

Interpretation: the repo now has a genuine large-universe C2 scaling result,
and it is a negative one for the current vine implementation.
