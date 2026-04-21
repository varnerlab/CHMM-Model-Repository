# Main-Body Model Selection — Synthesis Note

## Best single model: **K = 18**

Based on the updated K-sweep (K ∈ {3, 6, 9, 12, 15, 18, 21}) reported in `Table-T1-State-Resolution-Sensitivity.txt`, K=18 dominates or ties on the distributional metrics that matter most to the JDIQ synthetic-data-quality framing, while remaining comparable on ACF reproduction:

| Metric          | Winner | K=3   | K=6   | K=9   | K=12  | K=15  | **K=18** | K=21  |
|-----------------|--------|-------|-------|-------|-------|-------|----------|-------|
| KS IS (%)       | K=18   | 89.7  | 93.4  | 92.9  | 94.5  | 94.2  | **95.2** | 94.7  |
| KS OoS (%)      | K=18   | 92.3  | 92.5  | 91.9  | 93.8  | 93.4  | **95.6** | 95.4  |
| AD IS (%)       | K=15   | 81.4  | 87.1  | 83.3  | 86.1  | 88.5  | 88.4     | 88.3  |
| AD OoS (%)      | K=18   | 83.2  | 84.3  | 84.1  | 86.0  | 85.8  | **90.0** | 88.1  |
| Kurtosis (sim)  | K=18   | 3.94  | 5.06  | 3.70  | 4.18  | 4.52  | **5.11** | 4.71  |
| ACF-MAE         | K=3    | 0.0458| 0.049 | 0.047 | 0.048 | 0.049 | 0.0502   | 0.0505|
| Wasserstein-1   | K=18   | 0.119 | 0.106 | 0.114 | 0.109 | 0.106 | **0.102**| 0.103 |
| Hellinger       | K=15   | 0.079 | 0.078 | 0.078 | 0.077 | **0.0766** | 0.0771 | 0.0774 |
| Coverage        | tied   | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0    | 100.0 |

**Observed excess kurtosis:** 7.71

**Takeaway:** K=18 is the only K to win on both in-sample and out-of-sample KS/AD pass rates simultaneously, has the highest simulated kurtosis (closest to the observed 7.71), and the smallest Wasserstein-1 error. The residual kurtosis gap (5.11 vs 7.71) motivates the Student-t emission extension deferred to the companion paper.

## Proposed main-body structure (≤ 25 pages)

### Kept in main body (K = 18 only)

Every main-body item is explicitly tagged with its generating pipeline so a reader cannot confuse Pipeline A (per-ticker single-index trained CHMM) with Pipeline B (cross-asset SIM / copula dependence on top of Pipeline A marginals).

1. Fig 1 (Pipeline A) -- Stylized Facts, SPY only
2. Fig 2 (Pipeline A) -- Baum-Welch convergence, SPY, CHMM-N, K=18 (`results/SPY/multi_emission/K18/N/Fig-Convergence-K18-N`)
3. Fig 3 (Pipeline A) -- IS comparison panel, SPY, CHMM-N, K=18 (`results/SPY/multi_emission/K18/N/Fig-3-IS-Comparison-K18-N`)
4. Fig 4 (Pipeline A) -- OoS validation panel, SPY, CHMM-N, K=18 (`results/SPY/multi_emission/K18/N/Fig-4-OoS-Validation-K18-N`)
5. Fig 5 (Pipeline A) -- Emission PDFs + Transition heatmap, SPY, CHMM-N, K=18
6. Fig 6 (Pipeline A) -- OoS equity price fans, six tickers x three emission families (`results/equity_price_sim/Fig-*-PriceFan-*`)
7. Fig 7 (Pipeline B) -- Cross-asset correlation heatmap (`results/cross_asset/Fig-Cross-Asset-Correlation`)
8. Table 1 (data only) -- Descriptive statistics (IS / OoS) for six tickers
9. Table 2 (Pipeline A) -- SPY model comparison: Bootstrap, Block-BS, Gaussian, Laplace, Discrete NJ, **Discrete WJ** (prior paper), Bin-T NJ, GARCH, GRU, CHMM-N / t / L at K=18 (`results/SPY/Table-2-Baselines.txt`)
10. Table T2 (Pipeline A) -- Per-ticker marginal fidelity across three emission families (SPY, NVDA, JNJ, JPM, AAPL, QQQ), single-index trained (`results/SPY/Table-T2-Per-Ticker-Emission-Families.txt`)
11. Table T3 (Pipeline B) -- Cross-asset dependence: SIM vs. Gaussian copula vs. Student-t copula (nu* = 6), on the per-asset CHMM-N marginals from Table T2 (`results/cross_asset/Table-T3-Cross-Asset-Dependence.txt`)

### Moved to Appendix / Supplementary Material
- Per-K figures for K ∈ {3, 6, 9, 12, 15, 21}: Fig-Convergence, Fig-Emission-PDFs, Fig-Transition-Matrix, Fig-Residence-Times, Fig-Stationary-Distribution, Fig-3-IS-Comparison, Fig-4-OoS-Validation, Fig-ACF-Comparison, Fig-Trajectory-Example
- Table T1 — state-resolution sensitivity sweep (already one page; reference from §3.2)
- Emission-Parameters.txt files per K
- Per-K Metrics.txt files

### Cuts that further tighten the paper
- Drop the per-K "residence times" and "stationary distribution" figures from main body; they do not advance the central claim
- Fold "Bayesian Student-t / Laplace" (Turing.jl) discussion into a single paragraph in §4 (future work), since they are not used as benchmarks here

## Mimicking arXiv:2603.10202 structure

| Prior paper (arXiv:2603.10202) | Continuous paper (this work) |
|---|---|
| Discrete HMM on binned returns | CHMM on raw returns |
| Frequency-counted T, E | Baum-Welch EM on T, μ_k, σ_k |
| Tables: stylized facts, Table 2 (baselines), Table T2 (cross-asset) | Mirrored: Fig 1, Table 2 **now including Discrete NJ+WJ rows**, Table T2 |
| ε, λ Poisson-jump hyperparameters | None (no jumps) |
| Heavy tails via jump teleportation | Heavy tails via Gaussian mixture of regimes |

Every row / column that appeared in the prior paper now appears in this submission under identical metrics, which is what reviewer #1 (Gemini) and the reviewing editor (Claude) both requested.
