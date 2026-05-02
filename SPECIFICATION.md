# SPECIFICATION.md -- Design Philosophy and Architecture

## Motivation

The discrete HMM approach in [JumpHMM.jl](https://github.com/varnerlab/JumpHMM.jl) discretizes continuous returns into bins, fits transition matrices by frequency counting, and models emissions with Student-t distributions. While effective with Poisson jump augmentation, this introduces quantization error and limits fine-grained regime dynamics.

This project extends the methodology to **continuous emissions** -- each hidden state emits from a per-state density (Gaussian, Student-$t$ with per-state $\nu_k$, Laplace, or GED with per-state shape $p_k$) -- trained via the **Baum-Welch / ECM algorithm** under a unified scaffold where the M-step is the only architectural difference across emission families. At moderate $K$ the continuous HMM reproduces the three symmetric Cont (2001) stylized facts (heavy tails, negligible linear ACF, slow $|G_t|$ ACF) without requiring jump mechanisms; CHMM-GED's per-state $\hat p_k$ partitions bimodally into a Gaussian-bulk / Laplace-tail structure that replicates across seeds and tickers.

## Model Comparison

| Model | Training | Emissions |
|-------|----------|-----------|
| Discrete HMM + Jumps | Frequency counting on binned returns + Poisson teleportation | Categorical (bin indices) |
| Continuous HMM (Baum-Welch / ECM, four-family) | EM / ECM on raw returns; M-step branches on family | Gaussian / Student-$t$ ($\nu_k$) / Laplace / GED ($p_k$) per state |
| ML hidden semi-Markov | Yu (2010) explicit-duration EM | Gaussian per state, truncated Pareto / discrete-Gamma sojourn |
| GARCH family | Maximum likelihood estimation | Single regime, conditional Normal or Student-$t$ |
| Markov-switching GARCH | In-house Nelder-Mead or reference Bayesian (`MSGARCH` R) | Per-regime conditional Normal |

## Architecture

### Separation of Concerns

| Module | Responsibility |
|--------|---------------|
| `Types.jl` | Data structures only. No logic. |
| `Files.jl` | Data I/O. Maps file paths to JLD2 datasets. |
| `Factory.jl` | Construction. Every model via `build()`. |
| `Compute.jl` | Algorithms. Baum-Welch, GARCH MLE, simulation, growth calculations. |
| `CrossAsset.jl` | Single Index Model and Gaussian/Student-t copula generators. |
| `Visualize.jl` | Plotting. Validation visualization. |

### Factory Pattern

All model construction goes through `build(ModelType, data::NamedTuple)`:
- `build(MyHiddenMarkovModel, ...)` -- discrete HMM from T/E matrices
- `build(MyHiddenMarkovModelWithJumps, ...)` -- discrete + Poisson jumps
- `build(MyContinuousHiddenMarkovModel, ...)` -- trains via Baum-Welch
- `build(MyGARCHModel, ...)` -- fits via MLE

### Functor Interface

HMM models implement Julia's callable-object protocol:
```julia
(m::MyContinuousHiddenMarkovModel)(start, steps) = _simulate(m, start, steps)
```

## Algorithm: Baum-Welch (EM)

**Initialization**: Quantile-based. Sorted observations partitioned into K chunks.

**Forward-Backward in Log-Space**: All computations use `logsumexp` for numerical stability.

**Convergence**: Terminates when |ΔLL| < tol (1e-4) or max_iter reached.

## Algorithm: GARCH(1,1)

**Model**: σ²_t = ω + α * (r_{t-1} - μ)² + β * σ²_{t-1}

**Fitting**: Grid-initialized Nelder-Mead optimization of the Gaussian log-likelihood. Constraints: ω > 0, α ≥ 0, β ≥ 0, α + β < 1 (stationarity).

**Simulation**: Sequential sampling with conditional variance updating.

## Algorithm: Discrete HMM + Poisson Jumps

**Discretization**: Returns binned into K quantiles. Transition matrix from consecutive bin counts.

**Jump Mechanism**: At each step with probability ε, enter jump mode for Poisson(λ) steps. During jumps, coin flip (52/48) selects crash states (1:3) or boom states (K-2:K).

## Validation Strategy

1. **Convergence diagnostics** -- log-likelihood history
2. **Distributional match** -- density comparison, Q-Q plots
3. **ACF of returns** -- should show no autocorrelation
4. **ACF of |returns|** -- slow decay (volatility clustering)
5. **Statistical tests** -- KS test, Wasserstein distance, Hellinger distance
6. **Excess kurtosis** -- tail heaviness matching
7. **Out-of-sample** -- 2025 holdout validation

## Future Directions

- Multi-asset portfolio simulation with dependence modeling
- Automatic state-number selection (BIC/AIC)
- Student-t emissions in Baum-Welch loop
- Formal package registration and CI/CD pipeline
