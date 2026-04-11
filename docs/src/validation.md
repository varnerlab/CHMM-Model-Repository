# Validation

## Validation Strategy

Model quality is assessed across multiple statistical dimensions to ensure simulated paths reproduce the stylized facts of financial returns.

## Autocorrelation Analysis

The primary validation tool compares ACF profiles between observed and simulated data:

```julia
# Returns ACF (should show no significant autocorrelation)
plot_acf_comparison(observed, simulated, "Returns ACF", path_idx)

# Absolute returns ACF (should show slow decay = volatility clustering)
plot_acf_comparison(observed, simulated, "|Returns| ACF", path_idx; is_absolute=true)
```

### What to Look For

- **Returns ACF**: all lags should fall within the 99% confidence interval (gray dashed lines). Significant autocorrelation in returns would indicate a failure to capture market efficiency.
- **Absolute returns ACF**: should show positive, slowly decaying autocorrelation out to ~100+ lags. This is the signature of volatility clustering -- large moves followed by large moves.

## Distribution Comparison

Visual comparison of simulated vs. observed return distributions:

```julia
# Overlay PDFs
histogram(observed, normalize=true, label="Observed", alpha=0.5)
histogram!(simulated, normalize=true, label="Simulated", alpha=0.5)
```

### Key Properties

- **Heavy tails**: the simulated distribution should exhibit excess kurtosis comparable to the observed data
- **Near-zero skewness**: returns should be approximately symmetric
- **Peak height**: the mode of the simulated distribution should match the observed

## Statistical Tests

### Kolmogorov-Smirnov Test
```julia
using HypothesisTests
ks_result = ApproximateTwoSampleKSTest(observed, simulated)
```

A non-significant p-value indicates the simulated and observed distributions are not detectably different.

## Convergence Diagnostics

Always inspect the Baum-Welch convergence before using a model:

```julia
plot(model.log_likelihood_history,
    xlabel="Iteration", ylabel="Log-Likelihood",
    title="Baum-Welch Convergence", lw=2)
```

The log-likelihood should:
- Be **monotonically increasing** (EM guarantee)
- **Plateau** before `max_iter` is reached
- Not show **oscillation** (would indicate numerical issues)

## Out-of-Sample Testing

Train on historical data, validate on held-out future data:

```julia
# Train on 2014-2024
dataset = MyPortfolioDataSet()["dataset"]
R_train = log_growth_matrix(dataset, "SPY")
model = build(MyContinuousHiddenMarkovModel, (observations=R_train, number_of_states=13, max_iter=60))

# Validate on 2025
oos_dataset = MyOutOfSamplePortfolioDataSet()["dataset"]
R_test = log_growth_matrix(oos_dataset, "SPY")

# Compare distributions
ks_result = ApproximateTwoSampleKSTest(R_test, simulated_from_model)
```
