# Finance Utilities

## Excess Log-Growth Rates

The core financial calculation converts price series to annualized excess log returns:

```
G_t = (1/dt) * ln(P_t / P_{t-1}) - r_f
```

where `dt = 1/252` (daily) and `r_f` is the risk-free rate (default 0).

### Multiple Firms from Dataset Dictionary

```julia
# Returns a Matrix (time x firms)
R = log_growth_matrix(dataset, ["SPY", "AAPL", "NVDA"];
    Dt = 1/252, risk_free_rate = 0.0,
    keycol = :volume_weighted_average_price)
```

### Single Firm from Dataset Dictionary

```julia
# Returns a Vector
R = log_growth_matrix(dataset, "SPY")
```

### From a DataFrame

```julia
# When you already have a single firm's DataFrame
R = log_growth_matrix(spy_df; keycol = :close)
```

### From a Raw Price Vector

```julia
# From a plain array of prices
R = log_growth_matrix(price_vector; Dt = 1/252)
```

## Volume Weighted Average Price (VWAP)

```julia
vwap_values = vwap(df)
```

Computes cumulative VWAP from a DataFrame with `high`, `low`, `close`, and `volume` columns:

```
Typical Price = (High + Low + Close) / 3
VWAP_t = sum(TP_i * V_i, i=1..t) / sum(V_i, i=1..t)
```

## Data Loading

Three convenience functions load JLD2 datasets:

| Function | Dataset | Period |
|----------|---------|--------|
| `MyPortfolioDataSet()` | Training set | Jan 2014 -- Dec 2024 |
| `MyOutOfSamplePortfolioDataSet()` | Test set | Jan 2025 -- Nov 2025 |
| `MyOriginalPortfolioDataSet()` | Original full set | Jan 2014 -- Dec 2024 |

Each returns a `Dict` with key `"dataset"` containing a `Dict{String, DataFrame}` mapping ticker symbols to OHLCV DataFrames.
