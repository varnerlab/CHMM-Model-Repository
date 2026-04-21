# --- DATA LOADING ------------------------------------------------------------ #

"""
    _jld2(path::String) -> Dict{String,Any}

Private method: loads a JLD2 file and returns its contents as a dictionary.
"""
function _jld2(path::String)::Dict{String,Any}
    return load(path);
end

"""
    MyPortfolioDataSet() -> Dict{String,Any}

Loads the training dataset containing S&P 500 constituent OHLCV data.

### Coverage
- **Period**: January 3, 2014 -- December 31, 2024
- **Tickers**: 400+ US-listed equities and ETFs
- **Columns**: `date`, `open`, `high`, `low`, `close`, `volume`, `volume_weighted_average_price`

### Returns
- A `Dict` with key `"dataset"` mapping to `Dict{String, DataFrame}` (ticker => OHLCV DataFrame).
"""
MyPortfolioDataSet() = _jld2(joinpath(_PATH_TO_DATA, "CHMM-SP500-Train-10yr.jld2"));

"""
    MyOutOfSamplePortfolioDataSet() -> Dict{String,Any}

Loads the out-of-sample test dataset for model validation.

### Coverage
- **Period**: January 3, 2025 -- November 18, 2025
- **Tickers**: Same universe as training set

### Returns
- A `Dict` with key `"dataset"` mapping to `Dict{String, DataFrame}` (ticker => OHLCV DataFrame).
"""
MyOutOfSamplePortfolioDataSet() = _jld2(joinpath(_PATH_TO_DATA, "CHMM-SP500-OoS-Remainder.jld2"));

"""
    MyOriginalPortfolioDataSet() -> Dict{String,Any}

Loads the original (unsplit) dataset. Currently identical to the training set.

### Returns
- A `Dict` with key `"dataset"` mapping to `Dict{String, DataFrame}` (ticker => OHLCV DataFrame).
"""
MyOriginalPortfolioDataSet() = _jld2(joinpath(_PATH_TO_DATA, "CHMM-SP500-Train-10yr.jld2"));

"""
    MyVolatilityDataSet() -> Dict{String,Any}

Loads the VIX training dataset containing volatility index OHLCV data.

### Coverage
- **Period**: ~20 years up to December 31, 2024
- **Ticker**: VIX
- **Columns**: `date`, `open`, `high`, `low`, `close`, `volume`

### Returns
- A `Dict` with key `"dataset"` mapping to `Dict{String, DataFrame}` (ticker => OHLCV DataFrame).
"""
MyVolatilityDataSet() = _jld2(joinpath(_PATH_TO_DATA, "Volatility-Daily-OHLC-Train.jld2"));

"""
    MyOutOfSampleVolatilityDataSet() -> Dict{String,Any}

Loads the VIX out-of-sample test dataset.

### Coverage
- **Period**: January 2, 2025 onward
- **Ticker**: VIX

### Returns
- A `Dict` with key `"dataset"` mapping to `Dict{String, DataFrame}` (ticker => OHLCV DataFrame).
"""
MyOutOfSampleVolatilityDataSet() = _jld2(joinpath(_PATH_TO_DATA, "Volatility-Daily-OHLC-Test.jld2"));

# ----------------------------------------------------------------------------- #
