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
- **Period**: January 3, 2014 -- January 3, 2024 (ten-year in-sample window)
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
- **Period**: January 4, 2024 -- April 20, 2026 (held-out out-of-sample window)
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

# ----------------------------------------------------------------------------- #
