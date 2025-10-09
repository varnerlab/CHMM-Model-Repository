function _jld2(path::String)::Dict{String,Any}
    return load(path);
end

MyPortfolioDataSet() = _jld2(joinpath(_PATH_TO_DATA, "train_dataset_2014_2023.jld2")); # Training dataset
MyOutOfSamplePortfolioDataSet() = _jld2(joinpath(_PATH_TO_DATA, "test_dataset_2024_onward.jld2")); # Out-of-sample dataset
MyOriginalPortfolioDataSet() = _jld2(joinpath(_PATH_TO_DATA, "SP500-Daily-OHLC-1-3-2014-to-02-07-2025.jld2")); # Original dataset