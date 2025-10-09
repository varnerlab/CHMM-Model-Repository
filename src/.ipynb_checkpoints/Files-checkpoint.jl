function _jld2(path::String)::Dict{String,Any}
    return load(path);
end

MyPortfolioDataSet() = _jld2(joinpath(_PATH_TO_DATA, "SP500-Daily-OHLC-1-3-2014-to-02-07-2025.jld2"));
MyOutOfSamplePortfolioDataSet() = _jld2(joinpath(_PATH_TO_DATA, "SP500-Daily-OHLC-1-3-2024-to-10-25-2024.jld2"));