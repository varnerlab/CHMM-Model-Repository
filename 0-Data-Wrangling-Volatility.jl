# ========================================================================================= #
# 0-Data-Wrangling-Volatility.jl
#
# Reads VIX historical data from CSV and saves as JLD2 files for the analysis pipeline.
# Run the Python download first if /tmp/vix_full.csv does not exist:
#   python3 -c "import yfinance as yf; df=yf.Ticker('^VIX').history(period='20y'); ..."
#
# Output:
#   data/Volatility-Daily-OHLC-Train.jld2   (up to 2024-12-31)
#   data/Volatility-Daily-OHLC-Test.jld2    (2025-01-02 onward)
# ========================================================================================= #

using Pkg;
Pkg.activate(".");

using DataFrames;
using Dates;
using JLD2;
using FileIO;
using DelimitedFiles;

const _ROOT = @__DIR__;
const _PATH_TO_DATA = joinpath(_ROOT, "data");
mkpath(_PATH_TO_DATA);

# --- READ CSV ---
println("="^60);
println("  HMM-Vol Data Wrangling");
println("="^60);

csv_path = "/tmp/vix_full.csv";
if !isfile(csv_path)
    error("VIX CSV not found at $(csv_path). Run Python download first.");
end

# Parse CSV manually (avoid CSV.jl dependency)
lines = readlines(csv_path);
header = split(lines[1], ",");

dates = Date[];
open_vals = Float64[];
high_vals = Float64[];
low_vals = Float64[];
close_vals = Float64[];
volume_vals = Float64[];

for i in 2:length(lines)
    fields = split(lines[i], ",");
    if length(fields) >= 6
        try
            d = Date(fields[1][1:10]); # take first 10 chars for date
            o = parse(Float64, fields[2]);
            h = parse(Float64, fields[3]);
            l = parse(Float64, fields[4]);
            c = parse(Float64, fields[5]);
            v = tryparse(Float64, fields[6]);

            push!(dates, d);
            push!(open_vals, o);
            push!(high_vals, h);
            push!(low_vals, l);
            push!(close_vals, c);
            push!(volume_vals, isnothing(v) ? 0.0 : v);
        catch e
            continue;
        end
    end
end

full_df = DataFrame(
    date = dates,
    open = open_vals,
    high = high_vals,
    low = low_vals,
    close = close_vals,
    volume = volume_vals
);

println("  Total rows: $(nrow(full_df)) ($(full_df.date[1]) to $(full_df.date[end]))");

# --- SPLIT TRAIN/TEST ---
train_cutoff = Date(2024, 12, 31);
train_df = filter(row -> row.date <= train_cutoff, full_df);
test_df = filter(row -> row.date > train_cutoff, full_df);

println("  Train: $(nrow(train_df)) rows ($(train_df.date[1]) to $(train_df.date[end]))");
println("  Test:  $(nrow(test_df)) rows ($(test_df.date[1]) to $(test_df.date[end]))");

# --- SAVE AS JLD2 ---
# Package as Dict{String, DataFrame} matching the portfolio format
train_dataset = Dict{String, DataFrame}("VIX" => train_df);
test_dataset = Dict{String, DataFrame}("VIX" => test_df);

train_path = joinpath(_PATH_TO_DATA, "Volatility-Daily-OHLC-Train.jld2");
test_path = joinpath(_PATH_TO_DATA, "Volatility-Daily-OHLC-Test.jld2");

save(train_path, Dict("dataset" => train_dataset));
save(test_path, Dict("dataset" => test_dataset));

println("\n  Saved: $(train_path)");
println("  Saved: $(test_path)");
println("="^60);
println("  Done!");
