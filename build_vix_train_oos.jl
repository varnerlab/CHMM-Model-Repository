# ========================================================================================= #
# build_vix_train_oos.jl
#
# Downloads CBOE VIX daily history, aligns to the equity (AAPL-anchored) 10-year
# training window used by `build_new_train_oos.jl`, and saves matched train / OoS
# JLD2 files for the Block D volatility-index generalization.
#
# Source:
#   https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
#   (CBOE Global Indices daily-prices feed, public CSV, MM/DD/YYYY, OHLC)
#
# Outputs:
#   data/VIX-Train-10yr.jld2       (matched to equity IS window)
#   data/VIX-OoS-Remainder.jld2    (matched to equity OoS window)
#
# Usage: julia --project=. build_vix_train_oos.jl
# ========================================================================================= #

using Pkg; Pkg.activate(".");
using JLD2
using DataFrames
using Dates
using CSV
using Downloads

const VIX_URL       = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv";
const RAW_CSV_PATH  = "data/VIX_History.csv";
const EQUITY_TRAIN  = "data/CHMM-SP500-Train-10yr.jld2";
const EQUITY_OOS    = "data/CHMM-SP500-OoS-Remainder.jld2";
const TRAIN_OUT     = "data/VIX-Train-10yr.jld2";
const OOS_OUT       = "data/VIX-OoS-Remainder.jld2";
const REF_TICKER    = "AAPL";

println("="^72);
println("  VIX daily-history download and train / OoS alignment");
println("  Feed: CBOE Global Indices (VIX_History.csv, daily OHLC)");
println("="^72);

# --- Download raw CSV (idempotent; refresh if local copy is older than 24h) ---
function _need_download(path::String)::Bool
    isfile(path) || return true;
    age_hours = (time() - mtime(path)) / 3600;
    return age_hours > 24;
end

if _need_download(RAW_CSV_PATH)
    println("[fetch] downloading VIX history from CBOE...");
    mkpath(dirname(RAW_CSV_PATH));
    Downloads.download(VIX_URL, RAW_CSV_PATH);
    println("[fetch] saved to $(RAW_CSV_PATH)  ($(round(filesize(RAW_CSV_PATH)/1024, digits=1)) KB)");
else
    println("[fetch] using cached $(RAW_CSV_PATH)");
end

# --- Parse CBOE VIX CSV (DATE,OPEN,HIGH,LOW,CLOSE; date format MM/DD/YYYY) ---
raw = CSV.read(RAW_CSV_PATH, DataFrame;
    header=1, missingstring="", stripwhitespace=true);
rename!(raw, lowercase.(names(raw)));
raw.date = Date.(raw.date, dateformat"mm/dd/yyyy");
sort!(raw, :date);
println("[parse] VIX rows: $(nrow(raw))   range: $(raw.date[1]) to $(raw.date[end])");
println("[parse] columns : $(join(names(raw), ", "))");

# --- Pull equity train / OoS calendars (same AAPL-anchored split used in equity pipeline) ---
eq_train = load(EQUITY_TRAIN)["dataset"][REF_TICKER];
eq_oos   = load(EQUITY_OOS)["dataset"][REF_TICKER];
eq_train_dates = Date.(eq_train.timestamp);
eq_oos_dates   = Date.(eq_oos.timestamp);
eq_first = eq_train_dates[1];
eq_split = eq_oos_dates[1];
eq_last  = eq_oos_dates[end];
println("[equity] train window: $(eq_first) to $(eq_train_dates[end])  ($(length(eq_train_dates)) obs)");
println("[equity] OoS   window: $(eq_split) to $(eq_last)  ($(length(eq_oos_dates)) obs)");

# --- Subset VIX to equity's trading-day calendar (inner-join on date) ---
train_mask = [d >= eq_first && d < eq_split for d in raw.date];
oos_mask   = [d >= eq_split && d <= eq_last for d in raw.date];

train_df = raw[train_mask, :];
oos_df   = raw[oos_mask, :];

# Keep only rows whose date is also an equity trading day
train_df = train_df[[d in Set(eq_train_dates) for d in train_df.date], :];
oos_df   = oos_df[[d in Set(eq_oos_dates) for d in oos_df.date], :];

# Standardise column order and add a `timestamp` column for parity with equity frames
for d in (train_df, oos_df)
    if !(:timestamp in Symbol.(names(d)))
        d[!, :timestamp] = DateTime.(d.date);
    end
end
select!(train_df, [:timestamp, :date, :open, :high, :low, :close]);
select!(oos_df,   [:timestamp, :date, :open, :high, :low, :close]);

println("[align]  VIX train rows: $(nrow(train_df)) / $(length(eq_train_dates)) equity dates");
println("[align]  VIX OoS   rows: $(nrow(oos_df)) / $(length(eq_oos_dates)) equity dates");
missing_train = length(eq_train_dates) - nrow(train_df);
missing_oos   = length(eq_oos_dates) - nrow(oos_df);
if missing_train > 0 || missing_oos > 0
    println("[align]  WARN: missing $(missing_train) train / $(missing_oos) OoS rows (CBOE vs equity calendar drift)");
end

# --- Descriptive stats on log(VIX) for the training window ---
log_vix_train = log.(train_df.close);
dlog_vix_train = diff(log_vix_train);
using Statistics
μL = mean(log_vix_train); σL = std(log_vix_train);
μD = mean(dlog_vix_train); σD = std(dlog_vix_train);
skew_D = mean(((dlog_vix_train .- μD) ./ σD).^3);
kurt_D = mean(((dlog_vix_train .- μD) ./ σD).^4) - 3.0;
println();
println("[stats]  log(VIX) train:   μ=$(round(μL, digits=4))   σ=$(round(σL, digits=4))   " *
        "min=$(round(minimum(log_vix_train), digits=3))   max=$(round(maximum(log_vix_train), digits=3))");
println("[stats]  Δlog(VIX) train:  μ=$(round(μD, digits=5))   σ=$(round(σD, digits=4))   " *
        "skew=$(round(skew_D, digits=3))   excess-kurt=$(round(kurt_D, digits=3))");

# --- Save JLD2 datasets ---
jldsave(TRAIN_OUT; dataset=Dict("VIX" => train_df));
jldsave(OOS_OUT;   dataset=Dict("VIX" => oos_df));

println();
println("="^72);
println("  WROTE");
println("    $(TRAIN_OUT)   rows=$(nrow(train_df))  range: $(train_df.date[1]) to $(train_df.date[end])");
println("    $(OOS_OUT)     rows=$(nrow(oos_df))   range: $(oos_df.date[1]) to $(oos_df.date[end])");
println("="^72);
