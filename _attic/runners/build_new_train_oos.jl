# ========================================================================================= #
# build_new_train_oos.jl
#
# Builds two fresh, non-overlapping datasets by combining the existing raw
# Polygon/Massive files with the Alpaca extension:
#
#   IS (training):     first 10 years of AAPL's continuous history
#                        2014-01-03 through the trading-day closest to the
#                        10-year anniversary (i.e. first 2520 rows of AAPL)
#   OoS (held-out):    everything after that, ending 2026-04-20
#
# Only tickers with a full trading-day history matching AAPL are kept
# (same convention as `run_baselines_and_cross_asset.jl`).
#
# Inputs  (raw, NOT modified):
#   data/SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2
#   data/SP500-Daily-OHLC-1-3-2025-to-4-20-2026.jld2
#
# Outputs (new files):
#   data/CHMM-SP500-Train-10yr.jld2
#   data/CHMM-SP500-OoS-Remainder.jld2
# ========================================================================================= #

using Pkg; Pkg.activate(".");
using JLD2
using DataFrames
using Dates

const IS_IN   = "data/SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2";
const OOS_IN  = "data/SP500-Daily-OHLC-1-3-2025-to-4-20-2026.jld2";
const TRAIN_OUT = "data/CHMM-SP500-Train-10yr.jld2";
const OOS_OUT   = "data/CHMM-SP500-OoS-Remainder.jld2";
const REF_TICKER  = "AAPL";
const TRAIN_YEARS = 10;

println("="^72);
println("  Rebuilding train / OoS split over the combined 2014-2026 window");
println("  Reference ticker: $REF_TICKER   training length: $TRAIN_YEARS years");
println("="^72);

# --- Load both halves ---
is_raw  = load(IS_IN)["dataset"];
oos_raw = load(OOS_IN)["dataset"];
println("[load] IS tickers:  $(length(is_raw))   rows(AAPL) = $(nrow(is_raw[REF_TICKER]))");
println("[load] OoS tickers: $(length(oos_raw))  rows(AAPL) = $(nrow(oos_raw[REF_TICKER]))");

# --- Combine IS + OoS per ticker (stack rows contiguously) ---
combined = Dict{String, DataFrame}();
for sym in keys(is_raw)
    is_df  = is_raw[sym];
    oos_df = get(oos_raw, sym, nothing);
    if oos_df === nothing
        combined[sym] = is_df;
    else
        combined[sym] = vcat(is_df, oos_df; cols=:orderequal);
    end
end
println("[merge] combined tickers: $(length(combined))   rows(AAPL) = $(nrow(combined[REF_TICKER]))");
println("[merge] AAPL date range:  $(combined[REF_TICKER][1, :timestamp]) → $(combined[REF_TICKER][end, :timestamp])");

# --- Determine 10-year boundary on AAPL ---
aapl_full = combined[REF_TICKER];
n_aapl = nrow(aapl_full);
# 10 trading years: anniversary of the first timestamp plus 10 years, round to nearest AAPL row.
anchor_date = Date(aapl_full[1, :timestamp]);
target_date = anchor_date + Year(TRAIN_YEARS);
# Last AAPL row on or before target_date
boundary_idx = something(findlast(ts -> Date(ts) <= target_date, aapl_full.timestamp), n_aapl - 1);
train_len = boundary_idx;
oos_len   = n_aapl - train_len;
println("[split] 10-year anchor: $anchor_date → target boundary $target_date");
println("[split] last training row on AAPL: row $train_len ($(aapl_full[train_len, :timestamp]))");
println("[split] OoS rows on AAPL:          $oos_len ($(aapl_full[train_len + 1, :timestamp]) → $(aapl_full[end, :timestamp]))");

# --- Ticker universe: only those with the same total-row count as AAPL ---
full_syms = sort([sym for (sym, df) in combined if nrow(df) == n_aapl]);
println("[filter] tickers with full AAPL-matched coverage: $(length(full_syms))");

train_ds = Dict{String, DataFrame}();
oos_ds   = Dict{String, DataFrame}();
for sym in full_syms
    df = combined[sym];
    train_ds[sym] = df[1:train_len, :];
    oos_ds[sym]   = df[(train_len + 1):end, :];
end

# --- Save new datasets (new filenames, originals untouched) ---
jldsave(TRAIN_OUT; dataset=train_ds);
jldsave(OOS_OUT;   dataset=oos_ds);

println();
println("="^72);
println("  WROTE");
println("    $TRAIN_OUT");
println("      tickers: $(length(train_ds))   rows/ticker: $train_len");
println("      range (AAPL):  $(train_ds[REF_TICKER][1, :timestamp]) → $(train_ds[REF_TICKER][end, :timestamp])");
println("    $OOS_OUT");
println("      tickers: $(length(oos_ds))     rows/ticker: $oos_len");
println("      range (AAPL):  $(oos_ds[REF_TICKER][1, :timestamp]) → $(oos_ds[REF_TICKER][end, :timestamp])");
println("="^72);
