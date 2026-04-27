# ========================================================================================= #
# fetch_oos_extended.jl
#
# Extends the existing SP500 OoS dataset (2025-01-03 to 2025-11-18) by pulling
# fresh Alpaca daily bars from 2025-11-19 through 2026-04-20 and appending them
# to each ticker's DataFrame, preserving the Polygon/Massive column schema
# (volume, volume_weighted_average_price, open, close, high, low, timestamp,
#  number_of_transactions).
#
# Adjustment: "split" (matches the existing IS/OoS data, which is split-adjusted
# but not dividend-adjusted per Polygon's default).
# Feed:       "iex" (free tier).
#
# Output:
#   data/SP500-Daily-OHLC-1-3-2025-to-4-20-2026.jld2    (equity OoS, extended)
# ========================================================================================= #

using Pkg; Pkg.activate(".");
using Alpaca
using JLD2
using DataFrames
using Dates
using ProgressMeter

const CHUNK            = 50;       # symbols per Alpaca call
const START_EXT        = "2025-11-19";
const FINISH           = "2026-04-20";
const ADJUSTMENT       = "split";
const FEED             = "iex";
const TIMEFRAME        = "1Day";

const IN_OOS_PATH   = "data/SP500-Daily-OHLC-1-3-2025-to-11-18-2025.jld2";
const OUT_OOS_PATH  = "data/SP500-Daily-OHLC-1-3-2025-to-4-20-2026.jld2";

println("="^72);
println("  Extending OoS window via Alpaca daily bars");
println("  Existing OoS end:  2025-11-18");
println("  Extension:         $START_EXT → $FINISH   adj=$ADJUSTMENT  feed=$FEED");
println("="^72);

# --- Load client ---
client = load_client("../alpaca-markets-sdk/conf/apidata.toml");
println("[client] loaded OK");

# --- Load existing OoS frame ---
existing = load(IN_OOS_PATH)["dataset"];
tickers  = keys(existing) |> collect |> sort;
println("[load] existing OoS tickers: $(length(tickers)), rows per ticker: $(nrow(existing[first(tickers)]))");

# --- Helper: convert Alpaca Bar → NamedTuple matching Polygon schema ---
function _bar_to_row(b)::NamedTuple
    return (
        volume                         = b.v,
        volume_weighted_average_price  = isnothing(b.vw) ? (b.h + b.l) / 2 : b.vw,
        open                           = b.o,
        close                          = b.c,
        high                           = b.h,
        low                            = b.l,
        timestamp                      = b.t,
        number_of_transactions         = isnothing(b.n) ? 0 : b.n,
    );
end

# --- Batched Alpaca fetch for the extension window ---
println("\n[fetch] equities $START_EXT → $FINISH in chunks of $CHUNK...");
batches = [tickers[i:min(i + CHUNK - 1, end)] for i in 1:CHUNK:length(tickers)];
results = Dict{String,Vector{Alpaca.Bar}}();

prog = Progress(length(batches); desc="equity batches");
for (i, batch) in enumerate(batches)
    try
        part = get_bars(client, batch, TIMEFRAME;
            start=START_EXT, finish=FINISH, limit=10_000,
            adjustment=ADJUSTMENT, feed=FEED);
        merge!(results, part);
    catch err
        @warn "batch $i failed: $err"
    end
    next!(prog);
end

total_bars = sum(length(v) for v in values(results); init=0);
println("[fetch] pulled $total_bars bars across $(length(results)) tickers\n");

# --- Merge extension into existing DataFrames ---
println("[merge] appending extension rows to existing OoS frames...");

function _merge_extensions(existing, results)
    merged = Dict{String, DataFrame}();
    rows_added = 0;
    for (sym, old_df) in existing
        new_rows = get(results, sym, Alpaca.Bar[]);
        if isempty(new_rows)
            merged[sym] = old_df;
            continue;
        end
        ext_df = DataFrame([_bar_to_row(b) for b in new_rows]);
        select!(ext_df, names(old_df));
        merged[sym] = vcat(old_df, ext_df; cols=:orderequal);
        rows_added += nrow(ext_df);
    end
    return merged, rows_added;
end

merged, rows_added = _merge_extensions(existing, results);
println("[merge] appended $rows_added rows total (across $(length(merged)) tickers)");
println("[merge] AAPL new row count: $(nrow(merged["AAPL"])) (was $(nrow(existing["AAPL"])))");

# --- Save extended OoS JLD2 ---
jldsave(OUT_OOS_PATH; dataset=merged);
println("[save] wrote extended OoS → $OUT_OOS_PATH");

println("\n" * "="^72);
println("  DONE");
println("    $(basename(OUT_OOS_PATH)): $(length(merged)) tickers, last-day sample:");
sample = merged["SPY"];
println("    SPY last row: $(sample[end, :timestamp]) close=$(sample[end, :close])");
println("="^72);
