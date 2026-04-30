# ========================================================================================= #
# run_fetch_spy_independent_decade.jl
#
# Pull SPY daily bars covering 1994-01-01 to 2004-12-31 via the Alpaca Markets SDK
# (sibling repo `../alpaca-markets-sdk`). The body data pipeline (Polygon.io via Massive,
# Alpaca/IEX) covers 2014-01-03 onwards; for the independent-decade validation requested
# in the discussion (1994-2004 vs 2014-2024 SPY split) we need pre-2014 data outside the
# current pipeline.
#
# Alpaca's free IEX feed only covers 2016+; the SIP feed has full historical coverage but
# requires a paid subscription. We try IEX first and fall back to SIP if the user's
# credentials allow it. If neither feed returns data for 1994-2004, we report that and
# scope the validation to whatever decade is reachable.
#
# Output: data/spy_independent_decade.csv (alpaca-style bar columns)
# ========================================================================================= #

using Pkg; Pkg.activate(".");

const ALPACA_PATH = abspath(joinpath(@__DIR__, "..", "alpaca-markets-sdk"));
isdir(ALPACA_PATH) || error("Alpaca SDK not found at $ALPACA_PATH");

# Activate Alpaca SDK environment, then come back to the model project.
Pkg.develop(path=ALPACA_PATH);

using Alpaca, Dates

const OUT_DIR = joinpath(@__DIR__, "data", "independent_decade");
mkpath(OUT_DIR);

const TICKER       = "SPY";
const TARGET_START = Date(1994, 1, 1);
const TARGET_FINISH = Date(2004, 12, 31);

println("="^72)
println("  Fetching $TICKER daily bars $TARGET_START .. $TARGET_FINISH via Alpaca")
println("="^72)

const ALPACA_CRED = joinpath(ALPACA_PATH, "conf", "apidata.toml");
isfile(ALPACA_CRED) || error("Alpaca credentials not found at $ALPACA_CRED");
client = load_client(ALPACA_CRED);

# Try IEX first (free tier), then SIP (paid tier with full history).
function _try_fetch(client, ticker, start_date, finish_date, feed::String)
    try
        result = download_bars(client, ticker, "1Day";
            start = start_date, finish = finish_date,
            feed = feed, adjustment = "raw", verbose = true);
        bars = get(result, ticker, []);
        return (ok = !isempty(bars), bars = bars, err = nothing);
    catch e
        return (ok = false, bars = [], err = e);
    end
end

println("\n[1/2] IEX feed attempt...");
res_iex = _try_fetch(client, TICKER, TARGET_START, TARGET_FINISH, "iex");
println("  IEX: $(length(res_iex.bars)) bars");

println("\n[2/2] SIP feed attempt...");
res_sip = _try_fetch(client, TICKER, TARGET_START, TARGET_FINISH, "sip");
println("  SIP: $(length(res_sip.bars)) bars  $(res_sip.err === nothing ? "" : "(error: $(res_sip.err))")");

# Pick whichever has more bars
chosen = if length(res_sip.bars) > length(res_iex.bars)
    println("\n[choose] SIP feed has more bars; using SIP.");
    "sip", res_sip.bars
else
    println("\n[choose] IEX feed has equal or more bars; using IEX.");
    "iex", res_iex.bars
end
chosen_feed, chosen_bars = chosen;

if isempty(chosen_bars)
    println("\n[result] no data returned by either feed for the 1994-2004 window.");
    println("[result] Alpaca's data coverage on this account does not reach 1994.");
    println("[result] Logging the attempt; downstream runner should scope to available range.");
    open(joinpath(OUT_DIR, "fetch_log.txt"), "w") do io
        println(io, "Fetch attempt: $TICKER 1994-2004 via Alpaca SDK");
        println(io, "  iex bars: $(length(res_iex.bars))");
        println(io, "  sip bars: $(length(res_sip.bars))   (err: $(res_sip.err))");
        println(io, "  resolution: no data; scope downstream runner to available range");
    end
    exit(0);
end

# Diagnostic: what is the actual date range we got?
first_date = Date(first(chosen_bars).t);
last_date  = Date(last(chosen_bars).t);
println("\n[range] $chosen_feed feed returned $(length(chosen_bars)) bars: $first_date .. $last_date");

# Save as CSV in the same format as the rest of the pipeline
csv_path = joinpath(OUT_DIR, "$(TICKER)_independent_decade_$(chosen_feed).csv");
write_bars_csv(csv_path, chosen_bars);
println("[write] $csv_path");

# Brief summary file
open(joinpath(OUT_DIR, "fetch_log.txt"), "w") do io
    println(io, "Fetch attempt: $TICKER 1994-2004 via Alpaca SDK");
    println(io, "  iex bars: $(length(res_iex.bars))   first $(isempty(res_iex.bars) ? "n/a" : Date(first(res_iex.bars).t))   last $(isempty(res_iex.bars) ? "n/a" : Date(last(res_iex.bars).t))");
    println(io, "  sip bars: $(length(res_sip.bars))   first $(isempty(res_sip.bars) ? "n/a" : Date(first(res_sip.bars).t))   last $(isempty(res_sip.bars) ? "n/a" : Date(last(res_sip.bars).t))");
    println(io, "  err sip : $(res_sip.err)");
    println(io, "  chosen  : $chosen_feed");
    println(io, "  range   : $first_date .. $last_date");
    println(io, "  saved   : $csv_path");
end
println("\nDone.")
