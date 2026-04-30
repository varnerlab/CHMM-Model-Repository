# Probe Alpaca data depth for SPY: find the earliest year with data on each feed.

using Pkg; Pkg.activate(".");

const ALPACA_PATH = abspath(joinpath(@__DIR__, "..", "alpaca-markets-sdk"));
using Alpaca, Dates

const ALPACA_CRED = joinpath(ALPACA_PATH, "conf", "apidata.toml");
client = load_client(ALPACA_CRED);

println("Alpaca SPY data-depth probe");
for year in 1994:2:2018
    for feed in ["iex", "sip"]
        try
            res = get_bars(client, "SPY", "1Day";
                start = Date(year, 1, 1), finish = Date(year, 12, 31),
                feed = feed);
            bars = get(res, "SPY", []);
            n = length(bars);
            first_date = isempty(bars) ? "—" : string(Date(first(bars).t));
            println("  $year  $feed : $n bars   first=$first_date");
        catch e
            println("  $year  $feed : error $(e)");
        end
    end
end
println("Done.")
