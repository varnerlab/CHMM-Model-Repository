# Attic: stale data snapshots

Dated OHLC snapshots and intraday files from intermediate IS / OoS
extension passes. The arXiv preprint loads only the two clean bundles
in `../../data/`:

- `CHMM-SP500-Train-10yr.jld2`
- `CHMM-SP500-OoS-Remainder.jld2`

plus the per-ticker daily aggregates `HMM-WJ-{AAPL,NVDA,SPY}-daily-aggregate.jld2`.

The two raw OHLC sources used by `build_new_train_oos.jl` to rebuild
the clean bundles also stay in `../../data/`:

- `SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2` (IS source)
- `SP500-Daily-OHLC-1-3-2025-to-4-20-2026.jld2` (OoS source)

What lives here:

| File | Why archived |
|---|---|
| `SP500-Daily-OHLC-1-3-2014-to-02-07-2025.jld2` | Overlapping interim snapshot, superseded |
| `SP500-Daily-OHLC-1-3-2018-to-12-29-2023.jld2` | Older window, predates the 10-year IS |
| `SP500-Daily-OHLC-1-3-2024-to-10-25-2024.jld2` | Intermediate OoS extension snapshot |
| `SP500-Daily-OHLC-1-3-2025-to-09-26-2025.jld2` | Intermediate OoS extension snapshot |
| `SP500-Daily-OHLC-1-3-2025-to-11-18-2025.jld2` | Intermediate OoS extension snapshot |
| `train_dataset_2014_2023.jld2` | Legacy split predating the clean Train-10yr bundle |
| `test_dataset_2024_onward.jld2` | Legacy split |
| `HMM-SPY-1-min-aggregate.jld2` | Intraday 1-minute SPY; not used in arXiv scope |
| `SPY-OHLC-1-min-aggregate-2023.csv` | Intraday 1-minute SPY CSV; not used in arXiv scope |
