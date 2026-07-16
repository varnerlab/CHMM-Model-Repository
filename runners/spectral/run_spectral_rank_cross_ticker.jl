# ========================================================================================= #
# run_spectral_rank_cross_ticker.jl
#
# Cross-ticker spectral effective-rank diagnostic at K = 3 AND K = 18 across the 30-ticker
# sector-balanced panel (+ SPY control). The K = 18 block reports how concentrated the
# absolute-growth-rate ACF budget is when many modes are available; the K = 3 block runs
# the same grouped diagnostic at the paper's headline state count. Because a lag-1 budget
# alone is not an effective-rank diagnostic for the full lag-1..252 curve (a component
# small at lag 1 can matter at later lags if its eigenvalue is larger or its phase
# differs), every share is reported under TWO norms:
#   - lag-1 norm:      B  = Σ_c |contrib_c(1)|
#   - horizon-aware:   B∫ = Σ_c Σ_{τ=1..252} |contrib_c(τ)|
#
# The block that actually answers "would more modes help the ACF fit at the typical
# ticker?" is the model-vs-sample ACF comparison: per ticker, the MAE between the fitted
# population |G| ACF and the observed IS sample |G| ACF at K = 3 versus K = 18, on the
# short/medium band (lags 1-63) and the far band (64-252), with the zero-curve (i.i.d.)
# reference. If K = 18's extra modes do not reduce that MAE relative to K = 3, the
# decay-mode budget is not the binding constraint on the ACF fit at the typical ticker;
# if they do, it is, and the paper's claim must be scoped accordingly.
#
# Output: results/diagnostics/spectral_rank_cross_ticker.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
include(joinpath(@__DIR__, "spectral_common.jl"));

using Printf

const SEED      = 20260420;
const K_LIST    = [3, 18];
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const N_M_DRAW  = 200_000;
const MAXLAG    = 252;

const OUT_DIR = joinpath(_ROOT, "results", "diagnostics");
mkpath(OUT_DIR);

const SECTOR_PANEL = [
    ("Information Technology",   ["AAPL", "MSFT", "NVDA"]),
    ("Health Care",              ["JNJ",  "UNH",  "LLY"]),
    ("Financials",               ["JPM",  "BAC",  "WFC"]),
    ("Consumer Discretionary",   ["AMZN", "HD",   "MCD"]),
    ("Communication Services",   ["NFLX", "VZ",   "DIS"]),
    ("Industrials",              ["CAT",  "BA",   "HON"]),
    ("Consumer Staples",         ["PG",   "KO",   "WMT"]),
    ("Energy",                   ["XOM",  "CVX",  "COP"]),
    ("Utilities",                ["NEE",  "DUK",  "SO" ]),
    ("Materials",                ["FCX",  "NEM",  "APD"]),
];

println("="^88);
println("  Cross-ticker spectral effective-rank diagnostic at K in $(K_LIST)");
println("="^88);

# ----------------------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading IS panel...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
panel_tickers = sort(keys(filtered) |> collect);
all_R = log_growth_matrix(filtered, panel_tickers; Δt=DT, risk_free_rate=RISK_FREE);

function _sample_acf_abs(x::AbstractVector, maxlag::Int)
    a = abs.(x); n = length(a); μ = mean(a);
    v = sum((a .- μ).^2);
    return [sum((a[1:n-τ] .- μ) .* (a[1+τ:n] .- μ)) / v for τ in 1:maxlag];
end

# ----------------------------------------------------------------------------------------- #
# Loop over tickers x K
# ----------------------------------------------------------------------------------------- #
panel_results = Dict{Int, Dict{String, NamedTuple}}(K => Dict{String, NamedTuple}() for K in K_LIST);
acf_fit = Dict{String, NamedTuple}();   # per-ticker ACF-fit MAEs at both K + zero reference

for sector_name in vcat([s for (s, _) in SECTOR_PANEL], ["SPY (control)"])
    if sector_name == "SPY (control)"
        ticks = ["SPY"];
        sec = "Index";
    else
        sec = sector_name;
        ticks = first([ts for (s, ts) in SECTOR_PANEL if s == sec]);
    end
    for ticker in ticks
        idx = findfirst(==(ticker), panel_tickers);
        if idx === nothing
            @warn "ticker $ticker not in panel_tickers; skipping";
            continue;
        end
        R_is = all_R[:, idx];
        ρ_obs = _sample_acf_abs(R_is, MAXLAG);
        maes = Dict{Int, NamedTuple}();
        for K in K_LIST
            Random.seed!(SEED);
            try
                mdl = build(MyContinuousHiddenMarkovModel,
                    (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
                T, π̄, m, M = _T_pibar_m(mdl, K; n_draw=N_M_DRAW, seed=SEED);
                σ²_G, comps, κ_V, recon_err = _spectral_components(T, π̄, m, M);
                s = _component_summary(comps);
                panel_results[K][ticker] = (sector=sec, n_comps=length(comps),
                                            recon_err=recon_err, s...);
                ρ_mod = _theoretical_acf(T, π̄, m, M, MAXLAG);
                maes[K] = (near = mean(abs.(ρ_mod[1:63] .- ρ_obs[1:63])),
                           far  = mean(abs.(ρ_mod[64:MAXLAG] .- ρ_obs[64:MAXLAG])));
                @printf("  %-6s [%s] K=%-2d  dom = %.3f (int %.3f)   n95 = %d (int %d)   acf-mae 1-63 = %.4f\n",
                        ticker, sec, K, s.dom_share, s.dom_share_int,
                        s.n_for_95, s.n_for_95_int, maes[K].near);
            catch e
                @warn "fit failed for $ticker at K=$K: $e";
            end
        end
        if all(K -> haskey(maes, K), K_LIST)
            acf_fit[ticker] = (sector=sec,
                               near_k3 = maes[3].near,  near_k18 = maes[18].near,
                               far_k3  = maes[3].far,   far_k18  = maes[18].far,
                               near_zero = mean(abs.(ρ_obs[1:63])),
                               far_zero  = mean(abs.(ρ_obs[64:MAXLAG])));
        end
    end
end

# ----------------------------------------------------------------------------------------- #
# Summaries
# ----------------------------------------------------------------------------------------- #
function _dist_summary(K)
    rs = collect(values(panel_results[K]));
    dom  = [r.dom_share for r in rs];
    domI = [r.dom_share_int for r in rs];
    n95  = [r.n_for_95 for r in rs];
    n95I = [r.n_for_95_int for r in rs];
    return (n=length(rs), dom=dom, domI=domI, n95=n95, n95I=n95I);
end

Δ_near = [f.near_k3 - f.near_k18 for f in values(acf_fit)];
Δ_far  = [f.far_k3  - f.far_k18  for f in values(acf_fit)];

# ----------------------------------------------------------------------------------------- #
# Output
# ----------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "spectral_rank_cross_ticker.txt");
open(out_path, "w") do io
    println(io, "="^96);
    println(io, "Cross-ticker spectral effective-rank diagnostic  (grouped components; K = 3 and K = 18)");
    println(io, "="^96);
    println(io, "Setup: CHMM-N at K in $(K_LIST), sector-balanced 30-ticker panel + SPY control,");
    println(io, "       seed = $SEED, n_draw = $N_M_DRAW per state for m_k, IS window.");
    println(io, "Complex-conjugate eigenvalue pairs are grouped into single real damped-oscillatory");
    println(io, "components; contrib_c(τ) is each component's SIGNED real ACF contribution.");
    println(io, "Two ranking norms per ticker:");
    println(io, "  lag-1 norm      dom/n95      on B  = Σ_c |contrib_c(1)|");
    println(io, "  horizon-aware   domI/n95I    on B∫ = Σ_c Σ_{τ=1..$MAXLAG} |contrib_c(τ)|");
    println(io, "(B is not a percentage of ρ(1) itself: signed contributions can cancel.)");
    println(io, "recon = max |spectral - direct| ACF reconstruction error over τ = 1..$MAXLAG.");
    for K in K_LIST
        println(io);
        println(io, "-"^96);
        println(io, "K = $K per-ticker table");
        println(io, "-"^96);
        @printf(io, "%-6s %-26s %-10s %-10s %-6s %-7s %-6s %-9s\n",
                "ticker", "sector", "dom_share", "dom_int", "n95", "n95_int", "ncomp", "recon");
        println(io, "-"^96);
        for sector_name in vcat([s for (s, _) in SECTOR_PANEL], ["Index"])
            for ticker in sort([t for (t, r) in panel_results[K] if r.sector == sector_name])
                r = panel_results[K][ticker];
                @printf(io, "%-6s %-26s %-10.3f %-10.3f %-6d %-7d %-6d %-9.1e\n",
                        ticker, r.sector, r.dom_share, r.dom_share_int,
                        r.n_for_95, r.n_for_95_int, r.n_comps, r.recon_err);
            end
        end
        d = _dist_summary(K);
        println(io);
        println(io, "K = $K cross-ticker distribution (n = $(d.n) tickers):");
        @printf(io, "  dominant lag-1 share    : median %.3f  Q1 %.3f  Q3 %.3f  min %.3f\n",
                median(d.dom), quantile(d.dom, 0.25), quantile(d.dom, 0.75), minimum(d.dom));
        @printf(io, "  dominant horizon share  : median %.3f  Q1 %.3f  Q3 %.3f  min %.3f\n",
                median(d.domI), quantile(d.domI, 0.25), quantile(d.domI, 0.75), minimum(d.domI));
        @printf(io, "  n_for_95 (lag-1)        : median %.1f  Q1 %.1f  Q3 %.1f  max %d\n",
                median(d.n95), quantile(d.n95, 0.25), quantile(d.n95, 0.75), maximum(d.n95));
        @printf(io, "  n_for_95 (horizon)      : median %.1f  Q1 %.1f  Q3 %.1f  max %d\n",
                median(d.n95I), quantile(d.n95I, 0.25), quantile(d.n95I, 0.75), maximum(d.n95I));
    end
    println(io);
    println(io, "-"^96);
    println(io, "Does the K = 18 mode budget buy ACF fit over K = 3?  (model-vs-sample |G| ACF MAE, IS window)");
    println(io, "-"^96);
    println(io, "Per ticker: MAE between the fitted population ACF and the observed IS sample ACF,");
    println(io, "near band lags 1-63 and far band 64-$MAXLAG; 'zero' is the i.i.d. reference curve ρ = 0.");
    println(io);
    @printf(io, "%-6s %-26s %-9s %-9s %-9s %-9s %-9s %-9s\n",
            "ticker", "sector", "n_K3", "n_K18", "n_zero", "f_K3", "f_K18", "f_zero");
    println(io, "-"^96);
    for sector_name in vcat([s for (s, _) in SECTOR_PANEL], ["Index"])
        for ticker in sort([t for (t, f) in acf_fit if f.sector == sector_name])
            f = acf_fit[ticker];
            @printf(io, "%-6s %-26s %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f\n",
                    ticker, f.sector, f.near_k3, f.near_k18, f.near_zero,
                    f.far_k3, f.far_k18, f.far_zero);
        end
    end
    println(io);
    println(io, "Cross-ticker medians (n = $(length(acf_fit))):");
    @printf(io, "  near band 1-63   : K3 %.4f   K18 %.4f   zero %.4f   Δ(K3 - K18) median %.4f  Q1 %.4f  Q3 %.4f\n",
            median([f.near_k3 for f in values(acf_fit)]),
            median([f.near_k18 for f in values(acf_fit)]),
            median([f.near_zero for f in values(acf_fit)]),
            median(Δ_near), quantile(Δ_near, 0.25), quantile(Δ_near, 0.75));
    @printf(io, "  far band 64-%d  : K3 %.4f   K18 %.4f   zero %.4f   Δ(K3 - K18) median %.4f  Q1 %.4f  Q3 %.4f\n",
            MAXLAG,
            median([f.far_k3 for f in values(acf_fit)]),
            median([f.far_k18 for f in values(acf_fit)]),
            median([f.far_zero for f in values(acf_fit)]),
            median(Δ_far), quantile(Δ_far, 0.25), quantile(Δ_far, 0.75));
    @printf(io, "  tickers where K18 beats K3 on the near band: %d / %d\n",
            count(>(0), Δ_near), length(Δ_near));
    println(io);
    println(io, "Reading:");
    println(io, "  - The K = 18 share tables describe how concentrated the fitted budget is when 17");
    println(io, "    modes are available; a median dominant share below 0.90 means the fitted K = 18");
    println(io, "    chains spread their ACF budget over several modes at the typical ticker, so the");
    println(io, "    K = 18 cross-section alone does NOT establish that a two-mode (K = 3) budget is");
    println(io, "    sufficient.");
    println(io, "  - The binding-ness question is settled by the ACF-fit block: if the median");
    println(io, "    Δ(K3 - K18) near-band MAE is small relative to the zero-curve margin, the extra");
    println(io, "    K = 18 modes buy little ACF fit at the typical ticker and the mode budget is not");
    println(io, "    the binding constraint at K = 3; a materially positive Δ says it is binding.");
    println(io, "  - This is an IS model-vs-sample comparison across the fitted panel; it is not an");
    println(io, "    out-of-sample forecast comparison.");
end
println();
println("[done] Wrote $out_path");
