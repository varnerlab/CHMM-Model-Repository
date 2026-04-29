# =========================================================================== #
# run_sector_panel_k6.jl
#
# Peer-review item 1 (Tier-1): re-run the 30-ticker sector-balanced panel at
# K* = 6 (the held-out-clean operating point) instead of K = 18 (the multi-
# objective rule that touches OoS). Same protocol as run_sector_panel.jl
# otherwise: penalised CHMM-t, λ = 20, N_paths = 1000, seed = 20260420.
#
# Output:
#   results/sector_panel/sector_panel_summary_k6.csv
#   results/sector_panel/sector_panel_summary_k6.txt
# =========================================================================== #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random, Statistics, HypothesisTests, StatsBase, Printf;

const SEED        = 20260420;
const K_MAIN      = 6;
const N_PATHS     = 1000;
const MAX_ITER    = 60;
const DT          = 1/252;
const RISK_FREE   = 0.0;
const L_LAGS      = 252;
const SHRINK_RATE = 20.0;
const ALPHA_KS    = 0.05;
const KS_FAIL_PP  = 60.0;

const OUT_DIR = joinpath(_ROOT, "results", "sector_panel");
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

println("="^80)
println("  Sector-balanced 30-ticker panel at K* = 6 (peer-review item 1)")
println("  Penalised CHMM-t  λ = $SHRINK_RATE")
println("="^80)

train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];

function _stationary(model, K::Int)
    T = zeros(K, K);
    for i in 1:K; T[i, :] = probs(model.transition[i]); end
    π = (T^1000)[1, :];
    return Categorical(π);
end

function _sim_paths(model, sd, n::Int, np::Int)
    sim = Matrix{Float64}(undef, n, np);
    for p in 1:np
        s0 = rand(sd);
        st = model(s0, n);
        for j in 1:n; sim[j, p] = rand(model.emission[st[j]]); end
    end
    return sim;
end

function _eval(R_obs::AbstractVector, sim::AbstractMatrix; L::Int=L_LAGS, α::Float64=ALPHA_KS)
    np = size(sim, 2);
    n_o = length(R_obs);
    μ_o = mean(R_obs); σ_o = std(R_obs);
    kurt_o = sum(((R_obs .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L, n_o - 1);
    acf_o = autocor(abs.(R_obs), 1:L_use);
    ks_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0;
    for i in 1:np
        s = sim[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= α; ks_pass += 1; end
        μ_s = mean(s); σ_s = std(s);
        kurt_s += sum(((s .- μ_s) ./ σ_s).^4) / length(s) - 3.0;
        acf_s = autocor(abs.(s), 1:L_use);
        acf_mae_s += mean(abs.(acf_o .- acf_s));
    end
    return (ks_pct = round(100*ks_pass/np, digits=1),
            sim_kurt = round(kurt_s/np, digits=2),
            obs_kurt = round(kurt_o, digits=2),
            acf_mae = round(acf_mae_s/np, digits=4));
end

results_rows = NamedTuple[];
let
    global n_fitted = 0;
    global skipped = String[];

for (sector, tickers) in SECTOR_PANEL
    println("\n[sector] $sector")
    for tk in tickers
        t_idx = findfirst(==(tk), all_tickers);
        if isnothing(t_idx); push!(skipped, tk); continue; end
        if !haskey(oos_dataset, tk); push!(skipped, tk); continue; end

        R_t_is  = Vector{Float64}(all_R[:, t_idx]);
        R_t_oos = log_growth_matrix(oos_dataset, tk; Δt=DT, risk_free_rate=RISK_FREE);
        n_is = length(R_t_is); n_oos = length(R_t_oos);

        Random.seed!(SEED + 13 * n_fitted + hash(tk) % 1000);
        chmm_t_pen = build(MyStudentTHiddenMarkovModel,
            (observations=R_t_is, number_of_states=K_MAIN, max_iter=MAX_ITER,
             ν_shrink_rate=SHRINK_RATE));
        sd_t = _stationary(chmm_t_pen, K_MAIN);

        nus = [params(chmm_t_pen.emission[k].ρ)[1] for k in 1:K_MAIN];

        Random.seed!(SEED + 11 + 13 * n_fitted);
        sim_is  = _sim_paths(chmm_t_pen, sd_t, n_is,  N_PATHS);
        sim_oos = _sim_paths(chmm_t_pen, sd_t, n_oos, N_PATHS);

        is_panel  = _eval(R_t_is,  sim_is);
        oos_panel = _eval(R_t_oos, sim_oos);

        @printf("  %-5s  IS KS = %5.1f%%  OoS KS = %5.1f%%  kurt obs/sim = %6.2f / %6.2f  |G| ACF-MAE = %.4f\n",
            tk, is_panel.ks_pct, oos_panel.ks_pct, is_panel.obs_kurt, is_panel.sim_kurt, is_panel.acf_mae);

        push!(results_rows, (sector=sector, ticker=tk,
            ks_is=is_panel.ks_pct, ks_oos=oos_panel.ks_pct,
            kurt_obs=is_panel.obs_kurt, kurt_sim=is_panel.sim_kurt,
            kurt_resid=round(is_panel.sim_kurt - is_panel.obs_kurt, digits=2),
            acf_mae=is_panel.acf_mae, nu_med=round(median(nus), digits=2)));
        n_fitted += 1;
    end
end
end

# --- Aggregate ---
ks_is_arr  = [r.ks_is  for r in results_rows];
ks_oos_arr = [r.ks_oos for r in results_rows];
acf_arr    = [r.acf_mae for r in results_rows];
res_arr    = [r.kurt_resid for r in results_rows];

q1(x) = quantile(x, 0.25); q3(x) = quantile(x, 0.75);
n_fail = sum(ks_oos_arr .< KS_FAIL_PP);

# --- Output ---
open(joinpath(OUT_DIR, "sector_panel_summary_k6.csv"), "w") do io
    write(io, "sector,ticker,ks_is_pct,ks_oos_pct,kurt_obs,kurt_sim,kurt_resid,acf_mae_abs,nu_median\n");
    for r in results_rows
        write(io, @sprintf("%s,%s,%.1f,%.1f,%.2f,%.2f,%.2f,%.4f,%.2f\n",
            r.sector, r.ticker, r.ks_is, r.ks_oos, r.kurt_obs, r.kurt_sim, r.kurt_resid, r.acf_mae, r.nu_med));
    end
end

open(joinpath(OUT_DIR, "sector_panel_summary_k6.txt"), "w") do io
    println(io, "="^96);
    println(io, "Sector-balanced 30-ticker panel (penalised CHMM-t, K = $K_MAIN, λ = $SHRINK_RATE)");
    println(io, "[peer-review item 1: held-out-clean K* = 6 cross-ticker rebuild]");
    println(io, "="^96);
    println(io, "Setup: per-ticker independent fit, paths = $N_PATHS, seed = $SEED, T_IS = $(size(all_R, 1)).");
    println(io, "Universe: 10 GICS sectors × 3 large-cap representatives.");
    println(io);
    println(io, "Per-ticker rows ranked by sector then ticker:");
    println(io);
    println(io, "Sector                   | Ticker | IS KS% | OoS KS% | Kurt obs | Kurt sim | Kurt resid | |G| ACF-MAE | ν median");
    println(io, "-"^120);
    for r in results_rows
        @printf(io, "%-24s | %-6s | %6.1f | %7.1f | %8.2f | %8.2f | %10.2f | %11.4f | %8.2f\n",
            r.sector, r.ticker, r.ks_is, r.ks_oos, r.kurt_obs, r.kurt_sim, r.kurt_resid, r.acf_mae, r.nu_med);
    end
    println(io);
    println(io, "Aggregate distribution across 30 tickers:");
    println(io, "-"^96);
    @printf(io, "  IS KS%%        median = %5.1f   [Q1, Q3] = [%5.1f, %5.1f]   mean ± sd = %5.1f ± %4.1f\n",
            median(ks_is_arr), q1(ks_is_arr), q3(ks_is_arr), mean(ks_is_arr), std(ks_is_arr));
    @printf(io, "  OoS KS%%       median = %5.1f   [Q1, Q3] = [%5.1f, %5.1f]   mean ± sd = %5.1f ± %4.1f\n",
            median(ks_oos_arr), q1(ks_oos_arr), q3(ks_oos_arr), mean(ks_oos_arr), std(ks_oos_arr));
    @printf(io, "  |G| ACF-MAE   median = %.4f  [Q1, Q3] = [%.4f, %.4f]\n",
            median(acf_arr), q1(acf_arr), q3(acf_arr));
    @printf(io, "  Kurt residual median = %5.2f   [Q1, Q3] = [%5.2f, %5.2f]\n",
            median(res_arr), q1(res_arr), q3(res_arr));
    @printf(io, "  Tickers with OoS KS < 60%%: %d / %d  (%.1f%%)\n",
            n_fail, length(ks_oos_arr), 100 * n_fail / length(ks_oos_arr));
    println(io);
    println(io, "Compare to the K = 18 panel (results/sector_panel/sector_panel_summary.txt):");
    println(io, "  K=18 OoS KS median = 73.4%, mean = 66.8 ± 29.5%, fail count = 11/30.");
end

println("\n[done] $OUT_DIR/sector_panel_summary_k6.{csv,txt}");
@printf("[summary] OoS KS median = %.1f, mean = %.1f ± %.1f, fail = %d/%d\n",
        median(ks_oos_arr), mean(ks_oos_arr), std(ks_oos_arr), n_fail, length(ks_oos_arr));
