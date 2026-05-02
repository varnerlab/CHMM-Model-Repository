# =========================================================================== #
# run_sector_panel_quarterly_refit_k6.jl
#
# Peer-review item P1.5 (R3 W3): quarterly-refit version of the cross-ticker
# panel at the held-out-clean sensitivity reference K*=6, mirroring the
# existing K=18 quarterly-refit script. Same protocol: penalised CHMM-t at
# λ=20, N_paths=1000, refit on a 5y rolling window every 63 OoS days, OoS path
# concatenation of per-quarter simulations.
#
# Output:
#   results/sector_panel/sector_panel_quarterly_refit_k6.{csv,txt}
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
const REFIT_CADENCE = 63;     # quarterly
const TRAIN_LEN     = 1260;    # 5y rolling train window

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
println("  Quarterly-refit cross-ticker panel at K*=6 (P1.5: R3 W3)")
println("  Penalised CHMM-t at K=$K_MAIN, λ=$SHRINK_RATE, refit every $REFIT_CADENCE OoS days")
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

function _eval_KS(R_obs::AbstractVector, sim::AbstractMatrix; α::Float64=ALPHA_KS)
    np = size(sim, 2);
    ks_pass = 0;
    for i in 1:np
        s = sim[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= α; ks_pass += 1; end
    end
    return round(100 * ks_pass / np, digits=1);
end

function _eval_full(R_obs::AbstractVector, sim::AbstractMatrix; L::Int=L_LAGS, α::Float64=ALPHA_KS)
    np = size(sim, 2);
    n_o = length(R_obs);
    L_use = min(L, n_o - 1);
    acf_o = autocor(abs.(R_obs), 1:L_use);
    μ_o = mean(R_obs); σ_o = std(R_obs);
    kurt_o = sum(((R_obs .- μ_o) ./ σ_o).^4) / n_o - 3.0;
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

# Build a quarterly-refit OoS simulation: at each refit point j, fit on the rolling
# 5y window ending at OoS day j-1 and simulate the next REFIT_CADENCE days.
# Concatenate all quarters into a single (T_OoS, N_PATHS) matrix.
function _quarterly_refit_oos_paths(R_full::AbstractVector, n_is::Int, n_oos::Int,
                                     K::Int, np::Int; ticker_seed::Integer = 0)
    sim_oos = Matrix{Float64}(undef, n_oos, np);
    j = 1;
    refit_count = 0;
    while j <= n_oos
        train_end = n_is + j - 1;
        train_start = max(1, train_end - TRAIN_LEN + 1);
        R_train = R_full[train_start:train_end];
        # quarter length (last quarter may be short)
        q_len = min(REFIT_CADENCE, n_oos - j + 1);
        Random.seed!(SEED + 13 * refit_count + ticker_seed);
        m = build(MyStudentTHiddenMarkovModel,
                  (observations=R_train, number_of_states=K, max_iter=MAX_ITER,
                   ν_shrink_rate=SHRINK_RATE));
        sd = _stationary(m, K);
        Random.seed!(SEED + 11 + 13 * refit_count + ticker_seed);
        sim_q = _sim_paths(m, sd, q_len, np);
        sim_oos[j:j+q_len-1, :] = sim_q;
        j += q_len;
        refit_count += 1;
    end
    return sim_oos, refit_count;
end

results_rows = NamedTuple[];
let
    global n_fitted = 0;
    global skipped = String[];
for (sector, tickers) in SECTOR_PANEL
    println("\n[sector] $sector");
    for tk in tickers
        t_idx = findfirst(==(tk), all_tickers);
        if isnothing(t_idx); push!(skipped, tk); continue; end
        if !haskey(oos_dataset, tk); push!(skipped, tk); continue; end

        R_t_is  = Vector{Float64}(all_R[:, t_idx]);
        R_t_oos = log_growth_matrix(oos_dataset, tk; Δt=DT, risk_free_rate=RISK_FREE);
        n_is  = length(R_t_is);
        n_oos = length(R_t_oos);
        R_full = vcat(R_t_is, R_t_oos);

        ticker_seed = Int(hash(tk) % 1000);

        # IS evaluation under the full-IS fit (matches body Table 4 by construction;
        # we re-fit here for self-containment so the IS row is comparable)
        Random.seed!(SEED + 13 * n_fitted + ticker_seed);
        m_is = build(MyStudentTHiddenMarkovModel,
                     (observations=R_t_is, number_of_states=K_MAIN, max_iter=MAX_ITER,
                      ν_shrink_rate=SHRINK_RATE));
        sd_is = _stationary(m_is, K_MAIN);
        Random.seed!(SEED + 11 + 13 * n_fitted + ticker_seed);
        sim_is = _sim_paths(m_is, sd_is, n_is, N_PATHS);
        is_panel = _eval_full(R_t_is, sim_is);

        # OoS evaluation under quarterly refit
        sim_oos, refits = _quarterly_refit_oos_paths(R_full, n_is, n_oos, K_MAIN, N_PATHS;
                                                      ticker_seed = ticker_seed);
        oos_panel = _eval_full(R_t_oos, sim_oos);

        @printf("  %-5s  IS KS = %5.1f%%  OoS KS (refit) = %5.1f%%  refits = %d  kurt obs/sim = %6.2f / %6.2f\n",
            tk, is_panel.ks_pct, oos_panel.ks_pct, refits, is_panel.obs_kurt, is_panel.sim_kurt);

        push!(results_rows, (sector=sector, ticker=tk,
            ks_is=is_panel.ks_pct, ks_oos=oos_panel.ks_pct,
            refits=refits,
            kurt_obs=is_panel.obs_kurt, kurt_sim=is_panel.sim_kurt,
            acf_mae=is_panel.acf_mae));
        n_fitted += 1;
    end
end
end

ks_is_arr  = [r.ks_is  for r in results_rows];
ks_oos_arr = [r.ks_oos for r in results_rows];
q1(x) = quantile(x, 0.25); q3(x) = quantile(x, 0.75);
n_fail = sum(ks_oos_arr .< KS_FAIL_PP);

open(joinpath(OUT_DIR, "sector_panel_quarterly_refit_k6.csv"), "w") do io
    write(io, "sector,ticker,ks_is_pct,ks_oos_pct_refit,refits,kurt_obs,kurt_sim,acf_mae_abs\n");
    for r in results_rows
        write(io, @sprintf("%s,%s,%.1f,%.1f,%d,%.2f,%.2f,%.4f\n",
            r.sector, r.ticker, r.ks_is, r.ks_oos, r.refits,
            r.kurt_obs, r.kurt_sim, r.acf_mae));
    end
end

open(joinpath(OUT_DIR, "sector_panel_quarterly_refit_k6.txt"), "w") do io
    println(io, "="^104);
    println(io, "Quarterly-refit 30-ticker cross-ticker panel (penalised CHMM-t, K=$K_MAIN, λ=$SHRINK_RATE)");
    println(io, "[peer-review item 5: R1 W4 / RE4; R3 W2 / RE3]");
    println(io, "Refit cadence: every $REFIT_CADENCE OoS trading days, train window $TRAIN_LEN obs.");
    println(io, "="^104);
    println(io);
    println(io, "Sector                   | Ticker | IS KS% | OoS KS% (refit) | refits | Kurt obs | Kurt sim | |G| ACF-MAE");
    println(io, "-"^104);
    for r in results_rows
        @printf(io, "%-24s | %-6s | %6.1f | %15.1f | %6d | %8.2f | %8.2f | %11.4f\n",
            r.sector, r.ticker, r.ks_is, r.ks_oos, r.refits,
            r.kurt_obs, r.kurt_sim, r.acf_mae);
    end
    println(io);
    println(io, "Aggregate distribution across 30 tickers:");
    println(io, "-"^96);
    @printf(io, "  IS KS%%               median = %5.1f   [Q1, Q3] = [%5.1f, %5.1f]   mean ± sd = %5.1f ± %4.1f\n",
            median(ks_is_arr), q1(ks_is_arr), q3(ks_is_arr), mean(ks_is_arr), std(ks_is_arr));
    @printf(io, "  OoS KS%% (refit)      median = %5.1f   [Q1, Q3] = [%5.1f, %5.1f]   mean ± sd = %5.1f ± %4.1f\n",
            median(ks_oos_arr), q1(ks_oos_arr), q3(ks_oos_arr), mean(ks_oos_arr), std(ks_oos_arr));
    @printf(io, "  Tickers with OoS KS (refit) < 60%%: %d / %d  (%.1f%%)\n",
            n_fail, length(ks_oos_arr), 100 * n_fail / length(ks_oos_arr));
    println(io);
    println(io, "Compare against the IS-fixed protocol (results/sector_panel/sector_panel_summary.txt):");
    println(io, "  IS-fixed OoS KS median = 73.4%, mean = 66.8 ± 29.5%, fail count = 11/30.");
end

@printf("\n[summary] OoS KS (refit) median = %.1f, mean = %.1f ± %.1f, fail = %d/%d\n",
        median(ks_oos_arr), mean(ks_oos_arr), std(ks_oos_arr), n_fail, length(ks_oos_arr));
println("[done] $OUT_DIR/sector_panel_quarterly_refit.{csv,txt}");
