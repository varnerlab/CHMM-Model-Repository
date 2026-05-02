# ========================================================================================= #
# run_cross_ticker_penalised.jl
#
# Per-ticker CHMM-t at K = 18 under the penalised ECM (1/ν_k shrinkage at λ = 20),
# matched to the body Table tab:cross_ticker columns: IS / OoS KS pass rate, observed and
# simulated IS kurtosis, |G_t| ACF-MAE. Addresses peer-review Tier-1 item 5: the body table
# currently reports the unpenalised CHMM-t numbers, but the discussion section recommends
# the penalised variant as the operational default for tail-conditional consumers.
#
# Output: results/chmm_t_penalised/Cross_Ticker_CHMM_t_Pen.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random
using Statistics
using HypothesisTests
using StatsBase

const SEED        = 20260420;
const K_MAIN      = 18;
const N_PATHS     = 1000;
const MAX_ITER    = 60;
const DT          = 1/252;
const RISK_FREE   = 0.0;
const L_LAGS      = 252;
const SHRINK_RATE = 20.0;

const OUT_DIR = joinpath(_ROOT, "results", "chmm_t_penalised");
mkpath(OUT_DIR);

println("="^80)
println("  Cross-ticker CHMM-t penalised ECM (λ = $SHRINK_RATE)  [peer-review T1.5]")
println("  Seed: $SEED  K: $K_MAIN  Paths: $N_PATHS")
println("="^80)

# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading IS / OoS portfolios...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];

# ----------------------------------------------------------------------------------------- #
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

function _eval(R_obs::AbstractVector, sim::AbstractMatrix; L::Int=L_LAGS)
    np = size(sim, 2);
    n_o = length(R_obs);
    μ_o = mean(R_obs); σ_o = std(R_obs);
    kurt_o = sum(((R_obs .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L, n_o - 1);
    acf_o = autocor(abs.(R_obs), 1:L_use);

    ks_pass = 0;
    kurt_s = 0.0; acf_mae_s = 0.0;

    for i in 1:np
        s = sim[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= 0.05
            ks_pass += 1;
        end
        μ_s = mean(s); σ_s = std(s);
        kurt_s += sum(((s .- μ_s) ./ σ_s).^4) / length(s) - 3.0;
        acf_s = autocor(abs.(s), 1:L_use);
        acf_mae_s += mean(abs.(acf_o .- acf_s));
    end
    return (
        ks_pct   = round(100*ks_pass/np, digits=1),
        sim_kurt = round(kurt_s/np, digits=2),
        obs_kurt = round(kurt_o, digits=2),
        acf_mae  = round(acf_mae_s/np, digits=4),
    );
end

# ----------------------------------------------------------------------------------------- #
cross_tickers = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];

results_rows = NamedTuple[];

for t in cross_tickers
    println("\n[$t] CHMM-t penalised at λ = $SHRINK_RATE, K = $K_MAIN...")

    t_idx = findfirst(==(t), all_tickers);
    if isnothing(t_idx); println("  $t not found, skipping."); continue; end
    R_t_is = all_R[:, t_idx];

    if !haskey(oos_dataset, t); println("  $t OoS not available, skipping."); continue; end
    R_t_oos = log_growth_matrix(oos_dataset, t; Δt=DT, risk_free_rate=RISK_FREE);

    n_is = length(R_t_is); n_oos = length(R_t_oos);

    Random.seed!(SEED);
    chmm_t_pen = build(MyStudentTHiddenMarkovModel,
        (observations=R_t_is, number_of_states=K_MAIN, max_iter=MAX_ITER,
         ν_shrink_rate=SHRINK_RATE));
    sd_t = _stationary(chmm_t_pen, K_MAIN);

    nus = [params(chmm_t_pen.emission[k].ρ)[1] for k in 1:K_MAIN];
    println("  per-state ν_k: median $(round(median(nus), digits=2))  min $(round(minimum(nus), digits=2))  max $(round(maximum(nus), digits=2))");

    Random.seed!(SEED + 11);
    sim_is  = _sim_paths(chmm_t_pen, sd_t, n_is,  N_PATHS);
    sim_oos = _sim_paths(chmm_t_pen, sd_t, n_oos, N_PATHS);

    is_panel  = _eval(R_t_is,  sim_is);
    oos_panel = _eval(R_t_oos, sim_oos);

    println("  IS  KS = $(is_panel.ks_pct)%   sim kurt = $(is_panel.sim_kurt)  (obs $(is_panel.obs_kurt))   |G| ACF-MAE = $(is_panel.acf_mae)")
    println("  OoS KS = $(oos_panel.ks_pct)%")

    push!(results_rows, (
        ticker      = t,
        ks_is       = is_panel.ks_pct,
        ks_oos      = oos_panel.ks_pct,
        kurt_obs    = is_panel.obs_kurt,
        kurt_sim    = is_panel.sim_kurt,
        acf_mae     = is_panel.acf_mae,
        nu_median   = round(median(nus), digits=2),
        nu_min      = round(minimum(nus), digits=2),
        nu_max      = round(maximum(nus), digits=2),
    ));
end

# ----------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "Cross_Ticker_CHMM_t_Pen.txt");
open(out_path, "w") do io
    println(io, "="^90);
    println(io, "Cross-ticker CHMM-t PENALISED ECM at λ = $SHRINK_RATE, K = $K_MAIN  [peer-review T1.5]");
    println(io, "="^90);
    println(io, "Setup: per-ticker independent fit, paths = $N_PATHS, seed = $SEED.");
    println(io, "Penalty: 1/ν_k shrinkage at rate λ = $SHRINK_RATE on the per-state ECM Q-function.");
    println(io, "Bracket: (ν_min, ν_max) = (2.1, 50).");
    println(io);
    println(io, "Body Table tab:cross_ticker columns (replacement panel):");
    println(io);
    println(io, "Ticker | IS KS% | OoS KS% | Kurt obs | Kurt sim | |G| ACF-MAE | ν median | ν range");
    println(io, "-"^90);
    for r in results_rows
        println(io, rpad(r.ticker, 6), " | ",
                    lpad(r.ks_is, 6), " | ",
                    lpad(r.ks_oos, 7), " | ",
                    lpad(r.kurt_obs, 8), " | ",
                    lpad(r.kurt_sim, 8), " | ",
                    lpad(r.acf_mae, 11), " | ",
                    lpad(r.nu_median, 8), " | ",
                    "[", r.nu_min, ", ", r.nu_max, "]");
    end
    println(io);
    println(io, "Compare against unpenalised CHMM-t in body Table 3 (sourced from");
    println(io, "results/SPY/Table-T2-Per-Ticker-Emission-Families.txt CHMM-t rows):");
    println(io, "  SPY :  IS KS 95.2  OoS KS 84.1  sim kurt 14.68");
    println(io, "  NVDA:  IS KS 99.1  OoS KS 57.2  sim kurt 46.40");
    println(io, "  JNJ :  IS KS 99.8  OoS KS 94.3  sim kurt 99.66");
    println(io, "  JPM :  IS KS 99.2  OoS KS 53.0  sim kurt 18.82");
    println(io, "  AAPL:  IS KS 99.1  OoS KS 94.7  sim kurt  7.54");
    println(io, "  QQQ :  IS KS 96.1  OoS KS 90.3  sim kurt  3.30");
end

println("\n[done] $out_path")
