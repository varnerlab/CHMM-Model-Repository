# ========================================================================================= #
# run_chmm_t_penalised_headline.jl
#
# Computes the headline-panel metrics (IS / OoS KS, IS / OoS kurtosis, |G_t| ACF-MAE,
# raw-G_t ACF-MAE, OoS CRPS) for CHMM-t under the penalised ECM with 1/ν_k shrinkage at
# λ = 20. Output augments the body Table tab:model_comparison row addressed by R2-W6.
#
# Why λ = 20: it brings the simulated IS kurtosis from 14.30 (unpenalised) to 8.43, near
# the observed 7.68, at a 1pp IS KS cost. See Section discussion.tex paragraph "Closing
# the kurtosis gap" and Appendix sec:supp_misc penalised-ECM rate-sweep.
#
# Output: results/chmm_t_penalised/Headline_CHMM_t_Pen.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
using StatsBase

const SEED      = 20260420;
const K_MAIN    = 18;
const N_PATHS   = 1000;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const L_LAGS    = 252;
const SHRINK_RATE = 20.0;

Random.seed!(SEED);

const OUT_DIR = joinpath(_ROOT, "results", "chmm_t_penalised");
mkpath(OUT_DIR);

println("="^80)
println("  CHMM-t penalised ECM headline panel (λ = $SHRINK_RATE)  [R2-W6]")
println("  Seed: $SEED  K: $K_MAIN  Paths: $N_PATHS")
println("="^80)

# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading SPY IS / OoS...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = all_R[:, idx_spy];
n_is = length(R_is);
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE);
n_oos = length(R_oos);
println("  IS = $n_is  OoS = $n_oos")

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

function _eval_panel(R_obs::AbstractVector, sim::AbstractMatrix; L::Int=L_LAGS)
    np = size(sim, 2);
    n_o = length(R_obs);
    μ_o = mean(R_obs); σ_o = std(R_obs);
    kurt_o = sum(((R_obs .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L, n_o - 1);
    acf_o     = autocor(abs.(R_obs), 1:L_use);
    acf_o_raw = autocor(R_obs,        1:L_use);

    ks_pass = 0;
    kurt_s = 0.0; acf_mae_s = 0.0; acf_mae_raw_s = 0.0;

    for i in 1:np
        s = sim[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= 0.05
            ks_pass += 1;
        end
        μ_s = mean(s); σ_s = std(s);
        kurt_s += sum(((s .- μ_s) ./ σ_s).^4) / length(s) - 3.0;
        acf_s     = autocor(abs.(s), 1:L_use);
        acf_s_raw = autocor(s,       1:L_use);
        acf_mae_s     += mean(abs.(acf_o     .- acf_s));
        acf_mae_raw_s += mean(abs.(acf_o_raw .- acf_s_raw));
    end
    return (
        ks_pct      = round(100*ks_pass/np, digits=1),
        sim_kurt    = round(kurt_s/np, digits=2),
        obs_kurt    = round(kurt_o, digits=2),
        acf_mae     = round(acf_mae_s/np, digits=4),
        acf_mae_raw = round(acf_mae_raw_s/np, digits=4),
    );
end

# ----------------------------------------------------------------------------------------- #
println("\n[fit] CHMM-t penalised at λ = $SHRINK_RATE on SPY IS, K = $K_MAIN...");
chmm_t_pen = build(MyStudentTHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER,
     ν_shrink_rate=SHRINK_RATE));
sd_t = _stationary(chmm_t_pen, K_MAIN);

# Per-state ν_k diagnostics  (TDist(ν).ν via the ρ field of the LocationScale wrapper)
nus = [params(chmm_t_pen.emission[k].ρ)[1] for k in 1:K_MAIN];
println("  per-state ν_k: median $(round(median(nus), digits=2))  min $(round(minimum(nus), digits=2))  max $(round(maximum(nus), digits=2))");
println("  states near lower bracket (ν < 5): $(count(<(5), nus))  /  states at upper bracket (ν == 50): $(count(==(50.0), nus))")

println("\n[sim] Simulating $N_PATHS IS + OoS paths under penalised CHMM-t...");
sim_is  = _sim_paths(chmm_t_pen, sd_t, n_is,  N_PATHS);
sim_oos = _sim_paths(chmm_t_pen, sd_t, n_oos, N_PATHS);

is_panel  = _eval_panel(R_is,  sim_is);
oos_panel = _eval_panel(R_oos, sim_oos);

println("\n[panel] CHMM-t penalised (λ = $SHRINK_RATE)  K = $K_MAIN");
println("  IS  KS pass = $(is_panel.ks_pct)%   sim kurt = $(is_panel.sim_kurt)  (obs $(is_panel.obs_kurt))");
println("  OoS KS pass = $(oos_panel.ks_pct)%   sim kurt = $(oos_panel.sim_kurt)  (obs $(oos_panel.obs_kurt))");
println("  |G_t| ACF-MAE = $(is_panel.acf_mae)   raw-G_t ACF-MAE = $(is_panel.acf_mae_raw)");

# ----------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "Headline_CHMM_t_Pen.txt");
open(out_path, "w") do io
    println(io, "="^80);
    println(io, "CHMM-t penalised ECM headline panel  (R2-W6)");
    println(io, "="^80);
    println(io, "Setup: SPY IS / OoS, K = $K_MAIN, paths = $N_PATHS, seed = $SEED.");
    println(io, "Penalty: 1/ν_k shrinkage at rate λ = $SHRINK_RATE on the per-state ECM Q-function.");
    println(io, "Bracket: (ν_min, ν_max) = (2.1, 50).");
    println(io);
    println(io, "Per-state ν_k:");
    println(io, "  median    : $(round(median(nus), digits=2))");
    println(io, "  min / max : $(round(minimum(nus), digits=2))  /  $(round(maximum(nus), digits=2))");
    println(io, "  near lower: $(count(<(5), nus)) / $K_MAIN  states with ν < 5");
    println(io, "  at upper  : $(count(==(50.0), nus)) / $K_MAIN  states with ν = 50");
    println(io);
    println(io, "Headline panel:");
    println(io);
    println(io, "Window | KS pass% | sim kurt | obs kurt | |G| ACF-MAE | raw ACF-MAE");
    println(io, "-"^70);
    println(io, "IS     | $(is_panel.ks_pct)     | $(is_panel.sim_kurt)    | $(is_panel.obs_kurt)     | $(is_panel.acf_mae)      | $(is_panel.acf_mae_raw)");
    println(io, "OoS    | $(oos_panel.ks_pct)     | $(oos_panel.sim_kurt)    | $(oos_panel.obs_kurt)     | $(oos_panel.acf_mae)      | $(oos_panel.acf_mae_raw)");
    println(io);
    println(io, "Compare against unpenalised CHMM-t (Table tab:model_comparison):");
    println(io, "  IS  KS pass = 95.6%   sim kurt = 14.35   |G| ACF-MAE = 0.0549");
    println(io, "  OoS KS pass = 85.7%   sim kurt = 10.71");
end

println("\n[done] $out_path");
