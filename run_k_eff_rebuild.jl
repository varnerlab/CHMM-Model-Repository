# ========================================================================================= #
# run_k_eff_rebuild.jl
#
# Reviewer Round 2 / Item B6 (peer-review.md R1#5, R3#req-3).
#
# The K_eff diagnostic in Appendix sec:state_distinctness reports K_eff = 11/18 effective
# states for CHMM-N at K = 18 on SPY IS under the standardized-distance single-linkage
# merge. R1 / R3 ask for the K=11 nominal rebuild: refit CHMM-N at K_nominal = 11 and
# report whether the headline metric panel matches the K=18 nominal row, i.e. whether
# the K_eff parameterization works.
#
# Output:
#   results/k_eff_rebuild/k_eff_rebuild.csv
#   results/k_eff_rebuild/k_eff_rebuild.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
using StatsBase
using Printf

const SEED      = 20260420;
const N_PATHS   = 1000;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const L_LAGS    = 252;

Random.seed!(SEED);

const OUT_DIR = joinpath(_ROOT, "results", "k_eff_rebuild");
mkpath(OUT_DIR);

println("="^80)
println("  K_eff = 11 nominal rebuild  (R1/R3 / Item B6)")
println("  Seed: $SEED  Paths: $N_PATHS")
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
    np = size(sim, 2); n_o = length(R_obs);
    μ_o = mean(R_obs); σ_o = std(R_obs);
    kurt_o = sum(((R_obs .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L, n_o - 1);
    acf_o     = autocor(abs.(R_obs), 1:L_use);
    acf_o_raw = autocor(R_obs,        1:L_use);
    ks_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0; acf_mae_raw_s = 0.0;
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
results = NamedTuple[]

for K in [11, 18]
    println("\n[fit] CHMM-N on SPY IS, K = $K ...")
    chmm_n = build(MyContinuousHiddenMarkovModel,
        (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    sd = _stationary(chmm_n, K)

    println("[sim] $N_PATHS IS + OoS paths ...")
    sim_is  = _sim_paths(chmm_n, sd, n_is,  N_PATHS)
    sim_oos = _sim_paths(chmm_n, sd, n_oos, N_PATHS)
    is_p = _eval_panel(R_is, sim_is)
    oos_p = _eval_panel(R_oos, sim_oos)
    push!(results, (K=K, is=is_p, oos=oos_p))
    @printf("  K = %d : IS KS %.1f%% kurt %.2f  |G| ACF-MAE %.4f\n",
            K, is_p.ks_pct, is_p.sim_kurt, is_p.acf_mae)
    @printf("           OoS KS %.1f%% kurt %.2f  |G| ACF-MAE %.4f\n",
            oos_p.ks_pct, oos_p.sim_kurt, oos_p.acf_mae)
end

# ----------------------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "k_eff_rebuild.csv")
open(csv_path, "w") do io
    println(io, "K,is_KS,is_kurt,is_acf_mae,is_acf_mae_raw,oos_KS,oos_kurt,oos_acf_mae,oos_acf_mae_raw")
    for r in results
        @printf(io, "%d,%.1f,%.2f,%.4f,%.4f,%.1f,%.2f,%.4f,%.4f\n",
                r.K, r.is.ks_pct, r.is.sim_kurt, r.is.acf_mae, r.is.acf_mae_raw,
                r.oos.ks_pct, r.oos.sim_kurt, r.oos.acf_mae, r.oos.acf_mae_raw)
    end
end

txt_path = joinpath(OUT_DIR, "k_eff_rebuild.txt")
open(txt_path, "w") do io
    println(io, "="^110)
    println(io, "K_eff = 11 nominal rebuild  (Reviewer Round 2 / Item B6)")
    println(io, "="^110)
    println(io)
    println(io, "Setup: SPY CHMM-N, IS = $n_is, OoS = $n_oos, paths = $N_PATHS, seed = $SEED.")
    println(io, "Question: at K_nominal = K_eff = 11, does the headline metric panel match the K = 18")
    println(io, "nominal row?")
    println(io)
    @printf(io, "  %-3s  %-7s  %-9s  %-12s  %-12s  %-8s  %-9s  %-12s  %-12s\n",
            "K", "IS_KS%", "IS_kurt", "|G|_MAE_IS", "raw_MAE_IS",
            "OoS_KS%", "OoS_kurt", "|G|_MAE_OoS", "raw_MAE_OoS")
    println(io, "  ", "-"^110)
    for r in results
        @printf(io, "  %-3d  %7.1f  %9.2f  %12.4f  %12.4f  %8.1f  %9.2f  %12.4f  %12.4f\n",
                r.K, r.is.ks_pct, r.is.sim_kurt, r.is.acf_mae, r.is.acf_mae_raw,
                r.oos.ks_pct, r.oos.sim_kurt, r.oos.acf_mae, r.oos.acf_mae_raw)
    end
    println(io)
    println(io, "Reading.")
    println(io, "  If K=11 metrics ≈ K=18 metrics within sampling noise, the K=18 over-parameterization is")
    println(io, "  purely a high-K artefact and the K_eff parameterization is the operationally identical")
    println(io, "  but more parsimonious choice. If K=11 underperforms K=18 substantively, the K_eff merge")
    println(io, "  loses information that the K=18 nominal panel exploits.")
end

println("\n[done] $csv_path")
println("[done] $txt_path")
