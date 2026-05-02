# =========================================================================== #
# run_sv_msm_jd_baselines.jl
#
# Peer-review item 2: SV-AR(1) (Taylor 1986 / Harvey-Ruiz-Shephard 1994),
# Markov-Switching Multifractal (Calvet-Fisher 2004), and Merton 1976
# jump-diffusion baselines for the headline panel. Same SPY IS / OoS windows,
# 1000 paths, seed = 20260420, evaluated on KS pass rate, simulated kurtosis,
# absolute-return ACF-MAE, and raw-return ACF-MAE.
#
# Output: results/sv_msm_jd/sv_msm_jd_baselines.txt (and .csv)
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, StatsBase, HypothesisTests, Printf;

const SEED      = 20260420;
const N_PATHS   = 1000;
const ALPHA_KS  = 0.05;
const L_LAGS    = 252;
const DT        = 1/252;
const RISK_FREE = 0.0;
const OUT_DIR = joinpath(_ROOT, "results", "sv_msm_jd");
mkpath(OUT_DIR);

println("="^70);
println("  Peer-review item 2: SV-AR(1) / MSM / Merton-JD baseline panel on SPY");
println("="^70);

train_dataset = MyPortfolioDataSet()["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet()["dataset"];
R_is  = log_growth_matrix(train_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE);
R_oos = log_growth_matrix(oos_dataset,   "SPY"; Δt=DT, risk_free_rate=RISK_FREE);
@printf("[data] T_IS = %d   T_OoS = %d\n", length(R_is), length(R_oos));

function _eval(R_obs::AbstractVector, sim::AbstractMatrix; L::Int = L_LAGS, α::Float64 = ALPHA_KS)
    np = size(sim, 2);
    n_o = length(R_obs);
    L_use = min(L, n_o - 1);
    acf_o_abs = autocor(abs.(R_obs), 1:L_use);
    acf_o_raw = autocor(R_obs, 1:L_use);
    μ_o = mean(R_obs); σ_o = std(R_obs);
    kurt_o = sum(((R_obs .- μ_o) ./ σ_o) .^ 4) / n_o - 3.0;
    ks_pass = 0; kurt_s = 0.0; acf_mae_abs = 0.0; acf_mae_raw = 0.0;
    for i in 1:np
        s = sim[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= α; ks_pass += 1; end
        μ_s = mean(s); σ_s = std(s);
        kurt_s += sum(((s .- μ_s) ./ σ_s) .^ 4) / length(s) - 3.0;
        acf_mae_abs += mean(abs.(autocor(abs.(s), 1:L_use) .- acf_o_abs));
        acf_mae_raw += mean(abs.(autocor(s, 1:L_use) .- acf_o_raw));
    end
    return (ks_pct = round(100 * ks_pass / np, digits = 1),
            sim_kurt = round(kurt_s / np, digits = 2),
            obs_kurt = round(kurt_o, digits = 2),
            acf_abs = round(acf_mae_abs / np, digits = 4),
            acf_raw = round(acf_mae_raw / np, digits = 4));
end

function _run_baseline(name::String, fit_fn::Function, sim_fn::Function)
    println("\n[fit] $name on SPY IS...");
    Random.seed!(SEED);
    p = fit_fn(R_is);
    @printf("  fitted parameters: %s\n", string(p));
    Random.seed!(SEED + 11);
    sim_is  = sim_fn(p, length(R_is);  n_paths = N_PATHS);
    Random.seed!(SEED + 13);
    sim_oos = sim_fn(p, length(R_oos); n_paths = N_PATHS);
    is_panel  = _eval(R_is,  sim_is);
    oos_panel = _eval(R_oos, sim_oos);
    @printf("  IS  KS = %5.1f%%  kurt obs/sim = %5.2f / %5.2f  ACF-MAE |G| = %.4f  ACF-MAE raw = %.4f\n",
        is_panel.ks_pct, is_panel.obs_kurt, is_panel.sim_kurt, is_panel.acf_abs, is_panel.acf_raw);
    @printf("  OoS KS = %5.1f%%  kurt obs/sim = %5.2f / %5.2f  ACF-MAE |G| = %.4f  ACF-MAE raw = %.4f\n",
        oos_panel.ks_pct, oos_panel.obs_kurt, oos_panel.sim_kurt, oos_panel.acf_abs, oos_panel.acf_raw);
    return (model = name, params = p, is = is_panel, oos = oos_panel);
end

results = NamedTuple[];
push!(results, _run_baseline("SV-AR(1)", fit_sv_ar1, simulate_sv_ar1));
push!(results, _run_baseline("MSM kbar=8", obs -> fit_msm(obs; kbar = 8), simulate_msm));
push!(results, _run_baseline("Merton-JD", fit_jump_diffusion, simulate_jump_diffusion));

# --- Output ---
open(joinpath(OUT_DIR, "sv_msm_jd_baselines.csv"), "w") do io
    write(io, "model,window,KS_pct,kurt_obs,kurt_sim,ACF_MAE_abs,ACF_MAE_raw\n");
    for r in results
        write(io, @sprintf("%s,IS,%.1f,%.2f,%.2f,%.4f,%.4f\n",
            r.model, r.is.ks_pct, r.is.obs_kurt, r.is.sim_kurt, r.is.acf_abs, r.is.acf_raw));
        write(io, @sprintf("%s,OoS,%.1f,%.2f,%.2f,%.4f,%.4f\n",
            r.model, r.oos.ks_pct, r.oos.obs_kurt, r.oos.sim_kurt, r.oos.acf_abs, r.oos.acf_raw));
    end
end

open(joinpath(OUT_DIR, "sv_msm_jd_baselines.txt"), "w") do io
    println(io, "Peer-review item 2: SV / MSM / Merton-JD baseline panel on SPY");
    println(io, "Same protocol as Table 1 of the paper: $N_PATHS paths, seed = $SEED.");
    println(io, "="^96);
    @printf(io, "%-15s | %-7s | %-7s | %-9s | %-9s | %-12s | %-12s\n",
            "model", "window", "KS%", "kurt_obs", "kurt_sim", "ACF-MAE |G|", "ACF-MAE raw");
    println(io, "-"^96);
    for r in results
        @printf(io, "%-15s | %-7s | %7.1f | %9.2f | %9.2f | %12.4f | %12.4f\n",
                r.model, "IS",  r.is.ks_pct,  r.is.obs_kurt,  r.is.sim_kurt,  r.is.acf_abs,  r.is.acf_raw);
        @printf(io, "%-15s | %-7s | %7.1f | %9.2f | %9.2f | %12.4f | %12.4f\n",
                r.model, "OoS", r.oos.ks_pct, r.oos.obs_kurt, r.oos.sim_kurt, r.oos.acf_abs, r.oos.acf_raw);
    end
    println(io);
    println(io, "Reading:");
    println(io, "  SV-AR(1): Taylor 1986 / Harvey-Ruiz-Shephard 1994 lognormal SV.");
    println(io, "  MSM:      Calvet-Fisher 2004 binomial multifractal with kbar = 8.");
    println(io, "  Merton-JD: Merton 1976 jump-diffusion (Poisson-Gaussian jump mixture).");
    println(io);
    for r in results
        @printf(io, "  %-15s parameters: %s\n", r.model, string(r.params));
    end
end

println("\n[done] $OUT_DIR/sv_msm_jd_baselines.{csv,txt}");
