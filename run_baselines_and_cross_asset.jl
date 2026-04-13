# ========================================================================================= #
# run_baselines_and_cross_asset.jl
#
# Generates:
#   1. Table 2: Baseline comparisons for SPY (Bootstrap, Gaussian, Laplace, GARCH, CHMM)
#   2. Table T2: Cross-asset generalization for NVDA, JNJ, JPM at K=13
#
# Mirrors the discrete paper's Table 2 and Table T2 structure exactly,
# with Anderson-Darling test and quantile coverage added.
#
# Usage:
#   include("run_baselines_and_cross_asset.jl")
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

const TICKER = "SPY";
const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const N_PATHS = 1000;
const L = 252;
const K = 13;
const MAX_ITER = 60;
const RESULTS_DIR = joinpath(@__DIR__, "results");

# ========================================================================================= #
# LOAD DATA
# ========================================================================================= #
println("Loading data...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) ∈ train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
list_of_all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, list_of_all_tickers; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
idx_spy = findfirst(x -> x == TICKER, list_of_all_tickers);
R_is = all_R[:, idx_spy];
n_is = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset, TICKER; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
n_oos = length(R_oos);

# Observed stats
μ_obs = mean(R_is); σ_obs = std(R_is);
kurt_obs_is = sum(((R_is .- μ_obs) ./ σ_obs).^4) / n_is - 3.0;
println("IS: $n_is obs, OoS: $n_oos obs")

# ========================================================================================= #
# METRICS FUNCTION (matching paper: KS, AD, kurtosis, ACF-MAE, W1, Hellinger, Coverage)
# ========================================================================================= #
function eval_full(observed, sim_archive; L_val=252)
    np = size(sim_archive, 2); n_o = length(observed);
    μ_o = mean(observed); σ_o = std(observed);
    kurt_o = sum(((observed .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L_val, n_o - 1);
    acf_o = autocor(abs.(observed), 1:L_use);

    ks_pass = 0; ad_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0;
    w1_s = 0.0; hell_s = 0.0;

    # Coverage setup
    obs_qprobs = range(0.01, 0.99, length=99);
    obs_quantiles = quantile(observed, obs_qprobs);
    sim_qmatrix = zeros(99, np);

    for i in 1:np
        sim = sim_archive[:, i];

        # KS test
        pval_ks = pvalue(ApproximateTwoSampleKSTest(observed, sim));
        if pval_ks > 0.05; ks_pass += 1; end

        # AD test
        pval_ad = pvalue(KSampleADTest(observed, sim));
        if pval_ad > 0.05; ad_pass += 1; end

        # Kurtosis
        μ_s = mean(sim); σ_s = std(sim);
        kurt_s += sum(((sim .- μ_s) ./ σ_s).^4) / length(sim) - 3.0;

        # ACF-MAE
        acf_sim = autocor(abs.(sim), 1:L_use);
        acf_mae_s += mean(abs.(acf_o .- acf_sim));

        # Wasserstein-1
        obs_sorted = sort(observed); sim_sorted = sort(sim);
        n_min = min(length(obs_sorted), length(sim_sorted));
        obs_q = [obs_sorted[max(1, round(Int, k*length(obs_sorted)/n_min))] for k in 1:n_min];
        sim_q = [sim_sorted[max(1, round(Int, k*length(sim_sorted)/n_min))] for k in 1:n_min];
        w1_s += mean(abs.(obs_q .- sim_q));

        # Hellinger
        lo = min(minimum(observed), minimum(sim)) - 10;
        hi = max(maximum(observed), maximum(sim)) + 10;
        edges = range(lo, hi, length=101);
        h_o = fit(Histogram, observed, edges).weights ./ n_o;
        h_s = fit(Histogram, sim, edges).weights ./ length(sim);
        hell_s += sqrt(sum((sqrt.(h_o) .- sqrt.(h_s)).^2)) / sqrt(2);

        # Quantiles for coverage
        sim_qmatrix[:, i] = quantile(sim, obs_qprobs);
    end

    # Coverage
    cov_count = 0;
    for q in 1:99
        lo_env = quantile(sim_qmatrix[q, :], 0.05);
        hi_env = quantile(sim_qmatrix[q, :], 0.95);
        if obs_quantiles[q] >= lo_env && obs_quantiles[q] <= hi_env
            cov_count += 1;
        end
    end

    return (ks=round(100*ks_pass/np, digits=1),
            ad=round(100*ad_pass/np, digits=1),
            kurt=round(kurt_s/np, digits=2), kurt_obs=round(kurt_o, digits=2),
            acf_mae=round(acf_mae_s/np, digits=4),
            w1=round(w1_s/np, digits=3), hell=round(hell_s/np, digits=4),
            cov=round(100.0*cov_count/99, digits=1))
end

# ========================================================================================= #
# PART 1: BASELINE COMPARISONS (Table 2)
# ========================================================================================= #
println("\n" * "="^70)
println("PART 1: Baseline Comparisons for $TICKER")
println("="^70)

# --- 1. Bootstrap ---
println("  Bootstrap...")
boot_is = Array{Float64,2}(undef, n_is, N_PATHS);
boot_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    boot_is[:, i] = R_is[rand(1:n_is, n_is)];
    boot_oos[:, i] = R_is[rand(1:n_is, n_oos)];
end
m_boot_is = eval_full(R_is, boot_is);
m_boot_oos = eval_full(R_oos, boot_oos);

# --- 2. Gaussian i.i.d. ---
println("  Gaussian i.i.d....")
d_gauss = Normal(μ_obs, σ_obs);
gauss_is = Array{Float64,2}(undef, n_is, N_PATHS);
gauss_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    gauss_is[:, i] = rand(d_gauss, n_is);
    gauss_oos[:, i] = rand(d_gauss, n_oos);
end
m_gauss_is = eval_full(R_is, gauss_is);
m_gauss_oos = eval_full(R_oos, gauss_oos);

# --- 3. Laplace i.i.d. ---
println("  Laplace i.i.d....")
μ_lap = median(R_is); b_lap = mean(abs.(R_is .- μ_lap));
d_lap = Laplace(μ_lap, b_lap);
lap_is = Array{Float64,2}(undef, n_is, N_PATHS);
lap_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    lap_is[:, i] = rand(d_lap, n_is);
    lap_oos[:, i] = rand(d_lap, n_oos);
end
m_lap_is = eval_full(R_is, lap_is);
m_lap_oos = eval_full(R_oos, lap_oos);

# --- 4. GARCH(1,1) ---
println("  GARCH(1,1)...")
garch_model = build(MyGARCHModel, (observations=R_is,));
garch_is = Array{Float64,2}(undef, n_is, N_PATHS);
garch_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    garch_is[:, i] = simulate_garch(garch_model, n_is);
    garch_oos[:, i] = simulate_garch(garch_model, n_oos);
end
m_garch_is = eval_full(R_is, garch_is);
m_garch_oos = eval_full(R_oos, garch_oos);

# --- 5. CHMM (K=13, no jumps) ---
println("  CHMM (K=$K)...")
chmm = build(MyContinuousHiddenMarkovModel, (
    observations=R_is, number_of_states=K, max_iter=MAX_ITER));

T_mat = zeros(K, K);
for i in 1:K; T_mat[i, :] = probs(chmm.transition[i]); end
π_stat = (T_mat^1000)[1, :];
start_dist = Categorical(π_stat);

chmm_is = Array{Float64,2}(undef, n_is, N_PATHS);
chmm_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    s0 = rand(start_dist);
    states = chmm(s0, n_is);
    for j in 1:n_is; chmm_is[j,i] = rand(chmm.emission[states[j]]); end
    s0 = rand(start_dist);
    states = chmm(s0, n_oos);
    for j in 1:n_oos; chmm_oos[j,i] = rand(chmm.emission[states[j]]); end
end
m_chmm_is = eval_full(R_is, chmm_is);
m_chmm_oos = eval_full(R_oos, chmm_oos);

# Print Table 2
println("\nTable 2: Model Comparison — $TICKER ($N_PATHS paths, α=0.05)")
println("="^130)
println("Model          | KS IS(%) | AD IS(%) | KS OoS(%) | Kurt IS | ACF-MAE  | W1 IS  | H IS   | Cov IS(%)")
println("-"^130)
for (name, m_is_val, m_oos_val) in [
    ("Bootstrap", m_boot_is, m_boot_oos),
    ("Gaussian", m_gauss_is, m_gauss_oos),
    ("Laplace", m_lap_is, m_lap_oos),
    ("GARCH(1,1)", m_garch_is, m_garch_oos),
    ("CHMM", m_chmm_is, m_chmm_oos)]
    println("$(rpad(name,14)) | $(lpad(m_is_val.ks,7)) | $(lpad(m_is_val.ad,7)) | $(lpad(m_oos_val.ks,8))  | $(lpad(m_is_val.kurt,6)) | $(m_is_val.acf_mae) | $(m_is_val.w1) | $(m_is_val.hell) | $(m_is_val.cov)")
end
println("="^130)
println("Observed kurtosis: IS=$(round(kurt_obs_is,digits=2))")

# Save Table 2
mkpath(joinpath(RESULTS_DIR, TICKER));
open(joinpath(RESULTS_DIR, TICKER, "Table-2-Baselines.txt"), "w") do io
    println(io, "Table 2: Model Comparison — $TICKER ($N_PATHS paths, α=0.05)")
    println(io, "="^140)
    println(io, "")
    println(io, "                | KS IS (%) | AD IS (%) | KS OoS (%) | AD OoS (%) | Kurt IS | Kurt OoS | ACF-MAE  | W1 IS  | H IS   | Cov IS(%) | Cov OoS(%)")
    println(io, "-"^140)
    println(io, "Observed        |           |           |            |            | $(lpad(round(kurt_obs_is,digits=2),6)) |          |          |        |        |           |")
    for (name, m_is_val, m_oos_val) in [
        ("Bootstrap", m_boot_is, m_boot_oos),
        ("Gaussian", m_gauss_is, m_gauss_oos),
        ("Laplace", m_lap_is, m_lap_oos),
        ("GARCH(1,1)", m_garch_is, m_garch_oos),
        ("CHMM", m_chmm_is, m_chmm_oos)]
        println(io, "$(rpad(name,15)) | $(lpad(m_is_val.ks,8)) | $(lpad(m_is_val.ad,8)) | $(lpad(m_oos_val.ks,9))  | $(lpad(m_oos_val.ad,9))  | $(lpad(m_is_val.kurt,6)) | $(lpad(m_oos_val.kurt,7))  | $(lpad(m_is_val.acf_mae,7)) | $(lpad(m_is_val.w1,5))  | $(m_is_val.hell) | $(lpad(m_is_val.cov,8))  | $(lpad(m_oos_val.cov,8))")
    end
    println(io, "="^140)
end

# ========================================================================================= #
# PART 2: CROSS-ASSET GENERALIZATION (Table T2)
# ========================================================================================= #
println("\n" * "="^70)
println("PART 2: Cross-Asset Generalization (K=$K)")
println("="^70)

cross_tickers = ["NVDA", "JNJ", "JPM"];

open(joinpath(RESULTS_DIR, TICKER, "Table-T2-Cross-Asset.txt"), "w") do io
    println(io, "Table T2: Cross-Asset Generalization — CHMM (K=$K, $N_PATHS paths, α=0.05)")
    println(io, "="^120)
    println(io, "Ticker | KS IS (%) | AD IS (%) | KS OoS (%) | Kurt Obs | Kurt Sim | ACF-MAE  | W1 IS  | H IS   | Cov IS(%)")
    println(io, "-"^120)

    for t in cross_tickers
        println("  Processing $t...")

        t_idx = findfirst(x -> x == t, list_of_all_tickers);
        if isnothing(t_idx); println("  $t not found, skipping."); continue; end
        R_t_is = all_R[:, t_idx];

        if !haskey(oos_dataset, t); println("  $t OoS not available, skipping."); continue; end
        R_t_oos = log_growth_matrix(oos_dataset, t; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);

        n_t_is = length(R_t_is); n_t_oos = length(R_t_oos);
        μ_t = mean(R_t_is); σ_t = std(R_t_is);
        kurt_t_obs = sum(((R_t_is .- μ_t) ./ σ_t).^4) / n_t_is - 3.0;

        # Train CHMM
        base = build(MyContinuousHiddenMarkovModel, (
            observations=R_t_is, number_of_states=K, max_iter=MAX_ITER));

        T_t = zeros(K, K);
        for i in 1:K; T_t[i, :] = probs(base.transition[i]); end
        π_t = (T_t^1000)[1, :];
        sd_t = Categorical(π_t);

        # Simulate
        sim_is = Array{Float64,2}(undef, n_t_is, N_PATHS);
        sim_oos = Array{Float64,2}(undef, n_t_oos, N_PATHS);
        for i in 1:N_PATHS
            s0 = rand(sd_t);
            states_is = base(s0, n_t_is);
            for j in 1:n_t_is; sim_is[j,i] = rand(base.emission[states_is[j]]); end
            s0 = rand(sd_t);
            states_oos = base(s0, n_t_oos);
            for j in 1:n_t_oos; sim_oos[j,i] = rand(base.emission[states_oos[j]]); end
        end

        m_t_is = eval_full(R_t_is, sim_is);
        m_t_oos = eval_full(R_t_oos, sim_oos);

        line = "$(rpad(t,6)) | $(lpad(m_t_is.ks,8)) | $(lpad(m_t_is.ad,8)) | $(lpad(m_t_oos.ks,9))  | $(lpad(round(kurt_t_obs,digits=1),7))  | $(lpad(m_t_is.kurt,7))  | $(lpad(m_t_is.acf_mae,7)) | $(lpad(m_t_is.w1,5))  | $(m_t_is.hell) | $(m_t_is.cov)"
        println("    $line")
        println(io, line)
    end
    println(io, "="^120)
end

println("\nDone. Results saved to $(joinpath(RESULTS_DIR, TICKER))")
