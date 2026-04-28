# ========================================================================================= #
# run_baselines_and_cross_asset.jl
#
# Pipeline A (single-index trained): each ticker is fit with its own CHMM, independently.
# No cross-asset dependence is introduced here. See run_cross_asset_sim_copula.jl for the
# Pipeline B (cross-asset dependence, Table T3) counterpart.
#
# Generates:
#   1. Table 2 (SPY only): seven-way baseline comparison
#      -> results/SPY/Table-2-Baselines.txt
#   2. Table T2 (six tickers, three emission families): per-ticker marginal fidelity
#      -> results/SPY/Table-T2-Per-Ticker-Emission-Families.txt
#
# Usage:
#   include("run_baselines_and_cross_asset.jl")
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");
using Random;

const TICKER = "SPY";
const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const N_PATHS = 1000;
const L = 252;
const K = 18;
const MAX_ITER = 60;
const SEED = 20260420;       # paper-canonical global seed
const RESULTS_DIR = joinpath(@__DIR__, "results");

# Discrete baseline hyper-parameters (from prior paper arXiv:2603.10202,
# Notebook 3 `3-HMM-WithJumps-Simulation-Notebook.ipynb`, final SPY run)
const EPSILON_JUMP = 0.00005;   # fraction of steps with a jump event
const LAMBDA_JUMP = 67.0;       # mean number of jump events (Poisson)
const DISCRETE_K = 90;          # number_of_states from prior paper

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
    acf_o_raw = autocor(observed, 1:L_use);

    ks_pass = 0; ad_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0; acf_mae_raw_s = 0.0;
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

        # ACF-MAE on |G_t| (volatility clustering)
        acf_sim = autocor(abs.(sim), 1:L_use);
        acf_mae_s += mean(abs.(acf_o .- acf_sim));

        # ACF-MAE on raw G_t (linear autocorrelation)
        acf_sim_raw = autocor(sim, 1:L_use);
        acf_mae_raw_s += mean(abs.(acf_o_raw .- acf_sim_raw));

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
            acf_mae_raw=round(acf_mae_raw_s/np, digits=4),
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
Random.seed!(SEED + 1);
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
Random.seed!(SEED + 2);
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
Random.seed!(SEED + 3);
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

# --- 4a. Discrete HMM (NJ) — frequency-counted bins, no jumps ---
println("  Discrete HMM (no jumps)...")
Random.seed!(SEED + 4);
qprobs = range(0.0, 1.0, length=DISCRETE_K+1) |> collect;
bin_edges = quantile(R_is, qprobs);
function _assign_bin(x, edges)
    Kb = length(edges) - 1;
    for k in 1:Kb
        if k == Kb || x < edges[k+1]; return k; end
    end
    return Kb;
end;
bin_idx = [_assign_bin(r, bin_edges) for r in R_is];
bin_centers = zeros(DISCRETE_K);
for k in 1:DISCRETE_K
    mk = findall(==(k), bin_idx);
    bin_centers[k] = isempty(mk) ? 0.0 : mean(R_is[mk]);
end
T_counts = zeros(DISCRETE_K, DISCRETE_K);
for t in 2:length(bin_idx); T_counts[bin_idx[t-1], bin_idx[t]] += 1.0; end
T_disc = copy(T_counts);
for i in 1:DISCRETE_K
    rs = sum(T_disc[i, :]);
    T_disc[i, :] = rs > 0 ? T_disc[i, :] ./ rs : fill(1.0/DISCRETE_K, DISCRETE_K);
end
E_disc = zeros(DISCRETE_K, DISCRETE_K);
floor_ = 0.001;
for i in 1:DISCRETE_K
    for j in 1:DISCRETE_K; E_disc[i,j] = (i==j) ? 1.0 : floor_; end
    E_disc[i, :] ./= sum(E_disc[i, :]);
end
π_disc = (T_disc^1000)[1, :];
sd_disc = Categorical(π_disc);

model_nj = build(MyHiddenMarkovModel, (
    states=collect(1:DISCRETE_K), T=T_disc, E=E_disc));
disc_is = Array{Float64,2}(undef, n_is, N_PATHS);
disc_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    s0 = rand(sd_disc);
    ch = model_nj(s0, n_is);
    for t in 1:n_is
        eb = rand(model_nj.emission[ch[t]]);
        disc_is[t, i] = bin_centers[eb];
    end
    s0 = rand(sd_disc);
    ch = model_nj(s0, n_oos);
    for t in 1:n_oos
        eb = rand(model_nj.emission[ch[t]]);
        disc_oos[t, i] = bin_centers[eb];
    end
end
m_disc_is = eval_full(R_is, disc_is);
m_disc_oos = eval_full(R_oos, disc_oos);

# --- 4b. Discrete HMM + Poisson jumps (WJ) — from prior paper ---
println("  Discrete HMM + Poisson jumps...")
Random.seed!(SEED + 5);
model_wj = build(MyHiddenMarkovModelWithJumps, (
    states=collect(1:DISCRETE_K), T=T_disc, E=E_disc,
    ϵ=EPSILON_JUMP, λ=LAMBDA_JUMP));
wj_is = Array{Float64,2}(undef, n_is, N_PATHS);
wj_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    s0 = rand(sd_disc);
    ch = model_wj(s0, n_is);
    for t in 1:n_is
        eb = rand(model_wj.emission[ch[t]]);
        wj_is[t, i] = bin_centers[eb];
    end
    s0 = rand(sd_disc);
    ch = model_wj(s0, n_oos);
    for t in 1:n_oos
        eb = rand(model_wj.emission[ch[t]]);
        wj_oos[t, i] = bin_centers[eb];
    end
end
m_wj_is = eval_full(R_is, wj_is);
m_wj_oos = eval_full(R_oos, wj_oos);

# --- 5. GARCH(1,1) ---
println("  GARCH(1,1)...")
Random.seed!(SEED + 6);
garch_model = build(MyGARCHModel, (observations=R_is,));
garch_is = Array{Float64,2}(undef, n_is, N_PATHS);
garch_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    garch_is[:, i] = simulate_garch(garch_model, n_is);
    garch_oos[:, i] = simulate_garch(garch_model, n_oos);
end
m_garch_is = eval_full(R_is, garch_is);
m_garch_oos = eval_full(R_oos, garch_oos);

# --- Helper to simulate 1000 paths from a trained continuous HMM ---
function _simulate_chmm_paths(model, start_dist, n_is, n_oos, n_paths)
    sim_is = Array{Float64,2}(undef, n_is, n_paths);
    sim_oos = Array{Float64,2}(undef, n_oos, n_paths);
    for i in 1:n_paths
        s0 = rand(start_dist);
        st = model(s0, n_is);
        for j in 1:n_is; sim_is[j,i] = rand(model.emission[st[j]]); end
        s0 = rand(start_dist);
        st = model(s0, n_oos);
        for j in 1:n_oos; sim_oos[j,i] = rand(model.emission[st[j]]); end
    end
    return sim_is, sim_oos;
end

function _stationary_start_dist(model, K)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    π_stat = (T_mat^1000)[1, :];
    return Categorical(π_stat);
end

# --- 6. CHMM-Gaussian (no jumps) ---
println("  CHMM-Gaussian (K=$K)...")
Random.seed!(SEED);
chmm = build(MyContinuousHiddenMarkovModel, (
    observations=R_is, number_of_states=K, max_iter=MAX_ITER));
start_dist = _stationary_start_dist(chmm, K);
Random.seed!(SEED + 7);
chmm_is, chmm_oos = _simulate_chmm_paths(chmm, start_dist, n_is, n_oos, N_PATHS);
m_chmm_is = eval_full(R_is, chmm_is);
m_chmm_oos = eval_full(R_oos, chmm_oos);

# --- 7. CHMM-Student-t (no jumps) ---
println("  CHMM-Student-t (K=$K)...")
Random.seed!(SEED);
chmm_t = build(MyStudentTHiddenMarkovModel, (
    observations=R_is, number_of_states=K, max_iter=MAX_ITER));
start_dist_t = _stationary_start_dist(chmm_t, K);
Random.seed!(SEED + 8);
chmm_t_is, chmm_t_oos = _simulate_chmm_paths(chmm_t, start_dist_t, n_is, n_oos, N_PATHS);
m_chmm_t_is = eval_full(R_is, chmm_t_is);
m_chmm_t_oos = eval_full(R_oos, chmm_t_oos);

# --- 8. CHMM-Laplace (no jumps) ---
println("  CHMM-Laplace (K=$K)...")
Random.seed!(SEED);
chmm_l = build(MyLaplaceHiddenMarkovModel, (
    observations=R_is, number_of_states=K, max_iter=MAX_ITER));
start_dist_l = _stationary_start_dist(chmm_l, K);
Random.seed!(SEED + 9);
chmm_l_is, chmm_l_oos = _simulate_chmm_paths(chmm_l, start_dist_l, n_is, n_oos, N_PATHS);
m_chmm_l_is = eval_full(R_is, chmm_l_is);
m_chmm_l_oos = eval_full(R_oos, chmm_l_oos);

# --- 9. CHMM-GED (no jumps; per-state shape on the Gaussian-Laplace axis) ---
println("  CHMM-GED (K=$K)...")
Random.seed!(SEED);
chmm_ged = build(MyGEDHiddenMarkovModel, (
    observations=R_is, number_of_states=K, max_iter=MAX_ITER));
start_dist_ged = _stationary_start_dist(chmm_ged, K);
Random.seed!(SEED + 10);
chmm_ged_is, chmm_ged_oos = _simulate_chmm_paths(chmm_ged, start_dist_ged, n_is, n_oos, N_PATHS);
m_chmm_ged_is = eval_full(R_is, chmm_ged_is);
m_chmm_ged_oos = eval_full(R_oos, chmm_ged_oos);

# Print Table 2
println("\nTable 2: Model Comparison — $TICKER ($N_PATHS paths, α=0.05)")
println("="^130)
println("Model          | KS IS(%) | AD IS(%) | KS OoS(%) | Kurt IS | ACF-MAE|G| | ACF-MAE raw | W1 IS  | H IS   | Cov IS(%)")
println("-"^150)
for (name, m_is_val, m_oos_val) in [
    ("Bootstrap", m_boot_is, m_boot_oos),
    ("Gaussian", m_gauss_is, m_gauss_oos),
    ("Laplace", m_lap_is, m_lap_oos),
    ("Discrete NJ", m_disc_is, m_disc_oos),
    ("Discrete WJ", m_wj_is, m_wj_oos),
    ("GARCH(1,1)", m_garch_is, m_garch_oos),
    ("CHMM-N (K=$K)", m_chmm_is, m_chmm_oos),
    ("CHMM-t (K=$K)", m_chmm_t_is, m_chmm_t_oos),
    ("CHMM-L (K=$K)", m_chmm_l_is, m_chmm_l_oos),
    ("CHMM-GED (K=$K)", m_chmm_ged_is, m_chmm_ged_oos)]
    println("$(rpad(name,14)) | $(lpad(m_is_val.ks,7)) | $(lpad(m_is_val.ad,7)) | $(lpad(m_oos_val.ks,8))  | $(lpad(m_is_val.kurt,6)) | $(lpad(m_is_val.acf_mae,9)) | $(lpad(m_is_val.acf_mae_raw,10)) | $(m_is_val.w1) | $(m_is_val.hell) | $(m_is_val.cov)")
end
println("="^150)
println("Observed kurtosis: IS=$(round(kurt_obs_is,digits=2))")

# Save Table 2
mkpath(joinpath(RESULTS_DIR, TICKER));
open(joinpath(RESULTS_DIR, TICKER, "Table-2-Baselines.txt"), "w") do io
    println(io, "="^100)
    println(io, "TABLE 2. Baseline and CHMM model comparison on $TICKER.")
    println(io, "         Pipeline A (single-index trained CHMM) on the market index SPY.")
    println(io, "="^100)
    println(io, "")
    println(io, "Pipeline   : A. Single-index trained CHMM; $TICKER fit independently, no cross-asset coupling.")
    println(io, "Ticker     : $TICKER only (six-ticker generalization is in Table T2).")
    println(io, "Paths      : $N_PATHS simulated paths per model; alpha = 0.05 for KS / AD.")
    println(io, "Baselines  : Bootstrap, Gaussian i.i.d., Laplace i.i.d., Discrete HMM (no jumps = NJ,")
    println(io, "             with jumps = WJ, K=$(DISCRETE_K) bins, epsilon=$(EPSILON_JUMP), lambda=$(LAMBDA_JUMP) from prior paper), GARCH(1,1).")
    println(io, "CHMM       : CHMM-N (Gaussian), CHMM-t (Student-t per-state nu), CHMM-L (Laplace), CHMM-GED (per-state shape p_k). K=$K, no jumps.")
    println(io, "="^150)
    println(io, "")
    println(io, "                | KS IS (%) | AD IS (%) | KS OoS (%) | AD OoS (%) | Kurt IS | Kurt OoS | ACF-MAE|G| | ACF-MAE raw | W1 IS  | H IS   | Cov IS(%) | Cov OoS(%)")
    println(io, "-"^170)
    println(io, "Observed        |           |           |            |            | $(lpad(round(kurt_obs_is,digits=2),6)) |          |            |             |        |        |           |")
    for (name, m_is_val, m_oos_val) in [
        ("Bootstrap", m_boot_is, m_boot_oos),
        ("Gaussian", m_gauss_is, m_gauss_oos),
        ("Laplace", m_lap_is, m_lap_oos),
        ("Discrete NJ", m_disc_is, m_disc_oos),
        ("Discrete WJ", m_wj_is, m_wj_oos),
        ("GARCH(1,1)", m_garch_is, m_garch_oos),
        ("CHMM-N (K=$K)", m_chmm_is, m_chmm_oos),
        ("CHMM-t (K=$K)", m_chmm_t_is, m_chmm_t_oos),
        ("CHMM-L (K=$K)", m_chmm_l_is, m_chmm_l_oos),
        ("CHMM-GED (K=$K)", m_chmm_ged_is, m_chmm_ged_oos)]
        println(io, "$(rpad(name,15)) | $(lpad(m_is_val.ks,8)) | $(lpad(m_is_val.ad,8)) | $(lpad(m_oos_val.ks,9))  | $(lpad(m_oos_val.ad,9))  | $(lpad(m_is_val.kurt,6)) | $(lpad(m_oos_val.kurt,7))  | $(lpad(m_is_val.acf_mae,9)) | $(lpad(m_is_val.acf_mae_raw,10)) | $(lpad(m_is_val.w1,5))  | $(m_is_val.hell) | $(lpad(m_is_val.cov,8))  | $(lpad(m_oos_val.cov,8))")
    end
    println(io, "="^170)
end

# ========================================================================================= #
# PART 2: CROSS-ASSET GENERALIZATION (Table T2)
# ========================================================================================= #
println("\n" * "="^70)
println("PART 2: Cross-Asset Generalization (K=$K)")
println("="^70)

cross_tickers = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];

open(joinpath(RESULTS_DIR, TICKER, "Table-T2-Per-Ticker-Emission-Families.txt"), "w") do io
    println(io, "="^100)
    println(io, "TABLE T2. Per-ticker marginal fidelity across CHMM emission families.")
    println(io, "          Pipeline A (single-index trained): each ticker fit INDEPENDENTLY, no cross-asset coupling.")
    println(io, "="^100)
    println(io, "")
    println(io, "Pipeline      : A. Single-index trained CHMM (per-ticker independent Baum-Welch fit).")
    println(io, "Tickers       : SPY, NVDA, JNJ, JPM, AAPL, QQQ (six main-study tickers).")
    println(io, "K             : $K states (fixed; selected in Table T1a).")
    println(io, "Emission fams : CHMM-N (Gaussian), CHMM-t (Student-t, per-state nu), CHMM-L (Laplace), CHMM-GED (per-state shape p_k).")
    println(io, "Paths / alpha : $N_PATHS simulated paths per (ticker, family); KS and AD thresholded at alpha = 0.05.")
    println(io, "IS / OoS      : see data loader; OoS window runs 2024-01-04 through 2026-04-17.")
    println(io, "")
    println(io, "What this table tests:")
    println(io, "  Can the CHMM reproduce each ticker's univariate return distribution across four emission")
    println(io, "  families? Each ticker is its own independent experiment; there is NO dependence modeling here.")
    println(io, "")
    println(io, "How this differs from Table T3:")
    println(io, "  Table T3 (results/cross_asset/Table-T3-Cross-Asset-Dependence.txt) re-uses these per-ticker")
    println(io, "  CHMM-N marginals and ASKS A DIFFERENT QUESTION: given the marginals, which dependence")
    println(io, "  mechanism (SIM, Gaussian copula, Student-t copula) best reproduces cross-asset correlations?")
    println(io, "="^140)
    println(io, "Ticker | Emission | KS IS (%) | AD IS (%) | KS OoS (%) | Kurt Obs | Kurt Sim | ACF-MAE|G| | ACF-MAE raw | W1 IS  | H IS   | Cov IS(%)")
    println(io, "-"^160)

    emission_types = [
        ("CHMM-N", MyContinuousHiddenMarkovModel),
        ("CHMM-t", MyStudentTHiddenMarkovModel),
        ("CHMM-L", MyLaplaceHiddenMarkovModel),
        ("CHMM-GED", MyGEDHiddenMarkovModel),
    ];

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

        for (tag, MType) in emission_types
            println("    $tag...");
            Random.seed!(SEED);
            base = build(MType, (
                observations=R_t_is, number_of_states=K, max_iter=MAX_ITER));
            sd_t = _stationary_start_dist(base, K);
            Random.seed!(SEED + 11);
            sim_is, sim_oos = _simulate_chmm_paths(base, sd_t, n_t_is, n_t_oos, N_PATHS);

            m_t_is = eval_full(R_t_is, sim_is);
            m_t_oos = eval_full(R_t_oos, sim_oos);

            line = "$(rpad(t,6)) | $(rpad(tag,8)) | $(lpad(m_t_is.ks,8)) | $(lpad(m_t_is.ad,8)) | $(lpad(m_t_oos.ks,9))  | $(lpad(round(kurt_t_obs,digits=1),7))  | $(lpad(m_t_is.kurt,7))  | $(lpad(m_t_is.acf_mae,9)) | $(lpad(m_t_is.acf_mae_raw,10)) | $(lpad(m_t_is.w1,5))  | $(m_t_is.hell) | $(m_t_is.cov)"
            println("      $line");
            println(io, line);
        end
    end
    println(io, "="^140)
end

println("\nDone. Results saved to $(joinpath(RESULTS_DIR, TICKER))")
