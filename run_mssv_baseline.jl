# ========================================================================================= #
# run_mssv_baseline.jl
#
# Markov-Switching Stochastic Volatility (MSSV) baseline as a fair modern regime-switching
# competitor for the body comparison panel. Addresses peer-review item P3.2 (R3.W2,
# R3.Exp3): the existing benchmarks miss a regime-switching SV row; the Carvalho-Lopes
# (2007) MSSV specification is the natural target. We use a tractable quasi-MLE
# implementation based on the Harvey-Ruiz-Shephard (1994) log-squared linearisation
# rather than full particle-MCMC, which is sufficient to anchor the comparison.
#
# Specification:
#   r_t = exp(h_t / 2) * eps_t,  eps_t ~ N(0, 1) iid,
#   h_t = c_{s_t} + phi_{s_t} * h_{t-1} + sigma_eta_{s_t} * eta_t,
#   s_t ∈ {1, 2} a 2-state Markov chain with transition matrix T.
#
# Quasi-MLE: y_t = log((r_t - mu_hat)^2) - E[log chi^2_1] is approximately h_t + xi_t
# with xi_t ~ N(0, pi^2/2) (HRS 1994). Treat (s_t, h_t) as joint hidden state and run
# a Hamilton-style filter with state-dependent linear-Gaussian h_t dynamics.
#
# This is a 2-state regime-switching SV (the "Markov-Switching SV" of Carvalho-Lopes
# and So-Lam-Li 1998) — distinct from MS-GARCH (Haas-Mittnik-Paolella 2004) in the body
# panel. It complements the single-regime SV-AR(1) row already in Appendix sv_msm_jd.
#
# Output: results/mssv_baseline/mssv_baseline.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random, Statistics, HypothesisTests, StatsBase, Printf, LinearAlgebra
using Distributions

const SEED      = 20260420;
const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const N_PATHS   = 1000;
const L_LAGS    = 252;
const ALPHA_KS  = 0.05;
const K_REGIMES = 2;
const MAX_ITER  = 200;
const TOL       = 1e-4;

const _LOG_CHI2_1_MEAN = -1.2703628454614781;
const _LOG_CHI2_1_VAR  = 4.9348022005446793;

const OUT_DIR = joinpath(_ROOT, "results", "mssv_baseline");
mkpath(OUT_DIR);

println("="^88);
println("  Markov-Switching Stochastic Volatility (MSSV) baseline  (peer-review P3.2)");
println("="^88);

# ----------------------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading SPY IS / OoS ...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
all_tickers = keys(filtered) |> collect |> sort;
all_R = log_growth_matrix(filtered, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==(TICKER), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
n_is = length(R_is);
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, TICKER; Δt=DT, risk_free_rate=RISK_FREE));
n_oos = length(R_oos);
println("  IS = $n_is, OoS = $n_oos");

# ----------------------------------------------------------------------------------------- #
# 2-state MSSV via Hamilton-style filter on log-squared linearisation
# State variables:
#   s_t ∈ {1, 2}: regime
#   h_t: latent log-volatility (continuous; collapsed via per-regime Kalman)
# Approximation: collapse the latent h_t conditional on (s_{t-1}, s_t) by a Gaussian
# (the "GPB(1)" / Kim-Nelson 1999 approximation). This is the standard MSSV quasi-MLE.
#
# Per-regime parameters: c_k (intercept), phi_k (persistence), sigma_eta2_k (innovation var)
# Transition matrix T (2x2, rows sum to 1).
# ----------------------------------------------------------------------------------------- #

mutable struct MSSVParams
    mu::Float64
    c::Vector{Float64}             # per-regime intercept
    phi::Vector{Float64}           # per-regime persistence
    sigma_eta2::Vector{Float64}    # per-regime innovation variance
    T::Matrix{Float64}             # transition matrix
    pi0::Vector{Float64}           # initial regime distribution
end

function _init_mssv(R::Vector{Float64})
    mu = mean(R);
    K = K_REGIMES;
    c = [-1.0, 1.0];               # low-vol vs high-vol intercepts (in log-vol scale)
    phi = [0.92, 0.85];
    sigma_eta2 = [0.05, 0.30];
    T = [0.97 0.03; 0.05 0.95];
    pi0 = [0.5, 0.5];
    return MSSVParams(mu, c, phi, sigma_eta2, T, pi0);
end

# Hamilton filter on (s_t, h_t):
# Given Pr(s_{t-1} = i | y_{1:t-1}) and (m_{t-1|t-1}, P_{t-1|t-1}) = filtered Gaussian for h_{t-1},
# for each transition (i -> j):
#   h_{t|t-1, i, j} = c_j + phi_j * m_{t-1|t-1, i}
#   P_{t|t-1, i, j} = phi_j^2 * P_{t-1|t-1, i} + sigma_eta2_j
#   y_t observation density: y_t = h_t + xi_t with xi_t ~ N(0, pi^2/2)
#   v = y_t - h_{t|t-1, i, j}
#   S = P_{t|t-1, i, j} + LOG_CHI2_VAR
#   loglik contribution = log Pr(s_{t-1}=i, s_t=j | y_{1:t-1}) + log N(v; 0, S)
#   Kalman update: K = P_{t|t-1} / S, h_{t|t,i,j} = h_{t|t-1,i,j} + K*v, P_{t|t,i,j} = (1-K)*P_{t|t-1,i,j}
# Then collapse over i: Pr(s_t = j | y_{1:t}) ∝ sum_i {prior * obs density};
# (m_{t|t, j}, P_{t|t, j}) = posterior weighted average over i (collapse step).
function _mssv_loglik(p::MSSVParams, y_centered::Vector{Float64})
    K = K_REGIMES; n = length(y_centered);
    # Initial regime probs
    pr = copy(p.pi0);
    # Initialise h_{0|0} per regime at unconditional regime mean
    h_filt = zeros(K);
    P_filt = zeros(K);
    for k in 1:K
        h_filt[k] = p.c[k] / max(1.0 - p.phi[k], 1e-3);
        P_filt[k] = p.sigma_eta2[k] / max(1.0 - p.phi[k]^2, 1e-3);
    end
    log_lik = 0.0;
    for t in 1:n
        # Predict h_{t|t-1}, P_{t|t-1} for each (i, j) pair
        h_pred = zeros(K, K);
        P_pred = zeros(K, K);
        for i in 1:K, j in 1:K
            h_pred[i, j] = p.c[j] + p.phi[j] * h_filt[i];
            P_pred[i, j] = p.phi[j]^2 * P_filt[i] + p.sigma_eta2[j];
        end
        # Joint prior Pr(s_{t-1}=i, s_t=j) = pr[i] * T[i, j]
        joint_prior = zeros(K, K);
        for i in 1:K, j in 1:K
            joint_prior[i, j] = pr[i] * p.T[i, j];
        end
        # Likelihood per (i, j)
        like_ij = zeros(K, K);
        K_gain = zeros(K, K);
        h_post = zeros(K, K);
        P_post = zeros(K, K);
        for i in 1:K, j in 1:K
            v = y_centered[t] - h_pred[i, j];
            S = P_pred[i, j] + _LOG_CHI2_1_VAR;
            S = max(S, 1e-8);
            like_ij[i, j] = exp(-0.5 * (log(2pi*S) + v^2 / S));
            K_gain[i, j] = P_pred[i, j] / S;
            h_post[i, j] = h_pred[i, j] + K_gain[i, j] * v;
            P_post[i, j] = (1.0 - K_gain[i, j]) * P_pred[i, j];
        end
        # Joint posterior Pr(s_{t-1}=i, s_t=j | y_{1:t}) = joint_prior * like_ij / total
        total = 0.0;
        joint_post = zeros(K, K);
        for i in 1:K, j in 1:K
            joint_post[i, j] = joint_prior[i, j] * like_ij[i, j];
            total += joint_post[i, j];
        end
        if total <= 0.0
            return -1e10;
        end
        log_lik += log(total);
        joint_post ./= total;
        # Marginalise over i to get Pr(s_t = j | y_{1:t})
        new_pr = zeros(K);
        for j in 1:K
            for i in 1:K
                new_pr[j] += joint_post[i, j];
            end
        end
        # Kim-Nelson collapse: m_{t|t, j} = sum_i (joint_post[i, j] / new_pr[j]) * h_post[i, j]
        new_h_filt = zeros(K);
        new_P_filt = zeros(K);
        for j in 1:K
            if new_pr[j] > 1e-12
                for i in 1:K
                    w = joint_post[i, j] / new_pr[j];
                    new_h_filt[j] += w * h_post[i, j];
                end
                for i in 1:K
                    w = joint_post[i, j] / new_pr[j];
                    new_P_filt[j] += w * (P_post[i, j] + (h_post[i, j] - new_h_filt[j])^2);
                end
            else
                new_h_filt[j] = h_post[1, j];
                new_P_filt[j] = P_post[1, j];
            end
        end
        pr = new_pr; h_filt = new_h_filt; P_filt = new_P_filt;
    end
    return log_lik;
end

# Pack/unpack to vector for Nelder-Mead
function _pack(p::MSSVParams)
    # params: mu (free), c1, c2 (free), phi1, phi2 (atanh), log sigma_eta2_{1,2}, transitions T_{12}, T_{21} (logit)
    return [
        p.mu, p.c[1], p.c[2],
        atanh(clamp(p.phi[1], -0.9999, 0.9999)),
        atanh(clamp(p.phi[2], -0.9999, 0.9999)),
        log(p.sigma_eta2[1]), log(p.sigma_eta2[2]),
        log(p.T[1, 2] / max(1.0 - p.T[1, 2], 1e-6)),
        log(p.T[2, 1] / max(1.0 - p.T[2, 1], 1e-6)),
    ];
end

function _unpack(v::Vector{Float64})::MSSVParams
    mu = v[1];
    # Clip c into a sensible log-vol range to prevent regime-explosion artefacts
    c = [clamp(v[2], -3.0, 3.0), clamp(v[3], -3.0, 3.0)];
    phi = [tanh(v[4]), tanh(v[5])];
    # Clip log sigma_eta2 to prevent the Nelder-Mead degenerate region where one regime
    # absorbs the outlier innovations with huge variance (gives kurtosis blow-up under
    # exp(h/2) scaling).
    se2_1 = exp(clamp(v[6], -5.0, 0.5));      # sigma_eta2 ∈ [exp(-5), exp(0.5)] ≈ [0.0067, 1.65]
    se2_2 = exp(clamp(v[7], -5.0, 0.5));
    sigma_eta2 = [se2_1, se2_2];
    p12 = 1.0 / (1.0 + exp(-v[8]));
    p21 = 1.0 / (1.0 + exp(-v[9]));
    T = [1.0-p12 p12; p21 1.0-p21];
    pi0 = [0.5, 0.5];
    return MSSVParams(mu, c, phi, sigma_eta2, T, pi0);
end

function _fit_mssv(R::Vector{Float64}; max_iter::Int=MAX_ITER)
    mu_hat = mean(R);
    res2 = max.((R .- mu_hat) .^ 2, 1e-12);
    y_centered = log.(res2) .- _LOG_CHI2_1_MEAN;

    p_init = _init_mssv(R);
    v_init = _pack(p_init);

    nll_count = Ref(0);
    function nll(v::Vector{Float64})
        try
            p = _unpack(v);
            ll = _mssv_loglik(p, y_centered);
            nll_count[] += 1;
            return -ll;
        catch
            return 1e10;
        end
    end

    # Nelder-Mead from existing module (loaded via Include.jl from GARCHFamily.jl)
    pbest, nllbest = _nelder_mead(nll, v_init; max_iter=max_iter);
    println("  ($(nll_count[]) NLL evaluations; final NLL = $(round(nllbest, digits=2)))");
    return _unpack(pbest), -nllbest;
end

function _simulate_mssv(p::MSSVParams, T_horizon::Int; n_paths::Int=N_PATHS, seed::Int=SEED)
    Random.seed!(seed);
    out = Matrix{Float64}(undef, T_horizon, n_paths);
    for path in 1:n_paths
        s = rand() < p.pi0[1] ? 1 : 2;
        h = p.c[s] / max(1.0 - p.phi[s], 1e-3);
        for t in 1:T_horizon
            # Transition
            u = rand();
            cum = 0.0; new_s = 1;
            for j in 1:K_REGIMES
                cum += p.T[s, j];
                if u < cum; new_s = j; break; end
            end
            s = new_s;
            # Update h
            h = p.c[s] + p.phi[s] * h + sqrt(p.sigma_eta2[s]) * randn();
            out[t, path] = p.mu + exp(h / 2) * randn();
        end
    end
    return out;
end

# ----------------------------------------------------------------------------------------- #
# Fit and evaluate
# ----------------------------------------------------------------------------------------- #
println("\n[fit] Fitting 2-state MSSV on SPY IS ...");
@time params_mssv, ll_mssv = _fit_mssv(R_is);
println("  mu = $(round(params_mssv.mu, digits=4))");
println("  c  = $(round.(params_mssv.c, digits=3))");
println("  phi = $(round.(params_mssv.phi, digits=4))");
println("  sigma_eta2 = $(round.(params_mssv.sigma_eta2, digits=4))");
println("  T = $(round.(params_mssv.T, digits=4))");
println("  log-lik = $(round(ll_mssv, digits=2))");

println("\n[sim] Simulating $N_PATHS paths IS / OoS ...");
sim_is  = _simulate_mssv(params_mssv, n_is;  n_paths=N_PATHS, seed=SEED);
sim_oos = _simulate_mssv(params_mssv, n_oos; n_paths=N_PATHS, seed=SEED+11);

# Metrics
function _metrics(R_obs::Vector{Float64}, sim::Matrix{Float64}; L::Int=L_LAGS, α::Float64=ALPHA_KS)
    np = size(sim, 2); n_o = length(R_obs);
    L_use = min(L, n_o - 1);
    acf_o = autocor(abs.(R_obs), 1:L_use);
    μ_o = mean(R_obs); σ_o = std(R_obs);
    kurt_o = sum(((R_obs .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    ks_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0;
    raw_acf_mae = 0.0;
    raw_acf_o = autocor(R_obs, 1:L_use);
    for i in 1:np
        s = sim[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= α; ks_pass += 1; end
        μ_s = mean(s); σ_s = std(s);
        kurt_s += sum(((s .- μ_s) ./ σ_s).^4) / length(s) - 3.0;
        acf_s = autocor(abs.(s), 1:L_use);
        acf_mae_s += mean(abs.(acf_o .- acf_s));
        raw_acf_s = autocor(s, 1:L_use);
        raw_acf_mae += mean(abs.(raw_acf_o .- raw_acf_s));
    end
    return (ks_pct=round(100*ks_pass/np, digits=1),
            sim_kurt=round(kurt_s/np, digits=2),
            obs_kurt=round(kurt_o, digits=2),
            acf_mae=round(acf_mae_s/np, digits=4),
            raw_acf_mae=round(raw_acf_mae/np, digits=4));
end

m_is  = _metrics(R_is, sim_is);
m_oos = _metrics(R_oos, sim_oos);

println();
@printf("[IS]  KS = %5.1f%%  sim kurt = %5.2f  obs kurt = %5.2f  |G| ACF-MAE = %.4f  raw ACF-MAE = %.4f\n",
        m_is.ks_pct, m_is.sim_kurt, m_is.obs_kurt, m_is.acf_mae, m_is.raw_acf_mae);
@printf("[OoS] KS = %5.1f%%  sim kurt = %5.2f  obs kurt = %5.2f  |G| ACF-MAE = %.4f  raw ACF-MAE = %.4f\n",
        m_oos.ks_pct, m_oos.sim_kurt, m_oos.obs_kurt, m_oos.acf_mae, m_oos.raw_acf_mae);

# ----------------------------------------------------------------------------------------- #
# Output
# ----------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "mssv_baseline.txt");
open(out_path, "w") do io
    println(io, "="^96);
    println(io, "Markov-Switching Stochastic Volatility (MSSV) baseline  (peer-review P3.2 / R3.W2)");
    println(io, "="^96);
    println(io, "Setup: 2-state MSSV with regime-dependent (c_k, phi_k, sigma_eta_k); Hamilton/Kim-Nelson");
    println(io, "       quasi-MLE on the log-squared linearisation y = log((r-mu)^2) - E[log chi2_1].");
    println(io, "       Fit by Nelder-Mead. Specification follows Carvalho-Lopes (2007) and So-Lam-Li (1998);");
    println(io, "       quasi-MLE shortcut following Harvey-Ruiz-Shephard (1994) / Kim-Nelson (1999).");
    println(io, "Source: results/mssv_baseline/mssv_baseline.txt");
    println(io);
    println(io, "Fitted parameters:");
    @printf(io, "  mu          = %+.4f\n", params_mssv.mu);
    @printf(io, "  c (regime 1, 2)         = (%.3f, %.3f)\n", params_mssv.c[1], params_mssv.c[2]);
    @printf(io, "  phi (regime 1, 2)       = (%.4f, %.4f)\n", params_mssv.phi[1], params_mssv.phi[2]);
    @printf(io, "  sigma_eta2 (regime 1, 2) = (%.4f, %.4f)\n", params_mssv.sigma_eta2[1], params_mssv.sigma_eta2[2]);
    @printf(io, "  T = [%.4f %.4f; %.4f %.4f]\n",
            params_mssv.T[1,1], params_mssv.T[1,2], params_mssv.T[2,1], params_mssv.T[2,2]);
    @printf(io, "  log-lik (quasi)   = %.2f\n", ll_mssv);
    println(io);
    println(io, "Comparison row for body Table tab:model_comparison");
    @printf(io, "%-15s %-10s %-10s %-10s %-10s %-12s %-12s\n",
            "Model", "IS KS%", "OoS KS%", "IS kurt", "OoS kurt", "|G| ACF-MAE", "raw ACF-MAE");
    println(io, "-"^96);
    @printf(io, "%-15s %-10.1f %-10.1f %-10.2f %-10.2f %-12.4f %-12.4f\n",
            "MSSV (K=2)", m_is.ks_pct, m_oos.ks_pct, m_is.sim_kurt, m_oos.sim_kurt,
            m_is.acf_mae, m_is.raw_acf_mae);
end

println();
println("[done] Wrote $out_path");
