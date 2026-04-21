# Quick VIX CHMM-N/t/L at K=18 with the seven-metric panel; complements the
# v7 revisions by providing the main-body VIX numbers referenced in Section 6.7.

using Pkg; Pkg.activate(".");
include("Include.jl");
using Random
const V7_SEED = 20260420;
Random.seed!(V7_SEED);

const K = 18; const N_PATHS = 1000; const L_LAGS = 252; const MAX_ITER = 60;
const V7_DIR = joinpath(_ROOT, "results", "v7", "vix"); mkpath(V7_DIR);

println("Loading VIX data...");
train = MyVolatilityDataSet() |> x -> x["dataset"];
test  = MyOutOfSampleVolatilityDataSet() |> x -> x["dataset"];
R_is = log_growth_matrix(train, "VIX"; Δt=1/252, risk_free_rate=0.0, keycol=:close);
R_oos = log_growth_matrix(test, "VIX"; Δt=1/252, risk_free_rate=0.0, keycol=:close);
n_is = length(R_is); n_oos = length(R_oos);
println("  VIX IS: $n_is | OoS: $n_oos");

function eval_full(observed, sim_archive; L_val=L_LAGS)
    np = size(sim_archive, 2); n_o = length(observed);
    μ_o = mean(observed); σ_o = std(observed);
    kurt_o = sum(((observed .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L_val, n_o - 1);
    acf_o = autocor(abs.(observed), 1:L_use);
    ks_pass = 0; ad_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0;
    w1_s = 0.0; hell_s = 0.0;
    qprobs = range(0.01, 0.99, length=99);
    obs_quantiles = quantile(observed, qprobs);
    sim_qmatrix = zeros(99, np);
    for i in 1:np
        sim = sim_archive[:, i];
        pval_ks = pvalue(ApproximateTwoSampleKSTest(observed, sim));
        if pval_ks > 0.05; ks_pass += 1; end
        pval_ad = pvalue(KSampleADTest(observed, sim));
        if pval_ad > 0.05; ad_pass += 1; end
        μ_s = mean(sim); σ_s = std(sim);
        kurt_s += sum(((sim .- μ_s) ./ σ_s).^4) / length(sim) - 3.0;
        acf_sim = autocor(abs.(sim), 1:L_use);
        acf_mae_s += mean(abs.(acf_o .- acf_sim));
        obs_sorted = sort(observed); sim_sorted = sort(sim);
        n_min = min(length(obs_sorted), length(sim_sorted));
        obs_q = [obs_sorted[max(1, round(Int, k*length(obs_sorted)/n_min))] for k in 1:n_min];
        sim_q = [sim_sorted[max(1, round(Int, k*length(sim_sorted)/n_min))] for k in 1:n_min];
        w1_s += mean(abs.(obs_q .- sim_q));
        lo = min(minimum(observed), minimum(sim)) - 10;
        hi = max(maximum(observed), maximum(sim)) + 10;
        edges = range(lo, hi, length=101);
        h_o = fit(Histogram, observed, edges).weights ./ n_o;
        h_s = fit(Histogram, sim, edges).weights ./ length(sim);
        hell_s += sqrt(sum((sqrt.(h_o) .- sqrt.(h_s)).^2)) / sqrt(2);
        sim_qmatrix[:, i] = quantile(sim, qprobs);
    end
    cov_count = 0;
    for q in 1:99
        lo_env = quantile(sim_qmatrix[q, :], 0.05);
        hi_env = quantile(sim_qmatrix[q, :], 0.95);
        if obs_quantiles[q] >= lo_env && obs_quantiles[q] <= hi_env; cov_count += 1; end
    end
    return (ks=round(100*ks_pass/np, digits=1), ad=round(100*ad_pass/np, digits=1),
            kurt=round(kurt_s/np, digits=2), kurt_obs=round(kurt_o, digits=2),
            acf_mae=round(acf_mae_s/np, digits=4),
            w1=round(w1_s/np, digits=3), hell=round(hell_s/np, digits=4),
            cov=round(100.0*cov_count/99, digits=1));
end

function _stationary(model, K::Int)
    T = zeros(K, K); for i in 1:K; T[i, :] = probs(model.transition[i]); end
    π = (T^1000)[1, :]; return T, Categorical(π);
end
function _sim_paths(model, start_dist, n_is, n_oos, n_paths)
    sis = Array{Float64,2}(undef, n_is, n_paths); sos = Array{Float64,2}(undef, n_oos, n_paths);
    for i in 1:n_paths
        s0 = rand(start_dist); st = model(s0, n_is)
        for j in 1:n_is; sis[j,i] = rand(model.emission[st[j]]); end
        s0 = rand(start_dist); st = model(s0, n_oos)
        for j in 1:n_oos; sos[j,i] = rand(model.emission[st[j]]); end
    end
    return sis, sos
end

println("Fitting CHMM-N on VIX...")
m_n = build(MyContinuousHiddenMarkovModel, (observations=R_is, number_of_states=K, max_iter=MAX_ITER))
_, sd_n = _stationary(m_n, K); sn_is, sn_oos = _sim_paths(m_n, sd_n, n_is, n_oos, N_PATHS)
println("Fitting CHMM-t on VIX..."); Random.seed!(V7_SEED+1);
m_t = build(MyStudentTHiddenMarkovModel, (observations=R_is, number_of_states=K, max_iter=MAX_ITER))
_, sd_t = _stationary(m_t, K); st_is, st_oos = _sim_paths(m_t, sd_t, n_is, n_oos, N_PATHS)
println("Fitting CHMM-L on VIX..."); Random.seed!(V7_SEED+2);
m_l = build(MyLaplaceHiddenMarkovModel, (observations=R_is, number_of_states=K, max_iter=MAX_ITER))
_, sd_l = _stationary(m_l, K); sl_is, sl_oos = _sim_paths(m_l, sd_l, n_is, n_oos, N_PATHS)

mn_is = eval_full(R_is, sn_is); mn_oos = eval_full(R_oos, sn_oos);
mt_is = eval_full(R_is, st_is); mt_oos = eval_full(R_oos, st_oos);
ml_is = eval_full(R_is, sl_is); ml_oos = eval_full(R_oos, sl_oos);

open(joinpath(V7_DIR, "VIX_Metrics.txt"), "w") do io
    println(io, "VIX CHMM Seven-Metric Panel at K = $K (seed=$V7_SEED, $N_PATHS paths)")
    println(io, "="^95)
    println(io, "  IS: $n_is obs | OoS: $n_oos obs")
    println(io, "  Observed kurtosis: IS $(mn_is.kurt_obs) | OoS $(mn_oos.kurt_obs)")
    println(io)
    println(io, rpad("Family",8), " | ", rpad("KS IS",6), " | ", rpad("AD IS",6), " | ",
                rpad("KS OoS",6), " | ", rpad("AD OoS",6), " | ",
                rpad("Kurt IS",7), " | ", rpad("Kurt OoS",8), " | ",
                rpad("ACF-MAE",8), " | ", rpad("W1",6), " | ", rpad("H",7), " | ", rpad("Cov",5))
    println(io, "-"^95)
    for (tag, m_is, m_oos) in [("CHMM-N", mn_is, mn_oos), ("CHMM-t", mt_is, mt_oos), ("CHMM-L", ml_is, ml_oos)]
        println(io, rpad(tag,8), " | ",
                    rpad(m_is.ks,6), " | ", rpad(m_is.ad,6), " | ",
                    rpad(m_oos.ks,6), " | ", rpad(m_oos.ad,6), " | ",
                    rpad(m_is.kurt,7), " | ", rpad(m_oos.kurt,8), " | ",
                    rpad(m_is.acf_mae,8), " | ", rpad(m_is.w1,6), " | ",
                    rpad(m_is.hell,7), " | ", rpad(m_is.cov,5))
    end
end
println("VIX metrics written to $V7_DIR/VIX_Metrics.txt")
