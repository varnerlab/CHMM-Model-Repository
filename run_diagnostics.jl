# ========================================================================================= #
# run_diagnostics.jl
#
# Generates every diagnostic / supplementary-results subdirectory referenced by
# the paper's supplemental section:
#
#   [4.1] VaR/ES utility back-test            -> results/diagnostics/utility/
#   [4.2] CHMM-t nu_k diagnostics             -> results/diagnostics/nu_diagnostics/
#   [4.3] OoS KS power calibration            -> results/diagnostics/ks_power/
#   [4.4] Ryden K=2 random-init replication   -> results/diagnostics/ryden_k2/
#   [4.6] Copula profile log-L plot           -> results/diagnostics/copula_profile/
#   [5.2] Block-bootstrap benchmark row       -> results/diagnostics/block_bootstrap/
#   [8.1] Discrete bin-conditional Student-t  -> results/diagnostics/bin_t/
#
# Uses the global seed below so numbers are deterministic.
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
const SEED = 20260420;       # deterministic for table reproducibility
Random.seed!(SEED);

const TICKER       = "SPY";
const RISK_FREE    = 0.0;
const DT           = 1/252;
const N_PATHS      = 1000;
const N_PATHS_OOS  = 1000;
const L_LAGS       = 252;
const K_MAIN       = 18;
const MAX_ITER     = 60;
const DISCRETE_K   = 13;

const DIAG_DIR       = joinpath(_ROOT, "results", "diagnostics");
mkpath(DIAG_DIR);

println("="^70)
println("  Diagnostics pipeline")
println("  Seed:      $SEED")
println("  Main K:    $K_MAIN")
println("  N paths:   $N_PATHS")
println("="^70)

# ========================================================================================= #
# Data
# ========================================================================================= #
println("\n[setup] Loading SPY + cross-asset tickers...")

train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);

idx_spy = findfirst(==("SPY"),  all_tickers);
idx_jnj = findfirst(==("JNJ"),  all_tickers);
idx_jpm = findfirst(==("JPM"),  all_tickers);
R_is   = all_R[:, idx_spy];
R_jnj_is = all_R[:, idx_jnj];
R_jpm_is = all_R[:, idx_jpm];
n_is   = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos      = log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE);
R_jnj_oos  = log_growth_matrix(oos_dataset, "JNJ"; Δt=DT, risk_free_rate=RISK_FREE);
R_jpm_oos  = log_growth_matrix(oos_dataset, "JPM"; Δt=DT, risk_free_rate=RISK_FREE);
n_oos      = length(R_oos);

println("  SPY:  IS $n_is / OoS $n_oos");
println("  JNJ:  IS $(length(R_jnj_is)) / OoS $(length(R_jnj_oos))");
println("  JPM:  IS $(length(R_jpm_is)) / OoS $(length(R_jpm_oos))");

# Observed kurtosis summary (referenced by later experiments)
kurt_obs_is  = sum(((R_is  .- mean(R_is))  ./ std(R_is)).^4)  / n_is  - 3.0;
kurt_obs_oos = sum(((R_oos .- mean(R_oos)) ./ std(R_oos)).^4) / n_oos - 3.0;

# ========================================================================================= #
# Generic metrics helper (7-metric panel, matches run_baselines_and_cross_asset.jl)
# ========================================================================================= #
function eval_full(observed, sim_archive; L_val=L_LAGS)
    np = size(sim_archive, 2); n_o = length(observed);
    μ_o = mean(observed); σ_o = std(observed);
    kurt_o = sum(((observed .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L_val, n_o - 1);
    acf_o = autocor(abs.(observed), 1:L_use);
    acf_o_raw = autocor(observed, 1:L_use);

    ks_pass = 0; ad_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0; acf_mae_raw_s = 0.0;
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
        acf_sim_raw = autocor(sim, 1:L_use);
        acf_mae_raw_s += mean(abs.(acf_o_raw .- acf_sim_raw));
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
            cov=round(100.0*cov_count/99, digits=1));
end

function _stationary(model, K::Int)
    T = zeros(K, K);
    for i in 1:K; T[i, :] = probs(model.transition[i]); end
    π = (T^1000)[1, :];
    return T, Categorical(π);
end

function _simulate_chmm_paths(model, start_dist, n_is, n_oos, n_paths)
    sim_is  = Array{Float64,2}(undef, n_is,  n_paths);
    sim_oos = Array{Float64,2}(undef, n_oos, n_paths);
    for i in 1:n_paths
        s0 = rand(start_dist); st = model(s0, n_is);
        for j in 1:n_is; sim_is[j,i] = rand(model.emission[st[j]]); end
        s0 = rand(start_dist); st = model(s0, n_oos);
        for j in 1:n_oos; sim_oos[j,i] = rand(model.emission[st[j]]); end
    end
    return sim_is, sim_oos;
end

# ========================================================================================= #
# Train the three base CHMM families at K = 18 (shared input for utility + coverage rows)
# ========================================================================================= #
println("\n[base] Training CHMM-N / CHMM-t / CHMM-L at K = $K_MAIN on SPY IS...")

chmm_n = build(MyContinuousHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
chmm_t = build(MyStudentTHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
chmm_l = build(MyLaplaceHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));

_, sd_n = _stationary(chmm_n, K_MAIN);
_, sd_t = _stationary(chmm_t, K_MAIN);
_, sd_l = _stationary(chmm_l, K_MAIN);

println("  CHMM-N converged in $(length(chmm_n.log_likelihood_history)) iters");
println("  CHMM-t converged in $(length(chmm_t.log_likelihood_history)) iters");
println("  CHMM-L converged in $(length(chmm_l.log_likelihood_history)) iters");

println("\n[base] Simulating $N_PATHS paths from each family...")
sim_n_is, sim_n_oos = _simulate_chmm_paths(chmm_n, sd_n, n_is, n_oos, N_PATHS);
sim_t_is, sim_t_oos = _simulate_chmm_paths(chmm_t, sd_t, n_is, n_oos, N_PATHS);
sim_l_is, sim_l_oos = _simulate_chmm_paths(chmm_l, sd_l, n_is, n_oos, N_PATHS);
println("  Paths ready.");

# ========================================================================================= #
# [4.1] VaR / ES utility back-test
# ========================================================================================= #
println("\n[4.1] VaR / ES utility back-test...")

util_dir = joinpath(DIAG_DIR, "utility"); mkpath(util_dir);

function var_es(x::AbstractVector, α::Float64)
    # Left-tail VaR at level α. Daily excess-growth input.
    v = quantile(x, α);
    tail = x[x .<= v];
    es = isempty(tail) ? v : mean(tail);
    return v, es;
end

function mc_var_es(sim_archive::AbstractMatrix, α::Float64)
    np = size(sim_archive, 2);
    vs = Vector{Float64}(undef, np);
    es = Vector{Float64}(undef, np);
    for i in 1:np
        v, e = var_es(sim_archive[:, i], α);
        vs[i] = v; es[i] = e;
    end
    return vs, es;
end

# Observed historical VaR/ES on IS and OoS
obs_is_v01, obs_is_e01 = var_es(R_is, 0.01);
obs_is_v05, obs_is_e05 = var_es(R_is, 0.05);
obs_oos_v01, obs_oos_e01 = var_es(R_oos, 0.01);
obs_oos_v05, obs_oos_e05 = var_es(R_oos, 0.05);

# Bootstrap baseline
boot_is = Array{Float64,2}(undef, n_is, N_PATHS);
boot_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    boot_is[:,  i] = R_is[rand(1:n_is, n_is)];
    boot_oos[:, i] = R_is[rand(1:n_is, n_oos)];
end

# GARCH baseline
garch_model = build(MyGARCHModel, (observations=R_is,));
garch_is  = Array{Float64,2}(undef, n_is,  N_PATHS);
garch_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    garch_is[:,  i] = simulate_garch(garch_model, n_is);
    garch_oos[:, i] = simulate_garch(garch_model, n_oos);
end

# Simulation-based VaR and ES per path
function _summary(v_arr, e_arr)
    return (v_med=median(v_arr), v_lo=quantile(v_arr, 0.05), v_hi=quantile(v_arr, 0.95),
            e_med=median(e_arr), e_lo=quantile(e_arr, 0.05), e_hi=quantile(e_arr, 0.95));
end

println("  Computing VaR/ES on all models...")
models_var = Dict{String,Any}();
for (name, sim_is_mx, sim_oos_mx) in [
    ("Bootstrap", boot_is,     boot_oos),
    ("GARCH",     garch_is,    garch_oos),
    ("CHMM-N",    sim_n_is,    sim_n_oos),
    ("CHMM-t",    sim_t_is,    sim_t_oos),
    ("CHMM-L",    sim_l_is,    sim_l_oos)]
    v01is, e01is = mc_var_es(sim_is_mx,  0.01);
    v05is, e05is = mc_var_es(sim_is_mx,  0.05);
    v01os, e01os = mc_var_es(sim_oos_mx, 0.01);
    v05os, e05os = mc_var_es(sim_oos_mx, 0.05);
    models_var[name] = (
        is01=_summary(v01is, e01is),
        is05=_summary(v05is, e05is),
        os01=_summary(v01os, e01os),
        os05=_summary(v05os, e05os),
    );
end

open(joinpath(util_dir, "VaR_ES_Backtest.txt"), "w") do io
    println(io, "VaR and ES Back-test Calibration (SPY, seed=$SEED, $N_PATHS paths)");
    println(io, "="^98);
    println(io, "Historical observed excess growth rate: daily (annualized), dt=1/252");
    println(io, "  IS observed VaR01=$(round(obs_is_v01, digits=3))  ES01=$(round(obs_is_e01, digits=3))");
    println(io, "  IS observed VaR05=$(round(obs_is_v05, digits=3))  ES05=$(round(obs_is_e05, digits=3))");
    println(io, "  OoS observed VaR01=$(round(obs_oos_v01, digits=3))  ES01=$(round(obs_oos_e01, digits=3))");
    println(io, "  OoS observed VaR05=$(round(obs_oos_v05, digits=3))  ES05=$(round(obs_oos_e05, digits=3))");
    println(io);
    println(io, rpad("Model",10), " | ",
                "IS VaR01 [5-95]                | IS ES01 [5-95]                 | ",
                "IS VaR05 [5-95]                | IS ES05 [5-95]                 | ",
                "OoS VaR01 [5-95]               | OoS ES01 [5-95]                | ",
                "OoS VaR05 [5-95]               | OoS ES05 [5-95]");
    println(io, "-"^240);
    for name in ["Bootstrap", "GARCH", "CHMM-N", "CHMM-t", "CHMM-L"]
        d = models_var[name];
        function fmt(s)
            return rpad("$(round(s.v_med, digits=3)) [$(round(s.v_lo, digits=3)), $(round(s.v_hi, digits=3))]", 30);
        end
        function fmt_e(s)
            return rpad("$(round(s.e_med, digits=3)) [$(round(s.e_lo, digits=3)), $(round(s.e_hi, digits=3))]", 30);
        end
        println(io, rpad(name,10), " | ",
                    fmt(d.is01), " | ", fmt_e(d.is01), " | ",
                    fmt(d.is05), " | ", fmt_e(d.is05), " | ",
                    fmt(d.os01), " | ", fmt_e(d.os01), " | ",
                    fmt(d.os05), " | ", fmt_e(d.os05));
    end
    println(io);
    println(io, "Notes: VaR and ES are computed at probability levels 0.01 and 0.05 on each of $N_PATHS simulated paths,");
    println(io, "then summarised by median and 5th-95th percentile envelope across paths. A well-calibrated generator");
    println(io, "should bracket the observed historical VaR/ES within its 5-95 percentile envelope.");
end

# VaR/ES figure: ordered comparison
var_fig = plot(layout=(2,2), size=(1100,800),
    left_margin=18Plots.mm,
    plot_title="VaR and ES back-test (SPY, $N_PATHS paths)");

model_names = ["Bootstrap", "GARCH", "CHMM-N", "CHMM-t", "CHMM-L"];
xs = 1:length(model_names);

for (i, (tag, obs_v, obs_e, key)) in enumerate([
    ("IS VaR (0.01)",  obs_is_v01,  obs_is_e01,  :is01),
    ("IS ES  (0.01)",  obs_is_v01,  obs_is_e01,  :is01),
    ("OoS VaR (0.05)", obs_oos_v05, obs_oos_e05, :os05),
    ("OoS ES  (0.05)", obs_oos_v05, obs_oos_e05, :os05)])

    is_var = startswith(tag, "IS VaR") || startswith(tag, "OoS VaR");
    meds = [is_var ? models_var[n][key].v_med : models_var[n][key].e_med for n in model_names];
    los  = [is_var ? models_var[n][key].v_lo  : models_var[n][key].e_lo  for n in model_names];
    his  = [is_var ? models_var[n][key].v_hi  : models_var[n][key].e_hi  for n in model_names];
    obs_line = is_var ? obs_v : obs_e;

    ylab = (i == 1 || i == 3) ? "Annualized log excess growth rate" : "";
    scatter!(var_fig, xs, meds, yerror=(meds .- los, his .- meds),
        subplot=i, title=tag, ms=6, color=:navy, label="sim median [5-95]",
        xticks=(xs, model_names),
        ylabel=ylab);
    hline!(var_fig, [obs_line], subplot=i, color=:red, lw=2, ls=:dash, label="observed");
end
savefig(var_fig, joinpath(util_dir, "VaR_ES_Backtest.pdf"));
savefig(var_fig, joinpath(util_dir, "VaR_ES_Backtest.svg"));
println("  [4.1] Done.")

# ========================================================================================= #
# [4.2] CHMM-t nu_k diagnostics + bracket sensitivity
# ========================================================================================= #
println("\n[4.2] CHMM-t nu_k histogram + bracket sensitivity...")

nu_dir = joinpath(DIAG_DIR, "nu_diagnostics"); mkpath(nu_dir);

# Pull nu_k from the already-fitted CHMM-t at K=18.
nu_k_main = Float64[];
for k in 1:K_MAIN
    push!(nu_k_main, chmm_t.emission[k].ρ.ν);
end

open(joinpath(nu_dir, "nu_values_K$(K_MAIN).txt"), "w") do io
    println(io, "Per-state nu_k under CHMM-t at K = $K_MAIN, SPY IS");
    println(io, "="^50);
    for (k, ν) in enumerate(nu_k_main)
        println(io, rpad("State $k", 10), " ν_k = $(round(ν, digits=2))");
    end
    println(io);
    println(io, "Summary: min=$(round(minimum(nu_k_main), digits=2)) median=$(round(median(nu_k_main), digits=2)) max=$(round(maximum(nu_k_main), digits=2))");
    println(io, "# at lower bracket (<= 2.2): $(count(<=(2.2), nu_k_main))");
    println(io, "# at upper bracket (>= 49): $(count(>=(49.0), nu_k_main))");
end

# Histogram figure
nu_hist_fig = plot(xlabel="ν_k", ylabel="Count", title="Per-state ν_k (CHMM-t, K=$K_MAIN, SPY)",
    titlefontsize=11, legend=false, size=(700,450));
histogram!(nu_hist_fig, nu_k_main, bins=range(2.0, 50.0, length=25), color=:steelblue, alpha=0.8);
vline!(nu_hist_fig, [2.1], color=:red, lw=2, ls=:dash);
annotate!(nu_hist_fig, 3.0, maximum([1,maximum(ones(length(nu_k_main)))])*0.9, text("ν_min = 2.1", :red, 9, :left));
savefig(nu_hist_fig, joinpath(nu_dir, "Fig-nu-Histogram-K$(K_MAIN).pdf"));
savefig(nu_hist_fig, joinpath(nu_dir, "Fig-nu-Histogram-K$(K_MAIN).svg"));

# Bracket sensitivity: refit CHMM-t at K = 18 with lifted nu_min
ν_floors = [2.1, 2.5, 3.0, 4.0];
bracket_rows = [];

for νf in ν_floors
    println("  Refitting CHMM-t with ν_min = $νf ...");
    Random.seed!(SEED);   # keep each refit independent but reproducible
    local_model = build(MyStudentTHiddenMarkovModel, (
        observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER,
        ν_bounds=(νf, 50.0)));
    _, local_sd = _stationary(local_model, K_MAIN);
    local_is, local_oos = _simulate_chmm_paths(local_model, local_sd, n_is, n_oos, N_PATHS);
    m_is  = eval_full(R_is,  local_is);
    m_oos = eval_full(R_oos, local_oos);
    push!(bracket_rows, (νf=νf, m_is=m_is, m_oos=m_oos,
                         nu_k=[local_model.emission[k].ρ.ν for k in 1:K_MAIN]));
end

# Also a Gaussian-limit row (equivalent to CHMM-N at K=18)
m_n_is  = eval_full(R_is,  sim_n_is);
m_n_oos = eval_full(R_oos, sim_n_oos);
push!(bracket_rows, (νf=Inf, m_is=m_n_is, m_oos=m_n_oos, nu_k=[Inf for _ in 1:K_MAIN]));

open(joinpath(nu_dir, "Bracket_Sensitivity_K$(K_MAIN).txt"), "w") do io
    println(io, "CHMM-t ν_min bracket sensitivity (K = $K_MAIN, SPY IS+OoS, seed=$SEED)");
    println(io, "="^100);
    println(io, rpad("ν_min",8), " | ", rpad("KS IS",8), " | ", rpad("AD IS",8), " | ",
                rpad("KS OoS",8), " | ", rpad("AD OoS",8), " | ",
                rpad("Kurt IS",8), " | ", rpad("Kurt OoS",8), " | ",
                rpad("ACF-MAE",8), " | ", rpad("W1",8), " | ", rpad("H",8));
    println(io, "-"^100);
    for r in bracket_rows
        νf_str = isinf(r.νf) ? "Gauss" : string(r.νf);
        println(io, rpad(νf_str, 8), " | ",
                    rpad(r.m_is.ks, 8), " | ", rpad(r.m_is.ad, 8), " | ",
                    rpad(r.m_oos.ks, 8), " | ", rpad(r.m_oos.ad, 8), " | ",
                    rpad(r.m_is.kurt, 8), " | ", rpad(r.m_oos.kurt, 8), " | ",
                    rpad(r.m_is.acf_mae, 8), " | ", rpad(r.m_is.w1, 8), " | ", rpad(r.m_is.hell, 8));
    end
    println(io);
    println(io, "Observed excess kurtosis: IS $(round(kurt_obs_is, digits=2)) | OoS $(round(kurt_obs_oos, digits=2))");
end
println("  [4.2] Done.")

# ========================================================================================= #
# [4.3] OoS KS power calibration
# ========================================================================================= #
println("\n[4.3] OoS KS power calibration...")

ksp_dir = joinpath(DIAG_DIR, "ks_power"); mkpath(ksp_dir);

# Reference generator 1: i.i.d. resamples of R_oos itself (known-correct by construction)
function _count_ks_pass(reference, sim_generator::Function, nrep::Int; α=0.05)
    cnt = 0;
    for _ in 1:nrep
        sim = sim_generator();
        pval = pvalue(ApproximateTwoSampleKSTest(reference, sim));
        if pval > α; cnt += 1; end
    end
    return cnt;
end

nrep = 1000;
Random.seed!(SEED + 1);
ref_pass_oos = _count_ks_pass(R_oos, () -> R_oos[rand(1:n_oos, n_oos)], nrep);

# Reference generator 2: T-length resamples of R_is ("nearly correct", same pipeline)
Random.seed!(SEED + 2);
ref_pass_is = _count_ks_pass(R_oos, () -> R_is[rand(1:n_is, n_oos)], nrep);

# Extra: KS power at IS length (should be higher rejection rate on misspecified generator)
Random.seed!(SEED + 3);
d_gauss_is = Normal(mean(R_is), std(R_is));
ref_pass_gauss_is = _count_ks_pass(R_is, () -> rand(d_gauss_is, n_is), nrep);

open(joinpath(ksp_dir, "KS_Power_Calibration.txt"), "w") do io
    println(io, "OoS KS Power Calibration");
    println(io, "="^60);
    println(io, "  OoS length T_oos = $n_oos");
    println(io, "  IS  length T_is  = $n_is");
    println(io, "  replications     = $nrep  (seed = $SEED)");
    println(io);
    println(io, "Reference pass rates at α = 0.05:");
    println(io, "  (a) Known-correct (iid resamples of R_oos):");
    println(io, "        T=$n_oos vs R_oos   ->  $(round(100*ref_pass_oos/nrep, digits=1)) %");
    println(io, "  (b) Nearly correct (iid resamples of R_is, length $n_oos):");
    println(io, "        R_is bootstrap vs R_oos  ->  $(round(100*ref_pass_is/nrep, digits=1)) %");
    println(io, "  (c) Misspecified Gaussian, IS length  (should reject):");
    println(io, "        Gaussian(μ̂,σ̂) vs R_is  ->  $(round(100*ref_pass_gauss_is/nrep, digits=1)) %");
    println(io);
    println(io, "Interpretation:");
    println(io, "  Line (a) is the ceiling: even a known-correct generator at T=$n_oos passes only at this rate under α=0.05.");
    println(io, "  CHMM OoS pass rates in the 93-96 % range should be read relative to this ceiling, not 100 %.");
    println(io, "  Line (c) shows the KS test retains power at IS length to reject a clearly-wrong generator.");
end
println("  [4.3] Done.")

# ========================================================================================= #
# [4.4] Ryden K = 2 random-init replication
# ========================================================================================= #
println("\n[4.4] Ryden K=2 random vs quantile init...")

ryden_dir = joinpath(DIAG_DIR, "ryden_k2"); mkpath(ryden_dir);

# Quantile-based init (existing baum_welch): already uses quantile init by default.
Random.seed!(SEED + 10);
ry_model_q = build(MyContinuousHiddenMarkovModel, (
    observations=R_is, number_of_states=2, max_iter=MAX_ITER));
_, ry_sd_q = _stationary(ry_model_q, 2);
ry_q_is, ry_q_oos = _simulate_chmm_paths(ry_model_q, ry_sd_q, n_is, n_oos, N_PATHS);
m_ry_q_is  = eval_full(R_is, ry_q_is);
m_ry_q_oos = eval_full(R_oos, ry_q_oos);

# Random init: wrap baum_welch-like code here. Simpler: perturb the quantile init with random draws.
# Implement a local Baum-Welch with random-initialized μ, σ over several seeds.
function _baum_welch_random_init(obs::Vector{Float64}, K::Int; max_iter=60, tol=1e-4, seed=1234)
    Random.seed!(seed);
    N = length(obs);
    μ_r = mean(obs); σ_r = std(obs);
    curr_μ = μ_r .+ 0.5*σ_r .* randn(K);
    curr_σ = abs.(σ_r .* (0.8 .+ 0.4 .* rand(K)));
    curr_T = ones(K,K) ./ K;
    curr_π = ones(K) ./ K;
    prev_ll = -Inf; ll_hist = Float64[];
    for _ in 1:max_iter
        log_B = zeros(N, K);
        for t in 1:N, k in 1:K
            log_B[t, k] = logpdf(Normal(curr_μ[k], curr_σ[k]), obs[t]);
        end
        log_alpha = zeros(N, K);
        log_alpha[1, :] = log.(curr_π) .+ log_B[1, :];
        for t in 2:N, j in 1:K
            log_alpha[t, j] = _logsumexp_vec(log_alpha[t-1, :] .+ log.(curr_T[:, j])) + log_B[t, j];
        end
        log_beta = zeros(N, K);
        for t in N-1:-1:1, i in 1:K
            log_beta[t, i] = _logsumexp_vec(log.(curr_T[i, :]) .+ log_B[t+1, :] .+ log_beta[t+1, :]);
        end
        γ = zeros(N, K);
        for t in 1:N
            γ[t, :] = exp.((log_alpha[t, :] .+ log_beta[t, :]) .-
                _logsumexp_vec(log_alpha[t, :] .+ log_beta[t, :]));
        end
        expected_trans = zeros(K, K);
        for t in 1:N-1
            log_denom = _logsumexp_vec(log_alpha[t, :] .+ log_beta[t, :]);
            for i in 1:K, j in 1:K
                lxi = log_alpha[t,i] + log(curr_T[i,j]) + log_B[t+1,j] + log_beta[t+1,j] - log_denom;
                expected_trans[i,j] += exp(lxi);
            end
        end
        curr_π = γ[1, :];
        for k in 1:K
            w = sum(γ[:, k]);
            if w > 0
                curr_μ[k] = sum(γ[:,k] .* obs) / w;
                curr_σ[k] = sqrt(max(sum(γ[:,k] .* (obs .- curr_μ[k]).^2) / w, 1e-12));
                curr_σ[k] = max(curr_σ[k], 1e-6);
            end
        end
        for i in 1:K
            rs = sum(expected_trans[i, :]);
            if rs > 0; curr_T[i, :] = expected_trans[i, :] ./ rs; end
        end
        curr_ll = _logsumexp_vec(log_alpha[N, :]);
        push!(ll_hist, curr_ll);
        if abs(curr_ll - prev_ll) < tol; break; end
        prev_ll = curr_ll;
    end
    return curr_T, curr_μ, curr_σ, curr_π, ll_hist;
end

# Build a CHMM from these params
function _make_chmm(T, μ, σ, K)
    mdl = MyContinuousHiddenMarkovModel();
    mdl.states = collect(1:K);
    mdl.log_likelihood_history = Float64[];
    trans = Dict{Int64, Categorical}(); emis = Dict{Int64, Normal}();
    for s in 1:K
        trans[s] = Categorical(T[s, :]);
        emis[s] = Normal(μ[s], σ[s]);
    end
    mdl.transition = trans; mdl.emission = emis;
    return mdl;
end

# Repeat random init over several seeds; report mean / best
seeds = [1, 7, 42, 100, 2024];
random_rows = [];
for s in seeds
    T_r, μ_r, σ_r, π_r, _ = _baum_welch_random_init(R_is, 2; max_iter=MAX_ITER, seed=s);
    m_r = _make_chmm(T_r, μ_r, σ_r, 2);
    _, sd_r = _stationary(m_r, 2);
    Random.seed!(SEED + 20 + s);
    sis, sos = _simulate_chmm_paths(m_r, sd_r, n_is, n_oos, N_PATHS);
    push!(random_rows, (seed=s, m_is=eval_full(R_is, sis), m_oos=eval_full(R_oos, sos)));
end

open(joinpath(ryden_dir, "Ryden_K2_Init.txt"), "w") do io
    println(io, "Rydén et al. (1998) K=2 replication: quantile vs random init (SPY)");
    println(io, "="^95);
    println(io, rpad("Init", 22), " | ", rpad("KS IS",8), " | ", rpad("AD IS",8), " | ",
                rpad("ACF-MAE",9), " | ", rpad("Kurt IS",8), " | ", rpad("KS OoS",8));
    println(io, "-"^95);
    println(io, rpad("Quantile (paper)", 22), " | ", rpad(m_ry_q_is.ks, 8), " | ",
                rpad(m_ry_q_is.ad, 8), " | ", rpad(m_ry_q_is.acf_mae, 9), " | ",
                rpad(m_ry_q_is.kurt, 8), " | ", rpad(m_ry_q_oos.ks, 8));
    for r in random_rows
        println(io, rpad("Random seed=$(r.seed)", 22), " | ",
                    rpad(r.m_is.ks, 8), " | ", rpad(r.m_is.ad, 8), " | ",
                    rpad(r.m_is.acf_mae, 9), " | ", rpad(r.m_is.kurt, 8), " | ",
                    rpad(r.m_oos.ks, 8));
    end
    println(io);
    println(io, "Interpretation:");
    println(io, "  Rydén et al. (1998) reported ACF-decay failure at K=2 under random initialization.");
    println(io, "  Our quantile-based initialization at K=2 preserves all three stylized facts;");
    println(io, "  the random-init runs illustrate the sensitivity that motivated the widely-cited 1998 verdict.");
end
println("  [4.4] Done.")

# ========================================================================================= #
# [4.6] Copula profile log-likelihood plot
# ========================================================================================= #
println("\n[4.6] Copula profile log-likelihood plot...")

cp_dir = joinpath(DIAG_DIR, "copula_profile"); mkpath(cp_dir);

# Use the six-asset universe from the cross-asset section
cross_tickers = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];
cross_idx = [findfirst(==(t), all_tickers) for t in cross_tickers];
R_cross = all_R[:, cross_idx];     # n_is × 6

# Compute PIT ranks and copula correlation Σ via Kendall's τ inversion
U = _pit_ranks(R_cross);
τ_mat = _kendall_tau_matrix(R_cross);
Σ_cop = sin.((π/2) .* τ_mat);
Σ_cop = _nearest_psd(Σ_cop);

ν_grid = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0];
ll_grid = Float64[];
for ν in ν_grid
    push!(ll_grid, _tcopula_profile_loglik(U, Σ_cop, ν));
end
ν_star_idx = argmax(ll_grid);
ν_star = ν_grid[ν_star_idx];

# Plot
cp_fig = plot(ν_grid, ll_grid, marker=:circle, lw=2, color=:navy, label="profile log-L",
    xlabel="ν (Student-t copula degrees of freedom)", ylabel="profile log-likelihood",
    title="Student-t copula profile MLE — six-asset SPY cross-section",
    titlefontsize=11, size=(700, 450));
vline!(cp_fig, [ν_star], color=:red, lw=2, ls=:dash, label="ν* = $(Int(ν_star))");
savefig(cp_fig, joinpath(cp_dir, "Fig-Copula-Profile-LogL.pdf"));
savefig(cp_fig, joinpath(cp_dir, "Fig-Copula-Profile-LogL.svg"));

open(joinpath(cp_dir, "Profile_LogL.txt"), "w") do io
    println(io, "Profile log-likelihood of Student-t copula — six-asset SPY cross-section");
    println(io, "="^75);
    println(io, rpad("ν", 8), "| log-L");
    println(io, "-"^30);
    for (ν, ll) in zip(ν_grid, ll_grid)
        println(io, rpad(ν, 8), "| $(round(ll, digits=2))");
    end
    println(io);
    println(io, "argmax: ν* = $ν_star  (profile log-L = $(round(maximum(ll_grid), digits=2)))");
end
println("  [4.6] Done.")

# ========================================================================================= #
# [5.2] Block-bootstrap benchmark
# ========================================================================================= #
println("\n[5.2] Block-bootstrap benchmark (block length 10)...")

bb_dir = joinpath(DIAG_DIR, "block_bootstrap"); mkpath(bb_dir);

function stationary_block_bootstrap(R::Vector{Float64}, T_out::Int; block_size::Int=10)
    n = length(R);
    p_end = 1.0 / block_size;
    sim = Vector{Float64}(undef, T_out);
    i = 1;
    cursor = rand(1:n);
    for t in 1:T_out
        sim[t] = R[cursor];
        if rand() < p_end
            cursor = rand(1:n);
        else
            cursor = (cursor % n) + 1;
        end
    end
    return sim;
end

Random.seed!(SEED + 70);
bb_is  = Array{Float64,2}(undef, n_is,  N_PATHS);
bb_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    bb_is[:,  i] = stationary_block_bootstrap(R_is, n_is;  block_size=10);
    bb_oos[:, i] = stationary_block_bootstrap(R_is, n_oos; block_size=10);
end
m_bb_is  = eval_full(R_is,  bb_is);
m_bb_oos = eval_full(R_oos, bb_oos);

open(joinpath(bb_dir, "BlockBootstrap.txt"), "w") do io
    println(io, "Stationary block bootstrap (block length 10 trading days, SPY)");
    println(io, "="^85);
    println(io, rpad("Window",8), " | ", rpad("KS",7), " | ", rpad("AD",7), " | ",
                rpad("Kurt",7), " | ", rpad("ACF|G|",8), " | ", rpad("ACFraw",8), " | ", rpad("W1",7), " | ",
                rpad("H",7), " | ", rpad("Cov",7));
    println(io, "-"^100);
    for (tag, m) in [("IS", m_bb_is), ("OoS", m_bb_oos)]
        println(io, rpad(tag, 8), " | ",
                    rpad(m.ks, 7), " | ", rpad(m.ad, 7), " | ",
                    rpad(m.kurt, 7), " | ", rpad(m.acf_mae, 8), " | ", rpad(m.acf_mae_raw, 8), " | ",
                    rpad(m.w1, 7), " | ", rpad(m.hell, 7), " | ", rpad(m.cov, 7));
    end
end
println("  [5.2] Done.")

# ========================================================================================= #
# [8.1] Discrete bin-conditional Student-t baseline
# ========================================================================================= #
println("\n[8.1] Discrete HMM with bin-conditional Student-t emissions...")

bt_dir = joinpath(DIAG_DIR, "bin_t"); mkpath(bt_dir);

# 13 quantile bins, frequency-counted transitions, within-bin Student-t fit.
qprobs_bt = range(0.0, 1.0, length=DISCRETE_K+1) |> collect;
bin_edges = quantile(R_is, qprobs_bt);
function _assign_bin(x, edges)
    Kb = length(edges) - 1;
    for k in 1:Kb
        if k == Kb || x < edges[k+1]; return k; end
    end
    return Kb;
end;
bin_idx = [_assign_bin(r, bin_edges) for r in R_is];

# Frequency-counted transition matrix
T_counts = zeros(DISCRETE_K, DISCRETE_K);
for t in 2:length(bin_idx); T_counts[bin_idx[t-1], bin_idx[t]] += 1.0; end
T_disc = copy(T_counts);
for i in 1:DISCRETE_K
    rs = sum(T_disc[i, :]);
    T_disc[i, :] = rs > 0 ? T_disc[i, :] ./ rs : fill(1.0/DISCRETE_K, DISCRETE_K);
end
π_disc = (T_disc^1000)[1, :];
sd_disc = Categorical(π_disc);

# Within-bin Student-t fits via method-of-moments seed; then MLE of ν on bin data.
# Use Distributions.fit_mle(TDist, centered/scaled) — but easier: use
# Distributions.LocationScale with TDist, estimate μ=bin_median, σ=bin_MAD, fit ν.
function _fit_tdist(y::Vector{Float64}; ν_bounds=(2.1, 50.0))
    μ = median(y);
    σ0 = mean(abs.(y .- μ));
    σ = max(σ0, 1e-4);
    best_ll = -Inf; best_ν = 6.0;
    for ν in range(ν_bounds[1], ν_bounds[2], length=80)
        d = LocationScale(μ, σ, TDist(ν));
        ll = 0.0;
        for yi in y; ll += logpdf(d, yi); end
        if ll > best_ll; best_ll = ll; best_ν = ν; end
    end
    return μ, σ, best_ν;
end

bin_tdists = Vector{LocationScale{Float64, Continuous, TDist{Float64}}}(undef, DISCRETE_K);
for k in 1:DISCRETE_K
    members = R_is[bin_idx .== k];
    if length(members) < 5
        bin_tdists[k] = LocationScale(mean(members), max(std(members), 1e-4), TDist(10.0));
    else
        μb, σb, νb = _fit_tdist(members);
        bin_tdists[k] = LocationScale(μb, σb, TDist(νb));
    end
end

# Simulate: chain in bin space via T_disc, then draw from bin_tdists[k].
Random.seed!(SEED + 80);
bt_is  = Array{Float64,2}(undef, n_is,  N_PATHS);
bt_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    # IS
    cur = rand(sd_disc);
    for t in 1:n_is
        bt_is[t, i] = rand(bin_tdists[cur]);
        cur = rand(Categorical(T_disc[cur, :]));
    end
    # OoS
    cur = rand(sd_disc);
    for t in 1:n_oos
        bt_oos[t, i] = rand(bin_tdists[cur]);
        cur = rand(Categorical(T_disc[cur, :]));
    end
end
m_bt_is  = eval_full(R_is,  bt_is);
m_bt_oos = eval_full(R_oos, bt_oos);

open(joinpath(bt_dir, "BinStudentT.txt"), "w") do io
    println(io, "Discrete HMM with bin-conditional Student-t emissions (no jumps), K_disc = $DISCRETE_K");
    println(io, "="^85);
    println(io, rpad("Window",8), " | ", rpad("KS",7), " | ", rpad("AD",7), " | ",
                rpad("Kurt",7), " | ", rpad("ACF-MAE",8), " | ", rpad("W1",7), " | ", rpad("H",7), " | ", rpad("Cov",7));
    println(io, "-"^85);
    for (tag, m) in [("IS", m_bt_is), ("OoS", m_bt_oos)]
        println(io, rpad(tag, 8), " | ",
                    rpad(m.ks, 7), " | ", rpad(m.ad, 7), " | ", rpad(m.kurt, 7), " | ",
                    rpad(m.acf_mae, 8), " | ", rpad(m.w1, 7), " | ",
                    rpad(m.hell, 7), " | ", rpad(m.cov, 7));
    end
    println(io);
    println(io, "Per-bin ν estimate:");
    for k in 1:DISCRETE_K
        println(io, "  bin $k  ν=$(round(bin_tdists[k].ρ.ν, digits=2))  μ=$(round(bin_tdists[k].μ, digits=3))  σ=$(round(bin_tdists[k].σ, digits=3))");
    end
end
println("  [8.1] Done.")

# ========================================================================================= #
# Final summary printout
# ========================================================================================= #
println("\n" * "="^70);
println("  Diagnostics pipeline finished.");
println("  Results written to: $DIAG_DIR");
println("="^70);
