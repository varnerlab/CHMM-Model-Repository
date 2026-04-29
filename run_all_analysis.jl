# ========================================================================================= #
# run_all_analysis.jl
#
# Generates ALL figures and analysis for the continuous HMM paper.
# Runs the full pipeline for K ∈ {3, 6, 9, 12, 15, 18, 21} hidden states on SPY.
#
# This study focuses on continuous HMM (no jump mechanisms). At small K,
# the CHMM alone reproduces all stylized facts to some extent.
#
# Output: results/<ticker>/K<N>/  with figures (.svg, .pdf) and metrics (.txt)
# ========================================================================================= #

println("="^70)
println("  Continuous HMM — Full Analysis Pipeline")
println("  States: K ∈ {3, 6, 9, 12, 15, 18, 21}")
println("="^70)

# --- SETUP ---
using Pkg;
Pkg.activate(".");

include("Include.jl");

# --- CONFIGURATION ---
const TICKER = "SPY";
const RETURN_LABEL = "Excess Growth Rate";
const K_VALUES = [3, 6, 9, 12, 15, 18, 21];
const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const MAX_ITER = 60;
const N_PATHS = 1000;
const L = 252;            # ACF max lag

# Output directory
const RESULTS_DIR = joinpath(_ROOT, "results");

println("  Ticker: $(TICKER)");

# ========================================================================================= #
# LOAD DATA
# ========================================================================================= #
println("\n[1/4] Loading data...")

train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
maximum_number_trading_days = nrow(train_dataset["AAPL"]);

dataset = Dict{String,DataFrame}();
for (t, data) ∈ train_dataset
    if nrow(data) == maximum_number_trading_days
        dataset[t] = data;
    end
end
list_of_all_tickers = keys(dataset) |> collect |> sort;
all_firms_R = log_growth_matrix(dataset, list_of_all_tickers; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
ticker_idx = findfirst(x -> x == TICKER, list_of_all_tickers);
R_is = all_firms_R[:, ticker_idx];
n_steps = length(R_is);

oos_dataset_raw = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset_raw, TICKER; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
n_steps_oos = length(R_oos);

println("  IS: $(n_steps) obs | OoS: $(n_steps_oos) obs | Tickers: $(length(list_of_all_tickers))")

# ========================================================================================= #
# FIGURE 1: STYLIZED FACTS (only once — independent of K)
# ========================================================================================= #
println("\n[2/4] Generating Figure 1: Stylized Facts...")

fig1_dir = joinpath(RESULTS_DIR, TICKER, "stylized_facts");
mkpath(fig1_dir);

# Descriptive stats
function descriptive_stats(R, label)
    n = length(R); μ = mean(R); σ = std(R);
    skew = sum(((R .- μ) ./ σ).^3) / n;
    kurt = sum(((R .- μ) ./ σ).^4) / n - 3.0;
    jb = (n/6) * (skew^2 + kurt^2/4);
    acf_raw = autocor(R, 1:20);
    lb_raw = n * (n+2) * sum(acf_raw.^2 ./ (n .- (1:20)));
    acf_abs = autocor(abs.(R), 1:20);
    lb_abs = n * (n+2) * sum(acf_abs.^2 ./ (n .- (1:20)));
    return (label=label, n=n, mean=μ, std=σ, skewness=skew, kurtosis=kurt, jb=jb, lb_raw=lb_raw, lb_abs=lb_abs)
end

stats_is = descriptive_stats(R_is, "IS (2014-2024)");
stats_oos = descriptive_stats(R_oos, "OoS (2025)");

# Save Table 1
open(joinpath(fig1_dir, "Table-1-Descriptive-Stats.txt"), "w") do io
    println(io, "Table 1: Descriptive Statistics — $TICKER")
    println(io, "="^65)
    for s in [stats_is, stats_oos]
        println(io, "\n--- $(s.label) (T=$(s.n)) ---")
        println(io, "Mean (annualized):    $(round(s.mean, digits=2))")
        println(io, "Std Dev (annualized): $(round(s.std, digits=2))")
        println(io, "Skewness:             $(round(s.skewness, digits=3))")
        println(io, "Excess Kurtosis:      $(round(s.kurtosis, digits=3))")
        println(io, "JB statistic:         $(round(s.jb, digits=1)) (critical ≈ 5.99)")
        println(io, "LB on Gₜ (lag 20):    $(round(s.lb_raw, digits=1)) (critical ≈ 31.4)")
        println(io, "LB on |Gₜ| (lag 20):  $(round(s.lb_abs, digits=1)) (critical ≈ 31.4)")
    end
end

# Gaussian and Laplace fits
μ_gauss = mean(R_is); σ_gauss = std(R_is);
d_gauss = Normal(μ_gauss, σ_gauss);
μ_lap = median(R_is); b_lap = mean(abs.(R_is .- μ_lap));
d_laplace = Laplace(μ_lap, b_lap);

# Percentile-based axis limits
x_lo_raw = quantile(R_is, 0.005); x_hi_raw = quantile(R_is, 0.995);
x_pad = 0.20 * (x_hi_raw - x_lo_raw);
x_lo = x_lo_raw - x_pad; x_hi = x_hi_raw + x_pad;
x_grid = range(x_lo, x_hi, length=1000);

# Stylized-facts panels saved as separate PDFs (no top titles; (a)/(b)/(c)/(d) come from LaTeX subcaptions).
_panel_size = (700, 500);

# Panel (a): Distribution
p1 = histogram(R_is, normalize=:pdf, bins=200, alpha=0.4, color=:lightgray, label="Observed",
    xlabel=RETURN_LABEL, ylabel="Probability Density (AU)", size=_panel_size);
plot!(p1, x_grid, pdf.(d_gauss, x_grid), lw=2, color=:blue, label="Gaussian", ls=:dash);
plot!(p1, x_grid, pdf.(d_laplace, x_grid), lw=2, color=:red, label="Laplace");
xlims!(p1, x_lo, x_hi);
savefig(p1, joinpath(fig1_dir, "Fig-1-Stylized-Facts-a.svg"));
savefig(p1, joinpath(fig1_dir, "Fig-1-Stylized-Facts-a.pdf"));

# Panel (b): Q-Q
sorted_R = sort(R_is); n_r = length(sorted_R);
theo_q = [quantile(d_gauss, (i-0.5)/n_r) for i in 1:n_r];
qq_lo = min(minimum(theo_q), minimum(sorted_R)); qq_hi = max(maximum(theo_q), maximum(sorted_R));
p2 = scatter(theo_q, sorted_R, ms=1, alpha=0.5, color=:steelblue, label="",
    xlabel="Theoretical", ylabel="Sample", size=_panel_size);
plot!(p2, [qq_lo, qq_hi], [qq_lo, qq_hi], lw=2, color=:red, ls=:dash, label="45°");
savefig(p2, joinpath(fig1_dir, "Fig-1-Stylized-Facts-b.svg"));
savefig(p2, joinpath(fig1_dir, "Fig-1-Stylized-Facts-b.pdf"));

# Panel (c): Returns ACF
τ = 1:(L-1); ci = 2.576/sqrt(n_r);
acf_raw = autocor(R_is, τ);
p3 = plot(τ, acf_raw, linetype=:steppost, lw=2, color=:steelblue, label="ACF(Gₜ)",
    xlabel="Lag", ylabel="ACF", size=_panel_size);
plot!(p3, τ, ci.*ones(length(τ)), lw=1.5, color=:gray, ls=:dash, label="99% CI");
plot!(p3, τ, -ci.*ones(length(τ)), lw=1.5, color=:gray, ls=:dash, label="");
savefig(p3, joinpath(fig1_dir, "Fig-1-Stylized-Facts-c.svg"));
savefig(p3, joinpath(fig1_dir, "Fig-1-Stylized-Facts-c.pdf"));

# Panel (d): |Returns| ACF
acf_abs = autocor(abs.(R_is), τ);
p4 = plot(τ, acf_abs, linetype=:steppost, lw=2, color=:darkorange, label="ACF(|Gₜ|)",
    xlabel="Lag", ylabel="ACF", size=_panel_size);
plot!(p4, τ, ci.*ones(length(τ)), lw=1.5, color=:gray, ls=:dash, label="99% CI");
plot!(p4, τ, -ci.*ones(length(τ)), lw=1.5, color=:gray, ls=:dash, label="");
savefig(p4, joinpath(fig1_dir, "Fig-1-Stylized-Facts-d.svg"));
savefig(p4, joinpath(fig1_dir, "Fig-1-Stylized-Facts-d.pdf"));
println("  Saved Figure 1 panels (a-d) + Table 1")

# ========================================================================================= #
# PER-K ANALYSIS LOOP
# ========================================================================================= #

# Observed targets
target_acf = autocor(abs.(R_is), 1:L);
obs_μ = mean(R_is); obs_σ = std(R_is);
target_kurtosis = sum(((R_is .- obs_μ) ./ obs_σ).^4) / n_steps - 3.0;

# Summary table storage
summary_rows = [];

for K in K_VALUES
    println("\n" * "="^70)
    println("[3/4] Processing K = $K states...")
    println("="^70)

    out_dir = joinpath(RESULTS_DIR, TICKER, "K$(K)");
    mkpath(out_dir);

    # ------------------------------------------------------------------- #
    # STEP 1: TRAIN BASE MODEL
    # ------------------------------------------------------------------- #
    println("  Training Baum-Welch (K=$K, max_iter=$MAX_ITER)...")

    model = build(MyContinuousHiddenMarkovModel, (
        observations = R_is,
        number_of_states = K,
        max_iter = MAX_ITER
    ));

    println("  Converged in $(length(model.log_likelihood_history)) iterations")

    # Transition matrix + stationary distribution
    T_mat = zeros(K, K);
    for i in 1:K
        T_mat[i, :] = probs(model.transition[i]);
    end
    π_stat = (T_mat^1000)[1, :];
    start_dist = Categorical(π_stat);

    # ------------------------------------------------------------------- #
    # FIGURE: Convergence
    # ------------------------------------------------------------------- #
    p_conv = plot(model.log_likelihood_history,
        xlabel="Iteration", ylabel="Log-Likelihood",
        legend=false, lw=2, color=:navy, marker=:circle, ms=3);
    savefig(p_conv, joinpath(out_dir, "Fig-Convergence.svg"));
    savefig(p_conv, joinpath(out_dir, "Fig-Convergence.pdf"));

    # ------------------------------------------------------------------- #
    # FIGURE: Emission Distributions
    # ------------------------------------------------------------------- #
    colors_k = cgrad(:RdYlBu, K, categorical=true);
    p_emit = plot(xlabel=RETURN_LABEL, ylabel="Probability Density (AU)", legend=:topright);
    histogram!(p_emit, R_is, normalize=:pdf, bins=200, alpha=0.3, color=:lightgray, label="Observed");
    for s in 1:K
        d = model.emission[s];
        plot!(p_emit, x_grid, pdf.(d, x_grid), lw=1.5, color=colors_k[s], label="S$s", alpha=0.8);
    end
    xlims!(p_emit, x_lo, x_hi);
    savefig(p_emit, joinpath(out_dir, "Fig-Emission-PDFs.svg"));
    savefig(p_emit, joinpath(out_dir, "Fig-Emission-PDFs.pdf"));

    # ------------------------------------------------------------------- #
    # FIGURE: Residence Times
    # ------------------------------------------------------------------- #
    res_times = [1.0 / (1.0 - T_mat[k,k]) for k in 1:K];
    p_res = bar(1:K, res_times,
        xlabel="State", ylabel="Steps", legend=false, color=:steelblue, alpha=0.7);
    savefig(p_res, joinpath(out_dir, "Fig-Residence-Times.svg"));
    savefig(p_res, joinpath(out_dir, "Fig-Residence-Times.pdf"));

    # NOTE: Per-K Fig-Transition-Matrix and Fig-Stationary-Distribution outputs
    # were removed (2026-04-28) because they are not panelled in the paper.
    # The K=18 cross-family transition-matrix panel uses the per-family files
    # produced by run_figures.jl (Fig-Transition-Matrix-K18-{N,t,L,GED}.pdf).

    # Save emission parameters
    open(joinpath(out_dir, "Emission-Parameters.txt"), "w") do io
        println(io, "Emission Parameters — $TICKER, K=$K")
        println(io, "="^50)
        println(io, "State | Mean (μ)     | Std Dev (σ)")
        println(io, "-"^50)
        for s in 1:K
            d = model.emission[s];
            println(io, "  $(lpad(s,2))  | $(lpad(round(mean(d),digits=2),12)) | $(lpad(round(std(d),digits=2),12))")
        end
    end

    # ------------------------------------------------------------------- #
    # STEP 2: SIMULATE N_PATHS PATHS (CHMM only)
    # ------------------------------------------------------------------- #
    println("  Simulating $N_PATHS paths...")

    decoded_is = Array{Float64,2}(undef, n_steps, N_PATHS);
    decoded_oos = Array{Float64,2}(undef, n_steps_oos, N_PATHS);

    for i in 1:N_PATHS
        # IS
        s0 = rand(start_dist);
        states = model(s0, n_steps);
        for j in 1:n_steps; decoded_is[j,i] = rand(model.emission[states[j]]); end

        # OoS
        s0 = rand(start_dist);
        states = model(s0, n_steps_oos);
        for j in 1:n_steps_oos; decoded_oos[j,i] = rand(model.emission[states[j]]); end
    end

    # ------------------------------------------------------------------- #
    # STEP 3: COMPUTE METRICS
    # ------------------------------------------------------------------- #
    println("  Computing validation metrics...")

    function eval_metrics(observed, sim_archive, L_val)
        np = size(sim_archive, 2);
        n_o = length(observed);
        μ_o = mean(observed); σ_o = std(observed);
        kurt_obs_val = sum(((observed .- μ_o) ./ σ_o).^4) / n_o - 3.0;
        L_use = min(L_val, n_o - 1);
        acf_obs_val = autocor(abs.(observed), 1:L_use);
        acf_obs_raw_val = autocor(observed, 1:L_use);

        ks_pass = 0; ad_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0; acf_mae_raw_s = 0.0;
        w1_s = 0.0; hell_s = 0.0;
        ks_pvals = Float64[];

        # Coverage: for each of 99 empirical quantiles, check if within [5th,95th] sim envelope
        obs_quantile_probs = range(0.01, 0.99, length=99);
        obs_quantiles = quantile(observed, obs_quantile_probs);
        sim_quantile_matrix = zeros(99, np);

        for i in 1:np
            sim = sim_archive[:, i];

            # KS test
            pval = pvalue(ApproximateTwoSampleKSTest(observed, sim));
            push!(ks_pvals, pval);
            if pval > 0.05; ks_pass += 1; end

            # AD test
            ad_pval = pvalue(KSampleADTest(observed, sim));
            if ad_pval > 0.05; ad_pass += 1; end

            # Kurtosis
            μ_s = mean(sim); σ_s = std(sim);
            kurt_s += sum(((sim .- μ_s) ./ σ_s).^4) / length(sim) - 3.0;

            # ACF-MAE on |Gₜ| (volatility clustering)
            acf_sim_val = autocor(abs.(sim), 1:L_use);
            acf_mae_s += mean(abs.(acf_obs_val .- acf_sim_val));

            # ACF-MAE on raw Gₜ (linear autocorrelation)
            acf_sim_raw_val = autocor(sim, 1:L_use);
            acf_mae_raw_s += mean(abs.(acf_obs_raw_val .- acf_sim_raw_val));

            # Wasserstein-1
            obs_s = sort(observed); sim_s = sort(sim);
            n_min = min(length(obs_s), length(sim_s));
            obs_q = [obs_s[max(1, round(Int, k*length(obs_s)/n_min))] for k in 1:n_min];
            sim_q = [sim_s[max(1, round(Int, k*length(sim_s)/n_min))] for k in 1:n_min];
            w1_s += mean(abs.(obs_q .- sim_q));

            # Hellinger
            lo = min(minimum(observed), minimum(sim)) - 10;
            hi = max(maximum(observed), maximum(sim)) + 10;
            edges = range(lo, hi, length=101);
            h_o = fit(Histogram, observed, edges).weights ./ n_o;
            h_s = fit(Histogram, sim, edges).weights ./ length(sim);
            hell_s += sqrt(sum((sqrt.(h_o) .- sqrt.(h_s)).^2)) / sqrt(2);

            # Quantiles for coverage
            sim_quantile_matrix[:, i] = quantile(sim, obs_quantile_probs);
        end

        # Coverage: fraction of 99 observed quantiles within [5th, 95th] percentile of sim quantiles
        coverage_count = 0;
        for q in 1:99
            lo_env = quantile(sim_quantile_matrix[q, :], 0.05);
            hi_env = quantile(sim_quantile_matrix[q, :], 0.95);
            if obs_quantiles[q] >= lo_env && obs_quantiles[q] <= hi_env
                coverage_count += 1;
            end
        end
        coverage = round(100.0 * coverage_count / 99, digits=1);

        return (ks_rate=round(100*ks_pass/np, digits=1),
                ad_rate=round(100*ad_pass/np, digits=1),
                kurtosis_obs=round(kurt_obs_val, digits=2),
                kurtosis_sim=round(kurt_s/np, digits=2),
                acf_mae=round(acf_mae_s/np, digits=4),
                acf_mae_raw=round(acf_mae_raw_s/np, digits=4),
                wasserstein=round(w1_s/np, digits=3),
                hellinger=round(hell_s/np, digits=4),
                coverage=coverage,
                ks_pvals=ks_pvals)
    end

    m_is = eval_metrics(R_is, decoded_is, L);
    m_oos = eval_metrics(R_oos, decoded_oos, L);

    # Store for summary
    push!(summary_rows, (K=K,
        ks_is=m_is.ks_rate, ad_is=m_is.ad_rate,
        ks_oos=m_oos.ks_rate, ad_oos=m_oos.ad_rate,
        kurt_obs=m_is.kurtosis_obs, kurt_sim_is=m_is.kurtosis_sim,
        acf_mae_is=m_is.acf_mae,
        acf_mae_raw_is=m_is.acf_mae_raw,
        w1_is=m_is.wasserstein, w1_oos=m_oos.wasserstein,
        hell_is=m_is.hellinger, hell_oos=m_oos.hellinger,
        cov_is=m_is.coverage, cov_oos=m_oos.coverage))

    # Save per-K metrics
    open(joinpath(out_dir, "Metrics.txt"), "w") do io
        println(io, "Validation Metrics — $TICKER, K=$K")
        println(io, "="^65)
        println(io, "")
        println(io, "                  | CHMM (IS)    | CHMM (OoS)")
        println(io, "-"^55)
        println(io, "KS pass rate (%) | $(lpad(m_is.ks_rate,12)) | $(lpad(m_oos.ks_rate,12))")
        println(io, "AD pass rate (%) | $(lpad(m_is.ad_rate,12)) | $(lpad(m_oos.ad_rate,12))")
        println(io, "Excess kurtosis  | $(lpad(m_is.kurtosis_sim,12)) | $(lpad(m_oos.kurtosis_sim,12))")
        println(io, "  (observed)     | $(lpad(m_is.kurtosis_obs,12)) | $(lpad(m_oos.kurtosis_obs,12))")
        println(io, "ACF-MAE |Gₜ|     | $(lpad(m_is.acf_mae,12)) |")
        println(io, "ACF-MAE Gₜ (raw) | $(lpad(m_is.acf_mae_raw,12)) |")
        println(io, "Wasserstein-1    | $(lpad(m_is.wasserstein,12)) | $(lpad(m_oos.wasserstein,12))")
        println(io, "Hellinger        | $(lpad(m_is.hellinger,12)) | $(lpad(m_oos.hellinger,12))")
        println(io, "Coverage (%)     | $(lpad(m_is.coverage,12)) | $(lpad(m_oos.coverage,12))")
    end

    # ------------------------------------------------------------------- #
    # FIGURE 3: In-Sample Comparison — 4 split panels (no top titles)
    # ------------------------------------------------------------------- #
    println("  Generating Figure 3: IS comparison (4 panels)...")
    _ps = (700, 500);

    # |Gₜ| ACF: observed + simulated mean / 10-90 band
    acf_obs_is = autocor(abs.(R_is), 1:L);
    n_acf_sample = min(200, N_PATHS);
    acf_arch = hcat([autocor(abs.(decoded_is[:,i]), 1:L) for i in 1:n_acf_sample]...);
    acf_m = mean(acf_arch, dims=2)[:];
    acf_10 = [quantile(acf_arch[t,:], 0.10) for t in 1:L];
    acf_90 = [quantile(acf_arch[t,:], 0.90) for t in 1:L];

    # Raw Gₜ ACF
    acf_obs_raw_is = autocor(R_is, 1:L);
    acf_arch_raw = hcat([autocor(decoded_is[:,i], 1:L) for i in 1:n_acf_sample]...);
    acf_m_raw = mean(acf_arch_raw, dims=2)[:];
    acf_10_raw = [quantile(acf_arch_raw[t,:], 0.10) for t in 1:L];
    acf_90_raw = [quantile(acf_arch_raw[t,:], 0.90) for t in 1:L];
    ci99_is = 2.576 / sqrt(length(R_is));

    # (a) Density
    p3a = plot(xlabel=RETURN_LABEL, ylabel="Probability Density (AU)", size=_ps);
    histogram!(p3a, R_is, normalize=:pdf, bins=200, alpha=0.3, color=:lightgray, label="Observed");
    density!(p3a, decoded_is[:,1], lw=2, color=:blue, alpha=0.7, label="CHMM");
    xlims!(p3a, x_lo, x_hi);
    savefig(p3a, joinpath(out_dir, "Fig-3-IS-Comparison-a.svg"));
    savefig(p3a, joinpath(out_dir, "Fig-3-IS-Comparison-a.pdf"));

    # (b) Tail Q-Q
    probs_qq = range(0.001, 0.999, length=200);
    q_obs = quantile(R_is, probs_qq);
    q_sim = quantile(vec(decoded_is), probs_qq);
    p3b = plot(q_obs, q_obs, lw=2, color=:black, ls=:dash, label="Perfect",
        xlabel="Observed Quantiles", ylabel="Simulated Quantiles", size=_ps);
    scatter!(p3b, q_obs, q_sim, ms=3, alpha=0.6, color=:blue, label="CHMM");
    savefig(p3b, joinpath(out_dir, "Fig-3-IS-Comparison-b.svg"));
    savefig(p3b, joinpath(out_dir, "Fig-3-IS-Comparison-b.pdf"));

    # (c) ACF of raw Gₜ
    p3c = plot(1:L, acf_obs_raw_is, lw=2, color=:red, ls=:dash, label="Observed",
        xlabel="Lag", ylabel="ACF of Gₜ", size=_ps);
    plot!(p3c, 1:L, acf_m_raw, lw=2, color=:navy, label="CHMM (mean)");
    plot!(p3c, 1:L, acf_10_raw, fillrange=acf_90_raw, alpha=0.15, color=:navy, label="10-90th pctl");
    hline!(p3c, [ci99_is, -ci99_is], lw=1, color=:gray, ls=:dash, label="99% CI");
    savefig(p3c, joinpath(out_dir, "Fig-3-IS-Comparison-c.svg"));
    savefig(p3c, joinpath(out_dir, "Fig-3-IS-Comparison-c.pdf"));

    # (d) ACF of |Gₜ|
    p3d = plot(1:L, acf_obs_is, lw=2, color=:red, ls=:dash, label="Observed",
        xlabel="Lag", ylabel="ACF of |Gₜ|", size=_ps);
    plot!(p3d, 1:L, acf_m, lw=2, color=:navy, label="CHMM (mean)");
    plot!(p3d, 1:L, acf_10, fillrange=acf_90, alpha=0.15, color=:navy, label="10-90th pctl");
    savefig(p3d, joinpath(out_dir, "Fig-3-IS-Comparison-d.svg"));
    savefig(p3d, joinpath(out_dir, "Fig-3-IS-Comparison-d.pdf"));

    # ------------------------------------------------------------------- #
    # FIGURE 4: OoS Validation — 4 split panels (no top titles)
    # ------------------------------------------------------------------- #
    println("  Generating Figure 4: OoS validation (4 panels)...")
    _ps = (700, 500);

    # (a) Density fan chart
    p4a = plot(xlabel=RETURN_LABEL, ylabel="Probability Density (AU)", size=_ps);
    _n_fan = min(50, N_PATHS);
    for i in 1:_n_fan
        _lbl = (i == 1) ? "CHMM simulated ($(_n_fan) paths)" : "";
        density!(p4a, decoded_oos[:,i], lw=1, color=:deepskyblue1, alpha=0.18, label=_lbl);
    end
    density!(p4a, R_oos, lw=3, color=:red, label="Observed OoS");
    oos_lo = quantile(R_oos, 0.005); oos_hi = quantile(R_oos, 0.995);
    oos_pad = 0.20 * (oos_hi - oos_lo);
    xlims!(p4a, oos_lo - oos_pad, oos_hi + oos_pad);
    savefig(p4a, joinpath(out_dir, "Fig-4-OoS-Validation-a.svg"));
    savefig(p4a, joinpath(out_dir, "Fig-4-OoS-Validation-a.pdf"));

    # (b) KS p-value histogram
    p4b = histogram(m_oos.ks_pvals, bins=50, normalize=true, alpha=0.6, color=:navy,
        label="CHMM", xlabel="p-value", ylabel="Density", size=_ps);
    vline!(p4b, [0.05], lw=2, color=:red, ls=:dash, label="α=0.05");
    savefig(p4b, joinpath(out_dir, "Fig-4-OoS-Validation-b.svg"));
    savefig(p4b, joinpath(out_dir, "Fig-4-OoS-Validation-b.pdf"));

    # (c) ACF of raw Gₜ (OoS)
    τ_oos = 1:min(L, n_steps_oos-1);
    acf_oos_obs_raw = autocor(R_oos, τ_oos);
    n_acf_oos = min(200, N_PATHS);
    acf_oos_arch_raw = hcat([autocor(decoded_oos[:,i], τ_oos) for i in 1:n_acf_oos]...);
    acf_oos_m_raw = mean(acf_oos_arch_raw, dims=2)[:];
    acf_oos_10_raw = [quantile(acf_oos_arch_raw[t,:], 0.10) for t in 1:length(τ_oos)];
    acf_oos_90_raw = [quantile(acf_oos_arch_raw[t,:], 0.90) for t in 1:length(τ_oos)];
    ci99_oos = 2.576 / sqrt(length(R_oos));

    p4c = plot(τ_oos, acf_oos_obs_raw, lw=2, color=:red, ls=:dash, label="Observed OoS",
        xlabel="Lag", ylabel="ACF of Gₜ", size=_ps);
    plot!(p4c, τ_oos, acf_oos_m_raw, lw=2, color=:navy, label="CHMM (mean)");
    plot!(p4c, τ_oos, acf_oos_10_raw, fillrange=acf_oos_90_raw, alpha=0.2, color=:navy, label="10-90th");
    hline!(p4c, [ci99_oos, -ci99_oos], lw=1, color=:gray, ls=:dash, label="99% CI");
    savefig(p4c, joinpath(out_dir, "Fig-4-OoS-Validation-c.svg"));
    savefig(p4c, joinpath(out_dir, "Fig-4-OoS-Validation-c.pdf"));

    # (d) ACF of |Gₜ| (OoS)
    acf_oos_obs = autocor(abs.(R_oos), τ_oos);
    acf_oos_arch = hcat([autocor(abs.(decoded_oos[:,i]), τ_oos) for i in 1:n_acf_oos]...);
    acf_oos_m = mean(acf_oos_arch, dims=2)[:];
    acf_oos_10 = [quantile(acf_oos_arch[t,:], 0.10) for t in 1:length(τ_oos)];
    acf_oos_90 = [quantile(acf_oos_arch[t,:], 0.90) for t in 1:length(τ_oos)];

    p4d = plot(τ_oos, acf_oos_obs, lw=2, color=:red, ls=:dash, label="Observed OoS",
        xlabel="Lag", ylabel="ACF of |Gₜ|", size=_ps);
    plot!(p4d, τ_oos, acf_oos_m, lw=2, color=:navy, label="CHMM (mean)");
    plot!(p4d, τ_oos, acf_oos_10, fillrange=acf_oos_90, alpha=0.2, color=:navy, label="10-90th");
    savefig(p4d, joinpath(out_dir, "Fig-4-OoS-Validation-d.svg"));
    savefig(p4d, joinpath(out_dir, "Fig-4-OoS-Validation-d.pdf"));

    # NOTE: Per-K Fig-Trajectory-Example and Fig-ACF-Comparison outputs were
    # removed (2026-04-28) because they are not panelled in the paper.

    println("  K=$K complete. Files saved to: $out_dir")
end

# ========================================================================================= #
# SUMMARY TABLE (State Resolution Sensitivity)
# ========================================================================================= #
println("\n" * "="^70)
println("[4/4] Writing Summary Table (State Resolution Sensitivity)...")
println("="^70)

summary_dir = joinpath(RESULTS_DIR, TICKER);
open(joinpath(summary_dir, "Table-T1-State-Resolution-Sensitivity.txt"), "w") do io
    println(io, "Table T1: State Resolution Sensitivity — $TICKER")
    println(io, "$(N_PATHS) simulated paths, α=0.05")
    println(io, "="^150)
    println(io, "  K  | KS IS(%) | AD IS(%) | KS OoS(%) | AD OoS(%) | Kurt(obs) | Kurt(sim) | ACF-MAE |G| | ACF-MAE raw | W1(IS) | H(IS)  | Cov IS(%) | Cov OoS(%)")
    println(io, "-"^150)
    for r in summary_rows
        println(io, "  $(lpad(r.K,2)) | $(lpad(r.ks_is,7)) | $(lpad(r.ad_is,7)) | $(lpad(r.ks_oos,8)) | $(lpad(r.ad_oos,8)) | $(lpad(r.kurt_obs,8)) | $(lpad(r.kurt_sim_is,8)) | $(lpad(r.acf_mae_is,10)) | $(lpad(r.acf_mae_raw_is,11)) | $(lpad(r.w1_is,5)) | $(lpad(r.hell_is,5)) | $(lpad(r.cov_is,8)) | $(lpad(r.cov_oos,8))")
    end
    println(io, "="^150)
end

# Also print to console
println("\nTable T1: State Resolution Sensitivity")
println("="^130)
println("  K  | KS IS(%) | AD IS(%) | KS OoS(%) | Kurt(obs) | Kurt(sim) | ACF-MAE |G| | ACF-MAE raw | W1(IS) | H(IS)  | Cov IS(%)")
println("-"^130)
for r in summary_rows
    println("  $(lpad(r.K,2)) | $(lpad(r.ks_is,7)) | $(lpad(r.ad_is,7)) | $(lpad(r.ks_oos,8)) | $(lpad(r.kurt_obs,8)) | $(lpad(r.kurt_sim_is,8)) | $(lpad(r.acf_mae_is,10)) | $(lpad(r.acf_mae_raw_is,11)) | $(lpad(r.w1_is,5)) | $(lpad(r.hell_is,5)) | $(lpad(r.cov_is,8))")
end
println("="^130)

# ========================================================================================= #
# DONE
# ========================================================================================= #
println("\n" * "="^70)
println("  ALL ANALYSIS COMPLETE")
println("  Output directory: $RESULTS_DIR")
println("="^70)
println("\nGenerated per K:")
println("  - Fig-Convergence (.svg/.pdf)")
println("  - Fig-Emission-PDFs (.svg/.pdf)")
println("  - Fig-Residence-Times (.svg/.pdf)")
println("  - Fig-3-IS-Comparison-{a,b,c,d} (.svg/.pdf)")
println("  - Fig-4-OoS-Validation-{a,b,c,d} (.svg/.pdf)")
println("  - Emission-Parameters.txt")
println("  - Metrics.txt")
println("\nGenerated once:")
println("  - Fig-1-Stylized-Facts-{a,b,c,d} (.svg/.pdf)")
println("  - Table-1-Descriptive-Stats.txt")
println("  - Table-T1-State-Resolution-Sensitivity.txt")

# ========================================================================================= #
# Copy paper-relevant figures into the paper's figs directory.
# ========================================================================================= #
const PAPER_FIGS = abspath(joinpath(_ROOT, "..", "CHMM-paper", "figs"));
if isdir(PAPER_FIGS)
    println("\nCopying figures to paper figs: $PAPER_FIGS")
    # Fig 1: stylized facts (4 split panels, ticker-level)
    for letter in ("a", "b", "c", "d"), ext in ("pdf",)
        src = joinpath(RESULTS_DIR, TICKER, "stylized_facts", "Fig-1-Stylized-Facts-$letter.$ext");
        if isfile(src); cp(src, joinpath(PAPER_FIGS, "Fig-1-Stylized-Facts-$letter.$ext"); force=true); end
    end
    # Per-K convergence (rename Fig-Convergence.{pdf,svg} -> Fig-Convergence-K{K}.{pdf,svg})
    for K in K_VALUES, ext in ("pdf",)
        src = joinpath(RESULTS_DIR, TICKER, "K$K", "Fig-Convergence.$ext");
        if isfile(src); cp(src, joinpath(PAPER_FIGS, "Fig-Convergence-K$K.$ext"); force=true); end
    end
else
    println("\n(Skipped paper copy: $PAPER_FIGS not found)")
end
