# ========================================================================================= #
# run_all_analysis.jl
#
# Generates ALL figures and analysis for the continuous HMM paper.
# Supports both equity (SPY) and volatility index (VIX) asset classes.
# Runs the full pipeline for K ∈ {3, 6, 9, 11, 13} hidden states.
#
# Usage:
#   ASSET_CLASS = :equity       # SPY (default)
#   ASSET_CLASS = :volatility   # VIX
#   include("run_all_analysis.jl")
#
# Output: results/<ticker>/K<N>/  with figures (.svg, .pdf) and metrics (.txt)
# ========================================================================================= #

println("="^70)
println("  Continuous HMM — Full Analysis Pipeline")
println("  States: K ∈ {3, 6, 9, 11, 13}")
println("="^70)

# --- SETUP ---
using Pkg;
Pkg.activate(".");

include("Include.jl");

# --- ASSET CLASS SELECTION ---
# Set ASSET_CLASS before including this file, or default to :equity
if !@isdefined(ASSET_CLASS)
    const ASSET_CLASS = :equity;
end

# --- CONFIGURATION ---
const TICKER = (ASSET_CLASS == :volatility) ? "VIX" : "SPY";
const RETURN_LABEL = (ASSET_CLASS == :volatility) ? "$TICKER Log Return (annualized)" : "Excess Growth Rate";
const K_VALUES = [3, 6, 9, 11, 13];
const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const MAX_ITER = 60;
const N_PATHS = 1000;
const N_PATHS_GRID = 200;
const L = 252;            # ACF max lag
const W_K = 0.20;         # kurtosis penalty weight

# Grid search parameters
const ε_GRID = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2.5e-2];
const λ_GRID = [10, 30, 55, 70, 85, 100, 130, 160];

# Output directory
const RESULTS_DIR = joinpath(_ROOT, "results");

println("  Asset class: $(ASSET_CLASS) | Ticker: $(TICKER)");

# ========================================================================================= #
# LOAD DATA
# ========================================================================================= #
println("\n[1/6] Loading data...")

if ASSET_CLASS == :volatility
    # Volatility index: single-ticker dataset (VIX)
    train_dataset = MyVolatilityDataSet() |> x -> x["dataset"];
    R_is = log_growth_matrix(train_dataset, TICKER; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
    n_steps = length(R_is);

    oos_dataset = MyOutOfSampleVolatilityDataSet() |> x -> x["dataset"];
    R_oos = log_growth_matrix(oos_dataset, TICKER; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
    n_steps_oos = length(R_oos);

    println("  IS: $(n_steps) obs | OoS: $(n_steps_oos) obs")
else
    # Equity: multi-ticker portfolio (SPY, AAPL, etc.)
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
end

# ========================================================================================= #
# FIGURE 1: STYLIZED FACTS (only once — independent of K)
# ========================================================================================= #
println("\n[2/6] Generating Figure 1: Stylized Facts...")

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

# Percentile-based axis limits (0.5th–99.5th with 20% padding)
x_lo_raw = quantile(R_is, 0.005); x_hi_raw = quantile(R_is, 0.995);
x_pad = 0.20 * (x_hi_raw - x_lo_raw);
x_lo = x_lo_raw - x_pad; x_hi = x_hi_raw + x_pad;
x_grid = range(x_lo, x_hi, length=1000);

# Panel (a): Distribution
p1 = histogram(R_is, normalize=:pdf, bins=200, alpha=0.4, color=:lightgray, label="Observed",
    title="(a) Marginal Distribution", titlefontsize=10, xlabel=RETURN_LABEL, ylabel="Probability Density (AU)");
plot!(p1, x_grid, pdf.(d_gauss, x_grid), lw=2, color=:blue, label="Gaussian", ls=:dash);
plot!(p1, x_grid, pdf.(d_laplace, x_grid), lw=2, color=:red, label="Laplace");
xlims!(p1, x_lo, x_hi);

# Panel (b): Q-Q
sorted_R = sort(R_is); n_r = length(sorted_R);
theo_q = [quantile(d_gauss, (i-0.5)/n_r) for i in 1:n_r];
qq_lo = min(minimum(theo_q), minimum(sorted_R)); qq_hi = max(maximum(theo_q), maximum(sorted_R));
p2 = scatter(theo_q, sorted_R, ms=1, alpha=0.5, color=:steelblue, label="",
    title="(b) Normal Q-Q Plot", titlefontsize=10, xlabel="Theoretical", ylabel="Sample");
plot!(p2, [qq_lo, qq_hi], [qq_lo, qq_hi], lw=2, color=:red, ls=:dash, label="45°");

# Panel (c): Returns ACF
τ = 1:(L-1); ci = 2.576/sqrt(n_r);
acf_raw = autocor(R_is, τ);
p3 = plot(τ, acf_raw, linetype=:steppost, lw=2, color=:steelblue, label="ACF(Gₜ)",
    title="(c) Returns ACF", titlefontsize=10, xlabel="Lag", ylabel="ACF");
plot!(p3, τ, ci.*ones(length(τ)), lw=1.5, color=:gray, ls=:dash, label="99% CI");
plot!(p3, τ, -ci.*ones(length(τ)), lw=1.5, color=:gray, ls=:dash, label="");

# Panel (d): |Returns| ACF
acf_abs = autocor(abs.(R_is), τ);
p4 = plot(τ, acf_abs, linetype=:steppost, lw=2, color=:darkorange, label="ACF(|Gₜ|)",
    title="(d) Volatility Clustering", titlefontsize=10, xlabel="Lag", ylabel="ACF");
plot!(p4, τ, ci.*ones(length(τ)), lw=1.5, color=:gray, ls=:dash, label="99% CI");
plot!(p4, τ, -ci.*ones(length(τ)), lw=1.5, color=:gray, ls=:dash, label="");

fig1 = plot(p1, p2, p3, p4, layout=(2,2), size=(1000,700),
    plot_title="Figure 1: Stylized Facts — $TICKER", plot_titlefontsize=12);
savefig(fig1, joinpath(fig1_dir, "Fig-1-Stylized-Facts.svg"));
savefig(fig1, joinpath(fig1_dir, "Fig-1-Stylized-Facts.pdf"));
println("  Saved Figure 1 + Table 1")

# ========================================================================================= #
# PER-K ANALYSIS LOOP
# ========================================================================================= #

# Observed targets for grid search
target_acf = autocor(abs.(R_is), 1:L);
obs_μ = mean(R_is); obs_σ = std(R_is);
target_kurtosis = sum(((R_is .- obs_μ) ./ obs_σ).^4) / n_steps - 3.0;

# Summary table storage
summary_rows = [];

for K in K_VALUES
    println("\n" * "="^70)
    println("[3/6] Processing K = $K states...")
    println("="^70)

    out_dir = joinpath(RESULTS_DIR, TICKER, "K$(K)");
    mkpath(out_dir);

    # ------------------------------------------------------------------- #
    # STEP 1: TRAIN BASE MODEL
    # ------------------------------------------------------------------- #
    println("  Training Baum-Welch (K=$K, max_iter=$MAX_ITER)...")

    base_model = build(MyContinuousHiddenMarkovModel, (
        observations = R_is,
        number_of_states = K,
        max_iter = MAX_ITER
    ));

    println("  Converged in $(length(base_model.log_likelihood_history)) iterations")

    # Transition matrix + stationary distribution
    T_mat = zeros(K, K);
    for i in 1:K
        T_mat[i, :] = probs(base_model.transition[i]);
    end
    π_stat = (T_mat^1000)[1, :];
    start_dist = Categorical(π_stat);

    # ------------------------------------------------------------------- #
    # FIGURE: Convergence
    # ------------------------------------------------------------------- #
    p_conv = plot(base_model.log_likelihood_history,
        title="Baum-Welch Convergence ($TICKER, K=$K)", titlefontsize=10,
        xlabel="Iteration", ylabel="Log-Likelihood",
        legend=false, lw=2, color=:navy, marker=:circle, ms=3);
    savefig(p_conv, joinpath(out_dir, "Fig-Convergence.svg"));
    savefig(p_conv, joinpath(out_dir, "Fig-Convergence.pdf"));

    # ------------------------------------------------------------------- #
    # FIGURE: Emission Distributions
    # ------------------------------------------------------------------- #
    colors_k = cgrad(:RdYlBu, K, categorical=true);
    p_emit = plot(title="Emission Distributions ($TICKER, K=$K)", titlefontsize=10,
        xlabel=RETURN_LABEL, ylabel="Probability Density (AU)", legend=:topright);
    histogram!(p_emit, R_is, normalize=:pdf, bins=200, alpha=0.3, color=:lightgray, label="Observed");
    for s in 1:K
        d = base_model.emission[s];
        plot!(p_emit, x_grid, pdf.(d, x_grid), lw=1.5, color=colors_k[s], label="S$s", alpha=0.8);
    end
    xlims!(p_emit, x_lo, x_hi);
    savefig(p_emit, joinpath(out_dir, "Fig-Emission-PDFs.svg"));
    savefig(p_emit, joinpath(out_dir, "Fig-Emission-PDFs.pdf"));

    # ------------------------------------------------------------------- #
    # FIGURE: Transition Matrix Heatmap
    # ------------------------------------------------------------------- #
    T_log = log10.(T_mat .+ 1e-10);
    p_trans = heatmap(T_log, title="Transition Matrix (log₁₀) — K=$K", titlefontsize=10,
        xlabel="To State", ylabel="From State", color=:viridis,
        yflip=true, aspect_ratio=:equal, size=(500,450));
    savefig(p_trans, joinpath(out_dir, "Fig-Transition-Matrix.svg"));
    savefig(p_trans, joinpath(out_dir, "Fig-Transition-Matrix.pdf"));

    # ------------------------------------------------------------------- #
    # FIGURE: Residence Times
    # ------------------------------------------------------------------- #
    res_times = [1.0 / (1.0 - T_mat[k,k]) for k in 1:K];
    p_res = bar(1:K, res_times, title="Natural Residence Time — K=$K", titlefontsize=10,
        xlabel="State", ylabel="Steps", legend=:topright, color=:steelblue, alpha=0.7);
    if K >= 6
        crash_label = (ASSET_CLASS == :volatility) ? "Low Vol" : "Crash";
        boom_label = (ASSET_CLASS == :volatility) ? "High Vol" : "Boom";
        bar!(p_res, 1:3, res_times[1:3], color=:red, alpha=0.5, label=crash_label);
        bar!(p_res, (K-2):K, res_times[(K-2):K], color=:teal, alpha=0.5, label=boom_label);
    end
    savefig(p_res, joinpath(out_dir, "Fig-Residence-Times.svg"));
    savefig(p_res, joinpath(out_dir, "Fig-Residence-Times.pdf"));

    # ------------------------------------------------------------------- #
    # FIGURE: Stationary Distribution
    # ------------------------------------------------------------------- #
    p_pi = bar(1:K, π_stat, title="Stationary Distribution π — K=$K", titlefontsize=10,
        xlabel="State", ylabel="Probability", legend=false, color=:steelblue, alpha=0.7);
    savefig(p_pi, joinpath(out_dir, "Fig-Stationary-Distribution.svg"));
    savefig(p_pi, joinpath(out_dir, "Fig-Stationary-Distribution.pdf"));

    # Save emission parameters
    open(joinpath(out_dir, "Emission-Parameters.txt"), "w") do io
        println(io, "Emission Parameters — $TICKER, K=$K")
        println(io, "="^50)
        println(io, "State | Mean (μ)     | Std Dev (σ)")
        println(io, "-"^50)
        for s in 1:K
            d = base_model.emission[s];
            println(io, "  $(lpad(s,2))  | $(lpad(round(mean(d),digits=2),12)) | $(lpad(round(std(d),digits=2),12))")
        end
    end

    # ------------------------------------------------------------------- #
    # STEP 2: GRID SEARCH FOR JUMP PARAMETERS
    # ------------------------------------------------------------------- #
    println("  Grid search for (ε, λ)...")

    obj_matrix = zeros(length(ε_GRID), length(λ_GRID));
    best_J = Inf; best_ε = ε_GRID[1]; best_λ = λ_GRID[1];

    for (ei, ε) in enumerate(ε_GRID)
        for (li, λ) in enumerate(λ_GRID)
            jm = build(MyContinuousHiddenMarkovModelWithJumps, (
                base_model=base_model, epsilon=ε, lambda=Float64(λ)));

            acf_sum = zeros(L); kurt_sum = 0.0;
            for p in 1:N_PATHS_GRID
                s0 = rand(start_dist);
                states = jm(s0, n_steps);
                returns = [rand(jm.emission[s]) for s in states];
                acf_sum .+= autocor(abs.(returns), 1:L);
                μ_r = mean(returns); σ_r = std(returns);
                kurt_sum += sum(((returns .- μ_r) ./ σ_r).^4) / length(returns) - 3.0;
            end

            acf_sim = acf_sum ./ N_PATHS_GRID;
            kurt_sim = kurt_sum / N_PATHS_GRID;
            J = sum((target_acf .- acf_sim).^2) + W_K * (target_kurtosis - kurt_sim)^2;
            obj_matrix[ei, li] = J;

            if J < best_J; best_J = J; best_ε = ε; best_λ = λ; end
        end
    end

    println("  Optimal: ε*=$best_ε, λ*=$best_λ")

    # Build optimal jump model
    jump_model = build(MyContinuousHiddenMarkovModelWithJumps, (
        base_model=base_model, epsilon=best_ε, lambda=Float64(best_λ)));

    # ------------------------------------------------------------------- #
    # FIGURE: Grid Search Heatmap
    # ------------------------------------------------------------------- #
    ε_labels = [string(round(e, sigdigits=2)) for e in ε_GRID];
    λ_labels = [string(Int(l)) for l in λ_GRID];

    p_heat = heatmap(λ_labels, ε_labels, log10.(obj_matrix),
        title="Grid Search J(ε,λ) — K=$K", titlefontsize=10,
        xlabel="λ", ylabel="ε", color=:viridis, size=(600,450));
    opt_li = findfirst(x->x==best_λ, λ_GRID);
    opt_ei = findfirst(x->x==best_ε, ε_GRID);
    scatter!(p_heat, [opt_li], [opt_ei], ms=12, color=:red, markershape=:star5, label="Optimal");
    savefig(p_heat, joinpath(out_dir, "Fig-Grid-Search.svg"));
    savefig(p_heat, joinpath(out_dir, "Fig-Grid-Search.pdf"));

    # ------------------------------------------------------------------- #
    # FIGURE: ACF at Optimal Parameters
    # ------------------------------------------------------------------- #
    n_val = 500;
    acf_archive = zeros(L, n_val);
    for p in 1:n_val
        s0 = rand(start_dist);
        states = jump_model(s0, n_steps);
        returns = [rand(jump_model.emission[s]) for s in states];
        acf_archive[:, p] = autocor(abs.(returns), 1:L);
    end
    acf_mean = mean(acf_archive, dims=2)[:];
    acf_p10 = [quantile(acf_archive[t,:], 0.10) for t in 1:L];
    acf_p90 = [quantile(acf_archive[t,:], 0.90) for t in 1:L];

    p_acf_opt = plot(1:L, target_acf, lw=2, color=:red, ls=:dash, label="Observed",
        title="ACF(|Gₜ|) at Optimal — K=$K (ε=$best_ε, λ=$best_λ)", titlefontsize=9,
        xlabel="Lag", ylabel="ACF(|Gₜ|)");
    plot!(p_acf_opt, 1:L, acf_mean, lw=2, color=:navy, label="Simulated (mean)");
    plot!(p_acf_opt, 1:L, acf_p10, fillrange=acf_p90, alpha=0.2, color=:navy, label="10-90th pctl");
    savefig(p_acf_opt, joinpath(out_dir, "Fig-ACF-Optimal.svg"));
    savefig(p_acf_opt, joinpath(out_dir, "Fig-ACF-Optimal.pdf"));

    # ------------------------------------------------------------------- #
    # STEP 3: SIMULATE 1000 PATHS (NJ + WJ)
    # ------------------------------------------------------------------- #
    println("  Simulating $N_PATHS paths (NJ + WJ)...")

    decoded_nj = Array{Float64,2}(undef, n_steps, N_PATHS);
    decoded_wj = Array{Float64,2}(undef, n_steps, N_PATHS);
    oos_decoded_nj = Array{Float64,2}(undef, n_steps_oos, N_PATHS);
    oos_decoded_wj = Array{Float64,2}(undef, n_steps_oos, N_PATHS);

    for i in 1:N_PATHS
        # IS - NJ
        s0 = rand(start_dist);
        states = base_model(s0, n_steps);
        for j in 1:n_steps; decoded_nj[j,i] = rand(base_model.emission[states[j]]); end

        # IS - WJ
        s0 = rand(start_dist);
        states = jump_model(s0, n_steps);
        for j in 1:n_steps; decoded_wj[j,i] = rand(jump_model.emission[states[j]]); end

        # OoS - NJ
        s0 = rand(start_dist);
        states = base_model(s0, n_steps_oos);
        for j in 1:n_steps_oos; oos_decoded_nj[j,i] = rand(base_model.emission[states[j]]); end

        # OoS - WJ
        s0 = rand(start_dist);
        states = jump_model(s0, n_steps_oos);
        for j in 1:n_steps_oos; oos_decoded_wj[j,i] = rand(jump_model.emission[states[j]]); end
    end

    # ------------------------------------------------------------------- #
    # STEP 4: COMPUTE METRICS
    # ------------------------------------------------------------------- #
    println("  Computing validation metrics...")

    function eval_metrics(observed, sim_archive, L_val)
        np = size(sim_archive, 2);
        n_o = length(observed);
        μ_o = mean(observed); σ_o = std(observed);
        kurt_obs_val = sum(((observed .- μ_o) ./ σ_o).^4) / n_o - 3.0;
        L_use = min(L_val, n_o - 1);
        acf_obs_val = autocor(abs.(observed), 1:L_use);

        ks_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0; w1_s = 0.0; hell_s = 0.0;
        ks_pvals = Float64[];

        for i in 1:np
            sim = sim_archive[:, i];
            pval = pvalue(ApproximateTwoSampleKSTest(observed, sim));
            push!(ks_pvals, pval);
            if pval > 0.05; ks_pass += 1; end

            μ_s = mean(sim); σ_s = std(sim);
            kurt_s += sum(((sim .- μ_s) ./ σ_s).^4) / length(sim) - 3.0;

            acf_sim_val = autocor(abs.(sim), 1:L_use);
            acf_mae_s += mean(abs.(acf_obs_val .- acf_sim_val));

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
        end

        return (ks_rate=round(100*ks_pass/np, digits=1),
                kurtosis_obs=round(kurt_obs_val, digits=2),
                kurtosis_sim=round(kurt_s/np, digits=2),
                acf_mae=round(acf_mae_s/np, digits=4),
                wasserstein=round(w1_s/np, digits=3),
                hellinger=round(hell_s/np, digits=4),
                ks_pvals=ks_pvals)
    end

    m_nj_is = eval_metrics(R_is, decoded_nj, L);
    m_wj_is = eval_metrics(R_is, decoded_wj, L);
    m_nj_oos = eval_metrics(R_oos, oos_decoded_nj, L);
    m_wj_oos = eval_metrics(R_oos, oos_decoded_wj, L);

    # Store for summary
    push!(summary_rows, (K=K, ε=best_ε, λ=best_λ,
        nj_ks_is=m_nj_is.ks_rate, wj_ks_is=m_wj_is.ks_rate,
        nj_ks_oos=m_nj_oos.ks_rate, wj_ks_oos=m_wj_oos.ks_rate,
        nj_kurt_is=m_nj_is.kurtosis_sim, wj_kurt_is=m_wj_is.kurtosis_sim,
        nj_acf_mae=m_nj_is.acf_mae, wj_acf_mae=m_wj_is.acf_mae,
        nj_w1_is=m_nj_is.wasserstein, wj_w1_is=m_wj_is.wasserstein,
        nj_hell_is=m_nj_is.hellinger, wj_hell_is=m_wj_is.hellinger,
        kurt_obs=m_nj_is.kurtosis_obs))

    # Save per-K metrics
    open(joinpath(out_dir, "Metrics.txt"), "w") do io
        println(io, "Validation Metrics — $TICKER, K=$K")
        println(io, "Optimal: ε*=$best_ε, λ*=$best_λ")
        println(io, "="^65)
        println(io, "")
        println(io, "                  | CHMM-NJ (IS) | CHMM-WJ (IS) | CHMM-NJ (OoS) | CHMM-WJ (OoS)")
        println(io, "-"^80)
        println(io, "KS pass rate (%) | $(lpad(m_nj_is.ks_rate,12)) | $(lpad(m_wj_is.ks_rate,12)) | $(lpad(m_nj_oos.ks_rate,13)) | $(lpad(m_wj_oos.ks_rate,13))")
        println(io, "Excess kurtosis  | $(lpad(m_nj_is.kurtosis_sim,12)) | $(lpad(m_wj_is.kurtosis_sim,12)) | $(lpad(m_nj_oos.kurtosis_sim,13)) | $(lpad(m_wj_oos.kurtosis_sim,13))")
        println(io, "  (observed)     | $(lpad(m_nj_is.kurtosis_obs,12)) |              | $(lpad(m_nj_oos.kurtosis_obs,13)) |")
        println(io, "ACF-MAE          | $(lpad(m_nj_is.acf_mae,12)) | $(lpad(m_wj_is.acf_mae,12)) |               |")
        println(io, "Wasserstein-1    | $(lpad(m_nj_is.wasserstein,12)) | $(lpad(m_wj_is.wasserstein,12)) | $(lpad(m_nj_oos.wasserstein,13)) | $(lpad(m_wj_oos.wasserstein,13))")
        println(io, "Hellinger        | $(lpad(m_nj_is.hellinger,12)) | $(lpad(m_wj_is.hellinger,12)) | $(lpad(m_nj_oos.hellinger,13)) | $(lpad(m_wj_oos.hellinger,13))")
    end

    # ------------------------------------------------------------------- #
    # FIGURE 3: In-Sample Comparison (Density + ACF + Q-Q)
    # ------------------------------------------------------------------- #
    println("  Generating Figure 3: IS comparison...")

    # (a) Density
    p3a = plot(title="(a) Density (KS: NJ=$(m_nj_is.ks_rate)%, WJ=$(m_wj_is.ks_rate)%)",
        titlefontsize=9, xlabel=RETURN_LABEL, ylabel="Probability Density (AU)");
    histogram!(p3a, R_is, normalize=:pdf, bins=200, alpha=0.3, color=:lightgray, label="Observed");
    density!(p3a, decoded_nj[:,1], lw=2, color=:blue, alpha=0.7, label="CHMM-NJ");
    density!(p3a, decoded_wj[:,1], lw=2, color=:red, alpha=0.7, label="CHMM-WJ");
    xlims!(p3a, x_lo, x_hi);

    # (b) ACF(|G|)
    acf_obs_is = autocor(abs.(R_is), 1:L);
    n_acf_sample = min(200, N_PATHS);
    acf_nj_arch = hcat([autocor(abs.(decoded_nj[:,i]), 1:L) for i in 1:n_acf_sample]...);
    acf_wj_arch = hcat([autocor(abs.(decoded_wj[:,i]), 1:L) for i in 1:n_acf_sample]...);
    acf_nj_m = mean(acf_nj_arch, dims=2)[:];
    acf_wj_m = mean(acf_wj_arch, dims=2)[:];
    acf_wj_10 = [quantile(acf_wj_arch[t,:], 0.10) for t in 1:L];
    acf_wj_90 = [quantile(acf_wj_arch[t,:], 0.90) for t in 1:L];

    p3b = plot(1:L, acf_obs_is, lw=2, color=:red, ls=:dash, label="Observed",
        title="(b) ACF(|Gₜ|)", titlefontsize=9, xlabel="Lag", ylabel="ACF");
    plot!(p3b, 1:L, acf_nj_m, lw=2, color=:blue, ls=:dot, label="NJ (mean)");
    plot!(p3b, 1:L, acf_wj_m, lw=2, color=:navy, label="WJ (mean)");
    plot!(p3b, 1:L, acf_wj_10, fillrange=acf_wj_90, alpha=0.15, color=:navy, label="WJ 10-90th");

    # (c) Q-Q
    probs_qq = range(0.001, 0.999, length=200);
    q_obs = quantile(R_is, probs_qq);
    q_nj = quantile(vec(decoded_nj), probs_qq);
    q_wj = quantile(vec(decoded_wj), probs_qq);

    p3c = plot(q_obs, q_obs, lw=2, color=:black, ls=:dash, label="Perfect",
        title="(c) Tail Q-Q (0.1st-99.9th)", titlefontsize=9,
        xlabel="Observed Quantiles", ylabel="Simulated Quantiles");
    scatter!(p3c, q_obs, q_nj, ms=3, alpha=0.6, color=:blue, label="NJ");
    scatter!(p3c, q_obs, q_wj, ms=3, alpha=0.6, color=:red, label="WJ");

    fig3 = plot(p3a, p3b, p3c, layout=(1,3), size=(1400,400),
        plot_title="Figure 3: IS Comparison — $TICKER, K=$K", plot_titlefontsize=12);
    savefig(fig3, joinpath(out_dir, "Fig-3-IS-Comparison.svg"));
    savefig(fig3, joinpath(out_dir, "Fig-3-IS-Comparison.pdf"));

    # ------------------------------------------------------------------- #
    # FIGURE 4: OoS Validation
    # ------------------------------------------------------------------- #
    println("  Generating Figure 4: OoS validation...")

    # (a) KS p-values
    p4a = histogram(m_wj_oos.ks_pvals, bins=50, normalize=true, alpha=0.6, color=:navy,
        label="CHMM-WJ", title="(a) OoS KS p-values", titlefontsize=9, xlabel="p-value", ylabel="Density");
    vline!(p4a, [0.05], lw=2, color=:red, ls=:dash, label="α=0.05");

    # (b) Density fan chart (discrete paper style: light simulated, bold observed)
    p4b = plot(title="(b) OoS Density Fan", titlefontsize=9, xlabel=RETURN_LABEL, ylabel="Probability Density (AU)");
    for i in 1:min(50, N_PATHS)
        density!(p4b, oos_decoded_wj[:,i], lw=1, color=:deepskyblue1, alpha=0.05, label="");
    end
    density!(p4b, R_oos, lw=3, color=:red, label="Observed OoS");
    oos_lo = quantile(R_oos, 0.005); oos_hi = quantile(R_oos, 0.995);
    oos_pad = 0.20 * (oos_hi - oos_lo);
    xlims!(p4b, oos_lo - oos_pad, oos_hi + oos_pad);

    # (c) OoS ACF
    τ_oos = 1:min(L, n_steps_oos-1);
    acf_oos_obs = autocor(abs.(R_oos), τ_oos);
    n_acf_oos = min(200, N_PATHS);
    acf_oos_arch = hcat([autocor(abs.(oos_decoded_wj[:,i]), τ_oos) for i in 1:n_acf_oos]...);
    acf_oos_m = mean(acf_oos_arch, dims=2)[:];
    acf_oos_10 = [quantile(acf_oos_arch[t,:], 0.10) for t in 1:length(τ_oos)];
    acf_oos_90 = [quantile(acf_oos_arch[t,:], 0.90) for t in 1:length(τ_oos)];

    p4c = plot(τ_oos, acf_oos_obs, lw=2, color=:red, ls=:dash, label="Observed OoS",
        title="(c) OoS ACF(|Gₜ|)", titlefontsize=9, xlabel="Lag", ylabel="ACF");
    plot!(p4c, τ_oos, acf_oos_m, lw=2, color=:navy, label="WJ (mean)");
    plot!(p4c, τ_oos, acf_oos_10, fillrange=acf_oos_90, alpha=0.2, color=:navy, label="10-90th");

    fig4 = plot(p4a, p4b, p4c, layout=(1,3), size=(1400,400),
        plot_title="Figure 4: OoS Validation — $TICKER, K=$K", plot_titlefontsize=12);
    savefig(fig4, joinpath(out_dir, "Fig-4-OoS-Validation.svg"));
    savefig(fig4, joinpath(out_dir, "Fig-4-OoS-Validation.pdf"));

    # ------------------------------------------------------------------- #
    # FIGURE: Example Trajectories
    # ------------------------------------------------------------------- #
    idx = rand(1:N_PATHS);
    traj_len = min(500, n_steps);
    p_traj = plot(R_is[1:traj_len], lw=1, color=:red, alpha=0.6, label="Observed",
        title="Return Trajectory (first $traj_len steps) — K=$K", titlefontsize=10,
        xlabel="Trading Day", ylabel=RETURN_LABEL);
    plot!(p_traj, decoded_wj[1:traj_len, idx], lw=1, color=:navy, alpha=0.6, label="CHMM-WJ (path $idx)");
    savefig(p_traj, joinpath(out_dir, "Fig-Trajectory-Example.svg"));
    savefig(p_traj, joinpath(out_dir, "Fig-Trajectory-Example.pdf"));

    # ------------------------------------------------------------------- #
    # FIGURE: ACF Comparison (NJ single path vs WJ single path)
    # ------------------------------------------------------------------- #
    idx_acf = rand(1:N_PATHS);
    p_acf_nj = plot_acf_comparison(R_is, decoded_nj[:, idx_acf], "Returns ACF — NJ, K=$K", idx_acf; L=L);
    p_acf_wj = plot_acf_comparison(R_is, decoded_wj[:, idx_acf], "Returns ACF — WJ, K=$K", idx_acf; L=L);
    p_acf_nj_abs = plot_acf_comparison(R_is, decoded_nj[:, idx_acf], "|Returns| ACF — NJ, K=$K", idx_acf; is_absolute=true, L=L);
    p_acf_wj_abs = plot_acf_comparison(R_is, decoded_wj[:, idx_acf], "|Returns| ACF — WJ, K=$K", idx_acf; is_absolute=true, L=L);

    fig_acf = plot(p_acf_nj, p_acf_wj, p_acf_nj_abs, p_acf_wj_abs, layout=(2,2), size=(1200,700),
        plot_title="ACF Comparison — $TICKER, K=$K", plot_titlefontsize=12);
    savefig(fig_acf, joinpath(out_dir, "Fig-ACF-Comparison.svg"));
    savefig(fig_acf, joinpath(out_dir, "Fig-ACF-Comparison.pdf"));

    println("  K=$K complete. Files saved to: $out_dir")
end

# ========================================================================================= #
# SUMMARY TABLE (Table T1 equivalent — State Resolution Sensitivity)
# ========================================================================================= #
println("\n" * "="^70)
println("[4/6] Writing Summary Table (State Resolution Sensitivity)...")
println("="^70)

summary_dir = joinpath(RESULTS_DIR, TICKER);
open(joinpath(summary_dir, "Table-T1-State-Resolution-Sensitivity.txt"), "w") do io
    println(io, "Table T1: State Resolution Sensitivity — $TICKER")
    println(io, "$(N_PATHS) simulated paths, α=0.05")
    println(io, "="^120)
    println(io, "  K  | ε*     | λ*   | KS IS(NJ) | KS IS(WJ) | KS OoS(NJ) | KS OoS(WJ) | Kurt(obs) | Kurt(NJ) | Kurt(WJ) | ACF-MAE(NJ) | ACF-MAE(WJ) | W1(NJ)  | W1(WJ)  | H(NJ)  | H(WJ)")
    println(io, "-"^120)
    for r in summary_rows
        println(io, "  $(lpad(r.K,2)) | $(lpad(r.ε,6)) | $(lpad(Int(r.λ),4)) | $(lpad(r.nj_ks_is,9)) | $(lpad(r.wj_ks_is,9)) | $(lpad(r.nj_ks_oos,10)) | $(lpad(r.wj_ks_oos,10)) | $(lpad(r.kurt_obs,8)) | $(lpad(r.nj_kurt_is,7)) | $(lpad(r.wj_kurt_is,7)) | $(lpad(r.nj_acf_mae,10)) | $(lpad(r.wj_acf_mae,10)) | $(lpad(r.nj_w1_is,6)) | $(lpad(r.wj_w1_is,6)) | $(lpad(r.nj_hell_is,5)) | $(lpad(r.wj_hell_is,5))")
    end
    println(io, "="^120)
end

# Also print to console
println("\nTable T1: State Resolution Sensitivity")
println("="^100)
println("  K  | ε*     | λ*   | KS IS(NJ)% | KS IS(WJ)% | KS OoS(NJ)% | KS OoS(WJ)% | ACF-MAE(NJ) | ACF-MAE(WJ)")
println("-"^100)
for r in summary_rows
    println("  $(lpad(r.K,2)) | $(lpad(r.ε,6)) | $(lpad(Int(r.λ),4)) |   $(lpad(r.nj_ks_is,8)) |   $(lpad(r.wj_ks_is,8)) |    $(lpad(r.nj_ks_oos,8))  |    $(lpad(r.wj_ks_oos,8))  |   $(lpad(r.nj_acf_mae,9)) |   $(lpad(r.wj_acf_mae,9))")
end
println("="^100)

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
println("  - Fig-Transition-Matrix (.svg/.pdf)")
println("  - Fig-Residence-Times (.svg/.pdf)")
println("  - Fig-Stationary-Distribution (.svg/.pdf)")
println("  - Fig-Grid-Search (.svg/.pdf)")
println("  - Fig-ACF-Optimal (.svg/.pdf)")
println("  - Fig-3-IS-Comparison (.svg/.pdf)")
println("  - Fig-4-OoS-Validation (.svg/.pdf)")
println("  - Fig-Trajectory-Example (.svg/.pdf)")
println("  - Fig-ACF-Comparison (.svg/.pdf)")
println("  - Emission-Parameters.txt")
println("  - Metrics.txt")
println("\nGenerated once:")
println("  - Fig-1-Stylized-Facts (.svg/.pdf)")
println("  - Table-1-Descriptive-Stats.txt")
println("  - Table-T1-State-Resolution-Sensitivity.txt")
