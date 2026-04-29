# Re-render Fig-3-IS-Comparison and Fig-4-OoS-Validation for the K=18 SPY
# fits of all three emission families, plus the transition-matrix heatmaps,
# with titlefontsize=10 (previously 9) and an explicit colorbar on every
# transition-matrix heatmap (review 6.5).

using Pkg; Pkg.activate(".");
include("Include.jl");
using Random
const SEED = 20260420;
Random.seed!(SEED);

const TICKER = "SPY";
const K = 18; const MAX_ITER = 60;
const N_PATHS = 1000; const L_LAGS = 252;
const PAPER_FIGS_DIR = joinpath(dirname(_ROOT), "CHMM-paper", "figs");
mkpath(PAPER_FIGS_DIR);

# Load SPY data
train = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train; if nrow(data) == max_days; dataset[t] = data; end; end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=1/252, risk_free_rate=0.0);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = all_R[:, idx_spy];
oos = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos, "SPY"; Δt=1/252, risk_free_rate=0.0);
n_is = length(R_is); n_oos = length(R_oos);
x_lo_raw = quantile(R_is, 0.005); x_hi_raw = quantile(R_is, 0.995);
x_pad = 0.20 * (x_hi_raw - x_lo_raw);
x_lo = x_lo_raw - x_pad; x_hi = x_hi_raw + x_pad;

println("SPY IS=$n_is OoS=$n_oos. Figs -> $PAPER_FIGS_DIR");

# ----- Metric helper (eval_metrics returns ks_pvals + ks etc.) -----
function eval_metrics(observed, sim_archive; L_val=L_LAGS)
    np = size(sim_archive, 2); n_o = length(observed);
    μ_o = mean(observed); σ_o = std(observed);
    kurt_o = sum(((observed .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L_val, n_o - 1);
    acf_o = autocor(abs.(observed), 1:L_use);
    ks_pass = 0; ad_pass = 0; ks_pvals = Float64[];
    for i in 1:np
        sim = sim_archive[:, i];
        pval_ks = pvalue(ApproximateTwoSampleKSTest(observed, sim));
        push!(ks_pvals, pval_ks);
        if pval_ks > 0.05; ks_pass += 1; end
        pval_ad = pvalue(KSampleADTest(observed, sim));
        if pval_ad > 0.05; ad_pass += 1; end
    end
    return (ks=round(100*ks_pass/np, digits=1),
            ad=round(100*ad_pass/np, digits=1),
            ks_pvals=ks_pvals);
end

function _stationary(model, K::Int)
    T = zeros(K, K); for i in 1:K; T[i, :] = probs(model.transition[i]); end
    π = (T^1000)[1, :]; return T, Categorical(π);
end
function _simulate(model, start_dist, n_is, n_oos, n_paths)
    sis = Array{Float64,2}(undef, n_is, n_paths); sos = Array{Float64,2}(undef, n_oos, n_paths);
    for i in 1:n_paths
        s0 = rand(start_dist); st = model(s0, n_is);
        for j in 1:n_is; sis[j,i] = rand(model.emission[st[j]]); end
        s0 = rand(start_dist); st = model(s0, n_oos);
        for j in 1:n_oos; sos[j,i] = rand(model.emission[st[j]]); end
    end
    return sis, sos;
end

# ----- Figure generators (V2 axis labels, V13 colorblind palette, V14 density fan) -----
const TFS = 11;                             # sub-panel title font size (was 10)
const PTFS = 12;                            # overall plot title font size
const _OBS_C = RGB(0.835, 0.369, 0.0);      # Okabe-Ito vermillion (observed)
const _SIM_C = RGB(0.0, 0.447, 0.698);      # Okabe-Ito blue (simulated paths)
const _MEAN_C = RGB(0.0, 0.620, 0.451);     # Okabe-Ito bluish green (mean over sims)
# Shared kwargs ensure axis labels / ticks / legends stay readable and do not
# clip after the paper LaTeX shrinks the figure to subfigure width. Margins
# are set project-wide in plots_defaults.jl (loaded via Include.jl); do not
# redeclare them here.
const _STYLE = (titlefontsize=TFS, guidefontsize=TFS, tickfontsize=TFS-1,
                legendfontsize=TFS-2);

function kfigs_save_is_comparison(sim_is, m_is, tag, K, out_path)
    # |G_t| ACF: observed + simulated mean / 10-90 band
    acf_obs_abs = autocor(abs.(R_is), 1:L_LAGS);
    n_acf = min(200, N_PATHS);
    acf_arch_abs = hcat([autocor(abs.(sim_is[:,i]), 1:L_LAGS) for i in 1:n_acf]...);
    acf_m_abs = mean(acf_arch_abs, dims=2)[:];
    acf_10_abs = [quantile(acf_arch_abs[t,:], 0.10) for t in 1:L_LAGS];
    acf_90_abs = [quantile(acf_arch_abs[t,:], 0.90) for t in 1:L_LAGS];

    # Raw G_t ACF: observed + simulated mean / 10-90 band
    acf_obs_raw = autocor(R_is, 1:L_LAGS);
    acf_arch_raw = hcat([autocor(sim_is[:,i], 1:L_LAGS) for i in 1:n_acf]...);
    acf_m_raw = mean(acf_arch_raw, dims=2)[:];
    acf_10_raw = [quantile(acf_arch_raw[t,:], 0.10) for t in 1:L_LAGS];
    acf_90_raw = [quantile(acf_arch_raw[t,:], 0.90) for t in 1:L_LAGS];
    ci99 = 2.576 / sqrt(length(R_is));

    # Panel (a): marginal density
    p_a = plot(title="IS density (KS: $(m_is.ks)%)",
        xlabel="Excess growth rate", ylabel="Probability density (AU)"; _STYLE...);
    histogram!(p_a, R_is, normalize=:pdf, bins=200, alpha=0.35, color=:lightgray, label="Observed");
    density!(p_a, sim_is[:,1], lw=2, color=_SIM_C, alpha=0.85, label="CHMM-$tag");
    xlims!(p_a, x_lo, x_hi);

    # Panel (b): tail Q-Q
    probs_qq = range(0.001, 0.999, length=200);
    q_obs = quantile(R_is, probs_qq);
    q_sim = quantile(vec(sim_is), probs_qq);
    p_b = plot(q_obs, q_obs, lw=2, color=:black, ls=:dash, label="Identity (perfect)",
        title="Tail Q-Q plot (0.1st-99.9th)",
        xlabel="Observed quantiles", ylabel="Simulated quantiles"; _STYLE...);
    scatter!(p_b, q_obs, q_sim, ms=3, alpha=0.7, color=_SIM_C, label="CHMM-$tag");

    # Panel (c): raw-return ACF (linear autocorrelation)
    p_c = plot(1:L_LAGS, acf_obs_raw, lw=2, color=_OBS_C, ls=:dash, label="Observed",
        title="ACF of G_t (raw returns)",
        xlabel="Lag (trading days)", ylabel="ACF of G_t"; _STYLE...);
    plot!(p_c, 1:L_LAGS, acf_m_raw, lw=2, color=_MEAN_C, label="CHMM-$tag (mean)");
    plot!(p_c, 1:L_LAGS, acf_10_raw, fillrange=acf_90_raw, alpha=0.2, color=_MEAN_C, label="10-90th pctl");
    hline!(p_c, [ci99, -ci99], lw=1, color=:gray, ls=:dash, label="99% CI");

    # Panel (d): |G_t| ACF (volatility clustering)
    p_d = plot(1:L_LAGS, acf_obs_abs, lw=2, color=_OBS_C, ls=:dash, label="Observed",
        title="ACF of |G_t|",
        xlabel="Lag (trading days)", ylabel="ACF of |G_t|"; _STYLE...);
    plot!(p_d, 1:L_LAGS, acf_m_abs, lw=2, color=_MEAN_C, label="CHMM-$tag (mean)");
    plot!(p_d, 1:L_LAGS, acf_10_abs, fillrange=acf_90_abs, alpha=0.2, color=_MEAN_C, label="10-90th pctl");

    fig = plot(p_a, p_b, p_c, p_d, layout=(2,2), size=(1100,800));
    savefig(fig, out_path * ".pdf");
    savefig(fig, out_path * ".svg");
end

function kfigs_save_oos_validation(sim_oos, m_oos, tag, K, out_path)
    # Panel (a): OoS density fan (marginal fidelity, visual)
    p_a = plot(title="OoS density fan",
        xlabel="Excess growth rate", ylabel="Probability density (AU)"; _STYLE...);
    _sim_paths_to_plot = min(50, N_PATHS);
    for i in 1:_sim_paths_to_plot
        _lbl = (i == 1) ? "CHMM-$tag simulated ($(_sim_paths_to_plot) paths)" : "";
        density!(p_a, sim_oos[:,i], lw=1, color=_SIM_C, alpha=0.18, label=_lbl);
    end
    density!(p_a, R_oos, lw=3, color=_OBS_C, label="Observed OoS");
    oos_lo = quantile(R_oos, 0.005); oos_hi = quantile(R_oos, 0.995);
    oos_pad = 0.20 * (oos_hi - oos_lo);
    xlims!(p_a, oos_lo - oos_pad, oos_hi + oos_pad);

    # Panel (b): KS p-value histogram (marginal fidelity, numerical)
    p_b = histogram(m_oos.ks_pvals, bins=50, normalize=true, alpha=0.75, color=_SIM_C,
        label="CHMM-$tag",
        title="OoS KS p-values (pass: $(m_oos.ks)%)",
        xlabel="p-value against OoS series", ylabel="Density"; _STYLE...);
    vline!(p_b, [0.05], lw=2, color=_OBS_C, ls=:dash, label="α = 0.05");

    # ACF computations
    τ_oos = 1:min(L_LAGS, n_oos-1);
    n_acf = min(200, N_PATHS);

    # Raw G_t ACF (linear autocorrelation)
    acf_oos_obs_raw = autocor(R_oos, τ_oos);
    acf_arch_raw = hcat([autocor(sim_oos[:,i], τ_oos) for i in 1:n_acf]...);
    acf_m_raw = mean(acf_arch_raw, dims=2)[:];
    acf_10_raw = [quantile(acf_arch_raw[t,:], 0.10) for t in 1:length(τ_oos)];
    acf_90_raw = [quantile(acf_arch_raw[t,:], 0.90) for t in 1:length(τ_oos)];
    ci99_oos = 2.576 / sqrt(length(R_oos));

    # |G_t| ACF (volatility clustering)
    acf_oos_obs_abs = autocor(abs.(R_oos), τ_oos);
    acf_arch_abs = hcat([autocor(abs.(sim_oos[:,i]), τ_oos) for i in 1:n_acf]...);
    acf_m_abs = mean(acf_arch_abs, dims=2)[:];
    acf_10_abs = [quantile(acf_arch_abs[t,:], 0.10) for t in 1:length(τ_oos)];
    acf_90_abs = [quantile(acf_arch_abs[t,:], 0.90) for t in 1:length(τ_oos)];

    # Panel (c): raw-return OoS ACF
    p_c = plot(τ_oos, acf_oos_obs_raw, lw=2, color=_OBS_C, ls=:dash, label="Observed OoS",
        title="OoS ACF of G_t (raw returns)",
        xlabel="Lag (trading days)", ylabel="ACF of G_t"; _STYLE...);
    plot!(p_c, τ_oos, acf_m_raw, lw=2, color=_MEAN_C, label="CHMM-$tag (mean)");
    plot!(p_c, τ_oos, acf_10_raw, fillrange=acf_90_raw, alpha=0.2, color=_MEAN_C, label="10-90th pctl");
    hline!(p_c, [ci99_oos, -ci99_oos], lw=1, color=:gray, ls=:dash, label="99% CI");

    # Panel (d): |G_t| OoS ACF
    p_d = plot(τ_oos, acf_oos_obs_abs, lw=2, color=_OBS_C, ls=:dash, label="Observed OoS",
        title="OoS ACF of |G_t|",
        xlabel="Lag (trading days)", ylabel="ACF of |G_t|"; _STYLE...);
    plot!(p_d, τ_oos, acf_m_abs, lw=2, color=_MEAN_C, label="CHMM-$tag (mean)");
    plot!(p_d, τ_oos, acf_10_abs, fillrange=acf_90_abs, alpha=0.2, color=_MEAN_C, label="10-90th pctl");

    fig = plot(p_a, p_b, p_c, p_d, layout=(2,2), size=(1100,800));
    savefig(fig, out_path * ".pdf");
    savefig(fig, out_path * ".svg");
end

function kfigs_save_transition_heatmap(T_mat, tag, K, out_path)
    T_log = log10.(T_mat .+ 1e-10);
    p = heatmap(T_log,
        title="Transition Matrix log₁₀ (CHMM-$tag, K=$K)",
        titlefontsize=TFS,
        xlabel="To State", ylabel="From State",
        xguidefontsize=TFS, yguidefontsize=TFS,
        color=:viridis, yflip=true, aspect_ratio=:equal,
        size=(520, 470),
        colorbar=true,
        colorbar_title="log₁₀ T_{ij}",
        colorbar_titlefontsize=TFS);
    savefig(p, out_path * ".pdf");
    savefig(p, out_path * ".svg");
end

# ----- Fit + simulate + render for each family -----
for (tag, build_fn) in [
    ("N",   () -> build(MyContinuousHiddenMarkovModel, (observations=R_is, number_of_states=K, max_iter=MAX_ITER))),
    ("t",   () -> build(MyStudentTHiddenMarkovModel,   (observations=R_is, number_of_states=K, max_iter=MAX_ITER))),
    ("L",   () -> build(MyLaplaceHiddenMarkovModel,    (observations=R_is, number_of_states=K, max_iter=MAX_ITER))),
    ("GED", () -> build(MyGEDHiddenMarkovModel,        (observations=R_is, number_of_states=K, max_iter=MAX_ITER)))]
    println("\n[$tag] Training CHMM-$tag at K=$K...")
    Random.seed!(SEED);
    model = build_fn();
    T_mat, sd = _stationary(model, K);
    sis, sos = _simulate(model, sd, n_is, n_oos, N_PATHS);
    m_is  = eval_metrics(R_is,  sis);
    m_oos = eval_metrics(R_oos, sos);

    kfigs_save_is_comparison(sis, m_is, tag, K,
        joinpath(PAPER_FIGS_DIR, "Fig-3-IS-Comparison-K$K-$tag"));
    kfigs_save_oos_validation(sos, m_oos, tag, K,
        joinpath(PAPER_FIGS_DIR, "Fig-4-OoS-Validation-K$K-$tag"));
    kfigs_save_transition_heatmap(T_mat, tag, K,
        joinpath(PAPER_FIGS_DIR, "Fig-Transition-Matrix-K$K-$tag"));
    println("  [$tag] Figures written.")

    # Also write the main-body unsuffixed versions from the Gaussian family (main-body figures)
    if tag == "N"
        kfigs_save_is_comparison(sis, m_is, tag, K,
            joinpath(PAPER_FIGS_DIR, "Fig-3-IS-Comparison-K$K"));
        kfigs_save_oos_validation(sos, m_oos, tag, K,
            joinpath(PAPER_FIGS_DIR, "Fig-4-OoS-Validation-K$K"));
        kfigs_save_transition_heatmap(T_mat, tag, K,
            joinpath(PAPER_FIGS_DIR, "Fig-Transition-Matrix-K$K"));
        println("  [$tag] Main-body Fig-*-K18 (unsuffixed) also rewritten.")
    end
end

println("\nDone. Figures regenerated with titlefontsize=$TFS and explicit colorbars.")
