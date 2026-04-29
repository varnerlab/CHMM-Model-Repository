# Re-render Fig-3-IS-Comparison-K<N> and Fig-4-OoS-Validation-K<N> for the
# K-sweep (CHMM-N / Gaussian emissions) at the same style and size as
# run_figures.jl, so the appendix K-sweep panels can be displayed at full
# \textwidth without label clipping after LaTeX shrink.

using Pkg; Pkg.activate(".");
include("Include.jl");
using Random
const SEED = 20260420;

const TICKER = "SPY";
const K_VALUES = [3, 6, 12, 21];   # K=9 and K=15 dropped: not displayed in the paper.
                                    # Re-add them here if you want to extend the K-sweep panel.
const MAX_ITER = 60;
const N_PATHS = 1000; const L_LAGS = 252;
const PAPER_FIGS_DIR = joinpath(dirname(_ROOT), "CHMM-paper", "figs");
mkpath(PAPER_FIGS_DIR);

# --- Data load (identical to run_figures.jl) ---
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

println("SPY IS=$n_is OoS=$n_oos. K_VALUES=$K_VALUES. Figs -> $PAPER_FIGS_DIR");

function eval_metrics(observed, sim_archive; L_val=L_LAGS)
    np = size(sim_archive, 2); n_o = length(observed);
    L_use = min(L_val, n_o - 1);
    ks_pass = 0; ks_pvals = Float64[];
    for i in 1:np
        sim = sim_archive[:, i];
        pval_ks = pvalue(ApproximateTwoSampleKSTest(observed, sim));
        push!(ks_pvals, pval_ks);
        if pval_ks > 0.05; ks_pass += 1; end
    end
    return (ks=round(100*ks_pass/np, digits=1), ks_pvals=ks_pvals);
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

# Style constants — keep in sync with run_figures.jl
const TFS = 11;
const PTFS = 12;
const _OBS_C = RGB(0.835, 0.369, 0.0);
const _SIM_C = RGB(0.0, 0.447, 0.698);
const _MEAN_C = RGB(0.0, 0.620, 0.451);
const _STYLE = (titlefontsize=TFS, guidefontsize=TFS, tickfontsize=TFS-1,
                legendfontsize=TFS-2);

function save_is_comparison(sim_is, m_is, tag, K, out_path)
    # |Gₜ| ACF: observed + simulated mean / 10-90 band
    acf_obs_abs = autocor(abs.(R_is), 1:L_LAGS);
    n_acf = min(200, N_PATHS);
    acf_arch_abs = hcat([autocor(abs.(sim_is[:,i]), 1:L_LAGS) for i in 1:n_acf]...);
    acf_m_abs = mean(acf_arch_abs, dims=2)[:];
    acf_10_abs = [quantile(acf_arch_abs[t,:], 0.10) for t in 1:L_LAGS];
    acf_90_abs = [quantile(acf_arch_abs[t,:], 0.90) for t in 1:L_LAGS];

    # Raw Gₜ ACF: observed + simulated mean / 10-90 band
    acf_obs_raw = autocor(R_is, 1:L_LAGS);
    acf_arch_raw = hcat([autocor(sim_is[:,i], 1:L_LAGS) for i in 1:n_acf]...);
    acf_m_raw = mean(acf_arch_raw, dims=2)[:];
    acf_10_raw = [quantile(acf_arch_raw[t,:], 0.10) for t in 1:L_LAGS];
    acf_90_raw = [quantile(acf_arch_raw[t,:], 0.90) for t in 1:L_LAGS];
    ci99 = 2.576 / sqrt(length(R_is));

    # Per-panel saves (no top titles; (a)/(b)/(c)/(d) come from LaTeX subcaptions).
    panel_size = (700, 500);

    # Panel (a): marginal density
    p_a = plot(xlabel="Excess growth rate", ylabel="Probability density (AU)",
        size=panel_size; _STYLE...);
    histogram!(p_a, R_is, normalize=:pdf, bins=200, alpha=0.35, color=:lightgray, label="Observed");
    density!(p_a, sim_is[:,1], lw=2, color=_SIM_C, alpha=0.85, label="CHMM-$tag");
    xlims!(p_a, x_lo, x_hi);
    savefig(p_a, out_path * "-a.pdf");

    # Panel (b): tail Q-Q
    probs_qq = range(0.001, 0.999, length=200);
    q_obs = quantile(R_is, probs_qq);
    q_sim = quantile(vec(sim_is), probs_qq);
    p_b = plot(q_obs, q_obs, lw=2, color=:black, ls=:dash, label="Identity (perfect)",
        xlabel="Observed quantiles", ylabel="Simulated quantiles",
        size=panel_size; _STYLE...);
    scatter!(p_b, q_obs, q_sim, ms=3, alpha=0.7, color=_SIM_C, label="CHMM-$tag");
    savefig(p_b, out_path * "-b.pdf");

    # Panel (c): raw-return ACF
    p_c = plot(1:L_LAGS, acf_obs_raw, lw=2, color=_OBS_C, ls=:dash, label="Observed",
        xlabel="Lag (trading days)", ylabel="ACF of Gₜ",
        size=panel_size; _STYLE...);
    plot!(p_c, 1:L_LAGS, acf_m_raw, lw=2, color=_MEAN_C, label="CHMM-$tag (mean)");
    plot!(p_c, 1:L_LAGS, acf_10_raw, fillrange=acf_90_raw, alpha=0.2, color=_MEAN_C, label="10-90th pctl");
    hline!(p_c, [ci99, -ci99], lw=1, color=:gray, ls=:dash, label="99% CI");
    savefig(p_c, out_path * "-c.pdf");

    # Panel (d): |Gₜ| ACF
    p_d = plot(1:L_LAGS, acf_obs_abs, lw=2, color=_OBS_C, ls=:dash, label="Observed",
        xlabel="Lag (trading days)", ylabel="ACF of |Gₜ|",
        size=panel_size; _STYLE...);
    plot!(p_d, 1:L_LAGS, acf_m_abs, lw=2, color=_MEAN_C, label="CHMM-$tag (mean)");
    plot!(p_d, 1:L_LAGS, acf_10_abs, fillrange=acf_90_abs, alpha=0.2, color=_MEAN_C, label="10-90th pctl");
    savefig(p_d, out_path * "-d.pdf");
end

function save_oos_validation(sim_oos, m_oos, tag, K, out_path)
    panel_size = (700, 500);

    # Panel (a): OoS density fan
    p_a = plot(xlabel="Excess growth rate", ylabel="Probability density (AU)",
        size=panel_size; _STYLE...);
    _sim_paths_to_plot = min(50, N_PATHS);
    for i in 1:_sim_paths_to_plot
        _lbl = (i == 1) ? "CHMM-$tag simulated ($(_sim_paths_to_plot) paths)" : "";
        density!(p_a, sim_oos[:,i], lw=1, color=_SIM_C, alpha=0.18, label=_lbl);
    end
    density!(p_a, R_oos, lw=3, color=_OBS_C, label="Observed OoS");
    oos_lo = quantile(R_oos, 0.005); oos_hi = quantile(R_oos, 0.995);
    oos_pad = 0.20 * (oos_hi - oos_lo);
    xlims!(p_a, oos_lo - oos_pad, oos_hi + oos_pad);
    savefig(p_a, out_path * "-a.pdf");

    # Panel (b): KS p-value histogram
    p_b = histogram(m_oos.ks_pvals, bins=50, normalize=true, alpha=0.75, color=_SIM_C,
        label="CHMM-$tag (pass: $(m_oos.ks)%)",
        xlabel="p-value against OoS series", ylabel="Density",
        size=panel_size; _STYLE...);
    vline!(p_b, [0.05], lw=2, color=_OBS_C, ls=:dash, label="α = 0.05");
    savefig(p_b, out_path * "-b.pdf");

    τ_oos = 1:min(L_LAGS, n_oos-1);
    n_acf = min(200, N_PATHS);

    # Raw Gₜ ACF
    acf_oos_obs_raw = autocor(R_oos, τ_oos);
    acf_arch_raw = hcat([autocor(sim_oos[:,i], τ_oos) for i in 1:n_acf]...);
    acf_m_raw = mean(acf_arch_raw, dims=2)[:];
    acf_10_raw = [quantile(acf_arch_raw[t,:], 0.10) for t in 1:length(τ_oos)];
    acf_90_raw = [quantile(acf_arch_raw[t,:], 0.90) for t in 1:length(τ_oos)];
    ci99_oos = 2.576 / sqrt(length(R_oos));

    # |Gₜ| ACF
    acf_oos_obs_abs = autocor(abs.(R_oos), τ_oos);
    acf_arch_abs = hcat([autocor(abs.(sim_oos[:,i]), τ_oos) for i in 1:n_acf]...);
    acf_m_abs = mean(acf_arch_abs, dims=2)[:];
    acf_10_abs = [quantile(acf_arch_abs[t,:], 0.10) for t in 1:length(τ_oos)];
    acf_90_abs = [quantile(acf_arch_abs[t,:], 0.90) for t in 1:length(τ_oos)];

    # Panel (c): raw-return OoS ACF
    p_c = plot(τ_oos, acf_oos_obs_raw, lw=2, color=_OBS_C, ls=:dash, label="Observed OoS",
        xlabel="Lag (trading days)", ylabel="ACF of Gₜ",
        size=panel_size; _STYLE...);
    plot!(p_c, τ_oos, acf_m_raw, lw=2, color=_MEAN_C, label="CHMM-$tag (mean)");
    plot!(p_c, τ_oos, acf_10_raw, fillrange=acf_90_raw, alpha=0.2, color=_MEAN_C, label="10-90th pctl");
    hline!(p_c, [ci99_oos, -ci99_oos], lw=1, color=:gray, ls=:dash, label="99% CI");
    savefig(p_c, out_path * "-c.pdf");

    # Panel (d): |Gₜ| OoS ACF
    p_d = plot(τ_oos, acf_oos_obs_abs, lw=2, color=_OBS_C, ls=:dash, label="Observed OoS",
        xlabel="Lag (trading days)", ylabel="ACF of |Gₜ|",
        size=panel_size; _STYLE...);
    plot!(p_d, τ_oos, acf_m_abs, lw=2, color=_MEAN_C, label="CHMM-$tag (mean)");
    plot!(p_d, τ_oos, acf_10_abs, fillrange=acf_90_abs, alpha=0.2, color=_MEAN_C, label="10-90th pctl");
    savefig(p_d, out_path * "-d.pdf");
end

for K in K_VALUES
    println("\n[K=$K] Training CHMM-N at K=$K (max_iter=$MAX_ITER)...")
    Random.seed!(SEED);
    model = build(MyContinuousHiddenMarkovModel,
        (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    _, sd = _stationary(model, K);
    sis, sos = _simulate(model, sd, n_is, n_oos, N_PATHS);
    m_is  = eval_metrics(R_is,  sis);
    m_oos = eval_metrics(R_oos, sos);

    save_is_comparison(sis, m_is, "N", K,
        joinpath(PAPER_FIGS_DIR, "Fig-3-IS-Comparison-K$K"));
    save_oos_validation(sos, m_oos, "N", K,
        joinpath(PAPER_FIGS_DIR, "Fig-4-OoS-Validation-K$K"));
    println("  [K=$K] IS KS=$(m_is.ks)%  OoS KS=$(m_oos.ks)%  → figures written.")
end

println("\nDone. K-sweep figures regenerated at size=(1500,450), titlefontsize=$TFS.")
