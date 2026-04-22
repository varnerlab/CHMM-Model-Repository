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
const PAPER_FIGS_DIR = joinpath(dirname(_ROOT), "CHMM-paper", "paper", "sections", "figs");
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

# ----- Figure generators (titlefontsize=10, colorbar=true on heatmaps) -----
const TFS = 10;   # sub-panel title font size (was 9)
const PTFS = 12;  # overall plot title font size

function save_is_comparison(sim_is, m_is, tag, K, out_path)
    acf_obs = autocor(abs.(R_is), 1:L_LAGS);
    n_acf = min(200, N_PATHS);
    acf_arch = hcat([autocor(abs.(sim_is[:,i]), 1:L_LAGS) for i in 1:n_acf]...);
    acf_m = mean(acf_arch, dims=2)[:];
    acf_10 = [quantile(acf_arch[t,:], 0.10) for t in 1:L_LAGS];
    acf_90 = [quantile(acf_arch[t,:], 0.90) for t in 1:L_LAGS];

    p_a = plot(title="(a) IS Density (KS: $(m_is.ks)%)",
        titlefontsize=TFS, xlabel="Excess Growth Rate", ylabel="Prob. Density (AU)",
        xguidefontsize=TFS, yguidefontsize=TFS, legendfontsize=TFS-1);
    histogram!(p_a, R_is, normalize=:pdf, bins=200, alpha=0.3, color=:lightgray, label="Observed");
    density!(p_a, sim_is[:,1], lw=2, color=:blue, alpha=0.7, label="CHMM-$tag");
    xlims!(p_a, x_lo, x_hi);

    p_b = plot(1:L_LAGS, acf_obs, lw=2, color=:red, ls=:dash, label="Observed",
        title="(b) ACF(|Gₜ|)", titlefontsize=TFS, xlabel="Lag", ylabel="ACF",
        xguidefontsize=TFS, yguidefontsize=TFS, legendfontsize=TFS-1);
    plot!(p_b, 1:L_LAGS, acf_m, lw=2, color=:navy, label="CHMM-$tag (mean)");
    plot!(p_b, 1:L_LAGS, acf_10, fillrange=acf_90, alpha=0.15, color=:navy, label="10-90th pctl");

    probs_qq = range(0.001, 0.999, length=200);
    q_obs = quantile(R_is, probs_qq);
    q_sim = quantile(vec(sim_is), probs_qq);
    p_c = plot(q_obs, q_obs, lw=2, color=:black, ls=:dash, label="Perfect",
        title="(c) Tail Q-Q (0.1st-99.9th)", titlefontsize=TFS,
        xlabel="Observed Quantiles", ylabel="Simulated Quantiles",
        xguidefontsize=TFS, yguidefontsize=TFS, legendfontsize=TFS-1);
    scatter!(p_c, q_obs, q_sim, ms=3, alpha=0.6, color=:blue, label="CHMM-$tag");

    fig = plot(p_a, p_b, p_c, layout=(1,3), size=(1400,400),
        plot_title="IS Comparison (CHMM-$tag, K=$K)", plot_titlefontsize=PTFS);
    savefig(fig, out_path * ".pdf");
    savefig(fig, out_path * ".svg");
end

function save_oos_validation(sim_oos, m_oos, tag, K, out_path)
    p_a = histogram(m_oos.ks_pvals, bins=50, normalize=true, alpha=0.6, color=:navy,
        label="CHMM-$tag", title="(a) OoS KS p-values (pass: $(m_oos.ks)%)", titlefontsize=TFS,
        xlabel="p-value", ylabel="Density",
        xguidefontsize=TFS, yguidefontsize=TFS, legendfontsize=TFS-1);
    vline!(p_a, [0.05], lw=2, color=:red, ls=:dash, label="α=0.05");

    p_b = plot(title="(b) OoS Density Fan", titlefontsize=TFS,
        xlabel="Excess Growth Rate", ylabel="Prob. Density (AU)",
        xguidefontsize=TFS, yguidefontsize=TFS, legendfontsize=TFS-1);
    for i in 1:min(50, N_PATHS)
        density!(p_b, sim_oos[:,i], lw=1, color=:deepskyblue1, alpha=0.05, label="");
    end
    density!(p_b, R_oos, lw=3, color=:red, label="Observed OoS");
    oos_lo = quantile(R_oos, 0.005); oos_hi = quantile(R_oos, 0.995);
    oos_pad = 0.20 * (oos_hi - oos_lo);
    xlims!(p_b, oos_lo - oos_pad, oos_hi + oos_pad);

    τ_oos = 1:min(L_LAGS, n_oos-1);
    acf_oos_obs = autocor(abs.(R_oos), τ_oos);
    n_acf = min(200, N_PATHS);
    acf_arch = hcat([autocor(abs.(sim_oos[:,i]), τ_oos) for i in 1:n_acf]...);
    acf_m = mean(acf_arch, dims=2)[:];
    acf_10 = [quantile(acf_arch[t,:], 0.10) for t in 1:length(τ_oos)];
    acf_90 = [quantile(acf_arch[t,:], 0.90) for t in 1:length(τ_oos)];

    p_c = plot(τ_oos, acf_oos_obs, lw=2, color=:red, ls=:dash, label="Observed OoS",
        title="(c) OoS ACF(|Gₜ|)", titlefontsize=TFS, xlabel="Lag", ylabel="ACF",
        xguidefontsize=TFS, yguidefontsize=TFS, legendfontsize=TFS-1);
    plot!(p_c, τ_oos, acf_m, lw=2, color=:navy, label="CHMM-$tag (mean)");
    plot!(p_c, τ_oos, acf_10, fillrange=acf_90, alpha=0.2, color=:navy, label="10-90th");

    fig = plot(p_a, p_b, p_c, layout=(1,3), size=(1400,400),
        plot_title="OoS Validation (CHMM-$tag, K=$K)", plot_titlefontsize=PTFS);
    savefig(fig, out_path * ".pdf");
    savefig(fig, out_path * ".svg");
end

function save_transition_heatmap(T_mat, tag, K, out_path)
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
    ("N", () -> build(MyContinuousHiddenMarkovModel, (observations=R_is, number_of_states=K, max_iter=MAX_ITER))),
    ("t", () -> build(MyStudentTHiddenMarkovModel,   (observations=R_is, number_of_states=K, max_iter=MAX_ITER))),
    ("L", () -> build(MyLaplaceHiddenMarkovModel,    (observations=R_is, number_of_states=K, max_iter=MAX_ITER)))]
    println("\n[$tag] Training CHMM-$tag at K=$K...")
    Random.seed!(SEED);
    model = build_fn();
    T_mat, sd = _stationary(model, K);
    sis, sos = _simulate(model, sd, n_is, n_oos, N_PATHS);
    m_is  = eval_metrics(R_is,  sis);
    m_oos = eval_metrics(R_oos, sos);

    save_is_comparison(sis, m_is, tag, K,
        joinpath(PAPER_FIGS_DIR, "Fig-3-IS-Comparison-K$K-$tag"));
    save_oos_validation(sos, m_oos, tag, K,
        joinpath(PAPER_FIGS_DIR, "Fig-4-OoS-Validation-K$K-$tag"));
    save_transition_heatmap(T_mat, tag, K,
        joinpath(PAPER_FIGS_DIR, "Fig-Transition-Matrix-K$K-$tag"));
    println("  [$tag] Figures written.")

    # Also write the main-body unsuffixed versions from the Gaussian family (main-body figures)
    if tag == "N"
        save_is_comparison(sis, m_is, tag, K,
            joinpath(PAPER_FIGS_DIR, "Fig-3-IS-Comparison-K$K"));
        save_oos_validation(sos, m_oos, tag, K,
            joinpath(PAPER_FIGS_DIR, "Fig-4-OoS-Validation-K$K"));
        save_transition_heatmap(T_mat, tag, K,
            joinpath(PAPER_FIGS_DIR, "Fig-Transition-Matrix-K$K"));
        println("  [$tag] Main-body Fig-*-K18 (unsuffixed) also rewritten.")
    end
end

println("\nDone. Figures regenerated with titlefontsize=$TFS and explicit colorbars.")
