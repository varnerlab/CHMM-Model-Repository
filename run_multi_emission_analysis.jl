# ========================================================================================= #
# run_multi_emission_analysis.jl
#
# Regenerates figures and the state-resolution sensitivity table (Table T1)
# for all three CHMM emission families (Gaussian, Student-t, Laplace) across
# K ∈ {3, 6, 9, 12, 15, 18, 21}. Full seven-metric validation panel per (family, K).
#
# Output:
#   results/SPY/multi_emission/K<N>/<family>/   per-(K, family) figures + metrics
#   results/SPY/Table-T1-Multi-Emission.txt     full (K, family) sensitivity table
#
# Usage: julia --project=. run_multi_emission_analysis.jl
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

const TICKER = "SPY";
const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const MAX_ITER = 60;
const N_PATHS = 1000;
const L = 252;
const K_VALUES = [3, 6, 9, 12, 15, 18, 21];
const EMISSION_FAMILIES = [:gaussian, :student_t, :laplace];
const RESULTS_DIR = joinpath(_ROOT, "results");

# Also copy selected K=18 figures to the paper's figs folder for LaTeX inclusion.
const PAPER_FIGS_DIR = joinpath(dirname(_ROOT), "CHMM-paper", "paper", "sections", "figs");

println("="^70)
println("  Multi-Emission Analysis Pipeline")
println("  Families: CHMM-N (Gaussian), CHMM-t (Student-t), CHMM-L (Laplace)")
println("  States:   K ∈ $(K_VALUES)")
println("="^70)

# ========================================================================================= #
# Load SPY data (shared across all runs)
# ========================================================================================= #
println("\n[1/4] Loading SPY data...")
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
n_steps = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset, TICKER; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
n_steps_oos = length(R_oos);
println("  IS: $n_steps obs | OoS: $n_steps_oos obs")

obs_μ = mean(R_is); obs_σ = std(R_is);
target_kurtosis = sum(((R_is .- obs_μ) ./ obs_σ).^4) / n_steps - 3.0;

x_lo = quantile(R_is, 0.001) * 1.2;
x_hi = quantile(R_is, 0.999) * 1.2;
x_grid = range(x_lo, x_hi, length=400);

# ========================================================================================= #
# Emission-family dispatch helpers
# ========================================================================================= #
const FAMILY_TAG = Dict(:gaussian => "N", :student_t => "t", :laplace => "L");
const FAMILY_LABEL = Dict(:gaussian => "CHMM-N (Gaussian)",
                          :student_t => "CHMM-t (Student-t)",
                          :laplace => "CHMM-L (Laplace)");

function _train_family(family::Symbol, obs::Vector{Float64}, K::Int, max_iter::Int)
    if family == :gaussian
        return build(MyContinuousHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    elseif family == :student_t
        return build(MyStudentTHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    elseif family == :laplace
        return build(MyLaplaceHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    else
        error("Unknown emission family: $family")
    end
end

function _stationary(model, K::Int)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    π_stat = (T_mat^1000)[1, :];
    return T_mat, Categorical(π_stat);
end

function _simulate_paths(model, start_dist, n_is::Int, n_oos::Int, n_paths::Int)
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

function eval_metrics(observed, sim_archive; L_val=L)
    np = size(sim_archive, 2); n_o = length(observed);
    μ_o = mean(observed); σ_o = std(observed);
    kurt_o = sum(((observed .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L_val, n_o - 1);
    acf_o = autocor(abs.(observed), 1:L_use);

    ks_pass = 0; ad_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0;
    w1_s = 0.0; hell_s = 0.0;
    ks_pvals = Float64[];

    obs_qprobs = range(0.01, 0.99, length=99);
    obs_quantiles = quantile(observed, obs_qprobs);
    sim_qmatrix = zeros(99, np);

    for i in 1:np
        sim = sim_archive[:, i];

        pval_ks = pvalue(ApproximateTwoSampleKSTest(observed, sim));
        push!(ks_pvals, pval_ks);
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

        sim_qmatrix[:, i] = quantile(sim, obs_qprobs);
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
            w1=round(w1_s/np, digits=3), hell=round(hell_s/np, digits=4),
            cov=round(100.0*cov_count/99, digits=1),
            ks_pvals=ks_pvals);
end

# ========================================================================================= #
# Figure generators (called only at K = 18)
# ========================================================================================= #

function save_convergence(model, family_tag::String, K::Int, out_dir::String)
    p = plot(model.log_likelihood_history,
        title="Pipeline A — Baum-Welch convergence | $TICKER | CHMM-$family_tag, K=$K\n" *
              "T_IS = $(n_steps) obs | max_iter = $MAX_ITER",
        titlefontsize=9,
        xlabel="EM iteration", ylabel="Data log-likelihood",
        legend=false, lw=2, color=:navy, marker=:circle, ms=3);
    savefig(p, joinpath(out_dir, "Fig-Convergence-K$K-$family_tag.svg"));
    savefig(p, joinpath(out_dir, "Fig-Convergence-K$K-$family_tag.pdf"));
end

function save_emission_pdfs(model, family_tag::String, K::Int, out_dir::String)
    colors_k = cgrad(:RdYlBu, K, categorical=true);
    p = plot(title="Pipeline A — Per-state emission densities | $TICKER | CHMM-$family_tag, K=$K\n" *
                  "Gray histogram = observed IS returns (T_IS = $(n_steps)); S1..S$K = fitted state emissions",
        titlefontsize=9,
        xlabel="Annualized excess log return G_t", ylabel="Probability density (arb. units)", legend=:topright);
    histogram!(p, R_is, normalize=:pdf, bins=200, alpha=0.3, color=:lightgray, label="Observed IS");
    for s in 1:K
        d = model.emission[s];
        plot!(p, x_grid, pdf.(d, x_grid), lw=1.5, color=colors_k[s], label="State $s", alpha=0.8);
    end
    xlims!(p, x_lo, x_hi);
    savefig(p, joinpath(out_dir, "Fig-Emission-PDFs-K$K-$family_tag.svg"));
    savefig(p, joinpath(out_dir, "Fig-Emission-PDFs-K$K-$family_tag.pdf"));
end

function save_transition_matrix(T_mat::Matrix{Float64}, family_tag::String, K::Int, out_dir::String)
    T_log = log10.(T_mat .+ 1e-10);
    p = heatmap(T_log,
        title="Pipeline A — Transition matrix log10 P(s_{t+1}=j | s_t=i) | $TICKER | CHMM-$family_tag, K=$K",
        titlefontsize=9,
        xlabel="To state j", ylabel="From state i", color=:viridis,
        yflip=true, aspect_ratio=:equal, size=(500,450));
    savefig(p, joinpath(out_dir, "Fig-Transition-Matrix-K$K-$family_tag.svg"));
    savefig(p, joinpath(out_dir, "Fig-Transition-Matrix-K$K-$family_tag.pdf"));
end

function save_stationary_distribution(π_stat::Vector{Float64}, family_tag::String, K::Int, out_dir::String)
    p = bar(1:K, π_stat,
        title="Stationary distribution pi (= left eigenvector of T) | $TICKER | CHMM-$family_tag, K=$K",
        titlefontsize=9,
        xlabel="State index", ylabel="Stationary probability pi_k", legend=false, color=:steelblue, alpha=0.7);
    savefig(p, joinpath(out_dir, "Fig-Stationary-Distribution-K$K-$family_tag.svg"));
    savefig(p, joinpath(out_dir, "Fig-Stationary-Distribution-K$K-$family_tag.pdf"));
end

function save_residence_times(T_mat::Matrix{Float64}, family_tag::String, K::Int, out_dir::String)
    res = [1.0 / max(1.0 - T_mat[k,k], 1e-12) for k in 1:K];
    p = bar(1:K, res,
        title="Natural residence time 1/(1 - T_ii) | $TICKER | CHMM-$family_tag, K=$K",
        titlefontsize=9,
        xlabel="State index", ylabel="Expected residence time (trading days)",
        legend=false, color=:steelblue, alpha=0.7);
    savefig(p, joinpath(out_dir, "Fig-Residence-Times-K$K-$family_tag.svg"));
    savefig(p, joinpath(out_dir, "Fig-Residence-Times-K$K-$family_tag.pdf"));
end

function save_is_comparison(sim_is::Matrix{Float64}, m_is, family_tag::String, K::Int, out_dir::String)
    acf_obs_is = autocor(abs.(R_is), 1:L);
    n_acf = min(200, N_PATHS);
    acf_arch = hcat([autocor(abs.(sim_is[:,i]), 1:L) for i in 1:n_acf]...);
    acf_m = mean(acf_arch, dims=2)[:];
    acf_10 = [quantile(acf_arch[t,:], 0.10) for t in 1:L];
    acf_90 = [quantile(acf_arch[t,:], 0.90) for t in 1:L];

    # V2/V13 style defaults (colorblind-safe palette, larger fonts + margins so
    # axis labels are not clipped in the compiled paper).
    _obs_c = RGB(0.0, 0.447, 0.698);        # Okabe-Ito blue
    _sim_c = RGB(0.835, 0.369, 0.0);        # Okabe-Ito vermillion
    _mean_c = RGB(0.0, 0.620, 0.451);       # Okabe-Ito bluish green
    _style = (titlefontsize=10, guidefontsize=10, tickfontsize=9, legendfontsize=8,
              left_margin=5Plots.mm, bottom_margin=5Plots.mm, top_margin=3Plots.mm);

    p_a = plot(title="(a) IS return density | KS pass rate = $(m_is.ks)% of $N_PATHS paths at alpha=0.05",
        xlabel="Annualized excess log return G_t", ylabel="Probability density (arb. units)"; _style...);
    histogram!(p_a, R_is, normalize=:pdf, bins=200, alpha=0.35, color=:lightgray, label="Observed IS (T=$(n_steps))");
    density!(p_a, sim_is[:,1], lw=2, color=_sim_c, alpha=0.85, label="CHMM-$family_tag (single sim path)");
    xlims!(p_a, x_lo, x_hi);

    p_b = plot(1:L, acf_obs_is, lw=2, color=_obs_c, ls=:dash, label="Observed |G_t|",
        title="(b) ACF of |G_t| | lag 1..$L (trading days)",
        xlabel="Lag (trading days)", ylabel="ACF of |G_t|"; _style...);
    plot!(p_b, 1:L, acf_m, lw=2, color=_mean_c, label="CHMM-$family_tag (mean over $n_acf sims)");
    plot!(p_b, 1:L, acf_10, fillrange=acf_90, alpha=0.2, color=_mean_c, label="CHMM-$family_tag 10-90 percentile");

    probs_qq = range(0.001, 0.999, length=200);
    q_obs = quantile(R_is, probs_qq);
    q_sim = quantile(vec(sim_is), probs_qq);

    p_c = plot(q_obs, q_obs, lw=2, color=:black, ls=:dash, label="Identity (perfect fit)",
        title="(c) Tail Q-Q plot | quantile grid 0.1% .. 99.9%",
        xlabel="Observed IS quantiles", ylabel="Simulated quantiles (pooled over paths)"; _style...);
    scatter!(p_c, q_obs, q_sim, ms=3, alpha=0.7, color=_sim_c, label="CHMM-$family_tag");

    fig = plot(p_a, p_b, p_c, layout=(1,3), size=(1500,450),
        plot_title="Pipeline A — IS validation | $TICKER | CHMM-$family_tag, K=$K | $N_PATHS sim paths",
        plot_titlefontsize=11);
    savefig(fig, joinpath(out_dir, "Fig-3-IS-Comparison-K$K-$family_tag.svg"));
    savefig(fig, joinpath(out_dir, "Fig-3-IS-Comparison-K$K-$family_tag.pdf"));
end

function save_oos_validation(sim_oos::Matrix{Float64}, m_oos, family_tag::String, K::Int, out_dir::String)
    # V2/V13/V14 style defaults (colorblind-safe palette, larger fonts + margins,
    # higher contrast for the density fan, explicit legend entry for simulations).
    _obs_c = RGB(0.835, 0.369, 0.0);        # Okabe-Ito vermillion (observed, high contrast)
    _sim_c = RGB(0.0, 0.447, 0.698);        # Okabe-Ito blue (simulated paths)
    _mean_c = RGB(0.0, 0.620, 0.451);       # Okabe-Ito bluish green (mean over sims)
    _style = (titlefontsize=10, guidefontsize=10, tickfontsize=9, legendfontsize=8,
              left_margin=5Plots.mm, bottom_margin=5Plots.mm, top_margin=3Plots.mm);

    p_a = histogram(m_oos.ks_pvals, bins=50, normalize=true, alpha=0.7, color=_sim_c,
        label="CHMM-$family_tag ($(length(m_oos.ks_pvals)) paths)",
        title="(a) OoS KS p-values | pass rate = $(m_oos.ks)% above alpha=0.05",
        xlabel="KS p-value against OoS series", ylabel="Density"; _style...);
    vline!(p_a, [0.05], lw=2, color=_obs_c, ls=:dash, label="alpha = 0.05 threshold");

    p_b = plot(title="(b) OoS return density fan | 50 sim paths vs. observed OoS (T=$(n_steps_oos))",
        xlabel="Annualized excess log return G_t", ylabel="Probability density (arb. units)"; _style...);
    # V14: single legend entry for the simulated fan; tripled alpha for contrast.
    _sim_paths_to_plot = min(50, N_PATHS);
    for i in 1:_sim_paths_to_plot
        _lbl = (i == 1) ? "CHMM-$family_tag simulated OoS ($(_sim_paths_to_plot) paths)" : "";
        density!(p_b, sim_oos[:,i], lw=1, color=_sim_c, alpha=0.15, label=_lbl);
    end
    density!(p_b, R_oos, lw=3, color=_obs_c, label="Observed OoS");
    oos_lo = quantile(R_oos, 0.005); oos_hi = quantile(R_oos, 0.995);
    oos_pad = 0.20 * (oos_hi - oos_lo);
    xlims!(p_b, oos_lo - oos_pad, oos_hi + oos_pad);

    τ_oos = 1:min(L, n_steps_oos-1);
    acf_oos_obs = autocor(abs.(R_oos), τ_oos);
    n_acf = min(200, N_PATHS);
    acf_oos_arch = hcat([autocor(abs.(sim_oos[:,i]), τ_oos) for i in 1:n_acf]...);
    acf_oos_m = mean(acf_oos_arch, dims=2)[:];
    acf_oos_10 = [quantile(acf_oos_arch[t,:], 0.10) for t in 1:length(τ_oos)];
    acf_oos_90 = [quantile(acf_oos_arch[t,:], 0.90) for t in 1:length(τ_oos)];

    p_c = plot(τ_oos, acf_oos_obs, lw=2, color=_obs_c, ls=:dash, label="Observed OoS |G_t|",
        title="(c) OoS ACF of |G_t| | lag 1..$(length(τ_oos)) (trading days)",
        xlabel="Lag (trading days)", ylabel="ACF of |G_t|"; _style...);
    plot!(p_c, τ_oos, acf_oos_m, lw=2, color=_mean_c, label="CHMM-$family_tag (mean over $n_acf sims)");
    plot!(p_c, τ_oos, acf_oos_10, fillrange=acf_oos_90, alpha=0.2, color=_mean_c, label="CHMM-$family_tag 10-90 percentile");

    fig = plot(p_a, p_b, p_c, layout=(1,3), size=(1500,450),
        plot_title="Pipeline A — OoS validation | $TICKER | CHMM-$family_tag, K=$K | " *
                   "T_OoS=$(n_steps_oos) obs, $N_PATHS sim paths",
        plot_titlefontsize=11);
    savefig(fig, joinpath(out_dir, "Fig-4-OoS-Validation-K$K-$family_tag.svg"));
    savefig(fig, joinpath(out_dir, "Fig-4-OoS-Validation-K$K-$family_tag.pdf"));
end

function save_trajectory(sim_is::Matrix{Float64}, family_tag::String, K::Int, out_dir::String)
    idx = rand(1:N_PATHS);
    traj_len = min(500, n_steps);
    p = plot(R_is[1:traj_len], lw=1, color=:red, alpha=0.6, label="Observed IS",
        title="Sample return trajectory (first $traj_len IS days) | $TICKER | CHMM-$family_tag, K=$K",
        titlefontsize=9,
        xlabel="Trading day (IS index)", ylabel="Annualized excess log return G_t");
    plot!(p, sim_is[1:traj_len, idx], lw=1, color=:navy, alpha=0.6, label="CHMM-$family_tag (single sim path)");
    savefig(p, joinpath(out_dir, "Fig-Trajectory-Example-K$K-$family_tag.svg"));
    savefig(p, joinpath(out_dir, "Fig-Trajectory-Example-K$K-$family_tag.pdf"));
end

# ========================================================================================= #
# Main loop: train + evaluate + (at K=18) generate figures
# ========================================================================================= #
println("\n[2/4] Training and evaluating all (K, family) combinations...")

summary = Vector{NamedTuple}();

for family in EMISSION_FAMILIES
    tag = FAMILY_TAG[family];
    label = FAMILY_LABEL[family];
    for K in K_VALUES
        println("\n  $(label) | K = $K")

        out_dir = joinpath(RESULTS_DIR, TICKER, "multi_emission", "K$K", tag);
        mkpath(out_dir);

        model = _train_family(family, R_is, K, MAX_ITER);
        T_mat, start_dist = _stationary(model, K);
        π_stat = T_mat^1000 |> x -> x[1, :];

        sim_is, sim_oos = _simulate_paths(model, start_dist, n_steps, n_steps_oos, N_PATHS);
        m_is = eval_metrics(R_is, sim_is);
        m_oos = eval_metrics(R_oos, sim_oos);

        push!(summary, (
            family=tag, K=K,
            ks_is=m_is.ks, ad_is=m_is.ad,
            ks_oos=m_oos.ks, ad_oos=m_oos.ad,
            kurt_obs=m_is.kurt_obs, kurt_sim=m_is.kurt,
            acf_mae=m_is.acf_mae,
            w1=m_is.w1, hell=m_is.hell,
            cov_is=m_is.cov, cov_oos=m_oos.cov,
        ));

        # Per-K per-family metrics file
        open(joinpath(out_dir, "Metrics.txt"), "w") do io
            println(io, "Validation Metrics (CHMM-$tag, K=$K, $TICKER)")
            println(io, "="^70)
            println(io, "KS IS (%):       $(m_is.ks)   | KS OoS (%):  $(m_oos.ks)")
            println(io, "AD IS (%):       $(m_is.ad)   | AD OoS (%):  $(m_oos.ad)")
            println(io, "Kurt (IS sim):   $(m_is.kurt)  | Obs IS:      $(m_is.kurt_obs)")
            println(io, "Kurt (OoS sim):  $(m_oos.kurt) | Obs OoS:     $(m_oos.kurt_obs)")
            println(io, "ACF-MAE:         $(m_is.acf_mae)")
            println(io, "Wasserstein-1:   $(m_is.w1)")
            println(io, "Hellinger:       $(m_is.hell)")
            println(io, "Coverage IS (%): $(m_is.cov) | OoS: $(m_oos.cov)")
        end

        # Figures: always save K=18; also save K=3 and K=12 for the appendix panels.
        if K == 18
            save_convergence(model, tag, K, out_dir);
            save_emission_pdfs(model, tag, K, out_dir);
            save_transition_matrix(T_mat, tag, K, out_dir);
            save_stationary_distribution(π_stat, tag, K, out_dir);
            save_residence_times(T_mat, tag, K, out_dir);
            save_is_comparison(sim_is, m_is, tag, K, out_dir);
            save_oos_validation(sim_oos, m_oos, tag, K, out_dir);
            save_trajectory(sim_is, tag, K, out_dir);
        elseif K in (3, 12)
            save_emission_pdfs(model, tag, K, out_dir);
            save_is_comparison(sim_is, m_is, tag, K, out_dir);
        end
    end
end

# ========================================================================================= #
# Table T1: all (K, family) combinations
# ========================================================================================= #
println("\n[3/4] Writing Table T1 (multi-emission)...")

open(joinpath(RESULTS_DIR, TICKER, "Table-T1-Multi-Emission.txt"), "w") do io
    println(io, "Table T1: State-Resolution Sensitivity across Emission Families ($TICKER)")
    println(io, "$(N_PATHS) simulated paths, α=0.05")
    println(io, "CHMM-N: Gaussian; CHMM-t: Student-t (per-state ν); CHMM-L: Laplace")
    println(io, "="^150)
    println(io, "Family | K  | KS IS(%) | AD IS(%) | KS OoS(%) | AD OoS(%) | Kurt Obs | Kurt Sim | ACF-MAE  | W1(IS) | H(IS)   | Cov IS(%) | Cov OoS(%)")
    println(io, "-"^150)
    for family in ("N", "t", "L")
        for K in K_VALUES
            r = summary[findfirst(s -> s.family == family && s.K == K, summary)];
            println(io, "CHMM-$(family) | $(lpad(r.K,2)) | $(lpad(r.ks_is,7)) | $(lpad(r.ad_is,7)) | $(lpad(r.ks_oos,8)) | $(lpad(r.ad_oos,8)) | $(lpad(r.kurt_obs,8)) | $(lpad(r.kurt_sim,8)) | $(lpad(r.acf_mae,7)) | $(lpad(r.w1,5))  | $(lpad(r.hell,5))  | $(lpad(r.cov_is,8))  | $(lpad(r.cov_oos,8))")
        end
        println(io, "-"^150)
    end
    println(io, "="^150)
end

# ========================================================================================= #
# Copy selected K=18 figures into the paper's sections/figs with suffixes.
# Also copy K=3 and K=12 IS comparison + emission PDFs for the appendix panels.
# ========================================================================================= #
println("\n[4/4] Copying figures to the paper's figs directory...")

if isdir(PAPER_FIGS_DIR)
    per_family_figs_K18 = [
        "Fig-Convergence-K18", "Fig-Emission-PDFs-K18",
        "Fig-Transition-Matrix-K18", "Fig-Stationary-Distribution-K18",
        "Fig-Residence-Times-K18", "Fig-Trajectory-Example-K18",
        "Fig-3-IS-Comparison-K18", "Fig-4-OoS-Validation-K18",
    ];
    for fam in ("N", "t", "L")
        src_dir = joinpath(RESULTS_DIR, TICKER, "multi_emission", "K18", fam);
        for stem in per_family_figs_K18
            for ext in (".pdf", ".svg")
                src = joinpath(src_dir, stem * "-" * fam * ext);
                if isfile(src)
                    dst = joinpath(PAPER_FIGS_DIR, stem * "-" * fam * ext);
                    cp(src, dst; force=true);
                end
            end
        end
    end
    # K=3 and K=12 side figures (emission PDFs + IS panel) for appendix
    for fam in ("N", "t", "L")
        for K_extra in (3, 12)
            src_dir = joinpath(RESULTS_DIR, TICKER, "multi_emission", "K$(K_extra)", fam);
            for stem in ("Fig-Emission-PDFs-K$(K_extra)", "Fig-3-IS-Comparison-K$(K_extra)")
                for ext in (".pdf", ".svg")
                    src = joinpath(src_dir, stem * "-" * fam * ext);
                    if isfile(src)
                        dst = joinpath(PAPER_FIGS_DIR, stem * "-" * fam * ext);
                        cp(src, dst; force=true);
                    end
                end
            end
        end
    end
    println("  Copied figures to: $PAPER_FIGS_DIR")
else
    println("  (Skipped copy: $PAPER_FIGS_DIR not found)")
end

println("\nDone. See results at: $(joinpath(RESULTS_DIR, TICKER))")
