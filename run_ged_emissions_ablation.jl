# ========================================================================================= #
# run_ged_emissions_ablation.jl
#
# Ablation test for the proposed CHMM-GED variant: per-state Generalized
# Error Distribution emissions PGeneralizedGaussian(μ_k, α_k, p_k), where
# p_k = 2 is Gaussian and p_k = 1 is Laplace, learned per state by ECM.
# This is the structural analog of CHMM-t's per-state ν_k, but on the
# Gaussian-Laplace shape axis.
#
# Headline question: does the data WANT per-state shape adaptivity on the
# Gaussian-Laplace axis? Diagnostic: the per-state p̂_k distribution. If
# p̂_k clusters at 2 for some states and at 1 for others, the data is
# asking for the hybrid; if all p̂_k cluster at one value, the global
# Gaussian or Laplace assumption is enough.
#
# Comparison panel: CHMM-N, CHMM-t, CHMM-L, CHMM-GED at K = 18 on SPY.
# Standard 7-metric IS/OoS validation panel.
#
# Output:
#   results/SPY/ged_ablation/             figures + summary table
#
# Usage: julia --project=. run_ged_emissions_ablation.jl
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");
using Random;

const TICKER = "SPY";
const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const K = 18;
const MAX_ITER = 60;
const N_PATHS = 1000;
const L = 252;
const SEED = 20260420;
const RESULTS_DIR = joinpath(_ROOT, "results", TICKER, "ged_ablation");
mkpath(RESULTS_DIR);

println("="^70)
println("  CHMM-GED Ablation (per-state Generalized Error Distribution)")
println("  Comparison: CHMM-N, CHMM-t, CHMM-L, CHMM-GED  |  K = $K  |  $TICKER")
println("="^70)

# ========================================================================================= #
# Load SPY data
# ========================================================================================= #
println("\n[1/5] Loading SPY data...")
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
println("  IS observed excess kurtosis: $(round(target_kurtosis, digits=2))")

# ========================================================================================= #
# Train all four CHMM variants from the same seed
# ========================================================================================= #
println("\n[2/5] Training CHMM-N, CHMM-t, CHMM-L, CHMM-GED at K = $K...")

Random.seed!(SEED);
println("  CHMM-N (Gaussian)...");
mN = build(MyContinuousHiddenMarkovModel,
    (observations=R_is, number_of_states=K, max_iter=MAX_ITER));

Random.seed!(SEED);
println("  CHMM-t (Student-t, per-state ν)...");
mT = build(MyStudentTHiddenMarkovModel,
    (observations=R_is, number_of_states=K, max_iter=MAX_ITER));

Random.seed!(SEED);
println("  CHMM-L (Laplace)...");
mL = build(MyLaplaceHiddenMarkovModel,
    (observations=R_is, number_of_states=K, max_iter=MAX_ITER));

Random.seed!(SEED);
println("  CHMM-GED (per-state p_k)...");
mG = build(MyGEDHiddenMarkovModel,
    (observations=R_is, number_of_states=K, max_iter=MAX_ITER));

models = (N=mN, T=mT, L=mL, GED=mG);
labels = Dict("N" => "CHMM-N (Gaussian)",
              "T" => "CHMM-t (Student-t)",
              "L" => "CHMM-L (Laplace)",
              "GED" => "CHMM-GED (per-state p)");

# ========================================================================================= #
# Simulate paths and evaluate the standard panel
# ========================================================================================= #
println("\n[3/5] Simulating $N_PATHS paths per family and computing metrics...")

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

    obs_qprobs = range(0.01, 0.99, length=99);
    obs_quantiles = quantile(observed, obs_qprobs);
    sim_qmatrix = zeros(99, np);

    for i in 1:np
        sim = sim_archive[:, i];
        if pvalue(ApproximateTwoSampleKSTest(observed, sim)) > 0.05; ks_pass += 1; end
        if pvalue(KSampleADTest(observed, sim)) > 0.05; ad_pass += 1; end

        μ_s = mean(sim); σ_s = std(sim);
        kurt_s += sum(((sim .- μ_s) ./ σ_s).^4) / length(sim) - 3.0;

        acf_mae_s += mean(abs.(acf_o .- autocor(abs.(sim), 1:L_use)));

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
            cov=round(100.0*cov_count/99, digits=1));
end

panel = Dict{String, NamedTuple}();
for (tag, model) in pairs(models)
    Random.seed!(SEED + 1);
    _, start_dist = _stationary(model, K);
    sim_is, sim_oos = _simulate_paths(model, start_dist, n_steps, n_steps_oos, N_PATHS);
    m_is = eval_metrics(R_is, sim_is);
    m_oos = eval_metrics(R_oos, sim_oos);
    panel[String(tag)] = (model=model, sim_is=sim_is, sim_oos=sim_oos, is=m_is, oos=m_oos);
    println("  $(labels[String(tag)]): KS_is=$(m_is.ks)% | KS_oos=$(m_oos.ks)% | kurt_sim=$(m_is.kurt) | ACF-MAE=$(m_is.acf_mae)")
end

# ========================================================================================= #
# Write summary table
# ========================================================================================= #
println("\n[4/5] Writing summary table...")
open(joinpath(RESULTS_DIR, "Table-GED-Ablation.txt"), "w") do io
    println(io, "Table: CHMM-GED Ablation vs CHMM-N / CHMM-t / CHMM-L  ($TICKER, K=$K)")
    println(io, "$(N_PATHS) simulated paths per family, α=0.05.  Seed $SEED.")
    println(io, "Observed IS excess kurtosis: $(round(target_kurtosis, digits=2))")
    println(io, "="^110)
    println(io, "Family    | KS IS(%) | AD IS(%) | KS OoS(%) | AD OoS(%) | Kurt(sim) | ACF-MAE | W1(IS) | Hell(IS) | Cov IS(%) | Cov OoS(%)")
    println(io, "-"^110)
    for tag in ("N", "T", "L", "GED")
        r = panel[tag];
        i = r.is; o = r.oos;
        name = rpad(labels[tag], 22);
        println(io, "$(name) | $(lpad(i.ks,7)) | $(lpad(i.ad,7)) | $(lpad(o.ks,8)) | $(lpad(o.ad,8)) | $(lpad(i.kurt,8)) | $(lpad(i.acf_mae,7)) | $(lpad(i.w1,5))  | $(lpad(i.hell,6))  | $(lpad(i.cov,8))  | $(lpad(o.cov,8))")
    end
    println(io, "="^110)
    println(io, "")
    println(io, "Final log-likelihoods (training, T_IS = $(n_steps) obs):")
    for tag in ("N", "T", "L", "GED")
        ll = panel[tag].model.log_likelihood_history;
        println(io, "  $(labels[tag]): final LL = $(round(ll[end], digits=3)) over $(length(ll)) ECM iters")
    end
end

# ========================================================================================= #
# CHMM-GED-specific diagnostic: per-state p̂_k distribution
# ========================================================================================= #
println("\n[5/5] CHMM-GED diagnostic: per-state p̂_k...")

p_k = [mG.emission[k].p for k in 1:K];
α_k = [mG.emission[k].α for k in 1:K];
μ_k = [mG.emission[k].μ for k in 1:K];

# Variance-equivalent σ for ranking (variance of GED is α²·Γ(3/p)/Γ(1/p)).
function _ged_std(α, p)
    return α * sqrt(gamma(3.0/p) / gamma(1.0/p));
end
σ_eq = [_ged_std(α_k[k], p_k[k]) for k in 1:K];
order = sortperm(σ_eq);

# Save the p̂_k diagnostic to a structured file.
open(joinpath(RESULTS_DIR, "GED-State-Diagnostics.txt"), "w") do io
    println(io, "CHMM-GED per-state shape diagnostics  ($TICKER, K=$K)")
    println(io, "States ordered by variance-equivalent sigma (calm → crash).")
    println(io, "p̂_k = 2 → Gaussian-shape state. p̂_k = 1 → Laplace-shape state.")
    println(io, "Bracket: p ∈ [0.5, 4.0].")
    println(io, "="^80)
    println(io, "rank | k  |  μ_k       |  α_k     |  p_k     |  σ_equiv  |  shape interpretation")
    println(io, "-"^80)
    for (rank, k) in enumerate(order)
        shape_label = if p_k[k] >= 1.85
            "Gaussian-like"
        elseif p_k[k] >= 1.30
            "Sub-Gaussian (intermediate)"
        elseif p_k[k] >= 0.85
            "Laplace-like"
        else
            "Super-Laplace (heavy tail)"
        end
        println(io, "$(lpad(rank,4)) | $(lpad(k,2)) | $(lpad(round(μ_k[k], digits=4), 10)) | $(lpad(round(α_k[k], digits=4), 7)) | $(lpad(round(p_k[k], digits=3), 7)) | $(lpad(round(σ_eq[k], digits=4), 8))  | $(shape_label)")
    end
    println(io, "="^80)

    # Aggregate: how many states fall in each shape regime?
    n_gaussian = count(p -> p >= 1.85, p_k);
    n_inter    = count(p -> 1.30 <= p < 1.85, p_k);
    n_laplace  = count(p -> 0.85 <= p < 1.30, p_k);
    n_super    = count(p -> p < 0.85, p_k);
    println(io, "")
    println(io, "Aggregate: Gaussian-like states = $(n_gaussian)/$(K)")
    println(io, "           Sub-Gaussian states  = $(n_inter)/$(K)")
    println(io, "           Laplace-like states  = $(n_laplace)/$(K)")
    println(io, "           Super-Laplace states = $(n_super)/$(K)")
    println(io, "")
    println(io, "Headline interpretation of the bimodality test:")
    if n_gaussian > 0 && (n_laplace + n_super) > 0
        println(io, "  -> Bimodal: GED uses per-state shape adaptivity. Some states converge to")
        println(io, "     Gaussian-like (bulk regimes) and others to Laplace-like / heavier")
        println(io, "     (tail regimes). The hybrid intuition is supported by the data.")
    elseif n_gaussian == K
        println(io, "  -> Collapsed to Gaussian: data does not need heavy tails per state at K=$K.")
        println(io, "     CHMM-N is enough; GED adds no value.")
    elseif (n_laplace + n_super) == K
        println(io, "  -> Collapsed to Laplace-or-heavier: every state wants heavy tails;")
        println(io, "     CHMM-L (or CHMM-t) captures the same effect with one less parameter.")
    else
        println(io, "  -> Unimodal-intermediate: states cluster around p ≈ 1.5 (between Gaussian")
        println(io, "     and Laplace). GED is exploring the shape continuum, but no clear")
        println(io, "     hand-classified hybrid would dominate.")
    end
end

# ========================================================================================= #
# Diagnostic plot: p̂_k vs state rank (volatility), plus emission overlay
# ========================================================================================= #
_obs_c  = RGB(0.0, 0.447, 0.698);
_sim_c  = RGB(0.835, 0.369, 0.0);
_mid_c  = RGB(0.0, 0.620, 0.451);

p_a = scatter(1:K, p_k[order], ms=6, color=_sim_c, label="Fitted p̂_k",
    title="(a) CHMM-GED per-state shape p̂_k vs volatility rank | $TICKER, K=$K",
    titlefontsize=10, guidefontsize=10, tickfontsize=9, legendfontsize=8,
    xlabel="State rank (1 = calm, $K = crash)", ylabel="Fitted p̂_k",
    ylims=(0.4, 4.1));
hline!(p_a, [2.0], lw=2, ls=:dash, color=:black, label="p = 2 (Gaussian)");
hline!(p_a, [1.0], lw=2, ls=:dash, color=_obs_c, label="p = 1 (Laplace)");

# Emission overlay (density curves, color-graded by p̂_k).
x_lo = quantile(R_is, 0.001) * 1.2;
x_hi = quantile(R_is, 0.999) * 1.2;
x_grid = range(x_lo, x_hi, length=400);
colors_p = [RGB(clamp((2.0 - p)/1.5, 0.0, 1.0), 0.3, clamp((p - 0.5)/3.5, 0.0, 1.0)) for p in p_k];
p_b = plot(title="(b) CHMM-GED per-state emission densities (colored by p̂_k)",
    titlefontsize=10, guidefontsize=10, tickfontsize=9, legendfontsize=7,
    xlabel="Annualized excess log return G_t",
    ylabel="Probability density (arb. units)", legend=false);
histogram!(p_b, R_is, normalize=:pdf, bins=200, alpha=0.3, color=:lightgray, label="Observed IS");
for k in 1:K
    plot!(p_b, x_grid, pdf.(mG.emission[k], x_grid), lw=1.4, color=colors_p[k], alpha=0.85);
end
xlims!(p_b, x_lo, x_hi);

# IS density comparison: CHMM-GED vs CHMM-N vs CHMM-L vs observed.
p_c = plot(title="(c) IS return density: GED vs N vs L vs observed",
    titlefontsize=10, guidefontsize=10, tickfontsize=9, legendfontsize=8,
    xlabel="Annualized excess log return G_t",
    ylabel="Probability density (arb. units)");
histogram!(p_c, R_is, normalize=:pdf, bins=200, alpha=0.35, color=:lightgray, label="Observed IS");
density!(p_c, panel["N"].sim_is[:,1],   lw=2, color=_obs_c, alpha=0.85, label="CHMM-N");
density!(p_c, panel["L"].sim_is[:,1],   lw=2, color=_mid_c, alpha=0.85, label="CHMM-L");
density!(p_c, panel["GED"].sim_is[:,1], lw=2, color=_sim_c, alpha=0.85, label="CHMM-GED");
xlims!(p_c, x_lo, x_hi);

fig = plot(p_a, p_b, p_c, layout=(1,3), size=(1700,500),
    plot_title="CHMM-GED ablation | $TICKER | K=$K | $N_PATHS sims",
    plot_titlefontsize=11);
savefig(fig, joinpath(RESULTS_DIR, "Fig-GED-Diagnostics.svg"));
savefig(fig, joinpath(RESULTS_DIR, "Fig-GED-Diagnostics.pdf"));

println("\nDone. Outputs in: $RESULTS_DIR")
println("  - Table-GED-Ablation.txt        (4-family metric panel)")
println("  - GED-State-Diagnostics.txt     (per-state p̂_k bimodality test)")
println("  - Fig-GED-Diagnostics.{pdf,svg} (visual diagnostic)")
