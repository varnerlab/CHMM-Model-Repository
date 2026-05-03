# =========================================================================== #
# run_oos_regime_trajectory.jl
#
# Peer-review item 17 (R2 Minor 5): per-state regime-probability trajectory on
# the SPY OoS window (2024-01-04 to 2026-04-20). Substantiates the
# "regime attenuation vs introduction" diagnosis in Section 5 by showing
# the OoS posterior Pr(s_t | F_t) under IS-fixed CHMM-N parameters.
#
# Output:
#   results/diagnostics/oos_regime_trajectory.csv
#   results/diagnostics/oos_regime_trajectory.txt
#   figs/Fig-OoS-Regime-Trajectory.{svg,pdf}
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, LinearAlgebra, Distributions, Printf, Plots, Dates;

const SEED      = 20260420;
const MAX_ITER  = 60;
const K         = 18;
const DT        = 1/252;
const RISK_FREE = 0.0;
const OUT_DIR   = joinpath(_ROOT, "results", "diagnostics");
const FIG_DIR   = joinpath(_ROOT, "figs");
mkpath(OUT_DIR); mkpath(FIG_DIR);
Random.seed!(SEED);

println("="^70);
println("  Item 17: OoS regime-probability trajectory under IS-fixed CHMM-N (K=$K)");
println("="^70);

# Load IS + OoS for SPY
train_dataset = MyPortfolioDataSet()["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet()["dataset"];
df_train = train_dataset["SPY"];
df_oos   = oos_dataset["SPY"];

R_is  = log_growth_matrix(train_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE);
R_oos = log_growth_matrix(oos_dataset,   "SPY"; Δt=DT, risk_free_rate=RISK_FREE);
@printf("[data] T_IS = %d   T_OoS = %d\n", length(R_is), length(R_oos));

# Fit CHMM-N on IS
println("[fit] CHMM-N at K=$K on IS...");
chmm = build(MyContinuousHiddenMarkovModel, (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
T_mat = zeros(K, K);
for i in 1:K; T_mat[i, :] = probs(chmm.transition[i]); end
μ = zeros(K); σ = zeros(K);
for k in 1:K
    d = chmm.emission[k];
    μ[k] = mean(d);
    σ[k] = std(d);
end
# π̄ as the stationary distribution from a high power of T.
π_bar = (T_mat^2000)[1, :];
π_bar ./= sum(π_bar);
@printf("[fit] T condition number = %.1f, ||π̄||1 = %.4f\n", cond(T_mat), sum(π_bar));

# Order states by emission variance (rank 1 = lowest σ, rank K = highest σ).
state_order = sortperm(σ);
σ_sorted = σ[state_order];
@printf("[states] σ_min = %.3f, σ_max = %.3f, ratio = %.1fx\n",
        σ_sorted[1], σ_sorted[end], σ_sorted[end] / σ_sorted[1]);

# --- Forward filter on the OoS series under IS-fixed (T, μ, σ) ---
# At each OoS day t, F_{t-1} = R_IS ∪ R_OoS[1:t-1].
# We initialise the filter with R_IS (so the prior at t=1 is the IS-end posterior).
# Then propagate through R_OoS, recording the filtered posterior Pr(s_t | F_t).
function _filter_posteriors(y::AbstractVector, T_mat::AbstractMatrix, μ::AbstractVector,
                            σ::AbstractVector, prior::AbstractVector)
    K = length(μ);
    n = length(y);
    post = zeros(n, K);
    p = copy(prior);
    @inbounds for t in 1:n
        b = [pdf(Normal(μ[k], σ[k]), y[t]) for k in 1:K];
        post_t = p .* b;
        Z = sum(post_t);
        if Z <= 0
            post_t .= p;
        else
            post_t ./= Z;
        end
        post[t, :] = post_t;
        # propagate to t+1: prior for next step
        p = vec(post_t' * T_mat);
    end
    return post;
end

# Run filter on IS first to get end-of-IS posterior.
post_is  = _filter_posteriors(R_is, T_mat, μ, σ, π_bar);
prior_oos = vec(post_is[end, :]' * T_mat);
post_oos = _filter_posteriors(R_oos, T_mat, μ, σ, prior_oos);

# Compute aggregate volatility-rank summary on OoS:
# at each t, mass in low-vol (rank 1-6), mid-vol (rank 7-12), high-vol (rank 13-18) bands.
function _band_mass(post::AbstractMatrix, state_order::Vector{Int}, lo::Int, hi::Int)
    states_in_band = state_order[lo:hi];
    return [sum(post[t, states_in_band]) for t in axes(post, 1)];
end

low_vol_oos  = _band_mass(post_oos, state_order, 1, 6);
mid_vol_oos  = _band_mass(post_oos, state_order, 7, 12);
high_vol_oos = _band_mass(post_oos, state_order, 13, K);

# Same on IS for comparison.
low_vol_is  = _band_mass(post_is, state_order, 1, 6);
mid_vol_is  = _band_mass(post_is, state_order, 7, 12);
high_vol_is = _band_mass(post_is, state_order, 13, K);

@printf("[OoS bands] mean low/mid/high = %.3f / %.3f / %.3f\n",
        mean(low_vol_oos), mean(mid_vol_oos), mean(high_vol_oos));
@printf("[IS  bands] mean low/mid/high = %.3f / %.3f / %.3f\n",
        mean(low_vol_is), mean(mid_vol_is), mean(high_vol_is));

# Stationary band masses for reference.
stat_low  = sum(π_bar[state_order[1:6]]);
stat_mid  = sum(π_bar[state_order[7:12]]);
stat_high = sum(π_bar[state_order[13:K]]);
@printf("[stat] low/mid/high = %.3f / %.3f / %.3f\n", stat_low, stat_mid, stat_high);

# OoS dates for the figure (one per R_oos entry: log returns are differences, so one
# fewer date; use df_oos.timestamp[2:end])
oos_dates = Date.(df_oos.timestamp[2:end]);
n_oos_used = min(length(oos_dates), length(R_oos));
oos_dates = oos_dates[1:n_oos_used];

# --- Plot stacked-area regime trajectory + observed |G_t| overlay ---
function _try_plot(plot_path_svg::String, plot_path_pdf::String)
    try
        gr();
        x = oos_dates;
        plt = plot(layout = (2, 1), size = (1100, 800), legend = :outertopright,
                   left_margin = 6Plots.mm, bottom_margin = 6Plots.mm);
        # top: stacked-area (low/mid/high vol bands)
        plot!(plt[1], x, low_vol_oos,  fillrange = 0,                       label = "low-vol  (σ-rank 1-6)",   linewidth = 0, fillalpha = 0.7);
        plot!(plt[1], x, low_vol_oos .+ mid_vol_oos,  fillrange = low_vol_oos,                       label = "mid-vol  (σ-rank 7-12)",  linewidth = 0, fillalpha = 0.7);
        plot!(plt[1], x, low_vol_oos .+ mid_vol_oos .+ high_vol_oos, fillrange = low_vol_oos .+ mid_vol_oos, label = "high-vol (σ-rank 13-18)", linewidth = 0, fillalpha = 0.7);
        title!(plt[1], "OoS regime-probability trajectory (CHMM-N K=$K, IS-fixed parameters)");
        ylabel!(plt[1], "Pr(σ-rank band)");
        ylims!(plt[1], 0, 1);
        # bottom: |G_t| with horizontal line at the observed OoS mean
        plot!(plt[2], x, abs.(R_oos[1:n_oos_used]), label = "|G_t|", color = :black, linewidth = 0.8);
        hline!(plt[2], [mean(abs.(R_oos))], label = "OoS mean |G_t|", linestyle = :dash, color = :red);
        ylabel!(plt[2], "|G_t| (annualised)");
        xlabel!(plt[2], "Date (OoS)");
        savefig(plt, plot_path_svg);
        savefig(plt, plot_path_pdf);
        println("[fig] $plot_path_svg, $plot_path_pdf");
    catch err
        @warn "Plot generation failed; continuing with CSV/TXT outputs only" exception=err;
    end
end
_try_plot(joinpath(FIG_DIR, "Fig-OoS-Regime-Trajectory.svg"),
          joinpath(FIG_DIR, "Fig-OoS-Regime-Trajectory.pdf"));

# --- CSV (full state-by-state posterior, in vol-rank order) ---
open(joinpath(OUT_DIR, "oos_regime_trajectory.csv"), "w") do io
    write(io, "date," * join(["rank_$(i)" for i in 1:K], ",") * "\n");
    for t in 1:n_oos_used
        write(io, string(oos_dates[t]) * ",");
        write(io, join([@sprintf("%.4f", post_oos[t, state_order[i]]) for i in 1:K], ","));
        write(io, "\n");
    end
end

# --- Text summary ---
open(joinpath(OUT_DIR, "oos_regime_trajectory.txt"), "w") do io
    println(io, "OoS regime-probability trajectory under IS-fixed CHMM-N at K=$K");
    println(io, "Peer-review item 17 (R2 Minor 5): substantiates the regime-attenuation/");
    println(io, "introduction diagnosis in the Discussion section.");
    println(io, "="^72);
    println(io);
    @printf(io, "Run config: seed=%d, max_iter=%d, IS T=%d, OoS T=%d\n",
            SEED, MAX_ITER, length(R_is), length(R_oos));
    println(io);
    println(io, "Volatility-rank band aggregates (mean Pr across the window):");
    println(io, "  band                | OoS mean | IS mean | stationary");
    println(io, "  --------------------|----------|---------|-----------");
    @printf(io, "  low-vol  rank  1-6  | %.3f    | %.3f   | %.3f\n", mean(low_vol_oos), mean(low_vol_is), stat_low);
    @printf(io, "  mid-vol  rank  7-12 | %.3f    | %.3f   | %.3f\n", mean(mid_vol_oos), mean(mid_vol_is), stat_mid);
    @printf(io, "  high-vol rank 13-18 | %.3f    | %.3f   | %.3f\n", mean(high_vol_oos), mean(high_vol_is), stat_high);
    println(io);
    println(io, "Reading:");
    println(io, "  Compare the OoS mean against the IS mean and the IS-stationary distribution.");
    println(io, "  The Discussion-section diagnosis predicts that the 2024-2026 OoS window is a");
    println(io, "  regime *attenuation* (return to a regime closer to the IS volatility level),");
    println(io, "  which on this band aggregation should manifest as OoS band masses inside the");
    println(io, "  IS-stationary band masses, with no novel high-vol-band mass that the IS");
    println(io, "  distribution does not span. By contrast, the W2 (COVID) and W4 (2022 rate-hike");
    println(io, "  onset) stress folds were regime *introductions* under the same diagnosis, and");
    println(io, "  the IS-fixed filter should drive most of the mass to the high-vol-band edge.");
    println(io);
    @printf(io, "  Empirical OoS-vs-stationary band ratios (>1 means OoS over-weights vs IS):\n");
    @printf(io, "    low / stat   = %.2f\n", mean(low_vol_oos)  / max(stat_low,  1e-9));
    @printf(io, "    mid / stat   = %.2f\n", mean(mid_vol_oos)  / max(stat_mid,  1e-9));
    @printf(io, "    high / stat  = %.2f\n", mean(high_vol_oos) / max(stat_high, 1e-9));
end

println("[done]");
println("  $OUT_DIR/oos_regime_trajectory.csv");
println("  $OUT_DIR/oos_regime_trajectory.txt");
