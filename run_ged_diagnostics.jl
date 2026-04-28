# ========================================================================================= #
# run_ged_diagnostics.jl
#
# CHMM-GED per-state shape diagnostics for the paper. Produces:
#   Fig-p-Histogram.{pdf,svg}    Two-panel paper-ready figure showing the bimodal
#                                distribution of fitted p_k across the K = 18 states
#                                on SPY: histogram with reference lines at p = 1
#                                (Laplace) and p = 2 (Gaussian), plus p_k vs state-
#                                rank ordered by variance-equivalent sigma. The
#                                analog of Fig-nu-Histogram for CHMM-t.
#   GED-Diagnostics-Table.txt    Per-state (μ_k, α_k, p_k, σ_eq) sorted by σ_eq with
#                                bimodality classification counts.
#   Fig-p-CrossTicker.{pdf,svg}  Stacked-bar of partition counts (Gaussian-like /
#                                intermediate / Laplace-like / super-Laplace) across
#                                {SPY, NVDA, JNJ, JPM, AAPL, QQQ}.
#
# Output: results/SPY/ged_diagnostics/
# Usage:  julia --project=. run_ged_diagnostics.jl
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");
using Random;
using SpecialFunctions: gamma;

const TICKER = "SPY";
const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const K = 18;
const MAX_ITER = 60;
const SEED = 20260420;
const TICKERS = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];

const RESULTS_DIR = joinpath(_ROOT, "results", "SPY", "ged_diagnostics");
mkpath(RESULTS_DIR);

const PAPER_FIGS = abspath(joinpath(_ROOT, "..", "CHMM-paper", "figs"));

println("="^70)
println("  CHMM-GED Per-State Shape Diagnostics  |  K = $K  |  seed = $SEED")
println("="^70)

# ========================================================================================= #
# Data loading
# ========================================================================================= #
println("\n[1/4] Loading panel data...");
train = MyPortfolioDataSet() |> x -> x["dataset"];
oos = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train["AAPL"]);
ds = Dict{String,DataFrame}();
for (t, data) in train; if nrow(data) == max_days; ds[t] = data; end; end
all_tickers = keys(ds) |> collect |> sort;
all_R = log_growth_matrix(ds, all_tickers; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);

function get_is(ticker::String)
    idx = findfirst(==(ticker), all_tickers);
    return Vector{Float64}(all_R[:, idx]);
end

R_is = get_is(TICKER);
println("  IS: $(length(R_is)) obs");

# ========================================================================================= #
# Helpers
# ========================================================================================= #
function _ged_sigma_eq(α, p)
    return α * sqrt(gamma(3.0/p) / gamma(1.0/p));
end

function classify_p(p_vec)
    n_g = count(p -> p >= 1.85, p_vec);
    n_i = count(p -> 1.30 <= p < 1.85, p_vec);
    n_l = count(p -> 0.85 <= p < 1.30, p_vec);
    n_s = count(p -> p < 0.85, p_vec);
    return (gauss=n_g, inter=n_i, lapl=n_l, super=n_s);
end

# ========================================================================================= #
# (1) Headline diagnostic on SPY
# ========================================================================================= #
println("\n[2/4] Fitting CHMM-GED on SPY at K = $K...");
Random.seed!(SEED);
m_spy = build(MyGEDHiddenMarkovModel,
    (observations=R_is, number_of_states=K, max_iter=MAX_ITER));

p_k = [m_spy.emission[k].p for k in 1:K];
α_k = [m_spy.emission[k].α for k in 1:K];
μ_k = [m_spy.emission[k].μ for k in 1:K];
σ_eq = [_ged_sigma_eq(α_k[k], p_k[k]) for k in 1:K];
order = sortperm(σ_eq);
cls_spy = classify_p(p_k);

println("  Fitted p_k range: [$(round(minimum(p_k), digits=3)), $(round(maximum(p_k), digits=3))], median = $(round(median(p_k), digits=3))");
println("  Partition: $(cls_spy.gauss) Gaussian-like, $(cls_spy.inter) intermediate, $(cls_spy.lapl) Laplace-like, $(cls_spy.super) super-Laplace");

# Per-state diagnostics table.
open(joinpath(RESULTS_DIR, "GED-Diagnostics-Table.txt"), "w") do io
    println(io, "CHMM-GED per-state shape diagnostics  ($TICKER, K = $K, seed = $SEED, p_bounds = (0.5, 3.0))");
    println(io, "States ordered by variance-equivalent sigma (calm to crash).");
    println(io, "p_k = 2 -> Gaussian-shape state. p_k = 1 -> Laplace-shape state.");
    println(io, "="^85);
    println(io, "rank | k  |  mu_k       |  alpha_k  |  p_k     |  sigma_eq |  shape interpretation");
    println(io, "-"^85);
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
        println(io, "$(lpad(rank,4)) | $(lpad(k,2)) | $(lpad(round(μ_k[k], digits=4), 10)) | $(lpad(round(α_k[k], digits=4), 8)) | $(lpad(round(p_k[k], digits=3), 7)) | $(lpad(round(σ_eq[k], digits=4), 8))  | $(shape_label)");
    end
    println(io, "="^85);
    println(io, "");
    println(io, "Aggregate partition:");
    println(io, "  Gaussian-like (p >= 1.85):     $(cls_spy.gauss) / $K");
    println(io, "  Intermediate (1.30 <= p < 1.85): $(cls_spy.inter) / $K");
    println(io, "  Laplace-like (0.85 <= p < 1.30): $(cls_spy.lapl) / $K");
    println(io, "  Super-Laplace (p < 0.85):      $(cls_spy.super) / $K");
end

# ========================================================================================= #
# (2) Fig-p-Histogram: paper-ready 2-panel figure
# ========================================================================================= #
println("\n[3/4] Building Fig-p-Histogram (2-panel)...");

# Okabe-Ito palette to match the rest of the paper figures.
_obs_c  = RGB(0.0, 0.447, 0.698);          # blue (observed / reference)
_sim_c  = RGB(0.835, 0.369, 0.0);          # vermillion (Laplace reference)
_mid_c  = RGB(0.0, 0.620, 0.451);          # bluish green (Gaussian reference)
_bar_c  = RGB(0.337, 0.337, 0.337);        # neutral gray for histogram bars
_style  = (titlefontsize=10, guidefontsize=10, tickfontsize=9, legendfontsize=8);

# Panel (a): histogram of fitted p_k.
p_a = histogram(p_k, bins=range(0.5, 3.05, length=22),
    color=_bar_c, alpha=0.7, lw=0, label="",
    title="Per-state p_k distribution (SPY, K = $K)",
    xlabel="Fitted shape parameter p_k", ylabel="Count (out of $K states)",
    legend=:topleft; _style...);
vline!(p_a, [1.0], lw=2.5, ls=:dash, color=_sim_c, label="p = 1 (Laplace)");
vline!(p_a, [2.0], lw=2.5, ls=:dash, color=_mid_c, label="p = 2 (Gaussian)");
xlims!(p_a, 0.5, 3.1);

# Panel (b): p_k vs volatility rank scatter with bimodal annotation.
p_b = scatter(1:K, p_k[order], ms=7, color=_obs_c,
    title="p_k vs volatility rank (1 = calm, $K = crash)",
    xlabel="State rank ordered by sigma_equivalent",
    ylabel="Fitted shape parameter p_k",
    legend=:topright, label="Fitted p_k"; _style...);
hline!(p_b, [2.0], lw=2.0, ls=:dash, color=_mid_c, label="p = 2 (Gaussian)");
hline!(p_b, [1.0], lw=2.0, ls=:dash, color=_sim_c, label="p = 1 (Laplace)");
xlims!(p_b, 0.5, K + 0.5);
ylims!(p_b, 0.4, 3.1);

fig = plot(p_a, p_b, layout=(1, 2), size=(1300, 450));

savefig(fig, joinpath(RESULTS_DIR, "Fig-p-Histogram.pdf"));
savefig(fig, joinpath(RESULTS_DIR, "Fig-p-Histogram.svg"));
println("  Saved Fig-p-Histogram.{pdf,svg}");

# ========================================================================================= #
# (3) Cross-ticker partition diagnostic
# ========================================================================================= #
println("\n[4/4] Cross-ticker shape partition...");

cross_counts = Vector{NamedTuple}();
for (i, t) in enumerate(TICKERS)
    R = get_is(t);
    Random.seed!(SEED);
    m_t = build(MyGEDHiddenMarkovModel,
        (observations=R, number_of_states=K, max_iter=MAX_ITER));
    p_vec = [m_t.emission[k].p for k in 1:K];
    cls = classify_p(p_vec);
    push!(cross_counts, (ticker=t, gauss=cls.gauss, inter=cls.inter, lapl=cls.lapl, super=cls.super));
    println("  $t: G=$(cls.gauss) I=$(cls.inter) L=$(cls.lapl) S=$(cls.super)");
end

# Stacked-bar diagnostic via StatsPlots.groupedbar.
labels_x = [r.ticker for r in cross_counts];
mat = hcat(
    [r.gauss for r in cross_counts],
    [r.inter for r in cross_counts],
    [r.lapl  for r in cross_counts],
    [r.super for r in cross_counts]
);
p_ct = groupedbar(labels_x, mat, bar_position=:stack, lw=0,
    color=[_mid_c :gold _sim_c :firebrick],
    label=["Gaussian-like (p >= 1.85)" "Intermediate (1.30 <= p < 1.85)" "Laplace-like (0.85 <= p < 1.30)" "Super-Laplace (p < 0.85)"],
    title="CHMM-GED state-shape partition across tickers  |  K = $K",
    xlabel="Ticker", ylabel="Count of states (out of $K)",
    legend=:outerright, size=(900, 420); _style...);

savefig(p_ct, joinpath(RESULTS_DIR, "Fig-p-CrossTicker.pdf"));
savefig(p_ct, joinpath(RESULTS_DIR, "Fig-p-CrossTicker.svg"));

# Append cross-ticker summary to the diagnostics text.
open(joinpath(RESULTS_DIR, "GED-Diagnostics-Table.txt"), "a") do io
    println(io, "");
    println(io, "Cross-ticker partition (K = $K, seed = $SEED):");
    println(io, "-"^60);
    println(io, "ticker | nGauss | nInter | nLapl | nSuper | total heavy");
    for r in cross_counts
        heavy = r.lapl + r.super;
        println(io, "$(rpad(r.ticker, 6)) | $(lpad(r.gauss, 6)) | $(lpad(r.inter, 6)) | $(lpad(r.lapl, 5)) | $(lpad(r.super, 6)) | $(lpad(heavy, 11))");
    end
end

# ========================================================================================= #
# Copy headline figure to paper figs/
# ========================================================================================= #
if isdir(PAPER_FIGS)
    cp(joinpath(RESULTS_DIR, "Fig-p-Histogram.pdf"),
       joinpath(PAPER_FIGS, "Fig-p-Histogram.pdf"); force=true);
    cp(joinpath(RESULTS_DIR, "Fig-p-Histogram.svg"),
       joinpath(PAPER_FIGS, "Fig-p-Histogram.svg"); force=true);
    cp(joinpath(RESULTS_DIR, "Fig-p-CrossTicker.pdf"),
       joinpath(PAPER_FIGS, "Fig-p-CrossTicker.pdf"); force=true);
    cp(joinpath(RESULTS_DIR, "Fig-p-CrossTicker.svg"),
       joinpath(PAPER_FIGS, "Fig-p-CrossTicker.svg"); force=true);
    println("\nCopied Fig-p-Histogram and Fig-p-CrossTicker to: $PAPER_FIGS");
else
    println("\n(Skipped paper copy: $PAPER_FIGS not found)");
end

println("\nDone. Outputs in: $RESULTS_DIR")
println("  - Fig-p-Histogram.{pdf,svg}      (paper headline GED diagnostic)")
println("  - Fig-p-CrossTicker.{pdf,svg}    (cross-ticker partition)")
println("  - GED-Diagnostics-Table.txt      (per-state + cross-ticker text dump)")
