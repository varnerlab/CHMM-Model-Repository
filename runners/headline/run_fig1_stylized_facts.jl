# ========================================================================================= #
# run_fig1_stylized_facts.jl
#
# Regenerates the Figure 1 stylized-facts panels (a)-(d) only, without running the full
# per-K analysis pipeline of run_all_analysis.jl. Extracted for the seventh-audit label
# correction: with RISK_FREE_RATE = 0 the analysed series is the annualised log growth
# rate, so the panel-(a) x-axis label "Excess Growth Rate" was stale relative to the
# conference paper's terminology. Everything except the axis label matches the original
# figure block verbatim (same data pipeline, bins, fits, axis limits, panel size, seed-free
# closed-form Gaussian/Laplace fits), so the panels are unchanged apart from the label.
#
# Output: results/SPY/stylized_facts/Fig-1-Stylized-Facts-{a,b,c,d}.{svg,pdf}
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Statistics, StatsBase, Distributions, DataFrames

const TICKER = "SPY";
const RETURN_LABEL = "Annualised Log Growth Rate";
const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const L = 252;            # ACF max lag
const RESULTS_DIR = joinpath(_ROOT, "results");

# --- Data (identical to run_all_analysis.jl) ---
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
ticker_idx = findfirst(==(TICKER), list_of_all_tickers);
R_is = all_firms_R[:, ticker_idx];
println("[setup] $TICKER IS observations: $(length(R_is))");

fig1_dir = joinpath(RESULTS_DIR, TICKER, "stylized_facts");
mkpath(fig1_dir);

# --- Gaussian and Laplace fits (identical) ---
μ_gauss = mean(R_is); σ_gauss = std(R_is);
d_gauss = Normal(μ_gauss, σ_gauss);
μ_lap = median(R_is); b_lap = mean(abs.(R_is .- μ_lap));
d_laplace = Laplace(μ_lap, b_lap);

# Percentile-based axis limits (identical)
x_lo_raw = quantile(R_is, 0.005); x_hi_raw = quantile(R_is, 0.995);
x_pad = 0.20 * (x_hi_raw - x_lo_raw);
x_lo = x_lo_raw - x_pad; x_hi = x_hi_raw + x_pad;
x_grid = range(x_lo, x_hi, length=1000);

_panel_size = (700, 500);

# Panel (a): Distribution — corrected x-axis label
p1 = histogram(R_is, normalize=:pdf, bins=200, alpha=0.4, color=:lightgray, label="Observed",
    xlabel=RETURN_LABEL, ylabel="Probability Density (AU)", size=_panel_size);
plot!(p1, x_grid, pdf.(d_gauss, x_grid), lw=2, color=:blue, label="Gaussian", ls=:dash);
plot!(p1, x_grid, pdf.(d_laplace, x_grid), lw=2, color=:red, label="Laplace");
xlims!(p1, x_lo, x_hi);
savefig(p1, joinpath(fig1_dir, "Fig-1-Stylized-Facts-a.svg"));
savefig(p1, joinpath(fig1_dir, "Fig-1-Stylized-Facts-a.pdf"));

# Panel (b): Q-Q (identical)
sorted_R = sort(R_is); n_r = length(sorted_R);
theo_q = [quantile(d_gauss, (i-0.5)/n_r) for i in 1:n_r];
qq_lo = min(minimum(theo_q), minimum(sorted_R)); qq_hi = max(maximum(theo_q), maximum(sorted_R));
p2 = scatter(theo_q, sorted_R, ms=1, alpha=0.5, color=:steelblue, label="",
    xlabel="Theoretical", ylabel="Sample", size=_panel_size);
plot!(p2, [qq_lo, qq_hi], [qq_lo, qq_hi], lw=2, color=:red, ls=:dash, label="45°");
savefig(p2, joinpath(fig1_dir, "Fig-1-Stylized-Facts-b.svg"));
savefig(p2, joinpath(fig1_dir, "Fig-1-Stylized-Facts-b.pdf"));

# Panel (c): Returns ACF (identical)
τ = 1:(L-1); ci = 2.576/sqrt(n_r);
acf_raw = autocor(R_is, τ);
p3 = plot(τ, acf_raw, linetype=:steppost, lw=2, color=:steelblue, label="ACF(Gₜ)",
    xlabel="Lag", ylabel="ACF", size=_panel_size);
plot!(p3, τ, ci.*ones(length(τ)), lw=1.5, color=:gray, ls=:dash, label="99% CI");
plot!(p3, τ, -ci.*ones(length(τ)), lw=1.5, color=:gray, ls=:dash, label="");
savefig(p3, joinpath(fig1_dir, "Fig-1-Stylized-Facts-c.svg"));
savefig(p3, joinpath(fig1_dir, "Fig-1-Stylized-Facts-c.pdf"));

# Panel (d): |Returns| ACF (identical)
acf_abs = autocor(abs.(R_is), τ);
p4 = plot(τ, acf_abs, linetype=:steppost, lw=2, color=:darkorange, label="ACF(|Gₜ|)",
    xlabel="Lag", ylabel="ACF", size=_panel_size);
plot!(p4, τ, ci.*ones(length(τ)), lw=1.5, color=:gray, ls=:dash, label="99% CI");
plot!(p4, τ, -ci.*ones(length(τ)), lw=1.5, color=:gray, ls=:dash, label="");
savefig(p4, joinpath(fig1_dir, "Fig-1-Stylized-Facts-d.svg"));
savefig(p4, joinpath(fig1_dir, "Fig-1-Stylized-Facts-d.pdf"));

println("[done] Saved Figure 1 panels (a-d) to $fig1_dir with corrected axis label");
