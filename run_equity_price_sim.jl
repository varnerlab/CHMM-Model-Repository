# ========================================================================================= #
# run_equity_price_sim.jl
#
# Trains the three continuous HMM families (Gaussian, Student-t, Laplace) on each
# main-study ticker's in-sample returns and generates equity *price* simulations
# over the out-of-sample horizon using the new public API
# (`simulate_returns`, `simulate_prices`).
#
# Outputs (results/equity_price_sim/):
#   - summary.csv                                 : per-ticker per-family summary stats
#   - Fig-{TICKER}-PriceFan-{Family}.svg          : simulated price fan + observed path
#   - Fig-{TICKER}-TerminalDist-{Family}.svg      : terminal price distribution histogram
#
# Uses the global seed below for reproducibility.
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
const EPS_SEED    = 20260421;
Random.seed!(EPS_SEED);

const TICKERS      = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];
const RISK_FREE    = 0.0;
const DT           = 1/252;
const N_PATHS      = 500;
const K_MAIN       = 18;
const MAX_ITER     = 60;

const OUT_DIR = joinpath(_ROOT, "results", "equity_price_sim");
mkpath(OUT_DIR);

println("="^72)
println("  Equity price simulation from CHMM family fits")
println("  Seed:      $EPS_SEED")
println("  Tickers:   ", join(TICKERS, ", "))
println("  K:         $K_MAIN")
println("  Paths:     $N_PATHS")
println("="^72)

# ------------------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------------------ #
println("\n[data] Loading training + out-of-sample datasets...")
train_raw = MyPortfolioDataSet() |> x -> x["dataset"];
oos_raw   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];

# Keep only tickers with the full daily panel, matching the convention used in
# the baseline pipelines.
max_days = nrow(train_raw["AAPL"]);
train_dataset = Dict{String,DataFrame}();
for (t, df) in train_raw
    if nrow(df) == max_days; train_dataset[t] = df; end
end

# ------------------------------------------------------------------------------ #
# Per-family simulation helper (uses the new public API)
# ------------------------------------------------------------------------------ #
function _fit_and_simulate(family::Symbol, R_is::Vector{Float64}, S0::Float64,
                           n_oos::Int; K::Int=K_MAIN, max_iter::Int=MAX_ITER,
                           n_paths::Int=N_PATHS)

    nt = (observations=R_is, number_of_states=K, max_iter=max_iter);
    if family === :Gaussian
        chmm = build(MyContinuousHiddenMarkovModel, nt);
    elseif family === :StudentT
        chmm = build(MyStudentTHiddenMarkovModel, nt);
    elseif family === :Laplace
        chmm = build(MyLaplaceHiddenMarkovModel, nt);
    else
        error("Unknown family: $family");
    end

    P_paths = simulate_prices(chmm, S0, n_oos;
        Δt=DT, risk_free_rate=RISK_FREE, n_paths=n_paths);

    return chmm, P_paths;
end

# ------------------------------------------------------------------------------ #
# Figure helpers
# ------------------------------------------------------------------------------ #
function _price_fan_plot(ticker::String, family_label::String,
                         P_obs::Vector{Float64}, P_paths::Matrix{Float64})
    n_steps = size(P_paths, 1) - 1;
    xs = 0:n_steps;

    # Quantile bands across simulated paths.
    q05 = [quantile(P_paths[i, :], 0.05) for i in 1:(n_steps + 1)];
    q25 = [quantile(P_paths[i, :], 0.25) for i in 1:(n_steps + 1)];
    q50 = [quantile(P_paths[i, :], 0.50) for i in 1:(n_steps + 1)];
    q75 = [quantile(P_paths[i, :], 0.75) for i in 1:(n_steps + 1)];
    q95 = [quantile(P_paths[i, :], 0.95) for i in 1:(n_steps + 1)];

    title_text = "Fig 6 (Pipeline A). $ticker OoS price fan | CHMM-$family_label, K=$K_MAIN\n" *
                 "$(size(P_paths,2)) simulated paths | S0 = last IS VWAP | dt = 1/252";
    p = plot(xs, q50, label="Simulated median", lw=2, c=:blue,
             title=title_text, titlefontsize=9,
             xlabel="Trading day after last IS close (out-of-sample)", ylabel="Price (USD)");
    plot!(p, xs, q05, fillrange=q95, fillalpha=0.15, c=:blue, label="Simulated 5-95 percentile band", lw=0);
    plot!(p, xs, q25, fillrange=q75, fillalpha=0.25, c=:blue, label="Simulated 25-75 percentile band", lw=0);

    # Overlay observed OoS price track (truncate/pad to min length).
    n_show = min(length(P_obs), n_steps + 1);
    plot!(p, 0:(n_show - 1), P_obs[1:n_show], label="Observed OoS price path", lw=2, c=:red);

    return p;
end

function _terminal_hist_plot(ticker::String, family_label::String,
                             P_paths::Matrix{Float64}, P_obs_end::Float64)
    terminal = P_paths[end, :];
    n_steps = size(P_paths, 1) - 1;
    title_text = "Fig 6 (Pipeline A). $ticker terminal-price distribution | CHMM-$family_label, K=$K_MAIN\n" *
                 "$(length(terminal)) simulated paths at day $n_steps OoS";
    p = histogram(terminal, bins=40, legend=:topright, label="Simulated terminal prices",
                  title=title_text, titlefontsize=9,
                  xlabel="Terminal price at end of OoS window (USD)", ylabel="Count",
                  fillalpha=0.6, c=:steelblue);
    vline!(p, [P_obs_end], lw=2, c=:red, label="Observed terminal");
    vline!(p, [median(terminal)], lw=2, c=:blue, ls=:dash, label="Simulated median");
    return p;
end

# ------------------------------------------------------------------------------ #
# Path-level probabilistic-forecast metrics
# ------------------------------------------------------------------------------ #

"""
    _band_coverage_rate(P_obs, P_paths; α=0.05) -> Float64

Fraction of OoS time steps at which the observed price P_obs[t] lies inside
the simulated [q_α, q_{1-α}] band across paths. t=1 is the shared start price
(trivially inside) and is excluded.
"""
function _band_coverage_rate(P_obs::Vector{Float64}, P_paths::Matrix{Float64}; α::Float64=0.05)::Float64
    n = length(P_obs);
    @assert size(P_paths, 1) == n;
    hits = 0;
    for t in 2:n
        q_lo = quantile(view(P_paths, t, :), α);
        q_hi = quantile(view(P_paths, t, :), 1 - α);
        if q_lo <= P_obs[t] <= q_hi
            hits += 1;
        end
    end
    return hits / (n - 1);
end

"""
    _median_mape(P_obs, P_paths) -> Float64

Mean absolute percentage error between the observed OoS price track and the
cross-path median at each time step (t=1 excluded: both are S0).
"""
function _median_mape(P_obs::Vector{Float64}, P_paths::Matrix{Float64})::Float64
    n = length(P_obs);
    err = 0.0;
    for t in 2:n
        med_t = quantile(view(P_paths, t, :), 0.5);
        err += abs(P_obs[t] - med_t) / P_obs[t];
    end
    return err / (n - 1);
end

"""
    _crps_sample(y, X) -> Float64

Bias-corrected sample-CRPS (Gneiting & Raftery 2007):
    CRPS = (1/N) Σ_i |X_i - y| - (1/(2N(N-1))) Σ_{i≠j} |X_i - X_j|.
Strictly proper scoring rule; smaller is better.
"""
function _crps_sample(y::Float64, X::AbstractVector{Float64})::Float64
    N = length(X);
    term1 = mean(abs.(X .- y));
    s = 0.0;
    @inbounds for i in 1:N, j in 1:N
        if i != j
            s += abs(X[i] - X[j]);
        end
    end
    term2 = s / (2.0 * N * (N - 1));
    return term1 - term2;
end

"""
    _mean_crps(P_obs, P_paths) -> Float64

Horizon-average CRPS between the observed OoS price P_obs[t] and the
simulated ensemble P_paths[t, :]. t=1 (S0) is excluded.
"""
function _mean_crps(P_obs::Vector{Float64}, P_paths::Matrix{Float64})::Float64
    n = length(P_obs);
    acc = 0.0;
    for t in 2:n
        acc += _crps_sample(P_obs[t], view(P_paths, t, :));
    end
    return acc / (n - 1);
end

# ------------------------------------------------------------------------------ #
# Main loop
# ------------------------------------------------------------------------------ #
summary_rows = DataFrame(
    ticker   = String[],
    family   = String[],
    S0       = Float64[],
    n_oos    = Int[],
    n_paths  = Int[],
    P_obs_end  = Float64[],
    q05_end    = Float64[],
    q50_end    = Float64[],
    q95_end    = Float64[],
    mean_end   = Float64[],
    std_end    = Float64[],
    hit_rate_90 = Float64[],       # terminal-only 90% band indicator
    cov90_path  = Float64[],       # fraction of OoS steps where observed is inside the simulated 90% band
    median_mape = Float64[],       # mean absolute percentage error between observed path and simulated median
    mean_crps   = Float64[],       # horizon-average CRPS (Gneiting & Raftery 2007)
    crps_rel_S0 = Float64[],       # CRPS divided by S0 (scale-free)
);

families = (:Gaussian, :StudentT, :Laplace);
family_labels = Dict(:Gaussian=>"N", :StudentT=>"t", :Laplace=>"L");

for ticker in TICKERS
    if !haskey(train_dataset, ticker) || !haskey(oos_raw, ticker)
        println("[warn] $ticker missing from train or OoS dataset. Skipping.");
        continue;
    end

    println("\n[ticker] $ticker");

    R_is  = log_growth_matrix(train_dataset, ticker; Δt=DT, risk_free_rate=RISK_FREE);
    R_oos = log_growth_matrix(oos_raw,       ticker; Δt=DT, risk_free_rate=RISK_FREE);

    # S0 = last in-sample close (vwap); observed OoS price track starts there.
    S0 = Float64(train_dataset[ticker][end, :volume_weighted_average_price]);
    P_obs = Vector{Float64}(undef, length(R_oos) + 1);
    P_obs[1] = S0;
    for t in eachindex(R_oos)
        P_obs[t + 1] = P_obs[t] * exp((R_oos[t] + RISK_FREE) * DT);
    end

    n_is = length(R_is); n_oos = length(R_oos);
    println("  IS days:  $n_is,  OoS days: $n_oos,  S0 = $(round(S0, digits=2))");

    for fam in families
        fam_label = family_labels[fam];
        Random.seed!(EPS_SEED);  # same RNG state per family for fair comparison
        print("  fitting CHMM-$fam_label ... ");
        chmm, P_paths = _fit_and_simulate(fam, R_is, S0, n_oos);
        println("done ($(size(P_paths, 2)) paths, $(size(P_paths, 1)) steps).");

        # Summary stats at the terminal step.
        terminal = P_paths[end, :];
        q05_end = quantile(terminal, 0.05);
        q50_end = quantile(terminal, 0.50);
        q95_end = quantile(terminal, 0.95);
        mu_end  = mean(terminal);
        sd_end  = std(terminal);
        hit_90  = (q05_end <= P_obs[end] <= q95_end) ? 1.0 : 0.0;

        # Path-level probabilistic-forecast metrics.
        cov90 = _band_coverage_rate(P_obs, P_paths; α=0.05);
        mape  = _median_mape(P_obs, P_paths);
        crps  = _mean_crps(P_obs, P_paths);
        crps_rel = crps / S0;

        push!(summary_rows, (
            ticker, String(fam), S0, n_oos, size(P_paths, 2),
            P_obs[end], q05_end, q50_end, q95_end, mu_end, sd_end, hit_90,
            cov90, mape, crps, crps_rel
        ));

        # Figures (SVG for inspection, PDF for inclusion in the paper).
        fan_fig = _price_fan_plot(ticker, fam_label, P_obs, P_paths);
        savefig(fan_fig, joinpath(OUT_DIR, "Fig-$ticker-PriceFan-$fam_label.svg"));
        savefig(fan_fig, joinpath(OUT_DIR, "Fig-$ticker-PriceFan-$fam_label.pdf"));

        term_fig = _terminal_hist_plot(ticker, fam_label, P_paths, P_obs[end]);
        savefig(term_fig, joinpath(OUT_DIR, "Fig-$ticker-TerminalDist-$fam_label.svg"));
        savefig(term_fig, joinpath(OUT_DIR, "Fig-$ticker-TerminalDist-$fam_label.pdf"));
    end
end

# ------------------------------------------------------------------------------ #
# Write summary CSV
# ------------------------------------------------------------------------------ #
summary_path = joinpath(OUT_DIR, "summary.csv");
CSV.write(summary_path, summary_rows);
println("\n[done] Summary written to: $summary_path");
println("[done] Figures in:          $OUT_DIR");

# Quick console table.
println("\nPer-ticker terminal-price summary (5/50/95 %ile, observed, 90% band hit):");
for row in eachrow(summary_rows)
    println(
        "  $(rpad(row.ticker,5)) $(rpad(row.family,9)) ",
        "obs=$(rpad(round(row.P_obs_end,digits=2),8)) ",
        "q05=$(rpad(round(row.q05_end,digits=2),8)) ",
        "q50=$(rpad(round(row.q50_end,digits=2),8)) ",
        "q95=$(rpad(round(row.q95_end,digits=2),8)) ",
        "hit90=$(Int(row.hit_rate_90))"
    );
end

println("\nPath-level probabilistic-forecast metrics (full OoS horizon):");
println("  ticker family     cov90      MAPE     CRPS      CRPS/S0");
for row in eachrow(summary_rows)
    println(
        "  $(rpad(row.ticker,6)) $(rpad(row.family,9)) ",
        "$(rpad(round(row.cov90_path, digits=3), 9)) ",
        "$(rpad(round(row.median_mape, digits=4), 9)) ",
        "$(rpad(round(row.mean_crps, digits=3), 9)) ",
        "$(round(row.crps_rel_S0, digits=4))"
    );
end
