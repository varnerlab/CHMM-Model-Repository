# ========================================================================================= #
# run_equity_price_sim_vix.jl
#
# VIX-conditioned single-asset equity price forecasting (Phases 1, 2, 3, and narrow 5
# of Plan_VIX_CHMM_CrossAsset_Forecasting.md).
#
# Pipeline per ticker:
#   1. Train CHMM-N on VIX log-returns (full VIX IS window).
#   2. Train CHMM-{N, t, L} on the ticker's excess growth returns (equity IS window).
#   3. Viterbi-decode both models on the date-aligned equity IS window.
#   4. Fit the per-ticker coupling matrix C[s^V, s^E] with lag h = 1 and Laplace smoothing.
#   5. Initialize the VIX state from the last in-sample forward-filter posterior π^V_T.
#   6. Simulate n_paths equity price paths via simulate_prices_vix_conditioned.
#   7. Evaluate against the observed OoS price track using the same battery as §4.10:
#      terminal 90% band hit, full-horizon C̄_{0.90}, median MAPE, CRPS, CRPS/S0.
#
# Outputs (results/equity_price_sim_vix/):
#   - summary.csv                                   : per-ticker per-family metrics
#   - Fig-VIX-{TICKER}-PriceFan-{Family}.{svg,pdf}
#   - Fig-VIX-{TICKER}-TerminalDist-{Family}.{svg,pdf}
#   - Fig-VIX-Coupling-{TICKER}.{svg,pdf}           : heatmap of the coupling matrix
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
const EPS_SEED   = 20260421;
Random.seed!(EPS_SEED);

const TICKERS   = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];
const RISK_FREE = 0.0;
const DT        = 1/252;
const N_PATHS   = 500;
const K_MAIN    = 18;
const MAX_ITER  = 60;
const LAG_H     = 1;
const LAPLACE_ALPHA = 1.0;

const OUT_DIR = joinpath(_ROOT, "results", "equity_price_sim_vix");
mkpath(OUT_DIR);

println("="^72)
println("  VIX-conditioned equity price forecasting")
println("  Seed:        $EPS_SEED")
println("  Tickers:     ", join(TICKERS, ", "))
println("  K:           $K_MAIN    lag: $LAG_H")
println("  Paths:       $N_PATHS")
println("="^72)

# ------------------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------------------ #
println("\n[data] Loading VIX + equity training / OoS datasets...")
vix_is_df = MyVolatilityDataSet()["dataset"]["VIX"];
vix_is_df = sort(vix_is_df, :date);
vix_R_is  = log_growth_matrix(vix_is_df; Δt=DT, risk_free_rate=RISK_FREE, keycol=:close);
vix_dates_ret = vix_is_df.date[2:end];  # date of each VIX return (prices at t-1, t)

train_raw = MyPortfolioDataSet() |> x -> x["dataset"];
oos_raw   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_raw["AAPL"]);
train_dataset = Dict{String, DataFrame}();
for (t, df) in train_raw
    if nrow(df) == max_days; train_dataset[t] = df; end
end

println("  VIX IS returns:      $(length(vix_R_is)) days ($(first(vix_dates_ret)) .. $(last(vix_dates_ret)))")
println("  Equity IS available: $(max_days) days per ticker")

# ------------------------------------------------------------------------------ #
# Train VIX CHMM-N once; reuse across tickers.
# ------------------------------------------------------------------------------ #
println("\n[vix] Training CHMM-N on VIX log-returns (K = $K_MAIN)...")
Random.seed!(EPS_SEED);
chmm_vix = build(MyContinuousHiddenMarkovModel,
    (observations=vix_R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
println("  VIX CHMM-N converged in $(length(chmm_vix.log_likelihood_history)) iters.")

# Full forward-filter on VIX IS and full VIX Viterbi decoding for alignment.
println("[vix] Forward-filtering VIX IS and decoding Viterbi states...")
vix_filter = forward_filter(chmm_vix, vix_R_is);
vix_states_full = viterbi(vix_R_is, chmm_vix);
pi_V_init = vix_filter[end, :];   # π^V_T : posterior at end of VIX IS
println("  argmax(π^V_T) = state ", argmax(pi_V_init),
        ";  max prob = ", round(maximum(pi_V_init), digits=3))

# ------------------------------------------------------------------------------ #
# Coupling-matrix heatmap helper (one per ticker, from CHMM-N equity decode).
# ------------------------------------------------------------------------------ #
function _coupling_heatmap(ticker::String, C::Matrix{Float64})
    K_v, K_e = size(C);
    hm = heatmap(1:K_e, 1:K_v, C;
                 xlabel="Equity regime", ylabel="VIX regime",
                 title="$ticker  coupling C[s^V, s^E]  (lag h=$LAG_H, Laplace α=$LAPLACE_ALPHA)",
                 titlefontsize=10, colorbar_title="P(S^E | S^V)",
                 yflip=true, c=:viridis);
    return hm;
end

# ------------------------------------------------------------------------------ #
# Evaluation helpers reused from run_equity_price_sim.jl (path-level metrics).
# ------------------------------------------------------------------------------ #
function _band_cov(P_obs::Vector{Float64}, P_paths::Matrix{Float64}; α::Float64=0.05)
    n = length(P_obs); hits = 0;
    for t in 2:n
        q_lo = quantile(view(P_paths, t, :), α);
        q_hi = quantile(view(P_paths, t, :), 1 - α);
        if q_lo <= P_obs[t] <= q_hi; hits += 1; end
    end
    return hits / (n - 1);
end
function _median_mape(P_obs::Vector{Float64}, P_paths::Matrix{Float64})
    n = length(P_obs); err = 0.0;
    for t in 2:n
        err += abs(P_obs[t] - quantile(view(P_paths, t, :), 0.5)) / P_obs[t];
    end
    return err / (n - 1);
end
function _crps_sample(y::Float64, X::AbstractVector{Float64})
    N = length(X);
    term1 = mean(abs.(X .- y));
    s = 0.0;
    @inbounds for i in 1:N, j in 1:N
        if i != j; s += abs(X[i] - X[j]); end
    end
    return term1 - s / (2.0 * N * (N - 1));
end
function _mean_crps(P_obs::Vector{Float64}, P_paths::Matrix{Float64})
    n = length(P_obs); acc = 0.0;
    for t in 2:n; acc += _crps_sample(P_obs[t], view(P_paths, t, :)); end
    return acc / (n - 1);
end

# ------------------------------------------------------------------------------ #
# Price-fan + terminal-histogram helpers (mirror §4.10 styling).
# ------------------------------------------------------------------------------ #
function _price_fan_plot(ticker::String, family_label::String,
                         P_obs::Vector{Float64}, P_paths::Matrix{Float64})
    n_steps = size(P_paths, 1) - 1;
    xs = 0:n_steps;
    q05 = [quantile(P_paths[i, :], 0.05) for i in 1:(n_steps + 1)];
    q25 = [quantile(P_paths[i, :], 0.25) for i in 1:(n_steps + 1)];
    q50 = [quantile(P_paths[i, :], 0.50) for i in 1:(n_steps + 1)];
    q75 = [quantile(P_paths[i, :], 0.75) for i in 1:(n_steps + 1)];
    q95 = [quantile(P_paths[i, :], 0.95) for i in 1:(n_steps + 1)];
    title_text = "$ticker - VIX-cond. CHMM-$family_label (K=$K_MAIN, $(size(P_paths,2)) paths, lag=$LAG_H)";
    p = plot(xs, q50, label="Median", lw=2, c=:darkgreen,
             title=title_text, titlefontsize=10,
             xlabel="Trading day (out-of-sample)", ylabel="Price");
    plot!(p, xs, q05, fillrange=q95, fillalpha=0.15, c=:green, label="5-95% band", lw=0);
    plot!(p, xs, q25, fillrange=q75, fillalpha=0.25, c=:green, label="25-75% band", lw=0);
    n_show = min(length(P_obs), n_steps + 1);
    plot!(p, 0:(n_show - 1), P_obs[1:n_show], label="Observed", lw=2, c=:red);
    return p;
end

function _terminal_hist_plot(ticker::String, family_label::String,
                             P_paths::Matrix{Float64}, P_obs_end::Float64)
    terminal = P_paths[end, :];
    title_text = "$ticker - VIX-cond. CHMM-$family_label terminal (n=$(length(terminal)))";
    p = histogram(terminal, bins=40, legend=:topright, label="Simulated",
                  title=title_text, titlefontsize=10,
                  xlabel="Terminal price", ylabel="Count",
                  fillalpha=0.6, c=:seagreen);
    vline!(p, [P_obs_end], lw=2, c=:red, label="Observed terminal");
    vline!(p, [median(terminal)], lw=2, c=:darkgreen, ls=:dash, label="Simulated median");
    return p;
end

# ------------------------------------------------------------------------------ #
# Main loop
# ------------------------------------------------------------------------------ #
summary_rows = DataFrame(
    ticker = String[], family = String[],
    S0 = Float64[], n_oos = Int[], n_paths = Int[],
    P_obs_end = Float64[], q05_end = Float64[], q50_end = Float64[], q95_end = Float64[],
    mean_end = Float64[], std_end = Float64[],
    hit_rate_90 = Float64[],
    cov90_path = Float64[], median_mape = Float64[], mean_crps = Float64[], crps_rel_S0 = Float64[],
    coupling_mean_entropy = Float64[],
);

families = (:Gaussian, :StudentT, :Laplace);
family_labels = Dict(:Gaussian => "N", :StudentT => "t", :Laplace => "L");

for ticker in TICKERS
    if !haskey(train_dataset, ticker) || !haskey(oos_raw, ticker)
        println("[warn] $ticker missing from train or OoS dataset. Skipping.");
        continue;
    end

    println("\n[ticker] $ticker");

    eq_is_df  = train_dataset[ticker];
    R_is      = log_growth_matrix(train_dataset, ticker; Δt=DT, risk_free_rate=RISK_FREE);
    R_oos     = log_growth_matrix(oos_raw,       ticker; Δt=DT, risk_free_rate=RISK_FREE);
    S0        = Float64(eq_is_df[end, :volume_weighted_average_price]);
    eq_dates_ret = Date.(eq_is_df.timestamp[2:end]);

    # Align equity IS dates with VIX IS dates.
    vix_date_index = Dict(d => i for (i, d) in enumerate(vix_dates_ret));
    vix_states_aligned = Int[]; eq_mask = Bool[];
    for d in eq_dates_ret
        if haskey(vix_date_index, d)
            push!(vix_states_aligned, vix_states_full[vix_date_index[d]]);
            push!(eq_mask, true);
        else
            push!(eq_mask, false);
        end
    end
    n_aligned = length(vix_states_aligned);
    println("  Equity IS: $(length(R_is)) returns, VIX-aligned: $n_aligned days");

    # Observed OoS price track.
    P_obs = Vector{Float64}(undef, length(R_oos) + 1);
    P_obs[1] = S0;
    for t in eachindex(R_oos)
        P_obs[t + 1] = P_obs[t] * exp((R_oos[t] + RISK_FREE) * DT);
    end
    n_oos = length(R_oos);
    println("  OoS days: $n_oos,  S0 = $(round(S0, digits=2))")

    for fam in families
        fam_label = family_labels[fam];
        Random.seed!(EPS_SEED);
        print("  CHMM-$fam_label ... ");

        nt = (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER);
        chmm_eq = if fam === :Gaussian
            build(MyContinuousHiddenMarkovModel, nt);
        elseif fam === :StudentT
            build(MyStudentTHiddenMarkovModel, nt);
        else
            build(MyLaplaceHiddenMarkovModel, nt);
        end;

        # Equity Viterbi on full IS, then subset to VIX-aligned days.
        eq_states_full = viterbi(R_is, chmm_eq);
        eq_states_aligned = eq_states_full[eq_mask];
        @assert length(eq_states_aligned) == n_aligned "alignment length mismatch";

        # Fit coupling with lag h=1 and save heatmap once per (ticker, CHMM-N).
        C = fit_coupling(vix_states_aligned, eq_states_aligned, K_MAIN, K_MAIN;
                         lag=LAG_H, alpha=LAPLACE_ALPHA);
        C_entropy = mean(coupling_entropy(C));

        if fam === :Gaussian
            hm = _coupling_heatmap(ticker, C);
            savefig(hm, joinpath(OUT_DIR, "Fig-VIX-Coupling-$ticker.svg"));
            savefig(hm, joinpath(OUT_DIR, "Fig-VIX-Coupling-$ticker.pdf"));
        end

        # Simulate VIX-conditioned price paths.
        P_paths = simulate_prices_vix_conditioned(chmm_eq, chmm_vix, C, S0, n_oos;
            pi_V_init=pi_V_init, Δt=DT, risk_free_rate=RISK_FREE, n_paths=N_PATHS);

        terminal = P_paths[end, :];
        q05e = quantile(terminal, 0.05);
        q50e = quantile(terminal, 0.50);
        q95e = quantile(terminal, 0.95);
        hit90 = (q05e <= P_obs[end] <= q95e) ? 1.0 : 0.0;
        cov90 = _band_cov(P_obs, P_paths; α=0.05);
        mape  = _median_mape(P_obs, P_paths);
        crps  = _mean_crps(P_obs, P_paths);

        push!(summary_rows, (
            ticker, String(fam), S0, n_oos, size(P_paths, 2),
            P_obs[end], q05e, q50e, q95e, mean(terminal), std(terminal),
            hit90, cov90, mape, crps, crps / S0, C_entropy
        ));

        fan = _price_fan_plot(ticker, fam_label, P_obs, P_paths);
        savefig(fan, joinpath(OUT_DIR, "Fig-VIX-$ticker-PriceFan-$fam_label.svg"));
        savefig(fan, joinpath(OUT_DIR, "Fig-VIX-$ticker-PriceFan-$fam_label.pdf"));

        th = _terminal_hist_plot(ticker, fam_label, P_paths, P_obs[end]);
        savefig(th, joinpath(OUT_DIR, "Fig-VIX-$ticker-TerminalDist-$fam_label.svg"));
        savefig(th, joinpath(OUT_DIR, "Fig-VIX-$ticker-TerminalDist-$fam_label.pdf"));

        println("cov90=$(round(cov90,digits=3))  MAPE=$(round(mape,digits=3))  CRPS/S0=$(round(crps/S0,digits=3))  H(C)=$(round(C_entropy,digits=2))");
    end
end

# ------------------------------------------------------------------------------ #
# Save and print.
# ------------------------------------------------------------------------------ #
summary_path = joinpath(OUT_DIR, "summary.csv");
CSV.write(summary_path, summary_rows);
println("\n[done] Summary written to: $summary_path")
println("[done] Figures in:          $OUT_DIR")

println("\nVIX-conditioned terminal + path metrics:")
println("  ticker family     hit90  cov90     MAPE    CRPS     CRPS/S0  H(C)")
for row in eachrow(summary_rows)
    println(
        "  $(rpad(row.ticker,6)) $(rpad(row.family,9)) ",
        "$(Int(row.hit_rate_90))      ",
        "$(rpad(round(row.cov90_path, digits=3), 8)) ",
        "$(rpad(round(row.median_mape, digits=4), 8)) ",
        "$(rpad(round(row.mean_crps, digits=3), 8)) ",
        "$(rpad(round(row.crps_rel_S0, digits=4), 8)) ",
        "$(round(row.coupling_mean_entropy, digits=2))"
    );
end
