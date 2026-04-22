# ========================================================================================= #
# run_gru_baseline.jl
#
# Trains the MyGRUGenerator on SPY in-sample data and evaluates the seven-metric
# panel on 1000 simulated paths (IS + OoS), producing the deep generative baseline
# row in Table 2.
#
# Output: results/diagnostics/gru/{Metrics.txt, Loss.pdf, Loss.svg}
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");
using Random

const SEED = 20260420;
Random.seed!(SEED);

const TICKER  = "SPY";
const N_PATHS = 1000;
const L_LAGS  = 252;

const EPOCHS     = 20;
const HIDDEN_DIM = 32;
const WINDOW     = 20;
const LR         = 1e-3;

const OUT_DIR = joinpath(_ROOT, "results", "diagnostics", "gru");
mkpath(OUT_DIR);

println("="^70)
println("  GRU deep-generative baseline")
println("  Seed:        $SEED")
println("  Hidden dim:  $HIDDEN_DIM   Window: $WINDOW   Epochs: $EPOCHS")
println("  N paths:     $N_PATHS")
println("="^70)

# ----- Data -----
println("\n[setup] Loading SPY...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=1/252, risk_free_rate=0.0);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = all_R[:, idx_spy];
oos = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos, "SPY"; Δt=1/252, risk_free_rate=0.0);
n_is = length(R_is); n_oos = length(R_oos);
println("  SPY IS=$n_is | OoS=$n_oos");

# ----- Train GRU -----
println("\n[train] Fitting MyGRUGenerator...")
gru = build(MyGRUGenerator, (
    observations=R_is, epochs=EPOCHS, lr=LR,
    hidden_dim=HIDDEN_DIM, window=WINDOW,
    seed=SEED, verbose=true));
println("  Final NLL = $(round(gru.loss_history[end], digits=4))")

# Convergence plot
fig_loss = plot(1:length(gru.loss_history), gru.loss_history,
    lw=2, marker=:circle, ms=3, color=:navy,
    xlabel="Epoch", ylabel="NLL (per window)",
    title="GRU training loss — SPY ($EPOCHS epochs, hidden=$HIDDEN_DIM, win=$WINDOW)",
    titlefontsize=10, legend=false, size=(700, 420));
savefig(fig_loss, joinpath(OUT_DIR, "Loss.pdf"));
savefig(fig_loss, joinpath(OUT_DIR, "Loss.svg"));

# ----- Simulate paths and evaluate -----
println("\n[sim] Generating $N_PATHS paths of length $n_is + $n_oos...")
gru_is_paths  = Array{Float64,2}(undef, n_is,  N_PATHS);
gru_oos_paths = Array{Float64,2}(undef, n_oos, N_PATHS);

# Use the last `WINDOW` IS observations to seed each rollout for warm-context
# generation; this is the standard "teacher-forcing seed + free run" protocol.
seed_window = R_is[(end - WINDOW + 1):end];

for i in 1:N_PATHS
    Random.seed!(SEED + i);
    gru_is_paths[:,  i] = simulate_gru(gru, n_is;  seed_window=seed_window);
    Random.seed!(SEED + 1_000_000 + i);
    gru_oos_paths[:, i] = simulate_gru(gru, n_oos; seed_window=seed_window);
    if i % 100 == 0
        println("  path $i / $N_PATHS");
    end
end

# ----- Seven-metric evaluation (shared with run_diagnostics.jl) -----
function eval_full(observed, sim_archive; L_val=L_LAGS)
    np = size(sim_archive, 2); n_o = length(observed);
    μ_o = mean(observed); σ_o = std(observed);
    kurt_o = sum(((observed .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L_val, n_o - 1);
    acf_o = autocor(abs.(observed), 1:L_use);

    ks_pass = 0; ad_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0;
    w1_s = 0.0; hell_s = 0.0;
    qprobs = range(0.01, 0.99, length=99);
    obs_quantiles = quantile(observed, qprobs);
    sim_qmatrix = zeros(99, np);

    for i in 1:np
        sim = sim_archive[:, i];
        # Guard against any pathological NaN/Inf paths from the GRU rollout
        if any(!isfinite, sim); continue; end
        pval_ks = pvalue(ApproximateTwoSampleKSTest(observed, sim));
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
        sim_qmatrix[:, i] = quantile(sim, qprobs);
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

println("\n[eval] Seven-metric panel...")
m_is  = eval_full(R_is,  gru_is_paths);
m_oos = eval_full(R_oos, gru_oos_paths);

open(joinpath(OUT_DIR, "Metrics.txt"), "w") do io
    println(io, "GRU Generator Seven-Metric Panel (SPY, seed=$SEED, $N_PATHS paths)");
    println(io, "="^85);
    println(io, "  Architecture: 1-layer GRU(hidden=$HIDDEN_DIM) + Dense(2) Gaussian head");
    println(io, "  Training:     $EPOCHS epochs, Adam(lr=$LR), window=$WINDOW, NLL loss");
    println(io, "  Final NLL:    $(round(gru.loss_history[end], digits=4))");
    println(io, "  IS: $n_is obs | OoS: $n_oos obs");
    println(io, "  Observed kurtosis: IS $(m_is.kurt_obs) | OoS $(m_oos.kurt_obs)");
    println(io);
    println(io, rpad("Window",8), " | ", rpad("KS",6), " | ", rpad("AD",6), " | ",
                rpad("Kurt",7), " | ", rpad("ACF-MAE",8), " | ",
                rpad("W1",6), " | ", rpad("H",7), " | ", rpad("Cov",5));
    println(io, "-"^85);
    for (tag, m) in [("IS", m_is), ("OoS", m_oos)]
        println(io, rpad(tag,8), " | ",
                    rpad(m.ks,6), " | ", rpad(m.ad,6), " | ",
                    rpad(m.kurt,7), " | ", rpad(m.acf_mae,8), " | ",
                    rpad(m.w1,6), " | ", rpad(m.hell,7), " | ", rpad(m.cov,5));
    end
end

println("\nGRU metrics written to $OUT_DIR/Metrics.txt")
println("Done.")
