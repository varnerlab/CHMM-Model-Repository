# ========================================================================================= #
# run_ged_robustness.jl
#
# Two robustness checks for CHMM-GED, building on the run_ged_emissions_ablation.jl
# headline result that GED's per-state p_k splits bimodally on SPY (mostly Gaussian-
# like states + a few Laplace-like tail states):
#
#   (i)  10-seed Monte Carlo on SPY at K = 18 — is the bimodality stable, or
#        an artifact of one EM init?
#   (ii) Cross-ticker generalization on {SPY, NVDA, JNJ, JPM, AAPL, QQQ} at K = 18 —
#        does every equity show the same Gaussian-bulk / Laplace-tail split?
#
# Refinement from the headline run: cap p_max at 3.0 (was 4.0). Above ~3 the GED
# density is essentially flat-topped and additional p doesn't change the fit; the
# tighter bracket also speeds up the per-state CM3 update.
#
# Output:
#   results/SPY/ged_ablation/MultiSeed-GED.txt
#   results/SPY/ged_ablation/CrossTicker-GED.txt
#
# Usage: julia --project=. run_ged_robustness.jl
# ========================================================================================= #

const _PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."));
using Pkg; Pkg.activate(_PROJECT_ROOT);
include(joinpath(_PROJECT_ROOT, "Include.jl"));
using Random;

const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const K = 18;
const MAX_ITER = 60;
const N_PATHS = 250;          # smaller than the headline 1000 for tractability across runs
const L = 252;
const P_BOUNDS = (0.5, 3.0);  # tightened upper bound (was 4.0)

const BASE_SEED = 20260420;
const SEED_GRID = [BASE_SEED + 100*k for k in 1:10];

const TICKERS = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];

const RESULTS_DIR = joinpath(_ROOT, "results", "SPY", "ged_ablation");
mkpath(RESULTS_DIR);

println("="^72)
println("  CHMM-GED robustness  (multiseed + cross-ticker)")
println("  K = $K  |  p_bounds = $P_BOUNDS  |  $N_PATHS sim paths per fit")
println("="^72)

# ========================================================================================= #
# Data loading
# ========================================================================================= #
println("\n[data] Loading IS + OoS panels...");
train = MyPortfolioDataSet() |> x -> x["dataset"];
oos = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train["AAPL"]);
ds = Dict{String,DataFrame}();
for (t, data) in train; if nrow(data) == max_days; ds[t] = data; end; end
all_tickers = keys(ds) |> collect |> sort;
all_R = log_growth_matrix(ds, all_tickers; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);

# Per-ticker IS + OoS series.
function get_series(ticker::String)
    idx = findfirst(==(ticker), all_tickers);
    R_is = Vector{Float64}(all_R[:, idx]);
    R_oos = Vector{Float64}(log_growth_matrix(oos, ticker; Δt=ΔT, risk_free_rate=RISK_FREE_RATE));
    return R_is, R_oos;
end

# ========================================================================================= #
# Shared helpers
# ========================================================================================= #
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

function quick_metrics(observed, sim_archive)
    np = size(sim_archive, 2); n_o = length(observed);
    μ_o = mean(observed); σ_o = std(observed);
    kurt_o = sum(((observed .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L, n_o - 1);
    acf_o = autocor(abs.(observed), 1:L_use);

    ks_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0;
    for i in 1:np
        sim = sim_archive[:, i];
        if pvalue(ApproximateTwoSampleKSTest(observed, sim)) > 0.05; ks_pass += 1; end
        μ_s = mean(sim); σ_s = std(sim);
        kurt_s += sum(((sim .- μ_s) ./ σ_s).^4) / length(sim) - 3.0;
        acf_mae_s += mean(abs.(acf_o .- autocor(abs.(sim), 1:L_use)));
    end
    return (ks=100*ks_pass/np, kurt=kurt_s/np, kurt_obs=kurt_o,
            acf_mae=acf_mae_s/np);
end

function classify_p(p_vec)
    n_g  = count(p -> p >= 1.85, p_vec);
    n_i  = count(p -> 1.30 <= p < 1.85, p_vec);
    n_l  = count(p -> 0.85 <= p < 1.30, p_vec);
    n_s  = count(p -> p < 0.85, p_vec);
    return (gauss=n_g, inter=n_i, lapl=n_l, super=n_s);
end

function fit_ged(obs::Vector{Float64}, K::Int, seed::Int)
    Random.seed!(seed);
    return build(MyGEDHiddenMarkovModel,
        (observations=obs, number_of_states=K, max_iter=MAX_ITER,
         p_bounds=P_BOUNDS));
end

# ========================================================================================= #
# (i) Multiseed on SPY
# ========================================================================================= #
println("\n[multiseed] CHMM-GED on SPY across $(length(SEED_GRID)) seeds...");
R_is_spy, R_oos_spy = get_series("SPY");

multi_rows = NamedTuple[];
all_p = Vector{Float64}[];

for (k, seed) in enumerate(SEED_GRID)
    println("  seed $seed ($k/$(length(SEED_GRID)))...");
    m = fit_ged(R_is_spy, K, seed);
    p_vec = [m.emission[s].p for s in 1:K];
    push!(all_p, p_vec);
    cls = classify_p(p_vec);

    Random.seed!(seed + 1);
    _, start_dist = _stationary(m, K);
    sim_is, sim_oos = _simulate_paths(m, start_dist, length(R_is_spy), length(R_oos_spy), N_PATHS);
    m_is = quick_metrics(R_is_spy, sim_is);
    m_oos = quick_metrics(R_oos_spy, sim_oos);

    push!(multi_rows, (
        seed = seed,
        ll = m.log_likelihood_history[end],
        n_gauss = cls.gauss, n_inter = cls.inter, n_lapl = cls.lapl, n_super = cls.super,
        p_min = minimum(p_vec), p_med = median(p_vec), p_max = maximum(p_vec),
        ks_is = round(m_is.ks, digits=1),
        ks_oos = round(m_oos.ks, digits=1),
        kurt_sim = round(m_is.kurt, digits=2),
        acf_mae = round(m_is.acf_mae, digits=4),
    ));
end

open(joinpath(RESULTS_DIR, "MultiSeed-GED.txt"), "w") do io
    println(io, "CHMM-GED multiseed Monte Carlo  (SPY, K=$K, p_bounds=$P_BOUNDS)")
    println(io, "$(N_PATHS) sim paths per seed, $(length(SEED_GRID)) seeds.")
    println(io, "Bimodality classification: p_k >= 1.85 → Gaussian-like; 1.30-1.85 → intermediate;")
    println(io, "                           0.85-1.30 → Laplace-like; < 0.85 → super-Laplace.")
    println(io, "="^110)
    println(io, "seed   |   LL    | nG | nI | nL | nS | p_min | p_med | p_max | KS_is | KS_oos | kurt_sim | ACF-MAE")
    println(io, "-"^110)
    for r in multi_rows
        println(io, "$(lpad(r.seed,6)) | $(lpad(round(r.ll, digits=1), 8)) | $(lpad(r.n_gauss,2)) | $(lpad(r.n_inter,2)) | $(lpad(r.n_lapl,2)) | $(lpad(r.n_super,2)) | $(lpad(round(r.p_min, digits=2), 5)) | $(lpad(round(r.p_med, digits=2), 5)) | $(lpad(round(r.p_max, digits=2), 5)) | $(lpad(r.ks_is, 5)) | $(lpad(r.ks_oos, 6)) | $(lpad(r.kurt_sim, 7)) | $(r.acf_mae)")
    end
    println(io, "="^110)

    n_g_arr = [r.n_gauss for r in multi_rows];
    n_l_arr = [r.n_lapl + r.n_super for r in multi_rows];   # combined heavy-shape count
    ks_is_arr = [r.ks_is for r in multi_rows];
    ks_oos_arr = [r.ks_oos for r in multi_rows];
    kurt_arr = [r.kurt_sim for r in multi_rows];
    acf_arr = [r.acf_mae for r in multi_rows];
    seed_with_lapl = count(r -> (r.n_lapl + r.n_super) > 0, multi_rows);

    println(io, "")
    println(io, "Multiseed summary (mean ± std across $(length(SEED_GRID)) seeds):")
    println(io, "  Gaussian-like states:        $(round(mean(n_g_arr), digits=1)) ± $(round(std(n_g_arr), digits=1)) / $K")
    println(io, "  Heavy-shape states (L + S):  $(round(mean(n_l_arr), digits=1)) ± $(round(std(n_l_arr), digits=1)) / $K")
    println(io, "  Seeds with >=1 heavy state:  $(seed_with_lapl) / $(length(SEED_GRID))")
    println(io, "  KS pass rate IS  (%):        $(round(mean(ks_is_arr), digits=1)) ± $(round(std(ks_is_arr), digits=1))")
    println(io, "  KS pass rate OoS (%):        $(round(mean(ks_oos_arr), digits=1)) ± $(round(std(ks_oos_arr), digits=1))")
    println(io, "  Simulated IS excess kurt:    $(round(mean(kurt_arr), digits=2)) ± $(round(std(kurt_arr), digits=2))   (target $(round(multi_rows[1].kurt_sim < 0 ? 0.0 : 7.68, digits=2)))")
    println(io, "  ACF-MAE on |G|:              $(round(mean(acf_arr), digits=4)) ± $(round(std(acf_arr), digits=4))")
end

# ========================================================================================= #
# (ii) Cross-ticker
# ========================================================================================= #
println("\n[cross-ticker] CHMM-GED on $(length(TICKERS)) tickers (single seed = $BASE_SEED)...");

cross_rows = NamedTuple[];
for (k, ticker) in enumerate(TICKERS)
    println("  $ticker ($k/$(length(TICKERS)))...");
    R_is, R_oos = get_series(ticker);
    m = fit_ged(R_is, K, BASE_SEED);
    p_vec = [m.emission[s].p for s in 1:K];
    cls = classify_p(p_vec);

    Random.seed!(BASE_SEED + 1);
    _, start_dist = _stationary(m, K);
    sim_is, sim_oos = _simulate_paths(m, start_dist, length(R_is), length(R_oos), N_PATHS);
    m_is = quick_metrics(R_is, sim_is);
    m_oos = quick_metrics(R_oos, sim_oos);

    push!(cross_rows, (
        ticker = ticker,
        n_obs_is = length(R_is), n_obs_oos = length(R_oos),
        kurt_obs = round(m_is.kurt_obs, digits=2),
        ll = m.log_likelihood_history[end],
        n_gauss = cls.gauss, n_inter = cls.inter, n_lapl = cls.lapl, n_super = cls.super,
        p_min = minimum(p_vec), p_med = median(p_vec), p_max = maximum(p_vec),
        ks_is = round(m_is.ks, digits=1),
        ks_oos = round(m_oos.ks, digits=1),
        kurt_sim = round(m_is.kurt, digits=2),
        acf_mae = round(m_is.acf_mae, digits=4),
    ));
end

open(joinpath(RESULTS_DIR, "CrossTicker-GED.txt"), "w") do io
    println(io, "CHMM-GED cross-ticker generalization  (K=$K, p_bounds=$P_BOUNDS, seed=$BASE_SEED)")
    println(io, "$(N_PATHS) sim paths per ticker.")
    println(io, "="^130)
    println(io, "ticker | T_IS  | T_OoS | k_obs | nG | nI | nL | nS | p_min | p_med | p_max | KS_is | KS_oos | kurt_sim | ACF-MAE")
    println(io, "-"^130)
    for r in cross_rows
        println(io, "$(rpad(r.ticker, 6)) | $(lpad(r.n_obs_is, 5)) | $(lpad(r.n_obs_oos, 5)) | $(lpad(r.kurt_obs, 5)) | $(lpad(r.n_gauss,2)) | $(lpad(r.n_inter,2)) | $(lpad(r.n_lapl,2)) | $(lpad(r.n_super,2)) | $(lpad(round(r.p_min, digits=2), 5)) | $(lpad(round(r.p_med, digits=2), 5)) | $(lpad(round(r.p_max, digits=2), 5)) | $(lpad(r.ks_is, 5)) | $(lpad(r.ks_oos, 6)) | $(lpad(r.kurt_sim, 7)) | $(r.acf_mae)")
    end
    println(io, "="^130)

    n_with_heavy = count(r -> (r.n_lapl + r.n_super) > 0, cross_rows);
    println(io, "")
    println(io, "Cross-ticker summary:")
    println(io, "  Tickers with >=1 heavy-shape state:  $(n_with_heavy) / $(length(TICKERS))")
    println(io, "  Mean Gaussian-like state count:      $(round(mean([r.n_gauss for r in cross_rows]), digits=1)) / $K")
    println(io, "  Mean heavy-shape state count (L+S):  $(round(mean([r.n_lapl + r.n_super for r in cross_rows]), digits=1)) / $K")
    println(io, "  Mean median p_k:                     $(round(mean([r.p_med for r in cross_rows]), digits=2))")
end

println("\nDone. Outputs in: $RESULTS_DIR")
println("  - MultiSeed-GED.txt    (10-seed bimodality stability)")
println("  - CrossTicker-GED.txt  (6-ticker generalization)")
