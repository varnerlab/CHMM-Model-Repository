# ========================================================================================= #
# run_garch_suite.jl
#
# expand the paper's GARCH-family
# baseline panel beyond the single GARCH(1,1) + two-regime MS-GARCH currently reported.
#
# New baselines:
#   - EGARCH(1,1) Gaussian
#   - GJR-GARCH(1,1) Gaussian
#   - GARCH(1,1) Student-t
#   - HAR-RV (on daily squared returns)
#   - MS-GARCH K=3
#
# Reference rows (already in the paper):
#   - GARCH(1,1) Gaussian
#   - MS-GARCH K=2
#
# Metrics per model: IS/OoS KS pass rate at α=0.05, IS/OoS AD pass rate, mean simulated
# excess kurtosis (IS), ACF-MAE of |r| over lags 1..252 (IS), and Kupiec / Christoffersen
# VaR statistics at α ∈ {0.01, 0.05} on OoS. FIGARCH is documented as a separate
# follow-up (see `revision-code-todo.md`).
#
# Outputs:
#   results/garch_suite/GARCH_Suite.txt
#   ../CHMM-paper/results/robustness/garch_suite.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
const SEED = 20260422;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const N_PATHS   = 1000;

const OUT_DIR       = joinpath(_ROOT, "results", "garch_suite");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^72)
println("  Expanded GARCH-family baseline panel")
println("  Seed $SEED")
println("="^72)

# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY IS + OoS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
n_is = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
n_oos = length(R_oos);
println("  IS $n_is, OoS $n_oos");

# --------------------------------------------------------------------------------------- #
# Fit all baselines
# --------------------------------------------------------------------------------------- #
println("\n[fit] GARCH(1,1) Gaussian (reference)...");
Random.seed!(SEED + 1);
ref_garch = build(MyGARCHModel, (observations=R_is,));

println("[fit] EGARCH(1,1)...");
Random.seed!(SEED + 2);
fit_eg = fit_egarch11(R_is);

println("[fit] GJR-GARCH(1,1)...");
Random.seed!(SEED + 3);
fit_gjr = fit_gjr11(R_is);

println("[fit] GARCH(1,1) Student-t...");
Random.seed!(SEED + 4);
fit_gt = fit_garcht11(R_is);

println("[fit] HAR-RV...");
Random.seed!(SEED + 5);
fit_har = fit_harrv(R_is);

println("[fit] MS-GARCH K=2 (reference)...");
Random.seed!(SEED + 6);
ref_ms2 = fit_msgarch_k2(R_is; max_iter=3000);

println("[fit] MS-GARCH K=3...");
Random.seed!(SEED + 7);
fit_ms3 = fit_msgarch_k3(R_is; max_iter=3000);

println("  fits complete.");
println("  GARCH(1,1):  ω=$(round(ref_garch.ω, digits=4)) α=$(round(ref_garch.α, digits=3)) β=$(round(ref_garch.β, digits=3))");
println("  EGARCH:      ω=$(round(fit_eg.ω, digits=3)) α=$(round(fit_eg.α, digits=3)) γ=$(round(fit_eg.γ, digits=3)) β=$(round(fit_eg.β, digits=3))");
println("  GJR:         ω=$(round(fit_gjr.ω, digits=4)) α=$(round(fit_gjr.α, digits=3)) γ=$(round(fit_gjr.γ, digits=3)) β=$(round(fit_gjr.β, digits=3))");
println("  GARCH-t:     ω=$(round(fit_gt.ω, digits=4)) α=$(round(fit_gt.α, digits=3)) β=$(round(fit_gt.β, digits=3)) ν=$(round(fit_gt.ν, digits=2))");
println("  HAR-RV:      β=[$(join([round(b, digits=3) for b in fit_har.β], ", "))], σ_η=$(round(fit_har.σ_η, digits=3))");
println("  MS-GARCH K=2: state σ=[$(round(sqrt(ref_ms2.ω[1]/max(1-ref_ms2.α[1]-ref_ms2.β[1], 1e-6)), digits=2)), $(round(sqrt(ref_ms2.ω[2]/max(1-ref_ms2.α[2]-ref_ms2.β[2], 1e-6)), digits=2))]");
println("  MS-GARCH K=3: state σ=[$(join([round(sqrt(fit_ms3.ω[k]/max(1-fit_ms3.α[k]-fit_ms3.β[k], 1e-6)), digits=2) for k in 1:3], ", "))]");

# --------------------------------------------------------------------------------------- #
# Simulators and metrics
# --------------------------------------------------------------------------------------- #

"""Simulates `n_paths` paths of length `T` from `sim_fn` (a zero-arg closure returning Vector{Float64}).
Returns a T x n_paths Matrix{Float64}."""
function _simulate_paths(sim_fn, T::Int, n_paths::Int)::Matrix{Float64}
    paths = Matrix{Float64}(undef, T, n_paths);
    for p in 1:n_paths
        paths[:, p] = sim_fn();
    end
    return paths;
end

function ks_pass_rate(R_ref::Vector{Float64}, sim::Matrix{Float64}; α::Float64=0.05)
    n_sim = size(sim, 2); n_pass = 0;
    for p in 1:n_sim
        pv = pvalue(ApproximateTwoSampleKSTest(R_ref, sim[:, p]));
        if pv >= α; n_pass += 1; end
    end
    return n_pass / n_sim;
end

function ad_pass_rate(R_ref::Vector{Float64}, sim::Matrix{Float64}; α::Float64=0.05)
    # Use the HypothesisTests two-sample Anderson-Darling if available; else fall back
    # to a p-value lookup via the KSampleADTest. For the metric panel this is a
    # proxy but consistent across rows.
    n_sim = size(sim, 2); n_pass = 0;
    for p in 1:n_sim
        try
            pv = pvalue(KSampleADTest(R_ref, sim[:, p]));
            if pv >= α; n_pass += 1; end
        catch
            # if AD fails for any reason (rare), count as fail
        end
    end
    return n_pass / n_sim;
end

function acf_mae(R_obs::Vector{Float64}, sim::Matrix{Float64}; max_lag::Int=252)
    obs_acf = autocor(abs.(R_obs), 1:max_lag);
    n_sim = size(sim, 2);
    sim_acf_mean = zeros(max_lag);
    for p in 1:n_sim
        sim_acf_mean .+= autocor(abs.(sim[:, p]), 1:max_lag);
    end
    sim_acf_mean ./= n_sim;
    return mean(abs.(obs_acf .- sim_acf_mean));
end

function acf_mae_raw(R_obs::Vector{Float64}, sim::Matrix{Float64}; max_lag::Int=252)
    obs_acf = autocor(R_obs, 1:max_lag);
    n_sim = size(sim, 2);
    sim_acf_mean = zeros(max_lag);
    for p in 1:n_sim
        sim_acf_mean .+= autocor(sim[:, p], 1:max_lag);
    end
    sim_acf_mean ./= n_sim;
    return mean(abs.(obs_acf .- sim_acf_mean));
end

function var_backtest(R_oos::Vector{Float64}, sim_oos_paths::Matrix{Float64})
    # Pooled-archive VaR: α-quantile of the stacked simulated OoS sample at each α.
    pooled = vec(sim_oos_paths);
    results = Dict{Float64, NamedTuple}();
    for α in [0.01, 0.05]
        v = quantile(pooled, α);
        br = R_oos .<= v;
        k = kupiec_lr(br, α);
        c = christoffersen_lr(br);
        results[α] = (VaR=v, br_rate=k.breach_rate, LR_uc=k.LR, p_uc=k.pvalue,
                      LR_ind=c.LR, p_ind=c.pvalue);
    end
    return results;
end

# --------------------------------------------------------------------------------------- #
# Run: simulate IS and OoS, compute metrics for each model
# --------------------------------------------------------------------------------------- #

function run_model(name::String, sim_fn_is, sim_fn_oos)
    Random.seed!(SEED + 100);
    sim_is = _simulate_paths(sim_fn_is, n_is, N_PATHS);
    Random.seed!(SEED + 101);
    sim_oos = _simulate_paths(sim_fn_oos, n_oos, N_PATHS);

    is_ks = ks_pass_rate(R_is, sim_is);
    oos_ks = ks_pass_rate(R_oos, sim_oos);
    is_ad = ad_pass_rate(R_is, sim_is);
    oos_ad = ad_pass_rate(R_oos, sim_oos);
    sim_kurt = mean([kurtosis(sim_is[:, p]) for p in 1:N_PATHS]);
    acf = acf_mae(R_is, sim_is);
    acf_raw = acf_mae_raw(R_is, sim_is);
    vb = var_backtest(R_oos, sim_oos);

    r = (
        name=name,
        is_ks=is_ks, oos_ks=oos_ks,
        is_ad=is_ad, oos_ad=oos_ad,
        sim_kurt=sim_kurt, acf_mae=acf, acf_mae_raw=acf_raw,
        br01=vb[0.01].br_rate, LRuc01=vb[0.01].LR_uc, LRind01=vb[0.01].LR_ind,
        br05=vb[0.05].br_rate, LRuc05=vb[0.05].LR_uc, LRind05=vb[0.05].LR_ind,
    );
    println("  $(rpad(name, 14))  IS KS $(round(100*r.is_ks, digits=1))%  OoS KS $(round(100*r.oos_ks, digits=1))%  Kurt $(round(r.sim_kurt, digits=2))  ACF-MAE $(round(r.acf_mae, digits=4))  ACF-MAE(raw) $(round(r.acf_mae_raw, digits=4))  LR_uc01 $(round(r.LRuc01, digits=2))  LR_uc05 $(round(r.LRuc05, digits=2))");
    return r;
end

results = NamedTuple[];

println("\n[metrics] simulating and scoring each model...");
push!(results, run_model("GARCH(1,1)",
    () -> simulate_garch(ref_garch, n_is),
    () -> simulate_garch(ref_garch, n_oos)));
push!(results, run_model("EGARCH",
    () -> simulate_egarch(fit_eg, n_is),
    () -> simulate_egarch(fit_eg, n_oos)));
push!(results, run_model("GJR-GARCH",
    () -> simulate_gjr(fit_gjr, n_is),
    () -> simulate_gjr(fit_gjr, n_oos)));
push!(results, run_model("GARCH-t",
    () -> simulate_garcht(fit_gt, n_is),
    () -> simulate_garcht(fit_gt, n_oos)));
push!(results, run_model("HAR-RV",
    () -> simulate_harrv(fit_har, n_is),
    () -> simulate_harrv(fit_har, n_oos)));
push!(results, run_model("MS-GARCH K=2",
    () -> simulate_msgarch(ref_ms2, n_is),
    () -> simulate_msgarch(ref_ms2, n_oos)));
push!(results, run_model("MS-GARCH K=3",
    () -> simulate_msgarch(fit_ms3, n_is),
    () -> simulate_msgarch(fit_ms3, n_oos)));

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
open(joinpath(OUT_DIR, "GARCH_Suite.txt"), "w") do io
    println(io, "="^170);
    println(io, "Expanded GARCH-family baseline panel.");
    println(io, "="^170);
    println(io, "");
    println(io, "Reference rows : GARCH(1,1) Gaussian, MS-GARCH K=2 (already in the paper, re-run here under the same seed policy for consistency).");
    println(io, "New rows       : EGARCH(1,1), GJR-GARCH(1,1), GARCH(1,1) Student-t, HAR-RV (daily squared returns), MS-GARCH K=3.");
    println(io, "Data           : SPY, IS $n_is obs, OoS $n_oos obs, N_paths = $N_PATHS per model.");
    println(io, "VaR            : pooled-archive α-quantile on stacked simulated OoS paths (unconditional VaR baseline, same convention as Table 3 in the paper).");
    println(io, "FIGARCH        : deferred to a follow-up code-repo pass (see revision-code-todo.md M7); requires a truncated-lag FI polynomial implementation.");
    println(io, "");
    println(io, rpad("Model",      14), " | ",
                rpad("IS KS%",     7), " | ", rpad("OoS KS%",    8), " | ",
                rpad("IS AD%",     7), " | ", rpad("OoS AD%",    8), " | ",
                rpad("Kurt",       6), " | ", rpad("ACF-MAE|G|", 10), " | ",
                rpad("ACF-MAE raw",11), " | ",
                rpad("br%01",      6), " | ", rpad("LR_uc01",    7), " | ",
                rpad("LR_ind01",   8), " | ", rpad("br%05",      6), " | ",
                rpad("LR_uc05",    7), " | ", rpad("LR_ind05",   8));
    println(io, "-"^180);
    for r in results
        println(io, rpad(r.name, 14), " | ",
                    rpad(round(100*r.is_ks, digits=1), 7), " | ",
                    rpad(round(100*r.oos_ks, digits=1), 8), " | ",
                    rpad(round(100*r.is_ad, digits=1), 7), " | ",
                    rpad(round(100*r.oos_ad, digits=1), 8), " | ",
                    rpad(round(r.sim_kurt, digits=2), 6), " | ",
                    rpad(round(r.acf_mae, digits=4), 10), " | ",
                    rpad(round(r.acf_mae_raw, digits=4), 11), " | ",
                    rpad(round(100*r.br01, digits=1), 6), " | ",
                    rpad(round(r.LRuc01, digits=2), 7), " | ",
                    rpad(round(r.LRind01, digits=2), 8), " | ",
                    rpad(round(100*r.br05, digits=1), 6), " | ",
                    rpad(round(r.LRuc05, digits=2), 7), " | ",
                    rpad(round(r.LRind05, digits=2), 8));
    end
    println(io, "="^180);
end

open(joinpath(PAPER_ROBUSTNESS_DIR, "garch_suite.csv"), "w") do io
    println(io, "model,IS_KS_pct,OoS_KS_pct,IS_AD_pct,OoS_AD_pct,sim_kurt,ACF_MAE,ACF_MAE_raw,br01_pct,LRuc01,LRind01,br05_pct,LRuc05,LRind05");
    for r in results
        println(io, "$(r.name),$(round(100*r.is_ks, digits=2)),$(round(100*r.oos_ks, digits=2)),$(round(100*r.is_ad, digits=2)),$(round(100*r.oos_ad, digits=2)),$(round(r.sim_kurt, digits=3)),$(round(r.acf_mae, digits=5)),$(round(r.acf_mae_raw, digits=5)),$(round(100*r.br01, digits=2)),$(round(r.LRuc01, digits=3)),$(round(r.LRind01, digits=3)),$(round(100*r.br05, digits=2)),$(round(r.LRuc05, digits=3)),$(round(r.LRind05, digits=3))");
    end
end

println("\n" * "="^72);
println("  GARCH suite complete.");
println("  Text report : $(joinpath(OUT_DIR, "GARCH_Suite.txt"))");
println("  Paper CSV   : $(joinpath(PAPER_ROBUSTNESS_DIR, "garch_suite.csv"))");
println("="^72);
