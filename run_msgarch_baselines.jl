# ========================================================================================= #
# run_msgarch_baselines.jl
#
# Markov-Switching GARCH(1,1), K=2 (Haas-Mittnik-Paolella 2004).
# Fit on SPY, simulate 1000 IS + 1000 OoS paths, append to the legacy harness
# panel, and emit the MS-GARCH row referenced by tab:extended_baselines.
#
# Outputs:
#   results/msgarch_baselines/ms_garch_model.jld2     fitted MS-GARCH(K=2) parameters
#   results/msgarch_baselines/extended_metrics.txt    metrics panel with MS-GARCH row
#   results/msgarch_baselines/sim_pvalues.txt
#   results/msgarch_baselines/var_lr_tests.txt
#   results/msgarch_baselines/summary.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
const SEED = 20260422;
Random.seed!(SEED);

const TICKER       = "SPY";
const RISK_FREE    = 0.0;
const DT           = 1/252;
const N_PATHS      = 1000;
const L_LAGS       = 252;
const WINDOW_LEN   = 20;
const N_WINDOWS    = 500;
const SIG_DEPTH    = 3;
const MAX_LAG_LEV  = 20;
const HORIZONS_AG  = [1, 5, 10, 21];

const SIM_ARCHIVE_PATH = joinpath(_ROOT, "results", "baselines_archive", "sim_archive_cache.jld2");
const MSGARCH_DIR      = joinpath(_ROOT, "results", "msgarch_baselines");
mkpath(MSGARCH_DIR);

println("="^72)
println("  MS-GARCH(1,1) K=2 baseline (Haas-Mittnik-Paolella 2004)")
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
# Fit MS-GARCH K=2 on SPY IS
# --------------------------------------------------------------------------------------- #
println("\n[fit] Fitting MS-GARCH K=2 on SPY IS...");
Random.seed!(SEED + 300);
t0 = time();
msg = fit_msgarch_k2(R_is; max_iter=3000);
dt_fit = time() - t0;
println("  fit complete in $(round(dt_fit, digits=1)) s");

println("  ω  = $(round.(msg.ω, digits=4))");
println("  α  = $(round.(msg.α, digits=3))");
println("  β  = $(round.(msg.β, digits=3))");
println("  α+β = $(round.(msg.α .+ msg.β, digits=3))");
println("  μ  = $(round(msg.μ, digits=4))");
println("  T:");
for i in 1:2
    println("     ", round.(msg.T[i, :], digits=4));
end
println("  uncond var per regime: ", round.([msg.ω[k] / max(1 - msg.α[k] - msg.β[k], 1e-6) for k in 1:2], digits=3));
println("  ll = $(round(msg.log_likelihood, digits=2))");

save(joinpath(MSGARCH_DIR, "ms_garch_model.jld2"), "model", msg);

# --------------------------------------------------------------------------------------- #
# Simulate 1000 IS + 1000 OoS paths
# --------------------------------------------------------------------------------------- #
println("\n[sim] Simulating $N_PATHS paths IS + OoS...");
Random.seed!(SEED + 310);
msg_is  = Array{Float64,2}(undef, n_is,  N_PATHS);
msg_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    msg_is[:, i]  = simulate_msgarch(msg, n_is);
    msg_oos[:, i] = simulate_msgarch(msg, n_oos);
end

# --------------------------------------------------------------------------------------- #
# Build a minimal archive {Observed-row + MS-GARCH row + CHMM-t for comparison}
# Load Track A cache to reuse observed + CHMM rows for the panel
# --------------------------------------------------------------------------------------- #
cache_path = SIM_ARCHIVE_PATH;
base_archive = load(cache_path)["archive"];
archive = merge(base_archive, Dict("MS-GARCH" => (is=msg_is, oos=msg_oos)));

# Append optional SM rows if the C1 archive is present
sm_cache = joinpath(_ROOT, "results", "smchmm_baseline", "sim_archive_sm.jld2");
if isfile(sm_cache)
    sm = load(sm_cache)["archive"];
    archive = merge(archive, sm);
end

MODEL_ORDER = [
    "Bootstrap", "Gaussian", "Laplace",
    "DiscreteNJ", "DiscreteWJ", "GARCH",
    "MS-GARCH",
    "CHMM-N", "CHMM-t", "CHMM-L",
];
if haskey(archive, "SM-CHMM-t")
    append!(MODEL_ORDER, ["SM-CHMM-N", "SM-CHMM-t", "SM-CHMM-L"]);
end

# --------------------------------------------------------------------------------------- #
# A1 / A2 / A3: MMD, sig-MMD, discriminator AUC
# --------------------------------------------------------------------------------------- #
println("\n[A1,A2,A3] Metrics on $(length(MODEL_ORDER)) models (W=$WINDOW_LEN)...");

Random.seed!(SEED + 400);
obs_windows_is = let
    W_all = windowize(R_is, WINDOW_LEN; stride=max(1, (n_is - WINDOW_LEN) ÷ N_WINDOWS));
    idx = randperm(size(W_all, 2))[1:min(N_WINDOWS, size(W_all, 2))];
    W_all[:, idx];
end;
obs_windows_oos = let
    W_all = windowize(R_oos, WINDOW_LEN; stride=max(1, (n_oos - WINDOW_LEN) ÷ N_WINDOWS));
    idx = randperm(size(W_all, 2))[1:min(N_WINDOWS, size(W_all, 2))];
    W_all[:, idx];
end;

mmd_results = Dict{String, NamedTuple}();
for m in MODEL_ORDER
    print("  $m...");
    rng = MersenneTwister(SEED + hash(m) % 10^6);
    s_is, s_oos = archive[m].is, archive[m].oos;
    syn_windows_is  = sample_windows_from_archive(s_is,  WINDOW_LEN, size(obs_windows_is, 2);  rng=rng);
    syn_windows_oos = sample_windows_from_archive(s_oos, WINDOW_LEN, size(obs_windows_oos, 2); rng=rng);
    mmd_is  = mmd2_rbf(obs_windows_is, syn_windows_is; rng=rng);
    mmd_oos = mmd2_rbf(obs_windows_oos, syn_windows_oos; rng=rng);
    sig_is  = sig_mmd2(obs_windows_is, syn_windows_is;  depth=SIG_DEPTH, rng=rng);
    sig_oos = sig_mmd2(obs_windows_oos, syn_windows_oos; depth=SIG_DEPTH, rng=rng);
    auc_is  = discriminator_auc(R_is,  s_is;  window=WINDOW_LEN, n_windows=N_WINDOWS, rng=rng);
    auc_oos = discriminator_auc(R_oos, s_oos; window=WINDOW_LEN, n_windows=N_WINDOWS, rng=rng);
    mmd_results[m] = (
        mmd_is=mmd_is, mmd_oos=mmd_oos,
        sig_is=sig_is, sig_oos=sig_oos,
        auc_is=auc_is.auc, auc_is_std=auc_is.auc_std,
        auc_oos=auc_oos.auc, auc_oos_std=auc_oos.auc_std,
    );
    println(" MMD=$(round(mmd_is, digits=5)) sig=$(round(sig_is, digits=5)) AUC=$(round(auc_is.auc, digits=3))");
end

# A9: p-values on key stylized-fact stats (kurtosis, agg kurt h=5, h=21, leverage)
println("\n[A9] p-values for MS-GARCH against observed...");

function _kurt(x)
    μ = mean(x); σ = std(x);
    return σ > 0 ? sum(((x .- μ) ./ σ) .^ 4) / length(x) - 3.0 : 0.0;
end

obs_kurt_is = _kurt(R_is); obs_kurt_oos = _kurt(R_oos);
obs_ak_is  = aggregational_kurtosis(R_is;  horizons=HORIZONS_AG);
obs_ak_oos = aggregational_kurtosis(R_oos; horizons=HORIZONS_AG);
obs_lev_is  = leverage_effect(R_is;  max_lag=MAX_LAG_LEV);
obs_lev_oos = leverage_effect(R_oos; max_lag=MAX_LAG_LEV);

pv_results = Dict{String, NamedTuple}();
for m in MODEL_ORDER
    s_is, s_oos = archive[m].is, archive[m].oos;
    kurt_is_s  = [_kurt(s_is[:, i])  for i in 1:size(s_is, 2)];
    kurt_oos_s = [_kurt(s_oos[:, i]) for i in 1:size(s_oos, 2)];
    lev_is_s = Float64[]; lev_oos_s = Float64[];
    ak5_is_s  = Float64[]; ak21_is_s  = Float64[];
    ak5_oos_s = Float64[]; ak21_oos_s = Float64[];
    for i in 1:size(s_is, 2)
        push!(lev_is_s,  leverage_effect(s_is[:, i]; max_lag=MAX_LAG_LEV).avg_neg);
        ak = aggregational_kurtosis(s_is[:, i]; horizons=HORIZONS_AG);
        push!(ak5_is_s,  ak[5]); push!(ak21_is_s,  ak[21]);
    end
    for i in 1:size(s_oos, 2)
        push!(lev_oos_s, leverage_effect(s_oos[:, i]; max_lag=MAX_LAG_LEV).avg_neg);
        ak = aggregational_kurtosis(s_oos[:, i]; horizons=HORIZONS_AG);
        push!(ak5_oos_s, ak[5]); push!(ak21_oos_s, ak[21]);
    end
    pv_kurt_is  = sim_pvalue(obs_kurt_is, kurt_is_s);
    pv_kurt_oos = sim_pvalue(obs_kurt_oos, kurt_oos_s);
    pv_lev_is  = sim_pvalue(obs_lev_is.avg_neg, lev_is_s);
    pv_lev_oos = sim_pvalue(obs_lev_oos.avg_neg, lev_oos_s);
    pv_ak5_is  = sim_pvalue(obs_ak_is[5], ak5_is_s);
    pv_ak21_is = sim_pvalue(obs_ak_is[21], ak21_is_s);
    pv_ak5_oos  = sim_pvalue(obs_ak_oos[5], ak5_oos_s);
    pv_ak21_oos = sim_pvalue(obs_ak_oos[21], ak21_oos_s);
    pv_joint_is  = mean([pv_kurt_is,  pv_lev_is,  pv_ak5_is,  pv_ak21_is]);
    pv_joint_oos = mean([pv_kurt_oos, pv_lev_oos, pv_ak5_oos, pv_ak21_oos]);
    pv_results[m] = (pv_joint_is=pv_joint_is, pv_joint_oos=pv_joint_oos);
    println("  $m pv̄ IS=$(round(pv_joint_is, digits=3)) pv̄ OoS=$(round(pv_joint_oos, digits=3))");
end

# A8 unconditional VaR Kupiec + Christoffersen
println("\n[A8] Unconditional VaR LR tests...");
var_results = Dict{String, NamedTuple}();
for m in MODEL_ORDER
    v01 = quantile(vec(archive[m].is), 0.01);
    v05 = quantile(vec(archive[m].is), 0.05);
    br01 = R_oos .<= v01; br05 = R_oos .<= v05;
    k01 = kupiec_lr(br01, 0.01); k05 = kupiec_lr(br05, 0.05);
    c01 = christoffersen_lr(br01); c05 = christoffersen_lr(br05);
    var_results[m] = (v01=v01, v05=v05, k01=k01, k05=k05, c01=c01, c05=c05);
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
println("\n[output] Writing MS-GARCH tables...");

open(joinpath(MSGARCH_DIR, "extended_metrics.txt"), "w") do io
    println(io, "="^130);
    println(io, "TABLE 4 (Track A + C1 + B4). MMD / sig-MMD / disc AUC, MS-GARCH(K=2) row appended.");
    println(io, "="^130);
    println(io, "");
    println(io, "Setup   : SPY daily log excess growth; IS n=$n_is, OoS n=$n_oos; seed=$SEED; N_paths=$N_PATHS.");
    println(io, "MS-GARCH: K=2 regimes (Haas-Mittnik-Paolella 2004), Hamilton-filter MLE + Nelder-Mead.");
    println(io, "          Fit parameters: ω=$(round.(msg.ω, digits=4)), α=$(round.(msg.α, digits=3)), β=$(round.(msg.β, digits=3)), μ=$(round(msg.μ, digits=4))");
    println(io, "          Transition diag: $(round.([msg.T[1,1], msg.T[2,2]], digits=4))");
    println(io, "          Unconditional var per regime: $(round.([msg.ω[k] / max(1 - msg.α[k] - msg.β[k], 1e-6) for k in 1:2], digits=3))");
    println(io, "");
    println(io, rpad("Model", 13), " | ",
                rpad("MMD IS",   11), " | ", rpad("MMD OoS",  11), " | ",
                rpad("sig IS",   11), " | ", rpad("sig OoS",  11), " | ",
                rpad("AUC IS",   14), " | ", rpad("AUC OoS",  14));
    println(io, "-"^130);
    for m in MODEL_ORDER
        r = mmd_results[m];
        auc_is_s  = "$(round(r.auc_is,  digits=3)) ± $(round(r.auc_is_std,  digits=3))";
        auc_oos_s = "$(round(r.auc_oos, digits=3)) ± $(round(r.auc_oos_std, digits=3))";
        println(io, rpad(m, 13), " | ",
                    rpad(round(r.mmd_is,  digits=5), 11), " | ",
                    rpad(round(r.mmd_oos, digits=5), 11), " | ",
                    rpad(round(r.sig_is,  digits=5), 11), " | ",
                    rpad(round(r.sig_oos, digits=5), 11), " | ",
                    rpad(auc_is_s, 14), " | ", rpad(auc_oos_s, 14));
    end
end

open(joinpath(MSGARCH_DIR, "sim_pvalues.txt"), "w") do io
    println(io, "MS-GARCH baseline. Joint p-value coverage pv̄ with MS-GARCH row appended.");
    println(io, rpad("Model", 13), " | pv̄ IS   | pv̄ OoS");
    for m in MODEL_ORDER
        pv = pv_results[m];
        println(io, rpad(m, 13), " | ",
                    rpad(round(pv.pv_joint_is,  digits=3), 7), " | ",
                    rpad(round(pv.pv_joint_oos, digits=3), 7));
    end
end

open(joinpath(MSGARCH_DIR, "var_lr_tests.txt"), "w") do io
    println(io, "MS-GARCH baseline. Unconditional VaR LR tests with MS-GARCH row appended (α ∈ {0.01, 0.05}).");
    println(io, rpad("Model", 13), " | VaR01   | br%01 | LR_uc01 | p_uc01 | LR_ind01 | VaR05   | br%05 | LR_uc05 | p_uc05 | LR_ind05");
    for m in MODEL_ORDER
        r = var_results[m];
        println(io, rpad(m, 13), " | ",
                    rpad(round(r.v01, digits=3), 7), " | ",
                    rpad(round(100*r.k01.breach_rate, digits=1), 5), " | ",
                    rpad(round(r.k01.LR, digits=2), 7), " | ",
                    rpad(round(r.k01.pvalue, digits=3), 6), " | ",
                    rpad(round(r.c01.LR, digits=2), 8), " | ",
                    rpad(round(r.v05, digits=3), 7), " | ",
                    rpad(round(100*r.k05.breach_rate, digits=1), 5), " | ",
                    rpad(round(r.k05.LR, digits=2), 7), " | ",
                    rpad(round(r.k05.pvalue, digits=3), 6), " | ",
                    rpad(round(r.c05.LR, digits=2), 8));
    end
end

open(joinpath(MSGARCH_DIR, "summary.txt"), "w") do io
    println(io, "MS-GARCH baseline summary: MS-GARCH(K=2) vs single-regime GARCH vs CHMMs (SPY)");
    println(io, "="^80);
    println(io, "");
    println(io, "Fit params (HMP 2004 path-independent recursion):");
    println(io, "  regime 1 (calm):   ω=$(round(msg.ω[1], digits=4)) α=$(round(msg.α[1], digits=3)) β=$(round(msg.β[1], digits=3))  uncond-σ=$(round(sqrt(msg.ω[1] / max(1 - msg.α[1] - msg.β[1], 1e-6)), digits=3))");
    println(io, "  regime 2 (stress): ω=$(round(msg.ω[2], digits=4)) α=$(round(msg.α[2], digits=3)) β=$(round(msg.β[2], digits=3))  uncond-σ=$(round(sqrt(msg.ω[2] / max(1 - msg.α[2] - msg.β[2], 1e-6)), digits=3))");
    println(io, "  common μ = $(round(msg.μ, digits=4))");
    println(io, "  stickiness: p_11=$(round(msg.T[1,1], digits=4)) p_22=$(round(msg.T[2,2], digits=4))");
    println(io, "  log-likelihood: $(round(msg.log_likelihood, digits=2))");
    println(io, "");
    println(io, "Headline metrics (lower is better for MMD / sig-MMD, closer to 0.50 for AUC):");
    for m in ("GARCH", "MS-GARCH", "CHMM-t")
        if haskey(mmd_results, m)
            r = mmd_results[m];
            println(io, "  $(rpad(m, 10))  MMD IS $(round(r.mmd_is, digits=5))  sig-MMD IS $(round(r.sig_is, digits=5))  disc AUC IS $(round(r.auc_is, digits=3))  pv̄ IS $(round(pv_results[m].pv_joint_is, digits=3))  pv̄ OoS $(round(pv_results[m].pv_joint_oos, digits=3))");
        end
    end
end

println("\n" * "="^72);
println("  MS-GARCH baseline complete.");
println("  Results: $MSGARCH_DIR");
println("="^72);
