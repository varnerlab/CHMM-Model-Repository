# ========================================================================================= #
# run_crps_dm.jl
#
#
# Each generator's archive is a marginal (unconditional) predictive ensemble at every t in
# the held-out OoS window. The sample CRPS for observation y_t against ensemble x_{t,1..N}
# uses the unbiased estimator
#
#   CRPS_t = (1/N) sum_i |x_{t,i} - y_t| - (1/(N(N-1))) sum_{i<j} |x_{t,(i)} - x_{t,(j)}|
#
# with the second term computed in O(N log N) via the sorted-ensemble identity
# sum_{i<j} (x_(j) - x_(i)) = sum_i x_(i) * (2i - N - 1).
#
# Diebold-Mariano statistic on the per-t CRPS loss differential d_t = CRPS_t^A - CRPS_t^B
# uses a Newey-West HAC variance with Bartlett kernel and bandwidth h = floor(T^(1/3)).
#
# Inputs : results/_attic_v10/track_a/sim_archive_cache.jld2  (legacy harness cache)
# Outputs: results/crps_dm/CRPS_DM.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random, Statistics, JLD2, FileIO, Printf
const SEED = 20260422;
Random.seed!(SEED);

const TICKER     = "SPY";
const RISK_FREE  = 0.0;
const DT         = 1/252;

const SIM_ARCHIVE_PATH = joinpath(_ROOT, "results", "_attic_v10", "track_a", "sim_archive_cache.jld2");
const OUT_DIR          = joinpath(_ROOT, "results", "crps_dm");
mkpath(OUT_DIR);

const ARCHIVE_PATH = SIM_ARCHIVE_PATH;
const OUT_PATH     = joinpath(OUT_DIR, "CRPS_DM.txt");

println("="^88)
println("  CRPS + Diebold-Mariano on the headline panel.")
println("  Seed: $SEED")
println("="^88)

if !isfile(ARCHIVE_PATH)
    error("Legacy harness archive cache not found at $ARCHIVE_PATH; restore from _attic_v10/.");
end

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
idx_spy = findfirst(==(TICKER), all_tickers);
R_is = all_R[:, idx_spy];
n_is = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset, TICKER; Δt=DT, risk_free_rate=RISK_FREE);
n_oos = length(R_oos);

println("  IS: $n_is obs; OoS: $n_oos obs");

# --------------------------------------------------------------------------------------- #
# Archive
# --------------------------------------------------------------------------------------- #
println("\n[cache] Loading simulation archives from $(basename(ARCHIVE_PATH))...");
archive = load(ARCHIVE_PATH)["archive"];

const MODEL_ORDER = [
    "Bootstrap", "Gaussian", "Laplace",
    "DiscreteNJ", "DiscreteWJ", "GARCH",
    "CHMM-N", "CHMM-t", "CHMM-L",
];

# --------------------------------------------------------------------------------------- #
# Sample CRPS via the sorted-ensemble identity (unbiased estimator).
# --------------------------------------------------------------------------------------- #
function sample_crps(ensemble::AbstractVector{<:Real}, y::Real)
    N = length(ensemble);
    N >= 2 || error("CRPS needs N >= 2 ensemble members");
    # Term 1: (1/N) sum_i |x_i - y|
    term1 = 0.0;
    @inbounds for x in ensemble
        term1 += abs(x - y);
    end
    term1 /= N;
    # Term 2 (unbiased): (1/(N(N-1))) sum_{i<j} |x_(j) - x_(i)|, with the
    # sorted-ensemble identity sum_{i<j}(x_(j) - x_(i)) = sum_i x_(i)*(2i - N - 1).
    xs = sort(ensemble);
    s2 = 0.0;
    @inbounds for i in 1:N
        s2 += xs[i] * (2*i - N - 1);
    end
    term2 = s2 / (N * (N - 1));
    return term1 - term2;
end

function crps_series(arch_oos::AbstractMatrix, y_oos::AbstractVector)
    T, N = size(arch_oos);
    length(y_oos) == T || error("Ensemble T=$T mismatch with y_oos length=$(length(y_oos))");
    losses = Array{Float64}(undef, T);
    @inbounds for t in 1:T
        losses[t] = sample_crps(view(arch_oos, t, :), y_oos[t]);
    end
    return losses;
end

# --------------------------------------------------------------------------------------- #
# Diebold-Mariano with Newey-West HAC variance (Bartlett kernel).
# --------------------------------------------------------------------------------------- #
function dm_statistic(loss_a::AbstractVector, loss_b::AbstractVector;
                      bandwidth::Union{Nothing,Int}=nothing)
    T = length(loss_a);
    length(loss_b) == T || error("DM requires equal-length loss series");
    d = loss_a .- loss_b;
    dbar = mean(d);
    h = bandwidth === nothing ? max(1, floor(Int, T^(1/3))) : bandwidth;
    # Newey-West (Bartlett kernel) long-run variance estimator
    γ0 = sum((d .- dbar) .^ 2) / T;
    γ_sum = γ0;
    for k in 1:h
        γk = sum((d[1:end-k] .- dbar) .* (d[1+k:end] .- dbar)) / T;
        w = 1.0 - k / (h + 1);
        γ_sum += 2 * w * γk;
    end
    σ_lr = sqrt(max(γ_sum, eps()));
    dm = dbar / (σ_lr / sqrt(T));
    # Two-sided p-value under standard normal null
    pval = 2 * (1 - cdf_std_normal(abs(dm)));
    return (dm=dm, dbar=dbar, σ_lr=σ_lr, h=h, pval=pval, T=T);
end

# Standard normal CDF without an extra dependency.
function cdf_std_normal(x::Real)
    return 0.5 * (1 + erf(x / sqrt(2)));
end

# --------------------------------------------------------------------------------------- #
# Compute per-model CRPS series and headline summaries
# --------------------------------------------------------------------------------------- #
println("\n[CRPS] Computing per-t sample CRPS for each generator (OoS, T=$n_oos)");

crps_loss = Dict{String,Vector{Float64}}();
for m in MODEL_ORDER
    print("  $m...");
    arch_oos = archive[m].oos;
    losses = crps_series(arch_oos, R_oos);
    crps_loss[m] = losses;
    println(@sprintf("  mean CRPS = %.5f   median = %.5f", mean(losses), median(losses)));
end

# Also compute IS CRPS for the headline summary table
println("\n[CRPS] Computing per-t sample CRPS for each generator (IS, T=$n_is)");
crps_loss_is = Dict{String,Vector{Float64}}();
for m in MODEL_ORDER
    print("  $m...");
    arch_is = archive[m].is;
    losses = crps_series(arch_is, R_is);
    crps_loss_is[m] = losses;
    println(@sprintf("  mean CRPS = %.5f   median = %.5f", mean(losses), median(losses)));
end

# --------------------------------------------------------------------------------------- #
# Pairwise DM tests against the headline CHMM rows
# --------------------------------------------------------------------------------------- #
const REFERENCE_MODELS = ["CHMM-N", "CHMM-t", "CHMM-L"];

println("\n[DM] Pairwise Diebold-Mariano (CHMM row vs each benchmark; OoS)");
dm_results = Dict{Tuple{String,String},NamedTuple}();
for ref in REFERENCE_MODELS
    for bench in MODEL_ORDER
        bench == ref && continue;
        out = dm_statistic(crps_loss[ref], crps_loss[bench]);
        dm_results[(ref, bench)] = out;
    end
end

# --------------------------------------------------------------------------------------- #
# Write report
# --------------------------------------------------------------------------------------- #
println("\n[write] $OUT_PATH");

open(OUT_PATH, "w") do io
    println(io, "="^110);
    println(io, "CRPS + Diebold-Mariano on the headline panel.");
    println(io, "="^110);
    println(io);
    println(io, "Setup    : SPY daily log excess growth; IS n=$n_is, OoS n=$n_oos; seed=$SEED.");
    println(io, "Ensemble : N_paths=1000 unconditional simulated paths per generator (legacy harness cache).");
    println(io, "CRPS     : sample CRPS via the unbiased sorted-ensemble identity.");
    println(io, "DM       : Newey-West HAC variance, Bartlett kernel, bandwidth h = floor(T^(1/3)).");
    println(io);
    println(io, "Reading  : Lower CRPS is better. DM > 0 means the reference (CHMM) loss exceeds the");
    println(io, "           benchmark loss, i.e. CHMM is worse on CRPS by that test. p-values are two-sided.");
    println(io, "-"^110);
    println(io);
    println(io, "Headline mean CRPS by generator:");
    println(io);
    @printf(io, "%-12s | %-12s | %-12s | %-12s | %-12s\n",
            "Model", "IS mean", "IS median", "OoS mean", "OoS median");
    println(io, "-"^72);
    for m in MODEL_ORDER
        @printf(io, "%-12s | %-12.5f | %-12.5f | %-12.5f | %-12.5f\n",
                m, mean(crps_loss_is[m]), median(crps_loss_is[m]),
                mean(crps_loss[m]), median(crps_loss[m]));
    end
    println(io);
    println(io, "-"^110);
    println(io);
    println(io, "Pairwise DM tests on OoS CRPS loss differentials.");
    println(io, "Sign convention: positive DM means the column row's CRPS exceeds the reference row's CRPS,");
    println(io, "i.e. the reference (CHMM) row is better.");
    println(io);
    for ref in REFERENCE_MODELS
        @printf(io, "Reference: %s   (DM bandwidth h = %d)\n", ref, dm_results[(ref, MODEL_ORDER[1])].h);
        println(io, "-"^72);
        @printf(io, "%-14s | %-10s | %-10s | %-10s | %-10s\n",
                "Benchmark", "CRPS bench", "ΔCRPS", "DM stat", "p-value");
        for bench in MODEL_ORDER
            bench == ref && continue;
            r = dm_results[(ref, bench)];
            @printf(io, "%-14s | %-10.5f | %-10.5f | %-10.4f | %-10.4f\n",
                    bench, mean(crps_loss[bench]), -r.dbar, -r.dm, r.pval);
        end
        println(io);
    end
    println(io, "="^110);
    println(io);
    println(io, "Notes:");
    println(io, "  - Δ CRPS column = CRPS(benchmark) - CRPS(reference). Positive means CHMM is better.");
    println(io, "  - DM stat sign flipped relative to dbar so positive DM stat tracks Δ CRPS sign.");
    println(io, "  - For the 'Bootstrap' benchmark: simulated paths are i.i.d. resamples of IS, so the IS");
    println(io, "    CRPS is artificially low; the OoS CRPS is the meaningful comparison.");
end

println();
println("[done] CRPS + DM results written to $OUT_PATH");

# Also save a JLD2 with the per-t loss series for downstream use.
const JLD_OUT = joinpath(OUT_DIR, "crps_loss_series.jld2");
save(JLD_OUT, "crps_loss_oos", crps_loss, "crps_loss_is", crps_loss_is, "dm_results", dm_results,
     "model_order", MODEL_ORDER, "reference_models", REFERENCE_MODELS);
println("[done] Per-t loss series saved to $JLD_OUT");
