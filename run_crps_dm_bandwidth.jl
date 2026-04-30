# ========================================================================================= #
# run_crps_dm_bandwidth.jl
#
# Diebold-Mariano bandwidth-sensitivity sweep for the headline CRPS panel. Addresses
# peer-review item P4.2 (R2.W5): the default bandwidth h = floor(T^(1/3)) = 8 at
# T_OoS = 572 may understate long-run variance under strong volatility clustering.
# Sweep h in {4, 8, 16, 32}; report DM p-values for the within-CHMM family pairs and
# the CHMM-vs-best-baseline pairs.
#
# Inputs : results/_attic_v10/track_a/sim_archive_cache.jld2 (legacy harness cache)
# Output : results/crps_dm/CRPS_DM_bandwidth_sweep.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random, Statistics, JLD2, FileIO, Printf, SpecialFunctions
const SEED = 20260422;
Random.seed!(SEED);

const TICKER     = "SPY";
const RISK_FREE  = 0.0;
const DT         = 1/252;

const SIM_ARCHIVE_PATH = joinpath(_ROOT, "results", "_attic_v10", "track_a", "sim_archive_cache.jld2");
const OUT_DIR          = joinpath(_ROOT, "results", "crps_dm");
mkpath(OUT_DIR);

const ARCHIVE_PATH = SIM_ARCHIVE_PATH;
const OUT_PATH     = joinpath(OUT_DIR, "CRPS_DM_bandwidth_sweep.txt");

const BANDWIDTHS  = [4, 8, 16, 32];
const FAMILY_PAIRS = [
    ("CHMM-N", "CHMM-t"),
    ("CHMM-N", "CHMM-L"),
    ("CHMM-t", "CHMM-L"),
    ("CHMM-N", "Bootstrap"),
    ("CHMM-N", "GARCH"),
    ("CHMM-t", "GARCH"),
];

println("="^88);
println("  CRPS Diebold-Mariano bandwidth-sensitivity sweep  (peer-review P4.2)");
println("  Bandwidths: $BANDWIDTHS");
println("="^88);

# ----------------------------------------------------------------------------------------- #
# Helpers (copied from run_crps_dm.jl to keep this script self-contained)
# ----------------------------------------------------------------------------------------- #
function dm_statistic(loss_a::AbstractVector, loss_b::AbstractVector; bandwidth::Int)
    T = length(loss_a);
    length(loss_b) == T || error("DM requires equal-length loss series");
    d = loss_a .- loss_b;
    dbar = mean(d);
    h = bandwidth;
    γ0 = sum((d .- dbar) .^ 2) / T;
    γ_sum = γ0;
    for k in 1:h
        γk = sum((d[1:end-k] .- dbar) .* (d[1+k:end] .- dbar)) / T;
        w = 1.0 - k / (h + 1);
        γ_sum += 2 * w * γk;
    end
    σ_lr = sqrt(max(γ_sum, eps()));
    dm = dbar / (σ_lr / sqrt(T));
    pval = 2 * (1 - 0.5 * (1 + erf(abs(dm) / sqrt(2))));
    return (dm=dm, dbar=dbar, σ_lr=σ_lr, h=h, pval=pval, T=T);
end

function crps_series(arch::AbstractMatrix, y::AbstractVector)
    T, N = size(arch);
    length(y) == T || error("Ensemble T mismatch");
    losses = zeros(Float64, T);
    for t in 1:T
        x = sort(arch[t, :]);
        s1 = mean(abs.(x .- y[t]));
        s2 = 0.0;
        for i in 1:N
            s2 += (2 * i - N - 1) * x[i];
        end
        s2 /= (N * (N - 1));
        losses[t] = s1 - s2;
    end
    return losses;
end

# ----------------------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY OoS...");
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset, TICKER; Δt=DT, risk_free_rate=RISK_FREE);
n_oos = length(R_oos);
println("  OoS T = $n_oos");

if !isfile(ARCHIVE_PATH)
    error("Legacy archive cache not found at $ARCHIVE_PATH");
end
println("[cache] Loading $(basename(ARCHIVE_PATH))...");
archive = load(ARCHIVE_PATH)["archive"];

# ----------------------------------------------------------------------------------------- #
# Compute per-model CRPS series on OoS
# ----------------------------------------------------------------------------------------- #
const NEEDED = unique(vcat([p[1] for p in FAMILY_PAIRS], [p[2] for p in FAMILY_PAIRS]));
println("\n[CRPS] Per-t CRPS for: $(join(NEEDED, ", "))");
crps_loss = Dict{String,Vector{Float64}}();
for m in NEEDED
    if !haskey(archive, m)
        @warn "Archive missing model $m; skipping";
        continue;
    end
    arch_oos = archive[m].oos;
    losses = crps_series(arch_oos, R_oos);
    crps_loss[m] = losses;
    @printf("  %-12s mean CRPS = %.5f   median = %.5f\n", m, mean(losses), median(losses));
end

# ----------------------------------------------------------------------------------------- #
# Sweep
# ----------------------------------------------------------------------------------------- #
println("\n[DM] Bandwidth sweep:");
results = Dict{Tuple{String,String,Int}, NamedTuple}();
for (a, b) in FAMILY_PAIRS
    if !haskey(crps_loss, a) || !haskey(crps_loss, b); continue; end
    for h in BANDWIDTHS
        r = dm_statistic(crps_loss[a], crps_loss[b]; bandwidth=h);
        results[(a, b, h)] = r;
    end
end

open(OUT_PATH, "w") do io
    println(io, "="^104);
    println(io, "CRPS Diebold-Mariano bandwidth-sensitivity sweep  (peer-review P4.2 / R2.W5)");
    println(io, "="^104);
    println(io, "Setup : SPY OoS T = $n_oos days, archive seed = 20260422.");
    println(io, "Test  : H0 : E[CRPS_A - CRPS_B] = 0,  HAC variance = Newey-West Bartlett kernel.");
    println(io, "        Default bandwidth in run_crps_dm.jl is h = floor(T^(1/3)) = $(floor(Int, n_oos^(1/3)))");
    println(io);
    println(io, "Reading: bandwidth-sensitivity is the question of whether DM p-values move when the");
    println(io, "         long-run variance estimator's truncation lag changes. If p-values are stable");
    println(io, "         across h ∈ {4, 8, 16, 32}, the within-CHMM equivalence finding is robust.");
    println(io);
    @printf(io, "%-22s | %s\n", "Pair (A vs B)",
            join([@sprintf("h=%-3d (DM, p)", h) for h in BANDWIDTHS], "  "));
    println(io, "-"^104);
    for (a, b) in FAMILY_PAIRS
        if !haskey(crps_loss, a) || !haskey(crps_loss, b); continue; end
        cells = String[];
        for h in BANDWIDTHS
            r = results[(a, b, h)];
            push!(cells, @sprintf("(%+.3f, %.3f)", r.dm, r.pval));
        end
        @printf(io, "%-22s | %s\n", "$a vs $b", join(cells, "  "));
    end
    println(io);
    println(io, "Robustness check:");
    println(io, "  Within-CHMM pairs (CHMM-N vs CHMM-t/-L) should retain p > 0.05 across all h.");
    println(io, "  CHMM-vs-baseline pairs should retain their direction across all h.");
end

println("\n[done] Wrote $OUT_PATH");
println();
@printf("%-22s | %s\n", "Pair", join([@sprintf("h=%-3d (DM, p)", h) for h in BANDWIDTHS], "  "));
println("-"^96);
for (a, b) in FAMILY_PAIRS
    if !haskey(crps_loss, a) || !haskey(crps_loss, b); continue; end
    cells = String[];
    for h in BANDWIDTHS
        r = results[(a, b, h)];
        push!(cells, @sprintf("(%+.3f, %.3f)", r.dm, r.pval));
    end
    @printf("%-22s | %s\n", "$a vs $b", join(cells, "  "));
end
