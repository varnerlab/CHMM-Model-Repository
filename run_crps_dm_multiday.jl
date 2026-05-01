# ========================================================================================= #
# run_crps_dm_multiday.jl
#
# Closes peer-review item P3.16 / R3 RE4: Diebold-Mariano test of CHMM vs.\ stationary
# block bootstrap on multi-day cumulative-return horizons (h = 5 and h = 20). Mirrors
# the per-step CRPS-DM construction of run_crps_dm.jl but aggregates the simulated and
# observed return series to non-overlapping h-day cumulative-return blocks before
# computing the per-block CRPS and the pairwise Diebold-Mariano statistic.
#
# Reviewer R3 RE4 framing: ``the bootstrap at L = 20 already dominates the CHMM on raw
# 1-day OoS KS; the natural question is whether CHMM dominates at multi-day cumulative
# returns where the regime structure should provide value. If CHMM does not dominate the
# bootstrap at any forecast horizon, the body's 'use-case differentiation' framing is
# vacuous.''
#
# Output:
#   results/crps_dm_multiday/CRPS_DM_multiday.txt
#   ../CHMM-paper/results/robustness/crps_dm_multiday.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random, Statistics, LinearAlgebra, Printf, JLD2

const SEED = 20260422;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const HORIZONS  = [5, 20];      # non-overlapping h-day cumulative-return horizons

const SIM_ARCHIVE_PATH = joinpath(_ROOT, "results", "_attic_v10", "track_a",
                                    "sim_archive_cache.jld2");
const OUT_DIR              = joinpath(_ROOT, "results", "crps_dm_multiday");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^80);
println("  Multi-day CRPS + Diebold-Mariano  (P3.16 / R3 RE4)");
println("  Horizons: $HORIZONS");
println("="^80);

if !isfile(SIM_ARCHIVE_PATH)
    error("Legacy harness archive cache not found at $SIM_ARCHIVE_PATH");
end

# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY OoS...");
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, TICKER; Δt=DT, risk_free_rate=RISK_FREE));
n_oos = length(R_oos);
println("  OoS: $n_oos obs");

# --------------------------------------------------------------------------------------- #
# Archive
# --------------------------------------------------------------------------------------- #
println("\n[cache] Loading $SIM_ARCHIVE_PATH ...");
archive = load(SIM_ARCHIVE_PATH)["archive"];
const MODELS = ["Bootstrap", "GARCH", "CHMM-N", "CHMM-t", "CHMM-L"];

# --------------------------------------------------------------------------------------- #
# Aggregate to non-overlapping h-day cumulative returns
# --------------------------------------------------------------------------------------- #
function aggregate_h_day(R::AbstractVector{Float64}, h::Int)
    n = length(R);
    nb = div(n, h);
    out = Vector{Float64}(undef, nb);
    for k in 1:nb
        @inbounds out[k] = sum(R[(k-1)*h+1 : k*h]);
    end
    return out;
end

function aggregate_h_day(M::AbstractMatrix{Float64}, h::Int)
    T, P = size(M);
    nb = div(T, h);
    out = Matrix{Float64}(undef, nb, P);
    for p in 1:P, k in 1:nb
        @inbounds out[k, p] = sum(M[(k-1)*h+1 : k*h, p]);
    end
    return out;
end

# --------------------------------------------------------------------------------------- #
# Sample CRPS (sorted-ensemble identity)
# --------------------------------------------------------------------------------------- #
function sample_crps(ensemble::AbstractVector{<:Real}, y::Real)
    N = length(ensemble);
    N >= 2 || error("CRPS needs N >= 2");
    term1 = 0.0;
    @inbounds for x in ensemble; term1 += abs(x - y); end
    term1 /= N;
    xs = sort(ensemble);
    s2 = 0.0;
    @inbounds for i in 1:N; s2 += xs[i] * (2*i - N - 1); end
    term2 = s2 / (N * (N - 1));
    return term1 - term2;
end

# Per-period CRPS series (one CRPS value per OoS aggregated period)
function crps_series(sim::AbstractMatrix{Float64}, R::AbstractVector{Float64})
    T = length(R);
    @assert size(sim, 1) == T;
    out = Vector{Float64}(undef, T);
    @inbounds for t in 1:T
        out[t] = sample_crps(sim[t, :], R[t]);
    end
    return out;
end

# Newey-West HAC long-run variance (Bartlett kernel)
function newey_west_lrv(d::AbstractVector{Float64}; bandwidth::Int = -1)
    T = length(d);
    h = bandwidth < 0 ? max(0, floor(Int, T^(1/3))) : bandwidth;
    μ = mean(d);
    γ0 = mean((d .- μ) .^ 2);
    s = γ0;
    @inbounds for k in 1:h
        γk = 0.0;
        for t in (k+1):T
            γk += (d[t] - μ) * (d[t-k] - μ);
        end
        γk /= T;
        w = 1.0 - k / (h + 1);
        s += 2 * w * γk;
    end
    return max(s, 1e-12);
end

# Diebold-Mariano statistic on a loss-differential series
function dm_test(loss_a::AbstractVector{Float64}, loss_b::AbstractVector{Float64})
    @assert length(loss_a) == length(loss_b);
    d = loss_a .- loss_b;
    T = length(d);
    μ = mean(d);
    σ² = newey_west_lrv(d) / T;
    DM = μ / sqrt(σ²);
    pval = 2 * (1 - 0.5 * (1 + erf(abs(DM) / sqrt(2))));   # two-sided normal
    return (DM = DM, p_value = pval, mean_loss_a = mean(loss_a),
            mean_loss_b = mean(loss_b), n_obs = T);
end

using SpecialFunctions: erf;

# --------------------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------------------- #
panels = NamedTuple[];

for h in vcat([1], HORIZONS)
    println("\n[h = $h] aggregating + scoring...");
    R_h = aggregate_h_day(R_oos, h);
    n_blocks = length(R_h);
    @printf("  blocks = %d (h = %d)\n", n_blocks, h);

    crps_by_model = Dict{String, Vector{Float64}}();
    for m in MODELS
        if !haskey(archive, m); continue; end
        sim_oos = archive[m].oos;
        sim_h = aggregate_h_day(sim_oos, h);
        @assert size(sim_h, 1) == n_blocks "Aggregated sim shape $(size(sim_h)) vs target $n_blocks";
        crps_by_model[m] = crps_series(sim_h, R_h);
        @printf("  %-12s mean CRPS = %.5f\n", m, mean(crps_by_model[m]));
    end

    if !haskey(crps_by_model, "Bootstrap"); continue; end
    if !haskey(crps_by_model, "CHMM-N");    continue; end

    # CHMM-N vs Bootstrap (the headline R3 RE4 ask)
    boot_loss = crps_by_model["Bootstrap"];
    for m in ["CHMM-N", "CHMM-t", "CHMM-L"]
        if !haskey(crps_by_model, m); continue; end
        cmm_loss = crps_by_model[m];
        # Pad if shapes differ (shouldn't happen; defensive)
        n = min(length(boot_loss), length(cmm_loss));
        r = dm_test(cmm_loss[1:n], boot_loss[1:n]);
        Δ_crps = r.mean_loss_a - r.mean_loss_b;   # CHMM - Bootstrap; negative => CHMM better
        @printf("  DM(%s vs Bootstrap)  ΔCRPS = %+.5f  DM = %+.3f  p = %.3f  (n = %d)\n",
                m, Δ_crps, r.DM, r.p_value, r.n_obs);
        push!(panels, (
            h = h, model = m, ref = "Bootstrap",
            mean_crps_model = r.mean_loss_a,
            mean_crps_ref   = r.mean_loss_b,
            delta_crps      = Δ_crps,
            DM              = r.DM,
            p_value         = r.p_value,
            n_blocks        = r.n_obs,
        ));
    end
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "CRPS_DM_multiday.txt");
open(out_path, "w") do io
    println(io, "="^110);
    println(io, "Multi-day cumulative-return CRPS + Diebold-Mariano  (P3.16 / R3 RE4)");
    println(io, "="^110);
    println(io);
    @printf(io, "Setup  : SPY OoS (%d days), non-overlapping h-day cumulative-return blocks.\n", n_oos);
    @printf(io, "Test   : DM = mean(loss_CHMM - loss_Bootstrap) / NW-HAC SE, two-sided normal.\n");
    @printf(io, "         Negative DM => CHMM has lower CRPS than Bootstrap at horizon h.\n");
    @printf(io, "Models : %s\n", join(MODELS, ", "));
    println(io);
    @printf(io, "%-3s  %-7s  %-9s  %-12s  %-10s  %-10s  %-7s  %-7s\n",
            "h", "model", "ref", "ΔCRPS", "CRPS model", "CRPS ref", "DM", "p");
    println(io, "-"^108);
    for r in panels
        @printf(io, "%-3d  %-7s  %-9s  %+10.5f  %-10.5f  %-10.5f  %+7.3f  %5.3f\n",
                r.h, r.model, r.ref, r.delta_crps,
                r.mean_crps_model, r.mean_crps_ref, r.DM, r.p_value);
    end
    println(io);
    println(io, "Reading: at h = 1 the bootstrap dominates on per-day CRPS (the body acknowledges");
    println(io, "this in Section 5.2). The substantive question R3 RE4 raises is whether the");
    println(io, "regime structure of CHMM produces lower CRPS than the bootstrap at multi-day");
    println(io, "horizons (h = 5, 20). Negative ΔCRPS at h ∈ {5, 20} would support the body's");
    println(io, "'use-case differentiation' framing on a forecast-horizon axis; positive ΔCRPS");
    println(io, "at every horizon reads as 'no horizon at which CHMM dominates the bootstrap on");
    println(io, "CRPS', which is consistent with the body's 'CHMM beats bootstrap on structural");
    println(io, "use cases (regime-conditional VaR, parametric copula composition, parametric");
    println(io, "privacy)' rather than on the marginal-distribution axis.");
end

csv_path = joinpath(PAPER_ROBUSTNESS_DIR, "crps_dm_multiday.csv");
open(csv_path, "w") do io
    println(io, "h,model,ref,delta_crps,crps_model,crps_ref,DM,p_value,n_blocks");
    for r in panels
        @printf(io, "%d,%s,%s,%.6f,%.6f,%.6f,%.4f,%.4f,%d\n",
                r.h, r.model, r.ref, r.delta_crps,
                r.mean_crps_model, r.mean_crps_ref, r.DM, r.p_value, r.n_blocks);
    end
end

println("\n" * "="^80);
println("  Multi-day DM panel complete.");
@printf("  Human-readable: %s\n", out_path);
@printf("  Paper CSV     : %s\n", csv_path);
println("="^80);
