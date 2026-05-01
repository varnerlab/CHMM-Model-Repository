# ========================================================================================= #
# run_crps_dm_multiday_replication.jl
#
# Reviewer Round 2 / Item B1 (peer-review.md R1#2, R2#req-3, R3#1, R3#req-1).
#
# The body multi-day DM result (CHMM-N beats bootstrap at h=20 on SPY OoS, p=0.003,
# n=28 non-overlapping blocks) is the strongest empirical CHMM-vs-bootstrap differentiator.
# All three reviewers ask for replication: across the six-asset universe of Section 5
# (SPY, NVDA, JNJ, JPM, AAPL, QQQ), with bandwidth sensitivity at NW h ∈ {2, 4, 8, 16},
# and with overlapping vs non-overlapping blocks.
#
# This runner produces three panels:
#   (1) Cross-asset h=20 panel: per-asset DM at h=20 against block-bootstrap baseline,
#       n=28 non-overlapping blocks per asset (the body finding replicated 5 more times).
#   (2) Bandwidth sweep at h=20 on SPY: NW bandwidth h_NW ∈ {2, 4, 8, 16}, plus default.
#   (3) Overlapping vs non-overlapping at h=20 on SPY: overlapping blocks (n=552) vs
#       non-overlapping (n=28), under the same NW HAC adjustment.
#
# All fits at K* = 3 (body headline). 1,000 paths per generator per asset.
#
# Output:
#   results/crps_dm_multiday_replication/cross_asset_h20.csv
#   results/crps_dm_multiday_replication/spy_bandwidth_h20.csv
#   results/crps_dm_multiday_replication/spy_overlap_h20.csv
#   results/crps_dm_multiday_replication/crps_dm_multiday_replication.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using LinearAlgebra
using Printf
using SpecialFunctions: erf

const SEED = 20260422;
Random.seed!(SEED);

const RISK_FREE = 0.0;
const DT = 1/252;
const K_HEADLINE = 3;
const N_PATHS = 1000;
const MAX_ITER = 60;
const BLOCK_LEN = 20;
const TICKERS = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];

const OUT_DIR = joinpath(_ROOT, "results", "crps_dm_multiday_replication");
mkpath(OUT_DIR);

println("="^80)
println("  Multi-day CRPS DM replication panel (B1)")
println("  Tickers: $TICKERS    K = $K_HEADLINE    Paths = $N_PATHS    h = 20")
println("="^80)

# ----------------------------------------------------------------------------------------- #
# Data loading
# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading IS / OoS panels...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R_is = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];

R_is = Dict{String,Vector{Float64}}()
R_oos = Dict{String,Vector{Float64}}()
for tk in TICKERS
    idx = findfirst(==(tk), all_tickers)
    if idx === nothing; continue; end
    R_is[tk] = collect(all_R_is[:, idx])
    R_oos[tk] = Vector{Float64}(log_growth_matrix(oos_dataset, tk; Δt=DT, risk_free_rate=RISK_FREE))
    @printf("  %-5s : IS = %4d   OoS = %3d\n", tk, length(R_is[tk]), length(R_oos[tk]))
end

# ----------------------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------------------- #
function _stationary(model, K::Int)
    T = zeros(K, K)
    for i in 1:K; T[i, :] = probs(model.transition[i]); end
    π = (T^1000)[1, :]
    return Categorical(π)
end

function _sim_chmm_paths(model, sd, n::Int, np::Int)
    sim = Matrix{Float64}(undef, n, np)
    for p in 1:np
        s0 = rand(sd)
        st = model(s0, n)
        for j in 1:n; sim[j, p] = rand(model.emission[st[j]]); end
    end
    return sim
end

# Stationary block bootstrap (Politis-Romano 1994) on R_src to length n, np paths.
function _block_bootstrap(R_src::AbstractVector, n::Int, np::Int; mean_block_len::Int=BLOCK_LEN, rng=Random.default_rng())
    L = length(R_src)
    p_geo = 1.0 / mean_block_len
    sim = Matrix{Float64}(undef, n, np)
    for path in 1:np
        t = 0
        while t < n
            start = rand(rng, 1:L)
            blocklen = max(1, rand(rng, Distributions.Geometric(p_geo)) + 1)
            for k in 0:(blocklen-1)
                t += 1
                if t > n; break; end
                sim[t, path] = R_src[((start - 1 + k) % L) + 1]
            end
        end
    end
    return sim
end

function aggregate_h_day_nonoverlap(R::AbstractVector{Float64}, h::Int)
    nb = div(length(R), h)
    out = Vector{Float64}(undef, nb)
    for k in 1:nb; out[k] = sum(R[(k-1)*h+1 : k*h]); end
    return out
end

function aggregate_h_day_nonoverlap(M::AbstractMatrix{Float64}, h::Int)
    T_, P = size(M)
    nb = div(T_, h)
    out = Matrix{Float64}(undef, nb, P)
    for p in 1:P, k in 1:nb
        out[k, p] = sum(M[(k-1)*h+1 : k*h, p])
    end
    return out
end

function aggregate_h_day_overlap(R::AbstractVector{Float64}, h::Int)
    n = length(R) - h + 1
    out = Vector{Float64}(undef, n)
    for k in 1:n; out[k] = sum(R[k:k+h-1]); end
    return out
end

function aggregate_h_day_overlap(M::AbstractMatrix{Float64}, h::Int)
    T_, P = size(M)
    n = T_ - h + 1
    out = Matrix{Float64}(undef, n, P)
    for p in 1:P, k in 1:n
        out[k, p] = sum(M[k:k+h-1, p])
    end
    return out
end

function sample_crps(ensemble::AbstractVector{<:Real}, y::Real)
    N = length(ensemble); N >= 2 || error("CRPS needs N >= 2")
    term1 = 0.0
    @inbounds for x in ensemble; term1 += abs(x - y); end
    term1 /= N
    xs = sort(ensemble)
    s2 = 0.0
    @inbounds for i in 1:N; s2 += xs[i] * (2*i - N - 1); end
    term2 = s2 / (N * (N - 1))
    return term1 - term2
end

function crps_series(sim::AbstractMatrix{Float64}, R::AbstractVector{Float64})
    T_ = length(R); @assert size(sim, 1) == T_
    out = Vector{Float64}(undef, T_)
    @inbounds for t in 1:T_; out[t] = sample_crps(sim[t, :], R[t]); end
    return out
end

function newey_west_lrv(d::AbstractVector{Float64}; bandwidth::Int = -1)
    T_ = length(d)
    h = bandwidth < 0 ? max(0, floor(Int, T_^(1/3))) : bandwidth
    μ = mean(d)
    γ0 = mean((d .- μ) .^ 2)
    s = γ0
    for k in 1:h
        γk = 0.0
        for t in (k+1):T_; γk += (d[t] - μ) * (d[t-k] - μ); end
        γk /= T_
        w = 1.0 - k / (h + 1)
        s += 2 * w * γk
    end
    return max(s, 1e-12)
end

function dm_test(loss_a::AbstractVector{Float64}, loss_b::AbstractVector{Float64};
                 bandwidth::Int = -1)
    @assert length(loss_a) == length(loss_b)
    d = loss_a .- loss_b
    T_ = length(d)
    μ = mean(d)
    σ² = newey_west_lrv(d; bandwidth=bandwidth) / T_
    DM = μ / sqrt(σ²)
    pval = 2 * (1 - 0.5 * (1 + erf(abs(DM) / sqrt(2))))
    return (DM=DM, p=pval, mean_a=mean(loss_a), mean_b=mean(loss_b), n=T_)
end

# ----------------------------------------------------------------------------------------- #
# Per-ticker fits + simulations
# ----------------------------------------------------------------------------------------- #
println("\n[fit] Fitting CHMM-N at K = $K_HEADLINE per ticker...")
chmm_sim_oos = Dict{String,Matrix{Float64}}()
boot_sim_oos = Dict{String,Matrix{Float64}}()
for tk in TICKERS
    if !haskey(R_is, tk); continue; end
    println("  $tk : CHMM-N fit on IS T=$(length(R_is[tk])) ...")
    chmm_n = build(MyContinuousHiddenMarkovModel,
        (observations=R_is[tk], number_of_states=K_HEADLINE, max_iter=MAX_ITER));
    sd = _stationary(chmm_n, K_HEADLINE)
    n_oos_t = length(R_oos[tk])
    chmm_sim_oos[tk] = _sim_chmm_paths(chmm_n, sd, n_oos_t, N_PATHS)
    boot_sim_oos[tk] = _block_bootstrap(R_is[tk], n_oos_t, N_PATHS; mean_block_len=BLOCK_LEN)
end

# ----------------------------------------------------------------------------------------- #
# Panel 1: cross-asset h=20 DM (non-overlapping blocks)
# ----------------------------------------------------------------------------------------- #
println("\n[panel 1] cross-asset h=20 DM (non-overlapping)...")
cross_rows = NamedTuple[]
for tk in TICKERS
    if !haskey(chmm_sim_oos, tk); continue; end
    R_h = aggregate_h_day_nonoverlap(R_oos[tk], 20)
    chmm_h = aggregate_h_day_nonoverlap(chmm_sim_oos[tk], 20)
    boot_h = aggregate_h_day_nonoverlap(boot_sim_oos[tk], 20)
    chmm_loss = crps_series(chmm_h, R_h)
    boot_loss = crps_series(boot_h, R_h)
    r = dm_test(chmm_loss, boot_loss)
    push!(cross_rows, (ticker=tk, n_blocks=r.n,
        crps_chmm=r.mean_a, crps_boot=r.mean_b,
        delta_crps=r.mean_a - r.mean_b, DM=r.DM, p=r.p))
    @printf("  %-5s : n=%2d  ΔCRPS=%+.4f  CHMM=%.4f  Boot=%.4f  DM=%+.3f  p=%.3f\n",
            tk, r.n, r.mean_a - r.mean_b, r.mean_a, r.mean_b, r.DM, r.p)
end

# ----------------------------------------------------------------------------------------- #
# Panel 2: bandwidth sweep at h=20 on SPY only (CHMM-N vs Bootstrap)
# ----------------------------------------------------------------------------------------- #
println("\n[panel 2] bandwidth sweep at h=20 on SPY...")
bw_rows = NamedTuple[]
R_h_spy = aggregate_h_day_nonoverlap(R_oos["SPY"], 20)
chmm_h_spy = aggregate_h_day_nonoverlap(chmm_sim_oos["SPY"], 20)
boot_h_spy = aggregate_h_day_nonoverlap(boot_sim_oos["SPY"], 20)
chmm_loss_spy = crps_series(chmm_h_spy, R_h_spy)
boot_loss_spy = crps_series(boot_h_spy, R_h_spy)
for bw in [-1, 2, 4, 8, 16]
    r = dm_test(chmm_loss_spy, boot_loss_spy; bandwidth=bw)
    bw_label = bw < 0 ? "default(n^1/3)" : string(bw)
    push!(bw_rows, (bandwidth=bw_label, h_nw=(bw < 0 ? floor(Int, r.n^(1/3)) : bw),
        n_blocks=r.n, delta_crps=r.mean_a - r.mean_b, DM=r.DM, p=r.p))
    @printf("  bw=%-15s n=%2d  ΔCRPS=%+.4f  DM=%+.3f  p=%.3f\n",
            bw_label, r.n, r.mean_a - r.mean_b, r.DM, r.p)
end

# ----------------------------------------------------------------------------------------- #
# Panel 3: overlapping vs non-overlapping at h=20 on SPY
# ----------------------------------------------------------------------------------------- #
println("\n[panel 3] overlapping vs non-overlapping at h=20 on SPY...")
overlap_rows = NamedTuple[]
# non-overlapping (same as Panel 2 default bandwidth row)
r_nono = dm_test(chmm_loss_spy, boot_loss_spy)
push!(overlap_rows, (mode="non-overlapping", n_blocks=r_nono.n,
    delta_crps=r_nono.mean_a - r_nono.mean_b, DM=r_nono.DM, p=r_nono.p))

# overlapping at h=20 with default NW bandwidth (n_eff much larger)
R_h_ov = aggregate_h_day_overlap(R_oos["SPY"], 20)
chmm_h_ov = aggregate_h_day_overlap(chmm_sim_oos["SPY"], 20)
boot_h_ov = aggregate_h_day_overlap(boot_sim_oos["SPY"], 20)
chmm_loss_ov = crps_series(chmm_h_ov, R_h_ov)
boot_loss_ov = crps_series(boot_h_ov, R_h_ov)
# For overlapping blocks the natural NW bandwidth is at least h-1 = 19 (overlap induces
# h-1-step autocorrelation in the loss differential).
for bw in [-1, 19, 30, 60]
    r = dm_test(chmm_loss_ov, boot_loss_ov; bandwidth=bw)
    bw_label = bw < 0 ? "default" : string(bw)
    push!(overlap_rows, (mode="overlap (bw=$bw_label)", n_blocks=r.n,
        delta_crps=r.mean_a - r.mean_b, DM=r.DM, p=r.p))
end
for r in overlap_rows
    @printf("  %-22s n=%4d  ΔCRPS=%+.4f  DM=%+.3f  p=%.3f\n",
            r.mode, r.n_blocks, r.delta_crps, r.DM, r.p)
end

# ----------------------------------------------------------------------------------------- #
# Output
# ----------------------------------------------------------------------------------------- #
csv1 = joinpath(OUT_DIR, "cross_asset_h20.csv")
open(csv1, "w") do io
    println(io, "ticker,n_blocks,crps_chmm,crps_boot,delta_crps,DM,p_value")
    for r in cross_rows
        @printf(io, "%s,%d,%.6f,%.6f,%.6f,%.4f,%.4f\n",
                r.ticker, r.n_blocks, r.crps_chmm, r.crps_boot, r.delta_crps, r.DM, r.p)
    end
end

csv2 = joinpath(OUT_DIR, "spy_bandwidth_h20.csv")
open(csv2, "w") do io
    println(io, "bandwidth_label,h_nw,n_blocks,delta_crps,DM,p_value")
    for r in bw_rows
        @printf(io, "%s,%d,%d,%.6f,%.4f,%.4f\n",
                r.bandwidth, r.h_nw, r.n_blocks, r.delta_crps, r.DM, r.p)
    end
end

csv3 = joinpath(OUT_DIR, "spy_overlap_h20.csv")
open(csv3, "w") do io
    println(io, "mode,n_blocks,delta_crps,DM,p_value")
    for r in overlap_rows
        @printf(io, "%s,%d,%.6f,%.4f,%.4f\n",
                r.mode, r.n_blocks, r.delta_crps, r.DM, r.p)
    end
end

txt = joinpath(OUT_DIR, "crps_dm_multiday_replication.txt")
open(txt, "w") do io
    println(io, "="^110)
    println(io, "Multi-day CRPS DM replication panel (B1)")
    println(io, "="^110)
    println(io)
    println(io, "Setup: per-ticker CHMM-N fit at K=$K_HEADLINE on IS, $N_PATHS paths, OoS aggregated to non-overlapping h=20 blocks.")
    println(io, "Block bootstrap baseline: stationary block bootstrap of Politis-Romano 1994 at mean block length L=$BLOCK_LEN.")
    println(io)

    println(io, "PANEL 1 - Cross-asset h=20 DM (non-overlapping blocks, default NW bandwidth):")
    println(io, "  ticker  n     ΔCRPS    DM       p")
    println(io, "  ", "-"^60)
    for r in cross_rows
        @printf(io, "  %-5s   %2d  %+8.4f  %+7.3f   %.3f\n",
                r.ticker, r.n_blocks, r.delta_crps, r.DM, r.p)
    end
    println(io)

    println(io, "PANEL 2 - SPY h=20 NW-HAC bandwidth sweep:")
    println(io, "  bandwidth         h_nw  n   ΔCRPS    DM       p")
    println(io, "  ", "-"^60)
    for r in bw_rows
        @printf(io, "  %-15s  %3d   %2d  %+8.4f  %+7.3f   %.3f\n",
                r.bandwidth, r.h_nw, r.n_blocks, r.delta_crps, r.DM, r.p)
    end
    println(io)

    println(io, "PANEL 3 - SPY h=20 overlapping vs non-overlapping blocks:")
    println(io, "  mode                    n      ΔCRPS    DM       p")
    println(io, "  ", "-"^60)
    for r in overlap_rows
        @printf(io, "  %-22s  %4d  %+8.4f  %+7.3f   %.3f\n",
                r.mode, r.n_blocks, r.delta_crps, r.DM, r.p)
    end
    println(io)

    println(io, "Reading.")
    println(io, "  PANEL 1: replicates the body's SPY h=20 CHMM-vs-bootstrap DM result across the")
    println(io, "    six-asset universe of Section 5. ΔCRPS < 0 => CHMM has lower CRPS at h=20.")
    println(io, "    Median ΔCRPS and median p across the six assets are the headline robustness check.")
    println(io, "  PANEL 2: bandwidth sensitivity on SPY h=20. Stable p across bandwidths => robust;")
    println(io, "    p moves => bandwidth-dependent.")
    println(io, "  PANEL 3: overlapping blocks at h=20 give n_eff ≈ 552 with NW bandwidth ≥ h-1 = 19.")
    println(io, "    Both should reach the same conclusion if the underlying signal is real.")
end

println("\n[done] $csv1")
println("[done] $csv2")
println("[done] $csv3")
println("[done] $txt")
