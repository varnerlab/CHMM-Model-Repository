# ========================================================================================= #
# run_state_distinctness.jl
#
# Effective number of distinct states diagnostic at K = 18 on SPY IS for CHMM-N and
# CHMM-t. Addresses peer-review item P4.4 (R2.W3): identifiability at K = 18 is asserted
# by appeal to Allman-Matias-Rhodes (2009) and Yakowitz-Spragins (1968), but the per-state
# nu_k histogram showing 13/18 states pinning at the upper bracket nu = 50 raises the
# question of how many of the 18 states are operationally distinct.
#
# Method: standardize each state's parameter triple, compute pairwise standardized
# distances, single-linkage cluster at threshold tau = 0.20 (in the standardized metric),
# report effective number of distinct clusters.
#
# Output: results/diagnostics/state_distinctness.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random
using LinearAlgebra
using Statistics
using Printf
using Distributions

const SEED      = 20260420;
const K_MAIN    = 18;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const TAU       = 0.20;  # standardized-distance threshold for cluster merging

const OUT_DIR = joinpath(_ROOT, "results", "diagnostics");
mkpath(OUT_DIR);

println("="^90);
println("  Effective number of distinct states at K = $K_MAIN  (peer-review P4.4 / R2.W3)");
println("="^90);

println("\n[setup] Loading SPY IS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
all_tickers = keys(filtered) |> collect |> sort;
all_R = log_growth_matrix(filtered, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = all_R[:, idx_spy];
println("  IS = $(length(R_is)) days");

println("\n[fit] Training CHMM-N and CHMM-t at K = $K_MAIN...");
Random.seed!(SEED);
chmm_n = build(MyContinuousHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
Random.seed!(SEED);
chmm_t = build(MyStudentTHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));

# ----------------------------------------------------------------------------------------- #
# Extract per-state parameter triples
# ----------------------------------------------------------------------------------------- #
function _stationary_pi(model, K::Int)
    T = zeros(K, K);
    for i in 1:K; T[i, :] = probs(model.transition[i]); end
    π̄ = (T^2000)[1, :];
    return π̄;
end

function _params_n(model::MyContinuousHiddenMarkovModel, K::Int)
    μ = zeros(K); σ = zeros(K);
    for k in 1:K
        d = model.emission[k];
        μ[k] = d.μ; σ[k] = d.σ;
    end
    return μ, σ;
end

function _params_t(model::MyStudentTHiddenMarkovModel, K::Int)
    μ = zeros(K); σ = zeros(K); ν = zeros(K);
    for k in 1:K
        d = model.emission[k];  # LocationScale{TDist}
        μ[k] = d.μ; σ[k] = d.σ;
        ν[k] = d.ρ.ν;            # underlying TDist degrees of freedom
    end
    return μ, σ, ν;
end

"""
    _single_linkage(D::Matrix, tau::Float64) -> labels::Vector{Int}

Single-linkage agglomerative clustering at distance threshold tau.
Returns cluster labels in 1:n_clusters.
"""
function _single_linkage(D::AbstractMatrix, tau::Float64)
    n = size(D, 1);
    labels = collect(1:n);
    for i in 1:n, j in (i+1):n
        if D[i, j] < tau
            old = labels[j]; new = labels[i];
            for r in 1:n
                if labels[r] == old; labels[r] = new; end
            end
        end
    end
    # Renumber 1..n_clusters
    uniq = sort(unique(labels));
    remap = Dict(uniq[k] => k for k in eachindex(uniq));
    return [remap[l] for l in labels];
end

# ----------------------------------------------------------------------------------------- #
# CHMM-N: 2D (μ_z, σ_z)
# ----------------------------------------------------------------------------------------- #
μn, σn = _params_n(chmm_n, K_MAIN);
π̄n = _stationary_pi(chmm_n, K_MAIN);
# Standardize features by their own (μ, σ) range
μn_z = (μn .- mean(μn)) ./ std(μn);
σn_z = (σn .- mean(σn)) ./ std(σn);
Dn = zeros(K_MAIN, K_MAIN);
for i in 1:K_MAIN, j in 1:K_MAIN
    Dn[i, j] = sqrt((μn_z[i] - μn_z[j])^2 + (σn_z[i] - σn_z[j])^2);
end
labels_n = _single_linkage(Dn, TAU);
n_distinct_n = length(unique(labels_n));

# ----------------------------------------------------------------------------------------- #
# CHMM-t: 3D (μ_z, σ_z, 1/ν_z) — using 1/ν because the heaviness axis is naturally
# parametrised in 1/ν (CHMM-t reduces to CHMM-N as 1/ν -> 0).
# ----------------------------------------------------------------------------------------- #
μt, σt, νt = _params_t(chmm_t, K_MAIN);
π̄t = _stationary_pi(chmm_t, K_MAIN);
inv_νt = 1.0 ./ νt;
μt_z = (μt .- mean(μt)) ./ std(μt);
σt_z = (σt .- mean(σt)) ./ std(σt);
inv_νt_z = (inv_νt .- mean(inv_νt)) ./ std(inv_νt);
Dt = zeros(K_MAIN, K_MAIN);
for i in 1:K_MAIN, j in 1:K_MAIN
    Dt[i, j] = sqrt((μt_z[i] - μt_z[j])^2 + (σt_z[i] - σt_z[j])^2 +
                    (inv_νt_z[i] - inv_νt_z[j])^2);
end
labels_t = _single_linkage(Dt, TAU);
n_distinct_t = length(unique(labels_t));

# ν bracket pinning summary
n_pinned_upper = count(ν -> ν >= 49.5, νt);   # upper bracket = 50
n_pinned_lower = count(ν -> ν <= 2.2, νt);    # lower bracket = 2.1

# ----------------------------------------------------------------------------------------- #
# Output
# ----------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "state_distinctness.txt");
open(out_path, "w") do io
    println(io, "="^90);
    println(io, "Effective number of distinct states at K = $K_MAIN on SPY IS");
    println(io, "Peer-review P4.4 / R2.W3: do all K = $K_MAIN states carry distinct (μ, σ, ν)?");
    println(io, "="^90);
    println(io, "Method: standardize each emission-parameter feature to z-score, compute pairwise");
    println(io, "        Euclidean distance in the standardized space, single-linkage cluster at");
    println(io, "        tau = $TAU. Effective state count = number of clusters.");
    println(io, "Seed:   $SEED");
    println(io);

    println(io, "-"^90);
    println(io, "CHMM-N (2D feature: (μ_k, σ_k))");
    println(io, "-"^90);
    @printf(io, "%-6s %-10s %-10s %-10s %-8s\n", "k", "μ_k", "σ_k", "π̄_k", "cluster");
    for k in 1:K_MAIN
        @printf(io, "%-6d %+10.4f %-10.4f %-10.4f %-8d\n",
                k, μn[k], σn[k], π̄n[k], labels_n[k]);
    end
    @printf(io, "Effective distinct states (CHMM-N): %d / %d\n", n_distinct_n, K_MAIN);
    println(io);

    println(io, "-"^90);
    println(io, "CHMM-t (3D feature: (μ_k, σ_k, 1/ν_k))");
    println(io, "-"^90);
    @printf(io, "%-6s %-10s %-10s %-10s %-10s %-8s\n",
            "k", "μ_k", "σ_k", "ν_k", "π̄_k", "cluster");
    for k in 1:K_MAIN
        @printf(io, "%-6d %+10.4f %-10.4f %-10.2f %-10.4f %-8d\n",
                k, μt[k], σt[k], νt[k], π̄t[k], labels_t[k]);
    end
    @printf(io, "Effective distinct states (CHMM-t): %d / %d\n", n_distinct_t, K_MAIN);
    @printf(io, "States pinned at ν_max = 50 : %d / %d\n", n_pinned_upper, K_MAIN);
    @printf(io, "States pinned at ν_min = 2.1: %d / %d\n", n_pinned_lower, K_MAIN);
    println(io);
    println(io, "Reading: under unpenalised CHMM-t, states pinned at the upper ν bracket are");
    println(io, "operationally Gaussian for the second-moment behaviour, but their (μ, σ) location");
    println(io, "may still distinguish them from each other. The effective-state count above");
    println(io, "captures both axes jointly.");
end

println("\n[done] Wrote $out_path");
println();
@printf("CHMM-N effective distinct states: %d / %d\n", n_distinct_n, K_MAIN);
@printf("CHMM-t effective distinct states: %d / %d  (ν-pinned upper: %d, lower: %d)\n",
        n_distinct_t, K_MAIN, n_pinned_upper, n_pinned_lower);
