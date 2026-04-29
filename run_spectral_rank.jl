# ========================================================================================= #
# run_spectral_rank.jl
#
# Effective spectral rank diagnostic for the absolute-return ACF identity (theory.tex
# eq. acf_normalised). For CHMM-N at K = 18 and at K = 3 on SPY IS, compute
#
#   c_k = (m' diag(π̄) v_k)(w_k' m),         w_k(norm) = c_k / σ²_|G|,
#   ρ_|G|(τ) = Σ_{k≥2} w_k(norm) λ_k^τ
#
# and report the mode-by-mode contribution |w_k(norm)| against |λ_k|. Addresses peer-review
# Tier-2 item B7: the body framing is "rank K - 1 = 17 non-unit eigenvalues at K = 18", but
# the empirical question is how many of those modes carry non-trivial weight.
#
# Output: results/diagnostics/spectral_rank.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using LinearAlgebra
using Statistics
using Printf

const SEED      = 20260420;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const N_M_DRAW  = 200_000;  # samples per state for m_k (folded mean) estimate

const OUT_DIR = joinpath(_ROOT, "results", "diagnostics");
mkpath(OUT_DIR);

println("="^80);
println("  Effective spectral rank of ρ_|G| at K = 18 and K = 3  (peer-review B7)");
println("="^80);

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

# ----------------------------------------------------------------------------------------- #
"""
For a fitted CHMM, return:
  T   :: K×K transition matrix (rows sum to 1)
  π̄   :: K stationary vector
  m   :: K vector of per-state E[|G_t| | s_t = k]   (estimated by sampling)
  M   :: K vector of per-state E[G_t^2 | s_t = k]
"""
function _T_pibar_m(model, K::Int; n_draw::Int = N_M_DRAW, seed::Int = 0)
    Random.seed!(seed);
    T = zeros(K, K);
    for i in 1:K; T[i, :] = probs(model.transition[i]); end
    π̄ = (T^2000)[1, :];
    m = zeros(K); M = zeros(K);
    for k in 1:K
        s = [rand(model.emission[k]) for _ in 1:n_draw];
        m[k] = mean(abs.(s));
        M[k] = mean(s.^2);
    end
    return T, π̄, m, M;
end

"""
Compute the spectral decomposition c_k = (m' diag(π̄) v_k)(w_k' m) for each non-unit
eigenvalue, plus σ²_|G| = E[G^2] - (E[|G|])² (the marginal variance of |G_t|).

Returns a table sorted by |λ_k| descending. The dominant (λ = 1) eigenvalue is dropped.
"""
function _spectral_modes(T::AbstractMatrix, π̄::AbstractVector, m::AbstractVector,
                          M::AbstractVector)
    K = length(m);
    F = eigen(T);
    λ = F.values;
    V = F.vectors;          # columns: right eigenvectors
    W = inv(V);             # rows:    left eigenvectors, with w_j' v_k = δ_jk

    σ²_G = π̄' * M - (π̄' * m)^2;

    # find the unit eigenvalue (within numerical tolerance)
    idx_one = argmin(abs.(λ .- 1.0));
    rest = setdiff(1:K, idx_one);

    rows = NamedTuple[];
    for k in rest
        v_k = V[:, k];
        w_k = W[k, :];
        c_k = (m' * Diagonal(π̄) * v_k) * dot(w_k, m);
        w_k_norm = c_k / σ²_G;
        push!(rows, (
            lambda  = λ[k],
            abs_lam = abs(λ[k]),
            c_k     = c_k,
            w_k     = w_k_norm,
            abs_w   = abs(w_k_norm),
            re_w    = real(w_k_norm),
            im_w    = imag(w_k_norm),
        ));
    end
    sort!(rows, by = r -> -r.abs_lam);
    return σ²_G, rows;
end

# ----------------------------------------------------------------------------------------- #
function _run_at_K(K::Int)
    println("\n[fit] CHMM-N at K = $K on SPY IS...");
    Random.seed!(SEED);
    m_n = build(MyContinuousHiddenMarkovModel,
                (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    T, π̄, m, M = _T_pibar_m(m_n, K; seed=SEED);
    σ²_G, rows = _spectral_modes(T, π̄, m, M);

    # Per-mode contribution to ρ(τ) at canonical lags. The right metric for
    # "effective rank of the ACF" is |w_k λ_k^τ|, not |w_k| alone, because a
    # mode with tiny |λ_k| has zero ACF contribution regardless of |w_k|.
    enriched = NamedTuple[];
    for r in rows
        contribs = (
            t1  = r.w_k * r.lambda^1,
            t5  = r.w_k * r.lambda^5,
            t20 = r.w_k * r.lambda^20,
            t50 = r.w_k * r.lambda^50,
        );
        push!(enriched, merge(r, contribs));
    end

    abs_t1_sum = sum(abs(r.t1) for r in enriched);
    sort!(enriched, by = r -> -abs(r.t1));   # rank by lag-1 contribution magnitude
    cum = 0.0;
    final_rows = NamedTuple[];
    for r in enriched
        cum += abs(r.t1);
        push!(final_rows, merge(r, (cum_t1_share = cum / abs_t1_sum,)));
    end

    n_for_95_t1 = findfirst(r -> r.cum_t1_share >= 0.95, final_rows);
    n_for_99_t1 = findfirst(r -> r.cum_t1_share >= 0.99, final_rows);
    n_above_1pct_t1 = count(r -> abs(r.t1) / abs_t1_sum > 0.01, final_rows);

    return (K=K, sigma2_G=σ²_G, abs_t1_sum=abs_t1_sum, rows=final_rows,
            n_above_1pct=n_above_1pct_t1, n_for_95=n_for_95_t1, n_for_99=n_for_99_t1);
end

results_18 = _run_at_K(18);
results_3  = _run_at_K(3);

# ----------------------------------------------------------------------------------------- #
function _print_panel(io, r)
    println(io);
    println(io, "─"^110);
    println(io, "K = $(r.K)   σ²_|G| = $(round(r.sigma2_G, digits=6))   ρ_|G|(1) = Σ w_k λ_k = $(round(real(sum(row.t1 for row in r.rows)), digits=4))");
    println(io, "Effective rank (lag-1 ACF contribution): ");
    println(io, "  $(r.n_above_1pct) modes carry > 1% of Σ|w_k λ_k|;");
    println(io, "  $(r.n_for_95) modes carry ≥ 95% of cumulative |w_k λ_k|;");
    println(io, "  $(r.n_for_99) modes carry ≥ 99% of cumulative |w_k λ_k|.");
    println(io, "─"^110);
    @printf(io, "%4s | %-12s | %-12s | %-10s | %-12s | %-12s | %-12s | %-10s\n",
            "rank","|λ_k|","|w_k|","|w_k λ_k|","Re(w_k λ_k^5)","Re(w_k λ_k^20)","Re(w_k λ_k^50)","cum lag-1");
    println(io, "-"^110);
    for (i, row) in enumerate(r.rows)
        @printf(io, "%4d | %-12.4f | %-12.5f | %-10.5f | %-12.5f | %-12.6f | %-12.7f | %-10.4f\n",
                i, row.abs_lam, abs(row.w_k),
                abs(row.t1), real(row.t5), real(row.t20), real(row.t50),
                row.cum_t1_share);
    end
end

out_path = joinpath(OUT_DIR, "spectral_rank.txt");
open(out_path, "w") do io
    println(io, "="^90);
    println(io, "Effective spectral rank of the absolute-return ACF identity  (peer-review B7)");
    println(io, "="^90);
    println(io, "Setup: CHMM-N on SPY IS, seed = $SEED, n_draw = $N_M_DRAW per state for m_k.");
    println(io, "Identity: ρ_|G|(τ) = Σ_{k=2}^K w_k λ_k^τ,  w_k = (m' diag(π̄) v_k)(w_k' m) / σ²_|G|");
    println(io, "(theory.tex eq. acf_normalised; v_k right and w_k left eigenvectors of T̂).");
    println(io, "Modes are ordered by |λ_k| descending. Complex eigenvalues come in conjugate pairs.");
    _print_panel(io, results_18);
    _print_panel(io, results_3);
    println(io);
    println(io, "Reading note. The body framing 'rank K-1 = 17 non-unit eigenvalues at K = 18'");
    println(io, "is an upper bound from the algebraic rank of T - 1 π̄^T. The empirical question");
    println(io, "is how many of those 17 modes carry non-trivial weight w_k. The 'modes for ≥ 95%");
    println(io, "of cumulative |w_k|' line above is the effective rank of ρ_|G|(τ) on this fit.");
end

# also stdout summary
println();
println("[K = 18] effective rank: $(results_18.n_above_1pct) modes > 1%, ",
        "$(results_18.n_for_95) modes ≥ 95% cum, ",
        "$(results_18.n_for_99) modes ≥ 99% cum");
println("[K =  3] effective rank: $(results_3.n_above_1pct) modes > 1%, ",
        "$(results_3.n_for_95) modes ≥ 95% cum, ",
        "$(results_3.n_for_99) modes ≥ 99% cum");
println();
println("[done] $out_path");
