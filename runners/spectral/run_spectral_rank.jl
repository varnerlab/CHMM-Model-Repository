# ========================================================================================= #
# run_spectral_rank.jl
#
# Effective spectral rank diagnostic for the absolute-return ACF identity (theory.tex
# eq. acf_normalised). For CHMM-N at K = 18, 3, and 2 on SPY IS, group the non-unit
# eigenvalues of T̂ into real damped(-oscillatory) components (complex-conjugate pairs
# combined; third-review item 7), compute each component's signed real contribution
#
#   contrib_c(τ) = a_k λ_k^τ                      (real mode)
#   contrib_c(τ) = 2 Re(a_k λ_k^τ)                (conjugate pair)
#
# with a_k = (m' diag(π̄) v_k)(w_k' m) / σ²_|G|, and report the component-by-component
# share of the absolute component-magnitude budget B = Σ_c |contrib_c(1)|. B is not a
# percentage of ρ(1) itself (signed contributions can cancel); ρ(1) = Σ_c contrib_c(1)
# and the max reconstruction error of the spectral sum against the direct matrix formula
# over τ = 1..252 are reported alongside.
#
# Output: results/diagnostics/spectral_rank.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
include(joinpath(@__DIR__, "spectral_common.jl"));

using Printf

const SEED      = 20260420;
const MAX_ITER  = 4000;   # converged fits (sixth review, finding 2); tol 1e-4
const N_STARTS  = Dict(2 => 1, 3 => 3, 18 => 5);
const DT        = 1/252;
const RISK_FREE = 0.0;
const N_M_DRAW  = 200_000;  # samples per state for m_k (folded mean) estimate

const OUT_DIR = joinpath(_ROOT, "results", "diagnostics");
mkpath(OUT_DIR);

println("="^80);
println("  Effective spectral rank of ρ_|G| at K = 18, 3 and 2  (grouped components)");
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
function _run_at_K(K::Int)
    println("\n[fit] CHMM-N at K = $K on SPY IS (multistart, converged)...");
    Tm, μ, σ, πv, llh, γ, diags = baum_welch_multistart(collect(Float64, R_is), K;
        n_starts=get(N_STARTS, K, 3), max_iter=MAX_ITER, tol=1e-4, seed=SEED + K);
    d_best = diags[argmax([d.ll for d in diags])];
    @printf("  best start %d: n_evals = %d, final inc = %.2e, converged = %s\n",
            d_best.start, d_best.n_evals, d_best.final_increment, d_best.converged);
    # spectral inputs from the best fit (analytic folded-normal moments)
    T = Tm;
    π̄ = _stationary_pi(T);
    m = zeros(K); M = zeros(K);
    for k in 1:K
        z = μ[k] / σ[k];
        m[k] = σ[k] * sqrt(2 / π) * exp(-z^2 / 2) + μ[k] * (1 - 2 * cdf(Normal(), -z));
        M[k] = μ[k]^2 + σ[k]^2;
    end
    σ²_G, comps, κ_V, recon_err = _spectral_components(T, π̄, m, M);
    s = _component_summary(comps);

    cum = 0.0;
    final_rows = NamedTuple[];
    for c in comps
        cum += c.mag_t1;
        push!(final_rows, merge(c, (cum_share = cum / s.budget,)));
    end

    return (K=K, sigma2_G=σ²_G, comps=final_rows, cond_V=κ_V, recon_err=recon_err, s...);
end

results_18 = _run_at_K(18);
results_3  = _run_at_K(3);
results_2  = _run_at_K(2);   # K=2: rank-1 lower bound (single non-unit eigenvalue, mono-exponential ACF)

# ----------------------------------------------------------------------------------------- #
function _print_panel(io, r)
    println(io);
    println(io, "─"^110);
    println(io, "K = $(r.K)   σ²_|G| = $(round(r.sigma2_G, digits=6))   ρ_|G|(1) = Σ_c contrib_c(1) = $(round(r.rho1, digits=4))   B = Σ_c |contrib_c(1)| = $(round(r.budget, digits=4))");
    println(io, "cond(V) = $(round(r.cond_V, digits=1))   max reconstruction error vs direct matrix ACF (τ ≤ 252) = $(@sprintf("%.2e", r.recon_err))");
    println(io, "Components: $(length(r.comps)) grouped real components from $(r.K - 1) non-unit eigenvalues.");
    println(io, "Effective rank (share of the absolute component-magnitude budget B at lag 1):");
    println(io, "  $(r.n_above_1pct) components carry > 1% of B;");
    println(io, "  $(r.n_for_95) components carry ≥ 95% of cumulative B;");
    println(io, "  $(r.n_for_99) components carry ≥ 99% of cumulative B.");
    println(io, "─"^110);
    @printf(io, "%4s | %-5s | %-8s | %-8s | %-12s | %-12s | %-12s | %-12s | %-10s\n",
            "rank","kind","|λ|","θ","contrib(1)","contrib(5)","contrib(20)","contrib(50)","cum share");
    println(io, "-"^110);
    for (i, c) in enumerate(r.comps)
        @printf(io, "%4d | %-5s | %-8.4f | %-8.4f | %-+12.5f | %-+12.5f | %-+12.6f | %-+12.7f | %-10.4f\n",
                i, String(c.kind), c.abs_lam, c.theta,
                c.contrib.t1, c.contrib.t5, c.contrib.t20, c.contrib.t50,
                c.cum_share);
    end
end

out_path = joinpath(OUT_DIR, "spectral_rank.txt");
open(out_path, "w") do io
    println(io, "="^90);
    println(io, "Effective spectral rank of the absolute-return ACF identity  (grouped components)");
    println(io, "="^90);
    println(io, "Setup: CHMM-N on SPY IS, seed = $SEED, n_draw = $N_M_DRAW per state for m_k.");
    println(io, "Identity: ρ_|G|(τ) = Σ_c contrib_c(τ) over grouped real components (complex-");
    println(io, "conjugate eigenvalue pairs combined into single damped-oscillatory components;");
    println(io, "third-review item 7). Shares are of the absolute component-magnitude budget");
    println(io, "B = Σ_c |contrib_c(1)|, which upper-bounds |ρ(1)| and is NOT a percentage of");
    println(io, "ρ(1) itself: signed contributions can cancel. Components ranked by |contrib(1)|.");
    _print_panel(io, results_18);
    _print_panel(io, results_3);
    _print_panel(io, results_2);
    println(io);
    println(io, "Reading note. The body framing 'rank K-1 = 17 non-unit eigenvalues at K = 18'");
    println(io, "is an upper bound from the algebraic rank of T - 1 π̄^T. The empirical question");
    println(io, "is how many grouped components carry non-trivial absolute contribution. The");
    println(io, "'components for ≥ 95% of cumulative B' line above is the effective rank of");
    println(io, "ρ_|G|(τ) on this fit under that budget definition.");
end

println();
for r in (results_18, results_3, results_2)
    println("[K = $(r.K)] $(length(r.comps)) components: $(r.n_above_1pct) > 1%, ",
            "$(r.n_for_95) for 95%, $(r.n_for_99) for 99%;  recon err $(@sprintf("%.2e", r.recon_err))");
end
println();
println("[done] $out_path");
