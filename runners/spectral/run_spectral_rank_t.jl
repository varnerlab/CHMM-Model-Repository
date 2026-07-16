# ========================================================================================= #
# run_spectral_rank_t.jl
#
# Effective spectral rank of the absolute-return ACF identity (theory.tex eq. acf_normalised),
# on the HEADLINE shared-ν Student-t CHMM-t (ν≈5.81), at K = 2, 3 and 18 on SPY IS. Companion
# to run_spectral_rank.jl, which runs the same decomposition on the Gaussian CHMM-N. The
# decomposition (_spectral_modes) is emission-agnostic and copied verbatim; only the fit and the
# per-state m_k = E[|G| | k] / M_k = E[G² | k] sampling change (shared-ν Student-t emissions).
#
# The shared-ν fit is the canonical ecm_shared_nu from runners/headline/run_chmm_t_shared_nu.jl
# (a single ν across all K states, fit by ECM), copied verbatim so this runner is self-contained.
#
# Output: results/diagnostics/spectral_rank_t.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
include(joinpath(@__DIR__, "spectral_common.jl"));

using Printf
using Distributions

const SEED      = 20260420;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const N_M_DRAW  = 200_000;  # samples per state for m_k / M_k
const NU_BOUNDS = (2.1, 50.0);
const NU_INIT   = 6.0;

const OUT_DIR = joinpath(_ROOT, "results", "diagnostics");
mkpath(OUT_DIR);

println("="^80);
println("  Effective spectral rank of ρ_|G| on shared-ν CHMM-t at K = 18, 3 and 2");
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
# Shared-ν ECM (single ν across all K states), copied verbatim from
# runners/headline/run_chmm_t_shared_nu.jl. Returns (T, μ, σ, ν, π).
function _logsumexp(v::AbstractVector)
    m = maximum(v)
    isfinite(m) || return m
    return m + log(sum(exp.(v .- m)))
end

function ecm_shared_nu(obs::AbstractVector, K::Int;
        max_iter::Int=60, tol::Float64=1e-4,
        ν_init::Float64=NU_INIT, ν_bounds::Tuple{Float64,Float64}=NU_BOUNDS)
    N = length(obs)
    sorted_data = sort(obs)
    chunk = floor(Int, N/K)
    μ = zeros(K); σ = zeros(K)
    for s in 1:K
        a = (s-1)*chunk + 1
        b = (s == K) ? N : s*chunk
        μ[s] = mean(sorted_data[a:b])
        σ[s] = max(std(sorted_data[a:b]), 1e-6)
    end
    ν = ν_init
    T = ones(K, K) ./ K
    π = ones(K) ./ K

    function logpdf_t(x, μk, σk, νk)
        return logpdf(LocationScale(μk, σk, TDist(νk)), x)
    end

    function Q_shared(νv, γ, μ, σ, obs)
        acc = 0.0; n = length(obs); kk = size(γ, 2)
        for k in 1:kk
            d = LocationScale(μ[k], σ[k], TDist(νv))
            for t in 1:n
                acc += γ[t, k] * logpdf(d, obs[t])
            end
        end
        return acc
    end

    function gss_nu(γ, μ, σ, obs, lo, hi; iters=40)
        φ = (sqrt(5)-1)/2
        a = lo; b = hi
        c = b - φ*(b-a); d = a + φ*(b-a)
        fc = Q_shared(c, γ, μ, σ, obs); fd = Q_shared(d, γ, μ, σ, obs)
        for _ in 1:iters
            if fc > fd
                b = d; d = c; fd = fc
                c = b - φ*(b-a); fc = Q_shared(c, γ, μ, σ, obs)
            else
                a = c; c = d; fc = fd
                d = a + φ*(b-a); fd = Q_shared(d, γ, μ, σ, obs)
            end
        end
        return 0.5*(a+b)
    end

    last_μ = copy(μ); last_σ = copy(σ); last_ν = ν
    last_T = copy(T); last_π = copy(π)

    prev_ll = -Inf
    for _ in 1:max_iter
        # E-STEP
        log_B = zeros(N, K)
        for t in 1:N, k in 1:K
            log_B[t, k] = logpdf_t(obs[t], μ[k], σ[k], ν)
        end
        log_α = zeros(N, K)
        log_α[1, :] = log.(π) .+ log_B[1, :]
        for t in 2:N, j in 1:K
            log_α[t, j] = _logsumexp(log_α[t-1, :] .+ log.(T[:, j])) + log_B[t, j]
        end
        ll_now = _logsumexp(log_α[N, :])
        if !isfinite(ll_now)
            μ = last_μ; σ = last_σ; ν = last_ν; T = last_T; π = last_π
            break
        end
        last_μ = copy(μ); last_σ = copy(σ); last_ν = ν
        last_T = copy(T); last_π = copy(π)

        log_β = zeros(N, K)
        for t in N-1:-1:1, i in 1:K
            log_β[t, i] = _logsumexp(log.(T[i, :]) .+ log_B[t+1, :] .+ log_β[t+1, :])
        end

        γ = zeros(N, K)
        for t in 1:N
            denom = _logsumexp(log_α[t, :] .+ log_β[t, :])
            γ[t, :] = exp.((log_α[t, :] .+ log_β[t, :]) .- denom)
        end

        ξ_sum = zeros(K, K)
        for t in 1:N-1
            denom = _logsumexp(log_α[t, :] .+ log_β[t, :])
            for i in 1:K, j in 1:K
                ξ_sum[i, j] += exp(log_α[t, i] + log(T[i, j]) + log_B[t+1, j] + log_β[t+1, j] - denom)
            end
        end

        # Latent precisions with shared ν
        u = zeros(N, K)
        for t in 1:N, k in 1:K
            δ2 = ((obs[t] - μ[k]) / σ[k])^2
            u[t, k] = (ν + 1.0) / (ν + δ2)
        end

        # M-STEP
        π = γ[1, :]
        for k in 1:K
            wu = γ[:, k] .* u[:, k]
            Σwu = sum(wu); Σγ = sum(γ[:, k])
            if Σwu > 0
                μ[k] = sum(wu .* obs) / Σwu
            end
            if Σγ > 0
                σ2 = sum(wu .* (obs .- μ[k]).^2) / Σγ
                σ[k] = max(sqrt(max(σ2, 1e-12)), 1e-6)
            end
        end
        ν = gss_nu(γ, μ, σ, obs, ν_bounds[1], ν_bounds[2])

        for i in 1:K
            r = sum(ξ_sum[i, :])
            if r > 0
                T[i, :] = ξ_sum[i, :] ./ r
            end
        end

        if abs(ll_now - prev_ll) < tol; break; end
        prev_ll = ll_now
    end

    return (T=T, μ=μ, σ=σ, ν=ν, π=π)
end

# ----------------------------------------------------------------------------------------- #
"""
For a shared-ν CHMM-t fit (NamedTuple with fields T, μ, σ, ν, π), return:
  T   :: K×K transition matrix (rows sum to 1)
  π̄   :: K stationary vector
  m   :: K vector of per-state E[|G_t| | s_t = k]   (estimated by sampling)
  M   :: K vector of per-state E[G_t^2 | s_t = k]
The only difference from run_spectral_rank.jl's _T_pibar_m is the emission sampler:
here we sample the shared-ν location-scale Student-t instead of model.emission[k].
"""
function _T_pibar_m_t(fit; n_draw::Int = N_M_DRAW, seed::Int = 0)
    Random.seed!(seed);
    K = length(fit.μ);
    T = copy(fit.T);
    π̄ = (T^2000)[1, :];
    m = zeros(K); M = zeros(K);
    for k in 1:K
        d = LocationScale(fit.μ[k], fit.σ[k], TDist(fit.ν));
        s = [rand(d) for _ in 1:n_draw];
        m[k] = mean(abs.(s));
        M[k] = mean(s.^2);
    end
    return T, π̄, m, M;
end

# ----------------------------------------------------------------------------------------- #
function _run_at_K(K::Int)
    println("\n[fit] shared-ν CHMM-t at K = $K on SPY IS...");
    Random.seed!(SEED);
    fit = ecm_shared_nu(R_is, K; max_iter=MAX_ITER);
    println("  ν_shared = $(round(fit.ν, digits=4))");
    T, π̄, m, M = _T_pibar_m_t(fit; seed=SEED);
    σ²_G, comps, κ_V, recon_err = _spectral_components(T, π̄, m, M);
    s = _component_summary(comps);

    cum = 0.0;
    final_rows = NamedTuple[];
    for c in comps
        cum += c.mag_t1;
        push!(final_rows, merge(c, (cum_share = cum / s.budget,)));
    end

    return (K=K, nu=fit.ν, sigma2_G=σ²_G, comps=final_rows, cond_V=κ_V,
            recon_err=recon_err, s...);
end

results_18 = _run_at_K(18);
results_3  = _run_at_K(3);
results_2  = _run_at_K(2);   # K=2: rank-1 lower bound (single non-unit eigenvalue, mono-exponential ACF)

# ----------------------------------------------------------------------------------------- #
function _print_panel(io, r)
    println(io);
    println(io, "─"^110);
    println(io, "K = $(r.K)   ν_shared = $(round(r.nu, digits=4))   σ²_|G| = $(round(r.sigma2_G, digits=6))   ρ_|G|(1) = Σ_c contrib_c(1) = $(round(r.rho1, digits=4))   B = $(round(r.budget, digits=4))");
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

out_path = joinpath(OUT_DIR, "spectral_rank_t.txt");
open(out_path, "w") do io
    println(io, "="^90);
    println(io, "Effective spectral rank of the absolute-return ACF identity, shared-ν CHMM-t (headline)");
    println(io, "="^90);
    println(io, "Setup: shared-ν Student-t CHMM-t on SPY IS, seed = $SEED, n_draw = $N_M_DRAW per state for m_k.");
    println(io, "Identity: ρ_|G|(τ) = Σ_c contrib_c(τ) over grouped real components (conjugate pairs");
    println(io, "combined; third-review item 7). Shares are of the absolute component-magnitude");
    println(io, "budget B = Σ_c |contrib_c(1)|; B is not a percentage of ρ(1) itself.");
    println(io, "Companion to spectral_rank.txt (Gaussian CHMM-N). Fit: ecm_shared_nu (single ν across states).");
    _print_panel(io, results_18);
    _print_panel(io, results_3);
    _print_panel(io, results_2);
    println(io);
    println(io, "Reading note. Same identity and effective-rank definition as the CHMM-N run; only the");
    println(io, "emission family (shared-ν Student-t) differs.");
end

# stdout summary for quick capture
for r in (results_2, results_3, results_18)
    dom = r.comps[1];
    @printf("[K=%2d] ν=%.3f  ρ_|G|(1)=%.4f  dominant |λ|=%.4f  dom share=%.4f (%.1f%%)  recon=%.1e\n",
            r.K, r.nu, r.rho1, dom.abs_lam, r.dom_share, 100*r.dom_share, r.recon_err);
end
println("\n[done] $out_path");
